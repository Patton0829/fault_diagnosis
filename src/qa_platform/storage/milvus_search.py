"""第十三步：对照学习NumPy内存检索与Milvus检索；默认只展示计划。"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import httpx
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document

from qa_platform.ingestion.railway import KNOWLEDGE_SOURCES
from qa_platform.providers.cloud_embedding import cosine_scores, embed_texts
from qa_platform.retrieval.dense import (
    DEFAULT_QUERY,
    embed_documents,
    rank_documents,
    validate_vectors,
)
from qa_platform.settings import ENV_PATH
from qa_platform.storage.milvus_build import (
    COLLECTION,
    DEFAULT_INDEX_DIR,
    OUTPUT_FIELDS,
    SCHEMA_VERSION,
    document_rows,
    embedding_contract,
    json_text,
    vector_hash,
)
from qa_platform.storage.milvus_smoke import (
    VERIFIED_VERSIONS,
    installed_versions,
    require,
)


FILTER = 'kb_id == "test" and is_test == true and review_status == "test_data"'
SCORE_TOLERANCE = 1e-5


def load_snapshot(index_dir: Path) -> tuple[dict, dict, list[Document]]:
    """只读本地清单；不复用构建期load_plan，以免代码改动误判旧索引失效。"""
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((index_dir / "build_plan.json").read_text(encoding="utf-8"))
    written = json.loads((index_dir / "build_result.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "ready", "索引尚未ready，不能用于检索。")
    require(
        manifest["verification"]
        == "cross_process_readback_and_document_vector_self_search",
        "索引验证方式不兼容。",
    )
    for key, expected in [
        ("schema_version", SCHEMA_VERSION),
        ("collection", COLLECTION),
        ("database", "knowledge.db"),
        ("kb_id", "test"),
    ]:
        require(manifest[key] == plan[key] == expected, f"索引{key}不兼容。")
    require(
        plan["status"] == "building" and written["status"] == "written_not_verified",
        "构建留痕格式不兼容。",
    )
    contract = embedding_contract()
    require(
        plan["embedding"] == contract, "索引与当前Embedding模型、地址或编码约定不兼容。"
    )
    require(
        manifest["embedding"]
        == {
            **contract,
            "dimension": written["dimension"],
            "dimension_source": "embedding_response",
        },
        "清单中的向量契约不一致。",
    )
    require(
        type(written["dimension"]) is int and written["dimension"] >= 2,
        "索引维度不合法。",
    )
    require(
        written["dimension_source"] == "embedding_response"
        and written["vector_hash_encoding"] == "little_endian_float32",
        "向量留痕格式不兼容。",
    )
    for key in ("document_count", "document_snapshot_sha256"):
        require(manifest[key] == plan[key] == written[key], f"索引{key}不一致。")
    for key in ("batch_size", "chunking", "knowledge_versions", "milvus_packages"):
        require(manifest[key] == plan[key], f"构建清单中的{key}不一致。")
    count = manifest["document_count"]
    require(
        type(count) is int and 0 < count <= 1000, "本教学对照只支持1到1000块的小索引。"
    )
    require(
        type(plan["batch_size"]) is int and 1 <= plan["batch_size"] <= 10,
        "编码批量大小不合法。",
    )
    documents = [
        Document(id=row["chunk_id"], page_content=row["text"], metadata=row["metadata"])
        for row in plan["documents"]
    ]
    rows = document_rows(documents)
    require(
        len(rows) == count and rows == plan["documents"],
        "文档快照数量、身份或字段不一致。",
    )
    require(
        sha256(json_text(rows).encode()).hexdigest()
        == manifest["document_snapshot_sha256"],
        "文档快照hash不一致。",
    )
    require(
        set(written["vector_sha256"]) == {row["chunk_id"] for row in rows},
        "向量指纹缺失或多余。",
    )
    source = KNOWLEDGE_SOURCES["test"]
    versions = {str(source): sha256(source.read_bytes()).hexdigest()}
    require(
        versions == manifest["knowledge_versions"],
        "当前知识库版本已变化；请另建新索引，不能混用旧索引。",
    )
    require(
        all(
            row["metadata"]["source"] == str(source)
            and row["metadata"]["document_version"] == versions[str(source)]
            for row in rows
        ),
        "来源版本与快照不一致。",
    )
    require(
        (index_dir / "knowledge.db").is_dir()
        and not (index_dir / "knowledge.db").is_symlink(),
        "已有Milvus数据库目录不存在或是软链接；拒绝自动新建。",
    )
    require(
        manifest["milvus_packages"] == installed_versions() == VERIFIED_VERSIONS,
        "Milvus依赖版本与已验证组合不一致。",
    )
    return manifest, written, documents


def read_vectors(
    client, manifest: dict, written: dict, documents: list[Document]
) -> np.ndarray:
    """付费请求前核验集合、正文、元数据及向量；全量读回仅服务于小库教学对照。"""
    from pymilvus import DataType

    require(bool(client.has_collection(COLLECTION)), "已建集合不存在。")
    description = client.describe_collection(collection_name=COLLECTION)
    fields = {field["name"]: field for field in description["fields"]}
    expected_types = {
        "chunk_id": DataType.VARCHAR,
        "text": DataType.VARCHAR,
        "vector": DataType.FLOAT_VECTOR,
        "kb_id": DataType.VARCHAR,
        "is_test": DataType.BOOL,
        "review_status": DataType.VARCHAR,
        "metadata": DataType.JSON,
    }
    require(
        set(fields) == set(expected_types)
        and all(fields[name]["type"] == kind for name, kind in expected_types.items()),
        "集合实际字段类型不兼容。",
    )
    require(
        description["auto_id"] is False
        and description["enable_dynamic_field"] is False,
        "集合身份或动态字段设置不兼容。",
    )
    require(fields["chunk_id"].get("is_primary") is True, "集合主键不正确。")
    require(
        int(fields["vector"]["params"]["dim"]) == written["dimension"],
        "集合实际维度与清单不一致。",
    )
    indexes = client.list_indexes(collection_name=COLLECTION)
    require(len(indexes) == 1, "向量索引数量不符合本步约定。")
    index = client.describe_index(collection_name=COLLECTION, index_name=indexes[0])
    require(
        index["field_name"] == "vector"
        and index["index_type"] == "FLAT"
        and index["metric_type"] == "COSINE",
        "集合实际索引不是约定的FLAT/COSINE。",
    )
    client.load_collection(collection_name=COLLECTION)
    require(
        int(client.get_collection_stats(collection_name=COLLECTION)["row_count"])
        == len(documents),
        "集合记录数不一致。",
    )
    ids = [document.metadata["chunk_id"] for document in documents]
    records = client.get(
        collection_name=COLLECTION, ids=ids, output_fields=OUTPUT_FIELDS
    )
    by_id = {record["chunk_id"]: record for record in records}
    require(
        len(records) == len(by_id) == len(ids) and set(by_id) == set(ids),
        "集合记录主键不一致。",
    )
    vectors = []
    for expected in document_rows(documents):
        actual = by_id[expected["chunk_id"]]
        require(
            all(actual[field] == value for field, value in expected.items()),
            "数据库正文、元数据或测试标记与快照不一致。",
        )
        vector = np.asarray(actual["vector"], dtype=np.float32)
        require(
            vector_hash(vector) == written["vector_sha256"][expected["chunk_id"]],
            "已保存向量的指纹不一致。",
        )
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=np.float32)
    validate_vectors(matrix, len(documents), manifest["embedding"]["dimension"])
    return matrix


def search_milvus(
    client, documents: list[Document], query: np.ndarray, top_k: int
) -> list[tuple[Document, float]]:
    """只search，不insert；返回完整Document，后续可接回原有RAG。"""
    results = client.search(
        collection_name=COLLECTION,
        data=[query.tolist()],
        anns_field="vector",
        limit=top_k,
        filter=FILTER,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=[field for field in OUTPUT_FIELDS if field != "vector"],
    )
    require(
        len(results) == 1 and len(results[0]) == top_k, "Milvus返回数量不符合预期。"
    )
    by_id = {document.metadata["chunk_id"]: document for document in documents}
    hits, seen = [], set()
    for hit in results[0]:
        chunk_id = hit["chunk_id"]  # 本集合的实际主键不是示例中的id。
        require(chunk_id in by_id and chunk_id not in seen, "检索返回未知或重复的块。")
        expected = document_rows([by_id[chunk_id]])[0]
        require(
            all(hit["entity"][field] == value for field, value in expected.items()),
            "检索返回的原文或元数据不一致。",
        )
        score = float(hit["distance"])  # COSINE时越大越相似，不是越小越好。
        require(
            np.isfinite(score) and -1 - SCORE_TOLERANCE <= score <= 1 + SCORE_TOLERANCE,
            "检索分数不合法。",
        )
        hits.append(
            (
                Document(
                    id=chunk_id,
                    page_content=hit["entity"]["text"],
                    metadata=hit["entity"]["metadata"],
                ),
                score,
            )
        )
        seen.add(chunk_id)
    require(
        all(
            hits[i][1] >= hits[i + 1][1] - SCORE_TOLERANCE for i in range(len(hits) - 1)
        ),
        "检索结果没有按分数降序返回。",
    )
    return hits


def timed_search(search, repeats: int) -> tuple[list, float, float]:
    """首轮单列，后续取中位数；不把初始化或编码时间塞进搜索时间。"""
    started = perf_counter()
    hits = search()
    first = perf_counter() - started
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        search()
        samples.append(perf_counter() - started)
    return hits, first, median(samples)


def validate_search(
    hits: list,
    documents: list[Document],
    vectors: np.ndarray,
    query: np.ndarray,
    top_k: int,
) -> None:
    """允许第K名边界同分替换，但不能遗漏严格高于边界的知识块。"""
    scores = cosine_scores(query, vectors)
    expected = {doc.id: float(score) for doc, score in zip(documents, scores)}
    returned = {doc.id for doc, _ in hits}
    require(len(hits) == len(returned) == top_k, "Milvus的Top-K数量或身份不合法。")
    require(
        all(
            doc.id in expected and abs(score - expected[doc.id]) <= SCORE_TOLERANCE
            for doc, score in hits
        ),
        "Milvus与内存的同块分差超出容差。",
    )
    cutoff = float(np.sort(scores)[-top_k])
    required_ids = {
        chunk_id
        for chunk_id, score in expected.items()
        if score > cutoff + SCORE_TOLERANCE
    }
    require(required_ids <= returned, "Milvus遗漏了严格高于Top-K边界的候选。")
    require(
        min(score for _, score in hits) >= cutoff - SCORE_TOLERANCE,
        "Milvus返回了低于Top-K边界的候选。",
    )


def print_comparison(left: list, right: list, left_name: str, right_name: str) -> None:
    """按名次并排展示；分差另外按chunk_id对齐，不能按名次相减。"""
    print(f"名次 | {left_name:<32} | {right_name}")
    for rank, ((a, sa), (b, sb)) in enumerate(zip(left, right), 1):
        a_label = a.metadata["entry_id"] or a.metadata["title"]
        b_label = b.metadata["entry_id"] or b.metadata["title"]
        print(f"{rank:>4} | {a_label:<16} {sa:.7f}       | {b_label:<16} {sb:.7f}")
    a_scores = {doc.id: score for doc, score in left}
    b_scores = {doc.id: score for doc, score in right}
    common = set(a_scores) & set(b_scores)
    ordered = [doc.id for doc, _ in left] == [doc.id for doc, _ in right]
    print(
        f"[对照] Top-K交集 {len(common)}/{len(left)}；顺序相同：{'是' if ordered else '否'}"
    )
    if common:
        delta = max(abs(a_scores[key] - b_scores[key]) for key in common)
        print(f"[对照] 共同块按chunk_id对齐的最大分差：{delta:.9g}")


def run_comparison(
    args: argparse.Namespace, manifest: dict, written: dict, documents: list[Document]
) -> None:
    # 先确认已有库可读，再加载密钥；此脚本没有建集合、插入、更新或删除语句。
    started = perf_counter()
    from pymilvus import MilvusClient

    client = MilvusClient(uri=str(args.index_dir / "knowledge.db"), timeout=20)
    try:
        opened = perf_counter() - started
        started = perf_counter()
        saved_vectors = read_vectors(client, manifest, written, documents)
        readback = perf_counter() - started
        print(
            f"[校验] 已读回并核验{len(documents)}条原文、元数据及float32向量；未重新编码文档。",
            flush=True,
        )
        require(ENV_PATH.is_file(), "models/.env不存在。")
        load_dotenv(ENV_PATH, override=False, encoding="utf-8")
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        require(bool(api_key), "未配置DASHSCOPE_API_KEY；不显示密钥。")
        fresh_vectors = None
        document_time = None
        with httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        ) as http_client:
            print(
                "[请求] 只编码一次问题，供所有对照路线共享：text_type=query", flush=True
            )
            started = perf_counter()
            with np.errstate(over="ignore", invalid="ignore"):
                query_matrix = embed_texts(http_client, [args.query], "query").astype(
                    np.float32
                )
            validate_vectors(query_matrix, 1, written["dimension"])
            query = query_matrix[0]
            query_time = perf_counter() - started
            if args.fresh_baseline:
                print(
                    "[额外实验] 重新编码文档，模拟旧内存流程；这些向量不写入数据库。",
                    flush=True,
                )
                started = perf_counter()
                with np.errstate(over="ignore", invalid="ignore"):
                    fresh_vectors = embed_documents(
                        http_client,
                        documents,
                        manifest["batch_size"],
                        written["dimension"],
                    ).astype(np.float32)
                validate_vectors(fresh_vectors, len(documents), written["dimension"])
                document_time = perf_counter() - started

        top_k = min(args.top_k, len(documents))
        memory, memory_first, memory_warm = timed_search(
            lambda: rank_documents(documents, saved_vectors, query, top_k), args.repeat
        )
        milvus, milvus_first, milvus_warm = timed_search(
            lambda: search_milvus(client, documents, query, top_k), args.repeat
        )
        validate_search(milvus, documents, saved_vectors, query, top_k)
        print("\n[实验一：只改变搜索实现，固定同一问题向量与同一份已保存文档向量]")
        print_comparison(memory, milvus, "NumPy内存COSINE", "Milvus FLAT/COSINE")
        print(
            f"[搜索首轮] NumPy={memory_first * 1000:.3f} ms；Milvus={milvus_first * 1000:.3f} ms"
        )
        print(
            f"[后续{args.repeat}轮中位数] NumPy={memory_warm * 1000:.3f} ms；Milvus={milvus_warm * 1000:.3f} ms"
        )
        print(
            "[计时口径] 客户端搜索及校验；Milvus含RPC、结果重建与原文/元数据检查，NumPy含矩阵计算与向量检查。"
        )
        print(
            "[计时边界] NumPy先测、Milvus后测；数据已读入内存，不是冷启动、等工作量内核基准或大规模性能结论。"
        )
        print(
            f"[准备耗时] SDK导入/打开客户端={opened:.3f} s；集合检查/全量读回校验={readback:.3f} s；共享问题编码={query_time:.3f} s"
        )
        print(
            "[实验边界] NumPy使用从Milvus读回的向量，仅隔离搜索实现；不是独立的无数据库持久化方案。"
        )
        print(
            "[判断] 同块分数及Top-K边界在容差内一致；同分/浮点误差可影响次序，重合度不是相关性准确率。"
        )

        if fresh_vectors is not None:
            print("\n[实验二：改变文档准备方式，旧流程重新编码 vs 复用已建索引]")
            print(
                "[控制说明] 两路统一使用float32；此处模拟旧流程的重编码方式，不是逐项复刻旧脚本的计时。"
            )
            started = perf_counter()
            fresh_hits = rank_documents(documents, fresh_vectors, query, top_k)
            fresh_search = perf_counter() - started
            print_comparison(
                fresh_hits, milvus, "旧流程：重新编码后内存检索", "复用索引：Milvus检索"
            )
            drift = float(
                np.max(np.abs(fresh_vectors.astype(np.float64) - saved_vectors))
            )
            print(
                f"[向量变化] 重新编码与入库向量的最大分量差={drift:.9g}；相同模型名称不保证服务版本不变。"
            )
            print(
                f"[旧流程准备] 本次实测文档编码={document_time:.3f} s；编码后内存搜索={fresh_search * 1000:.3f} ms"
            )
            print(
                "[解释] 本实验不固定文档向量；若排名变化，不能直接归因于Milvus。上面的实验一才用于隔离搜索实现。"
            )

        document_requests = (len(documents) + manifest["batch_size"] - 1) // manifest[
            "batch_size"
        ]
        actual_document_requests = document_requests if args.fresh_baseline else 0
        print("\n[请求次数：单独执行一条路线时的计划，不含失败重试]")
        print(
            f"旧脚本不缓存：文档{document_requests}次 + 问题1次 = {document_requests + 1}次Embedding/提问"
        )
        print("已有Milvus索引：文档0次 + 问题1次 = 1次Embedding/提问")
        print(
            f"[本轮实际] 文档{actual_document_requests}次 + 共享问题1次 = {actual_document_requests + 1}次；Reranker/聊天模型0次。"
        )
        print(
            "[历史成本] 建库时已做过文档编码；内容或模型变更后仍需更新索引，不是永久零文档费用。"
        )
        print(
            "[归因] 节省重复编码来自持久化复用，并非Milvus独有；NumPy配合文件缓存也能做到，但需自行管理。"
        )
        print("[结论边界] 小库/FLAT不预设提速或更准；当前没有问答生成，也未替换原RAG。")
        print("\n[Milvus候选：测试资料，仅用于学习，不是现场维护规程]")
        for rank, (document, score) in enumerate(milvus, 1):
            metadata = document.metadata
            print(
                f"{rank}. {metadata['entry_id']} | COSINE={score:.7f} | chunk_id={document.id}"
            )
            print(
                f"   来源：{metadata['source']}；父记录第{metadata['line_start']}–{metadata['line_end']}行；is_test=True"
            )
            print(f"   标题：{metadata['title']}")
        print(
            "[成功] 对照检索完成；未插入、更新、删除记录，未写报告；本地数据库运行时可能更新锁或日志。"
        )
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Milvus与内存检索的教学对照；默认只展示计划"
    )
    parser.add_argument(
        "--run", action="store_true", help="读取已有索引并请求一次问题Embedding"
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="首次搜索之后的本地重复次数；不增加模型请求",
    )
    parser.add_argument(
        "--fresh-baseline",
        action="store_true",
        help="另做旧流程实测：重新编码全部文档，额外产生API请求",
    )
    args = parser.parse_args()
    if not args.query.strip() or args.top_k <= 0 or not 1 <= args.repeat <= 100:
        parser.error("问题不能为空，top-k须大于0，repeat须在1到100之间")
    args.index_dir = args.index_dir.expanduser().absolute()
    try:
        manifest, written, documents = load_snapshot(args.index_dir)
        document_requests = (len(documents) + manifest["batch_size"] - 1) // manifest[
            "batch_size"
        ]
        extra = document_requests if args.fresh_baseline else 0
        print(f"[配置] Python：{sys.executable}")
        print(
            f"[索引] {args.index_dir}；ready；{len(documents)}块；{written['dimension']}维；float32；FLAT/COSINE"
        )
        print(
            f"[模型] {manifest['embedding']['model']}；服务版本未固定；维度相同不代表不同模型兼容。"
        )
        print(f"[问题] {args.query}")
        print(
            f"[控制条件] 同一问题向量、已保存文档向量、Top-K={min(args.top_k, len(documents))}；过滤={FILTER}"
        )
        print(
            "[实验一] NumPy内存计算 vs Milvus搜索；NumPy对照向量从已有数据库读回，不重编码。"
        )
        print(
            f"[实验二] {'启用：重新编码文档，对比旧流程，额外产生费用。' if args.fresh_baseline else '未启用；加--fresh-baseline可实测旧流程重新编码文档的开销。'}"
        )
        print(
            f"[请求计划] 问题1次 + 文档{extra}次 = {extra + 1}次Embedding；所有路线共享问题向量。"
        )
        print(f"[搜索计划] 每路首次1次 + 后续{args.repeat}次；重复搜索不调用模型。")
        print("[边界] 仅测试知识库；分数不是故障概率；不预设Milvus更准或更快。")
        if not args.run:
            print(
                "[计划完成] 已核对本地清单与知识版本；未读取.env、未导入Milvus SDK、未创建客户端或文件；加--run才执行。"
            )
            return 0
        run_comparison(args, manifest, written, documents)
    except httpx.TimeoutException:
        print("[错误] Embedding请求超时。", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(
            f"[错误] Embedding返回HTTP {exc.response.status_code}；不显示响应体。",
            file=sys.stderr,
        )
        return 1
    except httpx.RequestError:
        print("[错误] Embedding连接失败；不显示请求信息。", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[错误] {exc}；未展示未经校验的检索结论。", file=sys.stderr)
        return 1
    except (OSError, TypeError, KeyError, OverflowError):
        print(
            "[错误] 本地清单、知识版本、数据库或向量校验未通过；不展示未经校验的检索结论。",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"[错误] {type(exc).__name__}：数据库操作未完成；不显示请求或服务响应。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
