"""Kubernetes真实问句的NumPy/Milvus对照；默认纯离线计划。

--run才编码一个问题：GET模型身份→POST问题Embedding→GET模型身份。
文档向量不重编码；不读.env、不回退云端、不写报告或数据库记录。
"""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

from qa_platform.ingestion.embedding_plan import canonical, request_body, require
from qa_platform.providers.local_embedding import (
    http_json,
    local_endpoint_base,
    model_digest,
    model_entry,
    validate_embeddings,
)
from qa_platform.storage import kubernetes_milvus as build


DEFAULT_QUERY = "Pod里访问服务域名时DNS解析失败，应该如何排查？"
SCORE_TOLERANCE = 1e-5
MAX_DOCUMENTS = 5000  # 有界的本机学习范围，不是Milvus容量上限。
MAX_VECTOR_BYTES = 128 * 1024 * 1024  # 仅原始float32矩阵，不是进程总内存上限。
READ_BATCH_SIZE = 128
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


def _check_capacity(plan: dict) -> tuple[int, int]:
    require(isinstance(plan, dict) and isinstance(plan.get("embedding"), dict), "索引容量描述结构无效")
    count, dimension = plan.get("document_count"), plan["embedding"].get("actual_dimension")
    require(type(count) is int and 1 <= count <= MAX_DOCUMENTS,
            f"本机学习入口只支持1～{MAX_DOCUMENTS}块的索引")
    require(type(dimension) is int and 2 <= dimension <= 8192
            and count * dimension * 4 <= MAX_VECTOR_BYTES,
            "索引向量维度或原始float32矩阵超过128MiB读取预算")
    return count, dimension


def load_ready(index_dir: Path) -> tuple[dict, dict]:
    """只读校验清单、冻结计划与缓存；不打开Milvus。"""
    require(
        index_dir.is_dir() and not index_dir.is_symlink(), "索引目录不存在或是软链接"
    )
    for name in (
        "manifest.json",
        "build_plan.json",
        "build_result.json",
        "verification.json",
        "model_observation.json",
    ):
        path = index_dir / name
        require(path.is_file() and not path.is_symlink(), f"缺少正常验收文件：{name}")
    manifest_path = index_dir / "manifest.json"
    require(0 < manifest_path.stat().st_size <= MAX_MANIFEST_BYTES, "索引清单大小超出读取范围")
    manifest = json.loads(manifest_path.read_text())
    early_scope = _check_capacity(manifest)  # 在加载SQLite向量前先拒绝超范围清单。
    plan, bundle = build.load_build(index_dir)
    require(
        _check_capacity(plan) == early_scope and len(plan["documents"]) == plan["document_count"],
        "索引清单、冻结计划的容量或维度不一致",
    )
    written = json.loads((index_dir / "build_result.json").read_text())
    verified = json.loads((index_dir / "verification.json").read_text())
    observed = json.loads((index_dir / "model_observation.json").read_text())
    require(json.loads(manifest_path.read_text()) == manifest, "读取期间索引清单发生变化")
    require(
        written["status"] == "written_not_verified"
        and verified["status"] == "verified",
        "构建/验收回执状态不符",
    )
    require(
        all(
            type(verified[key]) is int and verified[key] > 0
            for key in ("reader_pid", "writer_pid")
        )
        and verified["writer_pid"] == written["writer_pid"]
        and verified["reader_pid"] != verified["writer_pid"],
        "缺少有效的跨进程验收记录",
    )
    for key in ("document_count", "document_snapshot_sha256", "vector_snapshot_sha256"):
        require(written[key] == verified[key] == plan[key], f"验收{key}与冻结计划不符")
    delta = verified["max_cosine_difference"]
    require(
        type(delta) in (int, float)
        and isfinite(delta)
        and 0 <= delta <= SCORE_TOLERANCE,
        "历史验收分差不合法",
    )
    digest = model_digest({"digest": plan["embedding"]["server_model_digest"]})
    require(
        observed["observed_digest"] == digest
        and type(observed["http_requests"]) is int
        and observed["http_requests"] == 1,
        "入库时的模型身份观测不符",
    )
    expected = {key: value for key, value in plan.items() if key != "documents"}
    expected.update(status="ready", verification=verified, model_observation=observed)
    require(manifest == expected, "ready清单与构建计划/验收记录不一致")
    database = index_dir / build.DATABASE
    require(
        database.is_dir() and not database.is_symlink(),
        "已有Milvus数据库目录缺失，不会自动建库",
    )
    return plan, bundle


def describe_index(index_dir: Path) -> dict:
    """从已核验本地快照生成页面摘要；不打开Milvus或请求模型，不外露本机路径。"""
    index_dir = Path(index_dir).expanduser().absolute()
    plan, _ = load_ready(index_dir)
    return {
        "index_name": index_dir.name, "document_count": plan["document_count"],
        "parent_count": len({row["parent_id"] for row in plan["documents"]}),
        "dimension": plan["embedding"]["actual_dimension"],
        "notice": "资料尚未逐条审核，包含参考数据页。恢复的历史可能来自旧索引，未重新检索；新提问才使用当前知识库。",
    }


def prepare_search(
    index_dir: Path,
    query: str = DEFAULT_QUERY,
    top_k: int = 3,
    parent_id: str | None = None,
) -> dict:
    require(
        isinstance(query, str) and query.strip() and len(query) <= 2000,
        "问题须为1～2000字符；字符限制不等于Token限制",
    )
    require(type(top_k) is int and 1 <= top_k <= 50, "top-k须为1～50")
    plan, bundle = load_ready(index_dir)
    profile = plan["embedding"]
    base_url = local_endpoint_base(profile)
    require(
        profile["document_mode"] == "body-only"
        and profile["query_format"] == "Instruct: {instruction}\nQuery: {query}"
        and profile["truncate"] is False,
        "问题/文档输入合同不兼容",
    )
    commits = {row["snapshot_commit"] for row in plan["documents"]}
    require(len(commits) == 1, "本步只支持同一个上游commit的索引")
    commit = next(iter(commits))
    selected = [
        i
        for i, row in enumerate(plan["documents"])
        if parent_id is None or row["parent_id"] == parent_id
    ]
    require(bool(selected), "parent-id未命中当前索引；不发送模型请求")
    expression = f'kb_id == "kubernetes_zh" and snapshot_commit == {canonical(commit)}'
    if parent_id is not None:
        expression += f" and parent_id == {canonical(parent_id)}"
    query_input = profile["query_format"].format(
        instruction=profile["query_instruction"], query=query
    )
    return {
        "index_dir": index_dir,
        "plan": plan,
        "bundle": bundle,
        "records": [plan["documents"][i] for i in selected],
        "vectors": [bundle["vectors"][i] for i in selected],
        "query": query,
        "query_input": query_input,
        "base_url": base_url,
        "filter": expression,
        "top_k": min(top_k, len(selected)),
    }


def read_index(client, plan: dict):
    """先验实际集合及全部原文/元数据/向量，验证完成后才请求问题编码。"""
    import numpy as np
    from pymilvus import DataType

    _check_capacity(plan)
    require(client.has_collection(build.COLLECTION), "已有集合不存在，不会自动创建")
    description = client.describe_collection(collection_name=build.COLLECTION)
    fields = {field["name"]: field for field in description["fields"]}
    expected_types = {
        "chunk_id": DataType.VARCHAR,
        "text": DataType.VARCHAR,
        "vector": DataType.FLOAT_VECTOR,
        "metadata": DataType.JSON,
        **{field: DataType.VARCHAR for field in build.SCALARS},
    }
    require(
        set(fields) == set(expected_types)
        and all(fields[name]["type"] == kind for name, kind in expected_types.items())
        and fields["chunk_id"].get("is_primary") is True
        and description["auto_id"] is False
        and description["enable_dynamic_field"] is False
        and int(fields["vector"]["params"]["dim"])
        == plan["embedding"]["actual_dimension"],
        "实际集合schema与清单不符",
    )
    names = client.list_indexes(collection_name=build.COLLECTION)
    require(len(names) == 1, "实际索引数量不符")
    actual = client.describe_index(
        collection_name=build.COLLECTION, index_name=names[0]
    )
    require(
        actual["field_name"] == "vector"
        and actual["index_type"] == "FLAT"
        and actual["metric_type"] == "COSINE",
        "实际索引不是FLAT/COSINE",
    )
    client.load_collection(collection_name=build.COLLECTION)
    require(
        int(client.get_collection_stats(build.COLLECTION)["row_count"])
        == plan["document_count"],
        "集合记录数不符",
    )
    ids = [row["chunk_id"] for row in plan["documents"]]
    require(len(ids) == plan["document_count"] and len(ids) == len(set(ids)), "计划主键数量或唯一性不符")
    vectors = []
    # 小批读回并立即校验，不同时保留全部JSON浮点列表，降低瞬时内存峰值。
    for start in range(0, len(ids), READ_BATCH_SIZE):
        batch_ids = ids[start : start + READ_BATCH_SIZE]
        records = client.get(collection_name=build.COLLECTION, ids=batch_ids, output_fields=build.OUTPUT_FIELDS)
        by_id = {row["chunk_id"]: row for row in records}
        require(
            len(records) == len(by_id) == len(batch_ids) and set(by_id) == set(batch_ids),
            "读回主键不符",
        )
        for expected in plan["documents"][start : start + READ_BATCH_SIZE]:
            actual = by_id[expected["chunk_id"]]
            require(all(actual.get(key) == value for key, value in expected.items()),
                    "数据库原文或元数据与快照不符")
            vector = np.asarray(actual["vector"], dtype="<f4")
            require(vector.shape == (plan["embedding"]["actual_dimension"],), "读回维度不符")
            require(sha256(vector.tobytes()).hexdigest() == plan["vector_sha256"][expected["chunk_id"]],
                    "读回向量指纹不符")
            vectors.append(vector)
    return np.asarray(vectors, dtype="<f4")


def cosine_scores(vectors, query):
    import numpy as np

    matrix, question = (
        np.asarray(vectors, dtype=np.float64),
        np.asarray(query, dtype=np.float64),
    )
    require(
        matrix.ndim == 2
        and matrix.shape[0] > 0
        and question.shape == (matrix.shape[1],),
        "问题与文档向量形状不符",
    )
    require(np.isfinite(matrix).all() and np.isfinite(question).all(), "向量含非有限数")
    lengths, length = np.linalg.norm(matrix, axis=1), float(np.linalg.norm(question))
    require(
        np.isfinite(lengths).all()
        and isfinite(length)
        and (lengths > 0).all()
        and length > 0,
        "向量为零或范数不合法",
    )
    return matrix @ question / (lengths * length)


def rank_numpy(records: list, vectors, query, top_k: int) -> list:
    scores = cosine_scores(vectors, query)
    require(
        len(records) == len(scores) and 1 <= top_k <= len(records),
        "NumPy候选数/top-k不合法",
    )
    ranked = sorted(
        range(len(records)), key=lambda i: (-float(scores[i]), records[i]["chunk_id"])
    )[:top_k]
    return [
        {
            "chunk_id": records[i]["chunk_id"],
            "score": float(scores[i]),
            "record": records[i],
        }
        for i in ranked
    ]


def search_milvus(client, records: list, query, top_k: int, expression: str) -> list:
    results = client.search(
        collection_name=build.COLLECTION,
        data=[query.tolist()],
        anns_field="vector",
        limit=top_k,
        filter=expression,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=[field for field in build.OUTPUT_FIELDS if field != "vector"],
    )
    require(len(results) == 1 and len(results[0]) == top_k, "Milvus返回数量不符")
    by_id = {row["chunk_id"]: row for row in records}
    hits, seen = [], set()
    for item in results[0]:
        chunk_id = item["chunk_id"]
        require(chunk_id in by_id and chunk_id not in seen, "返回未知、越界或重复候选")
        expected = by_id[chunk_id]
        require(
            all(item["entity"].get(key) == value for key, value in expected.items()),
            "命中正文或元数据不符",
        )
        score = float(item["distance"])
        require(
            isfinite(score) and -1 - SCORE_TOLERANCE <= score <= 1 + SCORE_TOLERANCE,
            "返回余弦分数不合法",
        )
        hits.append({"chunk_id": chunk_id, "score": score, "record": expected})
        seen.add(chunk_id)
    require(
        all(
            hits[i]["score"] >= hits[i + 1]["score"] - SCORE_TOLERANCE
            for i in range(len(hits) - 1)
        ),
        "Milvus结果未按分数降序排列",
    )
    return hits


def validate_top_k(hits: list, records: list, vectors, query, top_k: int) -> dict:
    """用全量余弦作参考，允许K边界同分互换，但不允许漏掉更高分候选。"""
    import numpy as np

    scores = cosine_scores(vectors, query)
    require(
        len(records) == len(scores)
        and type(top_k) is int
        and 1 <= top_k <= len(records),
        "Top-K校验的候选数/数量不合法",
    )
    expected = {row["chunk_id"]: float(score) for row, score in zip(records, scores)}
    ids = {hit["chunk_id"] for hit in hits}
    require(
        len(hits) == len(ids) == top_k and ids <= set(expected), "Top-K主键/数量不符"
    )
    require(all(isfinite(hit["score"]) for hit in hits), "Top-K分数含非有限数")
    delta = max(abs(hit["score"] - expected[hit["chunk_id"]]) for hit in hits)
    require(delta <= SCORE_TOLERANCE, "同块余弦分数差超出容差")
    cutoff = float(np.sort(scores)[-top_k])
    required = {
        key for key, score in expected.items() if score > cutoff + SCORE_TOLERANCE
    }
    require(required <= ids, "Milvus遗漏了严格高于Top-K边界的候选")
    require(
        all(expected[key] >= cutoff - SCORE_TOLERANCE for key in ids),
        "Milvus返回了低于Top-K边界的候选",
    )
    return {"max_score_difference": delta, "cutoff": cutoff}


def timed_search(search, repeat: int) -> tuple:
    started = perf_counter()
    hits = search()
    first = perf_counter() - started
    samples = []
    for _ in range(repeat):
        started = perf_counter()
        search()
        samples.append(perf_counter() - started)
    return hits, first, median(samples)


def encode_query(check: dict, timeout: int = 180, session_factory=None):
    """同一个问题只编码一次；版本不符、返回异常时不重试也不展示候选。"""
    if session_factory is None:
        import requests

        session_factory = requests.Session
    profile = check["plan"]["embedding"]
    stats = {"embedding_requests": 0, "http_requests": 0}
    expected = profile["server_model_digest"]
    with session_factory() as session:
        session.trust_env = False

        def digest():
            stats["http_requests"] += 1
            print(
                f"[元数据] HTTP第{stats['http_requests']}次：GET /api/tags核对模型digest。",
                flush=True,
            )
            return model_digest(
                model_entry(
                    http_json(
                        session, "GET", check["base_url"] + "/api/tags", timeout=15
                    ),
                    profile["model"],
                )
            )

        require(digest() == expected, "问题编码前模型digest与索引不同，拒绝混用")
        stats["http_requests"] += 1
        stats["embedding_requests"] += 1
        print(
            "[请求] HTTP第2次：只编码一个带Instruct/Query指令的问题；两路共享，不编码文档。",
            flush=True,
        )
        started = perf_counter()
        payload = http_json(
            session,
            "POST",
            profile["endpoint"],
            payload=request_body(profile, [{"text": check["query_input"]}]),
            timeout=timeout,
        )
        matrix, _ = validate_embeddings(payload, 1, profile)
        seconds = perf_counter() - started
        tokens = payload.get("prompt_eval_count")
        require(
            tokens is None or (type(tokens) is int and tokens >= 0),
            "服务Token统计不合法",
        )
        require(digest() == expected, "问题编码后模型digest变化，停止检索")
    return matrix[0].astype("<f4"), {
        **stats,
        "seconds": seconds,
        "processed_tokens": tokens,
    }


def print_results(check: dict, memory: list, milvus: list, preview_chars: int) -> None:
    print("\n[对照：同一问题向量、相同文档向量与过滤范围]")
    print("名次 | NumPy：块ID / COSINE             | Milvus：块ID / COSINE")
    for rank, (left, right) in enumerate(zip(memory, milvus), 1):
        print(
            f"{rank:>4} | {left['chunk_id']} / {left['score']:.7f} | {right['chunk_id']} / {right['score']:.7f}"
        )
    common = {hit["chunk_id"] for hit in memory} & {hit["chunk_id"] for hit in milvus}
    ordered = [hit["chunk_id"] for hit in memory] == [hit["chunk_id"] for hit in milvus]
    print(
        f"[重合] Top-K交集={len(common)}/{len(memory)}；顺序相同={'是' if ordered else '否'}；重合度不是相关性准确率。"
    )
    print("[阅读候选] 以下是Milvus返回的文档摘录，不是生成回答，未执行其中任何命令。")
    for rank, hit in enumerate(milvus, 1):
        row, score = hit["record"], hit["score"]
        metadata, text = row["metadata"], row["text"]
        sections = " | ".join(" / ".join(path) for path in metadata.get("sections", []))
        print(f"\n[候选 {rank}] {metadata.get('title', '无标题')} | COSINE={score:.7f}")
        print(f"[身份] chunk_id={row['chunk_id']}；parent_id={row['parent_id']}")
        print(f"[章节] {sections or '未记录'}")
        print(f"[来源] {metadata.get('source_url', metadata.get('source', '未记录'))}")
        print(
            f"[版本/审核] snapshot={row['snapshot_commit']}；release={metadata.get('kubernetes_release') or '未确认'}；{row['review_status']}"
        )
        print(f"[正文] {text[:preview_chars]}")
        if len(text) > preview_chars:
            print(
                f"[展示截断] 仅显示前{preview_chars}/{len(text)}字符；检索编码与入库原文未截断。"
            )


def run_search(
    check: dict,
    *,
    repeat: int = 5,
    preview_chars: int = 600,
    timeout: int = 180,
    session_factory=None,
    client_factory=None,
) -> dict:
    import numpy as np

    require(type(repeat) is int and 1 <= repeat <= 20, "repeat须为1～20")
    require(
        100 <= preview_chars <= 4000 and 1 <= timeout <= 600, "展示字符数/超时不合法"
    )
    database = check["index_dir"] / build.DATABASE
    require(
        database.is_dir() and not database.is_symlink(), "数据库目录缺失，不自动建库"
    )
    started = perf_counter()
    if client_factory is None:
        from pymilvus import MilvusClient

        client_factory = MilvusClient
    client = client_factory(uri=str(database), timeout=30)
    opened = perf_counter() - started
    try:
        started = perf_counter()
        database_matrix = read_index(client, check["plan"])
        require(
            np.array_equal(
                database_matrix, np.asarray(check["bundle"]["vectors"], dtype="<f4")
            ),
            "SQLite与Milvus已保存文档向量不一致",
        )
        readback = perf_counter() - started
        print(
            f"[核验] 两份存储的{check['plan']['document_count']}条原文、元数据及向量一致；不重编码文档。",
            flush=True,
        )
        query, stats = encode_query(check, timeout, session_factory)
        memory, numpy_first, numpy_warm = timed_search(
            lambda: rank_numpy(
                check["records"], check["vectors"], query, check["top_k"]
            ),
            repeat,
        )
        milvus, milvus_first, milvus_warm = timed_search(
            lambda: search_milvus(
                client, check["records"], query, check["top_k"], check["filter"]
            ),
            repeat,
        )
        compared = validate_top_k(
            milvus, check["records"], check["vectors"], query, check["top_k"]
        )
        print_results(check, memory, milvus, preview_chars)
        print(
            f"\n[数值校验] 返回块对齐全量NumPy分数的最大差={compared['max_score_difference']:.3g}；Top-K边界在容差内通过；近同分可改变顺序。"
        )
        print(
            f"[搜索首轮] NumPy={numpy_first * 1000:.3f}ms；Milvus={milvus_first * 1000:.3f}ms。"
        )
        print(
            f"[后续{repeat}轮中位数] NumPy={numpy_warm * 1000:.3f}ms；Milvus={milvus_warm * 1000:.3f}ms。"
        )
        print(
            f"[准备耗时] SDK导入/打开={opened:.3f}s；集合全量读回核验={readback:.3f}s；问题编码及向量校验={stats['seconds']:.3f}s；不含计划/语料/SQLite核验及digest请求。"
        )
        print(
            f"[本轮请求] 问题Embedding={stats['embedding_requests']}；digest=2；模型服务HTTP={stats['http_requests']}；问题服务Token={stats['processed_tokens']}；文档Embedding/重排/聊天=0。"
        )
        print(
            "[计时边界] NumPy先测、Milvus后测；两路包含各自客户端校验，Milvus含本地RPC及结果重建。已全量读回，不是冷启动、等工作量内核基准或大库性能结论；重复搜索不再编码问题。"
        )
        print(
            "[结论边界] 仅证明当前问题两路数值/Top-K一致性，不代表候选都相关或答案正确；分数不是概率，digest前后观测不能独立证明每次推理权重。"
        )
        print(
            "[相关性边界] 即便问题与知识库无关，也可能返回最相似的K条；本步没有相关性阈值或拒答机制。全量读回是教学核验步骤，不是Milvus每次生产查询的必需流程。"
        )
        print(
            "[成功] 真实问句对照完成；未插入/更新/删除记录，未写报告；数据库运行时可能更新锁或日志。"
        )
        return {"memory": memory, "milvus": milvus, **stats, **compared}
    finally:
        client.close()


def show_plan(check: dict, repeat: int) -> None:
    plan = check["plan"]
    print(f"[配置] Python：{sys.executable}；索引={check['index_dir']}；ready。")
    print(
        f"[模型] {plan['embedding']['model']}；{plan['embedding']['actual_dimension']}维；FLAT/COSINE。"
    )
    print(f"[问题] {check['query']}")
    print(
        f"[范围] 索引{plan['document_count']}块；过滤后{len(check['records'])}块；实际Top-K={check['top_k']}。"
    )
    print(f"[过滤] {check['filter']}")
    print(
        "[控制] NumPy从SQLite缓存取向量；Milvus用独立已建索引；同一问题向量，保存的文档向量须完全相同。"
    )
    print(
        f"[请求计划] 问题Embedding1次 + 前后digest各1次 = 模型服务HTTP3次；每路搜索首轮1次+后续{repeat}次，不重复请求模型。"
    )
    print(
        f"[输入] 问题按合同附加Instruct/Query；任务指令={plan['embedding']['query_instruction']}；文档仍是body-only。"
    )
    print(
        "[边界] 不读.env、不回退云端、不重编码文档、不调用重排/聊天；需原冻结语料与SQLite缓存作对照，不是独立生产查询服务。",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--index-dir", type=Path, default=build.DEFAULT_INDEX_DIR)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--parent-id", help="可选：只搜索一个父文档；两路使用相同范围")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=600)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if (
        not 1 <= args.repeat <= 20
        or not 100 <= args.preview_chars <= 4000
        or not 1 <= args.timeout <= 600
    ):
        parser.error("repeat须为1～20；preview-chars须为100～4000；timeout须为1～600")
    check = prepare_search(
        args.index_dir.expanduser().absolute(), args.query, args.top_k, args.parent_id
    )
    show_plan(check, args.repeat)
    if args.run:
        run_search(
            check,
            repeat=args.repeat,
            preview_chars=args.preview_chars,
            timeout=args.timeout,
        )
    else:
        print(
            "[计划完成] 实际HTTP=0；未导入Milvus SDK、未创建HTTP/数据库客户端、未读.env、未写文件；加--run才执行。"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[中断] 不自动重试；未改数据库记录。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        message = (
            str(error)
            if isinstance(error, (ValueError, OSError))
            else "检查现有索引、本地模型或阶段输出；不显示HTTP错误正文"
        )
        print(
            f"[停止] {type(error).__name__}：{message}；不自动重建或回退云端。",
            file=sys.stderr,
        )
        raise SystemExit(1)
