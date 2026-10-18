"""第七步：复用BM25与稠密检索，按文本块身份进行等权RRF排名融合。"""

import argparse
from dataclasses import dataclass
import os
import sys
from time import perf_counter

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document

from qa_platform.ingestion.railway import load_records, split_records
from qa_platform.providers.cloud_embedding import API_URL, MODEL_NAME, embed_texts
from qa_platform.retrieval.bm25 import BM25Index, tokenize
from qa_platform.retrieval.dense import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_QUERY,
    MAX_BATCH_SIZE,
    embed_documents,
    rank_documents,
    validate_vectors,
)
from qa_platform.settings import ENV_PATH


@dataclass
class RRFHit:
    """保存一个融合候选及其分数来源，不改变原始Document。"""

    document: Document
    score: float
    ranks: dict[str, int]
    contributions: dict[str, float]


def reciprocal_rank_fusion(
    documents: list[Document],
    rankings: dict[str, list[Document]],
    rank_constant: int = 60,
    top_k: int = 3,
) -> list[RRFHit]:
    """每路先按chunk_id去重，再累加1/(常数+排名)；排名从1开始。"""
    if type(rank_constant) is not int or rank_constant <= 0:
        raise ValueError("本演示的RRF常数必须是正整数。")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是正整数。")
    canonical, positions = {}, {}
    for position, document in enumerate(documents):
        chunk_id = document.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("文本块缺少有效的chunk_id。")
        if chunk_id in canonical:
            raise ValueError("原始文本块列表中存在重复chunk_id。")
        canonical[chunk_id], positions[chunk_id] = document, position

    fused = {}
    for source, candidates in rankings.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError("检索路线名称不能为空。")
        seen = set()
        for candidate in candidates:
            chunk_id = candidate.metadata.get("chunk_id")
            if not isinstance(chunk_id, str) or chunk_id not in canonical:
                raise ValueError("候选块不在本次共同知识库中，无法融合。")
            original = canonical[chunk_id]
            if (
                candidate.page_content != original.page_content
                or candidate.metadata != original.metadata
            ):
                raise ValueError("同一chunk_id对应的正文或元数据不一致。")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank = len(seen)
            contribution = 1.0 / (rank_constant + rank)
            if chunk_id not in fused:
                fused[chunk_id] = RRFHit(original, 0.0, {}, {})
            hit = fused[chunk_id]
            hit.score += contribution
            hit.ranks[source] = rank
            hit.contributions[source] = contribution

    # 只融合候选并集；同分按原始知识块顺序排序，不由集合遍历顺序决定。
    ordered = sorted(
        fused.values(),
        key=lambda hit: (-hit.score, positions[hit.document.metadata["chunk_id"]]),
    )
    return ordered[:top_k]


def print_recall_results(
    bm25_hits: list[tuple[Document, float, list[str]]],
    dense_hits: list[tuple[Document, float]],
) -> None:
    """展示融合前的两份排名；原始分数只供观察，不参与RRF计算。"""
    print("\n[BM25召回]")
    if not bm25_hits:
        print("没有词汇匹配候选；仍可融合Dense一路的结果。")
    for rank, (document, score, matched) in enumerate(bm25_hits, start=1):
        label = document.metadata["entry_id"] or document.metadata["title"]
        print(
            f"{rank}. {label} | BM25={score:.4f} | 块={document.metadata['chunk_id']}"
        )
        print(f"   命中词：{' / '.join(matched)}")
    print("\n[Dense召回]")
    for rank, (document, score) in enumerate(dense_hits, start=1):
        label = document.metadata["entry_id"] or document.metadata["title"]
        print(
            f"{rank}. {label} | 余弦相似度={score:.4f} | 块={document.metadata['chunk_id']}"
        )


def main() -> int:
    """同一问题、同一批分块、两路召回、本地融合；不重排、不生成答案。"""
    parser = argparse.ArgumentParser(description="验证BM25 + Dense的RRF混合检索")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="检索问题")
    parser.add_argument(
        "--kb",
        choices=["test", "experience", "all"],
        default="test",
        help="选择知识库；默认仅测试库",
    )
    parser.add_argument("--recall-k", type=int, default=5, help="每路最多召回几个子块")
    parser.add_argument("--top-k", type=int, default=3, help="融合后最多返回几个子块")
    parser.add_argument(
        "--rrf-constant", type=int, default=60, help="RRF平滑常数，正整数；不是返回数量"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=800, help="子块最大字符数，包含标题"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="同一父条目内子块的目标重叠字符数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="每次文档编码请求的文本块数，1到10",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="仅检查配置、分块和请求计划，不发送请求",
    )
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query 不能为空或只有空白字符")
    if args.recall_k <= 0 or args.top_k <= 0 or args.rrf_constant <= 0:
        parser.error("--recall-k、--top-k、--rrf-constant 必须是正整数")
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size 必须在1到{MAX_BATCH_SIZE}之间")
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("必须满足 chunk_size > 0 且 0 <= chunk_overlap < chunk_size")

    try:
        records = load_records(args.kb)
        # 两路必须复用这一份chunks，不能各自使用不同的分块配置。
        chunks = split_records(records, args.chunk_size, args.chunk_overlap)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[错误] 知识库解析或分块失败：{exc}", file=sys.stderr)
        return 1
    if not ENV_PATH.is_file():
        print(f"[错误] 配置文件不存在：{ENV_PATH}", file=sys.stderr)
        return 1
    load_dotenv(dotenv_path=ENV_PATH, override=False, encoding="utf-8")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("[错误] 请配置 DASHSCOPE_API_KEY。", file=sys.stderr)
        return 1

    document_requests = (len(chunks) + args.batch_size - 1) // args.batch_size
    print(f"[配置] Python：{sys.executable}")
    print(f"[配置] 文件：{ENV_PATH}")
    print(f"[配置] Embedding模型：{MODEL_NAME}")
    print(f"[配置] 服务地址：{API_URL}")
    print("[配置] DASHSCOPE_API_KEY 已配置（不显示密钥）")
    print(
        f"[配置] 知识库：{args.kb}；每路召回：{args.recall_k}；融合Top-K：{args.top_k}；RRF常数：{args.rrf_constant}"
    )
    print(
        f"[配置] 最大块大小：{args.chunk_size} 字符；目标重叠：{args.chunk_overlap} 字符；批量大小：{args.batch_size}"
    )
    print(f"[索引] 父记录 {len(records)} 条，共用检索子块 {len(chunks)} 个")
    print(
        f"[计划] 文档编码 {document_requests} 次 + 问题编码1次，共 {document_requests + 1} 次Embedding请求；BM25和RRF仅本地计算"
    )
    print("[说明] 当前不缓存向量，每次实际运行都会重新编码，API调用可能产生费用。")
    if args.kb in {"test", "all"}:
        print("[提醒] 本次范围包含测试资料，候选内容不应直接作为设备维护规程。")
    if args.kb in {"experience", "all"}:
        print("[提醒] 经验库内容尚未审核，融合排名不代表专业确认。")
    if args.check_config:
        print("[成功] 本地配置、知识库与请求计划检查通过，未发送模型请求。")
        return 0

    print(f"\n[问题] {args.query}")
    started = perf_counter()
    try:
        query_tokens = tokenize(args.query)
        print(
            f"[分词] {' / '.join(query_tokens) if query_tokens else '过滤后无有效词，本次BM25为空路'}"
        )
        bm25_hits = (
            BM25Index(chunks).search(query_tokens, args.recall_k)
            if query_tokens
            else []
        )
        with httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        ) as client:
            print("[请求] 编码问题：text_type=query", flush=True)
            query_matrix = embed_texts(client, [args.query], "query")
            validate_vectors(query_matrix, 1)
            query_vector = query_matrix[0]
            document_vectors = embed_documents(
                client, chunks, args.batch_size, query_vector.shape[0]
            )
        dense_hits = rank_documents(
            chunks, document_vectors, query_vector, args.recall_k
        )
        rankings = {
            "BM25": [document for document, _, _ in bm25_hits],
            "Dense": [document for document, _ in dense_hits],
        }
        fused_hits = reciprocal_rank_fusion(
            chunks, rankings, args.rrf_constant, args.top_k
        )
    except httpx.TimeoutException:
        print("[错误] 请求超时，请检查服务状态或网络。", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"[错误] 服务返回 HTTP {exc.response.status_code}。", file=sys.stderr)
        return 1
    except httpx.RequestError:
        print("[错误] 连接失败，请检查服务地址和网络。", file=sys.stderr)
        return 1
    except (ValueError, TypeError, OverflowError) as exc:
        print(f"[错误] 检索或融合校验失败：{exc}", file=sys.stderr)
        return 1

    print_recall_results(bm25_hits, dense_hits)
    union_count = len(
        {
            document.metadata["chunk_id"]
            for candidates in rankings.values()
            for document in candidates
        }
    )
    print(
        f"\n[融合] 两路候选按chunk_id去重后共 {union_count} 块，展示前 {len(fused_hits)} 块"
    )
    for rank, hit in enumerate(fused_hits, start=1):
        metadata = hit.document.metadata
        label = metadata["entry_id"] or metadata["title"]
        print(f"\n[候选 {rank}] {label} | RRF={hit.score:.8f}")
        for source in rankings:
            source_rank = hit.ranks.get(source)
            rank_label = (
                f"第{source_rank}名" if source_rank is not None else "未进入本路候选"
            )
            print(
                f"[贡献] {source}：{rank_label}，贡献={hit.contributions.get(source, 0.0):.8f}"
            )
        print(
            f"[来源] {metadata['source']}；父记录第 {metadata['line_start']}–{metadata['line_end']} 行"
        )
        print(
            f"[块] {metadata['chunk_id']}；父记录 {metadata['parent_id']}；第 {metadata['part_index'] + 1}/{metadata['part_count']} 块"
        )
        print(hit.document.page_content)
    print("\n[说明] RRF只使用排名，不相加BM25与余弦原始分数；未召回的一路贡献为0。")
    print("[说明] RRF分数不是概率；融合不保证提升相关性，也找不回两路都遗漏的文本块。")
    print(
        f"[成功] 混合检索完成，耗时 {perf_counter() - started:.2f} 秒；未调用Reranker或聊天模型，未写入数据库。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
