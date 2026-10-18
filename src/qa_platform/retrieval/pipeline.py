"""第八步：BM25与Dense召回、RRF融合，再对真实候选做Reranker重排。"""

import argparse
from dataclasses import dataclass
import os
import sys
from time import perf_counter

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document

from qa_platform.ingestion.railway import load_records, split_records
from qa_platform.providers.cloud_embedding import (
    API_URL as EMBEDDING_API_URL,
    MODEL_NAME as EMBEDDING_MODEL,
    embed_texts,
)
from qa_platform.providers.cloud_reranker import (
    API_URL as RERANK_API_URL,
    MODEL_NAME as RERANK_MODEL,
    rerank_documents,
)
from qa_platform.retrieval.bm25 import BM25Index, tokenize
from qa_platform.retrieval.dense import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_QUERY,
    MAX_BATCH_SIZE,
    embed_documents,
    rank_documents,
    validate_vectors,
)
from qa_platform.retrieval.hybrid import (
    RRFHit,
    print_recall_results,
    reciprocal_rank_fusion,
)
from qa_platform.settings import ENV_PATH


@dataclass
class RerankedHit:
    """保留重排分数、输入位置和原RRF候选，方便追溯排名变化。"""

    candidate: RRFHit
    input_index: int
    score: float


def rerank_candidates(
    client: httpx.Client,
    query: str,
    candidates: list[RRFHit],
    top_k: int = 3,
) -> list[RerankedHit]:
    """只重排固定候选池；返回index必须映射到该池，而不是整个知识库。"""
    if not query.strip():
        raise ValueError("重排问题不能为空。")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是正整数。")
    if not candidates:
        print("[说明] RRF候选池为空，跳过Reranker请求。")
        return []

    pool = list(candidates)
    chunk_ids = [hit.document.metadata.get("chunk_id") for hit in pool]
    if any(
        not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in chunk_ids
    ):
        raise ValueError("重排候选缺少有效chunk_id。")
    if len(set(chunk_ids)) != len(pool):
        raise ValueError("重排候选池存在重复chunk_id。")
    if any(not hit.document.page_content.strip() for hit in pool):
        raise ValueError("重排候选正文不能为空。")

    top_n = min(top_k, len(pool))
    print(f"[请求] Reranker：提交 {len(pool)} 个候选，返回前 {top_n} 个", flush=True)
    results = rerank_documents(
        client, query, [hit.document.page_content for hit in pool], top_n
    )
    # helper已校验返回数量、index范围/重复与有限分数。
    # index是本次提交pool的位置；用它找回原Document和来源，绝不能用重排名次找。
    hits = [RerankedHit(pool[index], index, score) for index, score in results]
    # 同分时优先沿用原RRF顺序，不依赖服务返回同分项的先后顺序。
    return sorted(hits, key=lambda hit: (-hit.score, hit.input_index))


def print_fusion_pool(candidates: list[RRFHit]) -> None:
    """打印重排输入顺序；只有此列表的位置才是接口返回index的含义。"""
    print(f"\n[RRF候选池] 实际 {len(candidates)} 块；以下全部提交给Reranker")
    for index, hit in enumerate(candidates):
        metadata = hit.document.metadata
        label = metadata["entry_id"] or metadata["title"]
        print(
            f"RRF第{index + 1}名 | index={index} | {label} | RRF={hit.score:.8f} | 块={metadata['chunk_id']}"
        )


class RetrievalError(RuntimeError):
    """只携带经过脱敏的检索阶段错误，不暴露请求头或服务响应正文。"""


def run_retrieval(
    client: httpx.Client,
    query: str,
    chunks: list[Document],
    recall_k: int = 5,
    fusion_k: int = 6,
    top_k: int = 3,
    rrf_constant: int = 60,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_candidates: bool = True,
) -> list[RerankedHit]:
    """可复用的完整检索函数；调用方负责读取配置与管理HTTP客户端。"""
    if not query.strip():
        raise ValueError("检索问题不能为空。")
    if any(
        type(value) is not int or value <= 0
        for value in (recall_k, fusion_k, top_k, rrf_constant)
    ):
        raise ValueError("召回、融合、展示数量与RRF常数必须是正整数。")
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size 必须在1到{MAX_BATCH_SIZE}之间。")
    if not chunks:
        return []

    stage = "BM25检索"
    try:
        query_tokens = tokenize(query)
        if show_candidates:
            print(
                f"[分词] {' / '.join(query_tokens) if query_tokens else '过滤后无有效词，本次BM25为空路'}"
            )
        bm25_hits = (
            BM25Index(chunks).search(query_tokens, recall_k) if query_tokens else []
        )
        stage = "问题编码"
        print("[请求] 编码问题：text_type=query", flush=True)
        query_matrix = embed_texts(client, [query], "query")
        validate_vectors(query_matrix, 1)
        query_vector = query_matrix[0]
        stage = "文档编码"
        document_vectors = embed_documents(
            client, chunks, batch_size, query_vector.shape[0]
        )
        stage = "Dense检索与RRF融合"
        dense_hits = rank_documents(chunks, document_vectors, query_vector, recall_k)
        rankings = {
            "BM25": [document for document, _, _ in bm25_hits],
            "Dense": [document for document, _ in dense_hits],
        }
        candidates = reciprocal_rank_fusion(chunks, rankings, rrf_constant, fusion_k)
        if show_candidates:
            print_recall_results(bm25_hits, dense_hits)
            print_fusion_pool(candidates)
        stage = "Reranker重排"
        return rerank_candidates(client, query, candidates, top_k)
    except httpx.TimeoutException:
        raise RetrievalError(f"{stage}请求超时，请检查服务状态或网络。") from None
    except httpx.HTTPStatusError as exc:
        raise RetrievalError(
            f"{stage}服务返回 HTTP {exc.response.status_code}。"
        ) from None
    except httpx.RequestError:
        raise RetrievalError(f"{stage}连接失败，请检查服务地址和网络。") from None
    except (ValueError, TypeError, OverflowError) as exc:
        raise RetrievalError(f"{stage}校验失败：{exc}") from None


def main() -> int:
    """验证完整检索链路；只返回资料，不调用聊天模型生成答案。"""
    parser = argparse.ArgumentParser(
        description="验证BM25 + Dense + RRF + Reranker完整检索链路"
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="检索问题")
    parser.add_argument(
        "--kb",
        choices=["test", "experience", "all"],
        default="test",
        help="选择知识库；默认仅测试库",
    )
    parser.add_argument("--recall-k", type=int, default=5, help="每路最多召回几个子块")
    parser.add_argument(
        "--fusion-k", type=int, default=6, help="RRF最多保留几个子块交给重排"
    )
    parser.add_argument("--top-k", type=int, default=3, help="重排后最多展示几个子块")
    parser.add_argument(
        "--rrf-constant", type=int, default=60, help="RRF平滑常数，正整数"
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
    if min(args.recall_k, args.fusion_k, args.top_k, args.rrf_constant) <= 0:
        parser.error("--recall-k、--fusion-k、--top-k、--rrf-constant 必须是正整数")
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size 必须在1到{MAX_BATCH_SIZE}之间")
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("必须满足 chunk_size > 0 且 0 <= chunk_overlap < chunk_size")

    try:
        records = load_records(args.kb)
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
    pool_limit = min(len(chunks), 2 * args.recall_k, args.fusion_k)
    print(f"[配置] Python：{sys.executable}")
    print(f"[配置] 文件：{ENV_PATH}")
    print(f"[配置] Embedding模型：{EMBEDDING_MODEL}；地址：{EMBEDDING_API_URL}")
    print(f"[配置] Reranker模型：{RERANK_MODEL}；地址：{RERANK_API_URL}")
    print("[配置] DASHSCOPE_API_KEY 已配置（不显示密钥）")
    print(
        f"[配置] 知识库：{args.kb}；每路召回：{args.recall_k}；融合池上限：{args.fusion_k}；重排Top-K：{args.top_k}"
    )
    print(
        f"[配置] RRF常数：{args.rrf_constant}；最大块大小：{args.chunk_size} 字符；目标重叠：{args.chunk_overlap} 字符；批量大小：{args.batch_size}"
    )
    print(f"[索引] 父记录 {len(records)} 条，共用检索子块 {len(chunks)} 个")
    print(
        f"[计划] Embedding：文档 {document_requests} 次 + 问题1次；Reranker：1次；共 {document_requests + 2} 次模型请求"
    )
    print(
        f"[计划] 重排输入最多 {pool_limit} 块，实际取决于两路召回并集；最终最多展示 {min(args.top_k, pool_limit)} 块"
    )
    print(
        "[说明] BM25与RRF仅本地计算；当前不缓存向量，实际运行会重新编码并可能产生API费用。"
    )
    if args.kb in {"test", "all"}:
        print("[提醒] 本次范围包含测试资料，候选内容不应直接作为设备维护规程。")
    if args.kb in {"experience", "all"}:
        print("[提醒] 经验库内容尚未审核，重排分数不代表专业确认。")
    if args.check_config:
        print("[成功] 本地配置、知识库与请求计划检查通过，未发送模型请求。")
        return 0

    print(f"\n[问题] {args.query}")
    started = perf_counter()
    try:
        with httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        ) as client:
            reranked = run_retrieval(
                client,
                args.query,
                chunks,
                args.recall_k,
                args.fusion_k,
                args.top_k,
                args.rrf_constant,
                args.batch_size,
            )
    except RetrievalError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    for rank, hit in enumerate(reranked, start=1):
        metadata = hit.candidate.document.metadata
        label = metadata["entry_id"] or metadata["title"]
        print(f"\n[重排 {rank}] {label} | 相关性分数={hit.score:.4f}")
        print(
            f"[对照] 原RRF第 {hit.input_index + 1} 名；重排输入index={hit.input_index}；RRF={hit.candidate.score:.8f}"
        )
        print(
            f"[来源] {metadata['source']}；父记录第 {metadata['line_start']}–{metadata['line_end']} 行；is_test={metadata['is_test']}"
        )
        print(
            f"[块] {metadata['chunk_id']}；父记录 {metadata['parent_id']}；第 {metadata['part_index'] + 1}/{metadata['part_count']} 块"
        )
        print(hit.candidate.document.page_content)
    if not reranked:
        print("\n[结果] 没有可展示的重排候选。")
    print(
        "\n[说明] 最终顺序仅按重排分数决定，不与BM25、余弦或RRF分数混加；分数不是故障概率。"
    )
    print(
        "[说明] 返回前K条不代表每条都相关；未入池候选无法被重排找回，未返回项也无法确定完整名次。"
    )
    print(
        f"[成功] 完整检索链路完成，耗时 {perf_counter() - started:.2f} 秒；未调用聊天模型、未生成诊断回答、未写入数据库。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
