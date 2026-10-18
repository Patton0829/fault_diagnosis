"""第六步：将真实知识库分块编码为向量，在内存中进行语义检索。"""

import argparse
import os
import sys
from time import perf_counter

import httpx
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document

from qa_platform.ingestion.railway import load_records, split_records
from qa_platform.providers.cloud_embedding import (
    API_URL,
    MODEL_NAME,
    cosine_scores,
    embed_texts,
)
from qa_platform.settings import ENV_PATH


DEFAULT_QUERY = "电机还在转，但丝杆一点也不动，可能是什么原因？"
DEFAULT_BATCH_SIZE = 8
# 按接口文档的文本列表上限控制批次；默认8条方便演示测试库。
MAX_BATCH_SIZE = 10


def validate_vectors(
    matrix: np.ndarray, rows: int, dimension: int | None = None
) -> None:
    """检查矩阵形状、数值、维度与向量长度，拒绝不能计算相似度的数据。"""
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("向量矩阵必须是二维 NumPy 数组。")
    if matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise ValueError("向量数量与文本数量不一致，或向量维度为空。")
    if dimension is not None and matrix.shape[1] != dimension:
        raise ValueError("问题与文档、或不同批次的向量维度不一致。")
    if matrix.dtype.kind not in "fiu" or not np.isfinite(matrix).all():
        raise ValueError("向量必须是有限的数值，不能包含 NaN 或无穷大。")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("向量长度必须有限且大于零。")


def embed_documents(
    client: httpx.Client,
    documents: list[Document],
    batch_size: int = DEFAULT_BATCH_SIZE,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """按原始块顺序分批编码；矩阵第i行始终对应documents[i]。"""
    if not documents or any(
        not document.page_content.strip() for document in documents
    ):
        raise ValueError("待编码文本块不能为空。")
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size 必须是1到{MAX_BATCH_SIZE}之间的整数。")
    matrices = []
    dimension = expected_dimension
    batch_count = (len(documents) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(documents), batch_size), start=1):
        batch = documents[start : start + batch_size]
        print(
            f"[请求] 文档批次 {batch_index}/{batch_count}：{len(batch)} 个文本块，text_type=document",
            flush=True,
        )
        # 只发送检索正文，不发送元数据中的文件说明、测试建议或本地路径。
        matrix = embed_texts(
            client, [document.page_content for document in batch], "document"
        )
        validate_vectors(matrix, len(batch), dimension)
        dimension = matrix.shape[1]
        # embed_texts已按本批局部index还原顺序；不同批次按发送顺序拼接。
        matrices.append(matrix)
    return np.vstack(matrices)


def rank_documents(
    documents: list[Document],
    document_vectors: np.ndarray,
    query_vector: np.ndarray,
    top_k: int = 3,
) -> list[tuple[Document, float]]:
    """计算问题与所有文本块的余弦相似度，稳定地返回最高的K个子块。"""
    if not documents:
        raise ValueError("没有可检索的文本块。")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是正整数。")
    if not isinstance(query_vector, np.ndarray) or query_vector.ndim != 1:
        raise ValueError("问题向量必须是一维 NumPy 数组。")
    validate_vectors(query_vector[np.newaxis, :], 1)
    validate_vectors(document_vectors, len(documents), query_vector.shape[0])
    scores = cosine_scores(query_vector, document_vectors)
    if not np.isfinite(scores).all():
        raise ValueError("余弦相似度包含非有限值。")
    # 对负分数做稳定升序，相当于原分数降序；同分沿用原始块顺序。
    positions = np.argsort(-scores, kind="stable")[:top_k]
    return [
        (documents[int(position)], float(scores[position])) for position in positions
    ]


def main() -> int:
    """本地检查或在线Embedding检索；不调用聊天模型，不持久化向量。"""
    parser = argparse.ArgumentParser(description="验证真实知识库的内存向量检索")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="检索问题")
    parser.add_argument(
        "--kb",
        choices=["test", "experience", "all"],
        default="test",
        help="选择知识库；默认仅测试库",
    )
    parser.add_argument("--top-k", type=int, default=3, help="最多返回几个检索子块")
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
    if args.top_k <= 0:
        parser.error("--top-k 必须大于0")
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
    print(f"[配置] Python：{sys.executable}")
    print(f"[配置] 文件：{ENV_PATH}")
    print(f"[配置] Embedding模型：{MODEL_NAME}")
    print(f"[配置] 服务地址：{API_URL}")
    print("[配置] DASHSCOPE_API_KEY 已配置（不显示密钥）")
    print(f"[配置] 知识库：{args.kb}；Top-K：{args.top_k}；批量大小：{args.batch_size}")
    print(
        f"[配置] 最大块大小：{args.chunk_size} 字符；目标重叠：{args.chunk_overlap} 字符"
    )
    print(f"[索引] 父记录 {len(records)} 条，检索子块 {len(chunks)} 个")
    print(
        f"[计划] 文档编码 {document_requests} 次 + 问题编码1次，共 {document_requests + 1} 次Embedding请求"
    )
    print("[说明] 当前不缓存向量；每次实际运行都会重新编码，API调用可能产生费用。")
    if args.kb in {"test", "all"}:
        print("[提醒] 本次范围包含测试资料，候选内容不应直接作为设备维护规程。")
    if args.kb in {"experience", "all"}:
        print("[提醒] 经验库内容尚未审核，语义相似不代表内容已获得专业确认。")
    if args.check_config:
        print("[成功] 本地配置、知识库与请求计划检查通过，未发送模型请求。")
        return 0

    print(f"\n[问题] {args.query}")
    started = perf_counter()
    try:
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
        hits = rank_documents(chunks, document_vectors, query_vector, args.top_k)
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
        print(f"[错误] 响应解析或向量校验失败：{exc}", file=sys.stderr)
        return 1

    print(f"\n[向量] 问题形状：{query_vector.shape}")
    print(f"[向量] 文档矩阵形状：{document_vectors.shape}")
    for rank, (document, score) in enumerate(hits, start=1):
        metadata = document.metadata
        label = metadata["entry_id"] or metadata["title"]
        print(f"\n[候选 {rank}] {label} | 余弦相似度={score:.4f}")
        print(
            f"[来源] {metadata['source']}；父记录第 {metadata['line_start']}–{metadata['line_end']} 行"
        )
        print(
            f"[块] {metadata['chunk_id']}；父记录 {metadata['parent_id']}；第 {metadata['part_index'] + 1}/{metadata['part_count']} 块"
        )
        print(document.page_content)
    print(
        "\n[说明] 相似度不是故障概率；返回最相似的候选，也不代表知识库一定能够回答问题。"
    )
    print("[说明] 向量分数不能直接与BM25分数比较；这里没有调用Reranker或聊天模型。")
    print(
        f"[成功] 内存向量检索完成，耗时 {perf_counter() - started:.2f} 秒；未写入数据库。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
