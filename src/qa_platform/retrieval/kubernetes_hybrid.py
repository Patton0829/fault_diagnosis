"""Kubernetes 三路候选对照：共用正文/范围，不重编码文档，不生成答案。"""

from math import isfinite
from pathlib import Path

from qa_platform.ingestion.embedding_plan import require
from qa_platform.storage import kubernetes_search as search


RECALL_K = 8
TOP_K = 5
RRF_CONSTANT = 60
NOTICE = (
    "同一问题、正文和范围；BM25按关键词，Dense按语义，RRF只融合排名。"
    "三类分数不能横向比较，也不是正确率。未重排、补上下文或生成答案；"
    "文档中的命令不会被执行。当前沿用原分词基线，域名/IP/路径会拆词。"
)


def prepare(index_dir: Path, query: str = search.DEFAULT_QUERY) -> dict:
    """纯离线检查；使用已有索引快照，默认不创建任何客户端。"""
    return search.prepare_search(index_dir, query, top_k=RECALL_K)


def describe(check: dict) -> dict:
    return {
        "query": check["query"],
        "document_count": len(check["records"]),
        "recall_k": min(RECALL_K, len(check["records"])),
        "top_k": TOP_K,
        "rrf_constant": RRF_CONSTANT,
        "model_requests": {"embedding": 1, "metadata": 2, "reranker": 0, "chat": 0},
        "notice": NOTICE,
    }


def compare(
    check: dict,
    mode: str = "bm25",
    *,
    session_factory=None,
    client_factory=None,
) -> dict:
    """仅 hybrid 模式调用一次问题编码；三路每路展示前5，实际召回各8。"""
    require(mode in {"bm25", "hybrid"}, "未知检索模式")
    # 延迟导入：打开网页/显示计划不初始化分词器、HTTP或Milvus客户端。
    from langchain_core.documents import Document

    from qa_platform.retrieval.bm25 import BM25Index, tokenize
    from qa_platform.retrieval.hybrid import reciprocal_rank_fusion

    records = check["records"]
    require(bool(records), "本次范围没有文本块")
    ids = [row["chunk_id"] for row in records]
    require(len(ids) == len(set(ids)), "共同范围内有重复块")
    require(
        all(row["metadata"].get("chunk_id") == row["chunk_id"] for row in records),
        "块身份与元数据不一致",
    )
    documents = [
        Document(page_content=row["text"], metadata=row["metadata"])
        for row in records
    ]
    recall_k = min(RECALL_K, len(records))
    require(check["top_k"] == recall_k, "Dense与BM25召回数量合同不符")
    tokens = list(dict.fromkeys(tokenize(check["query"])))
    index = BM25Index(documents)
    lexical = index.search(tokens, recall_k) if tokens else []
    bm25_hits = [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "rank": rank,
            "score": score,
            "matched_terms": matched,
        }
        for rank, (doc, score, matched) in enumerate(lexical, 1)
    ]
    dense_hits, fused_hits = [], []
    counts = {"embedding": 0, "http": 0, "reranker": 0, "chat": 0}
    if mode == "hybrid":
        import numpy as np

        from qa_platform.storage import kubernetes_milvus as build

        database = check["index_dir"] / build.DATABASE
        require(database.is_dir() and not database.is_symlink(), "已有数据库缺失")
        if client_factory is None:
            from pymilvus import MilvusClient

            client_factory = MilvusClient
        client = client_factory(uri=str(database), timeout=30)
        try:
            matrix = search.read_index(client, check["plan"])
            require(
                np.array_equal(matrix, np.asarray(check["bundle"]["vectors"], dtype="<f4")),
                "数据库与冻结缓存向量不同",
            )
            query, stats = search.encode_query(check, session_factory=session_factory)
            dense = search.search_milvus(
                client, records, query, recall_k, check["filter"]
            )
            # 这是同向量的数值验算，不把重合度当作相关性准确率。
            search.validate_top_k(dense, records, check["vectors"], query, recall_k)
        finally:
            client.close()
        counts.update(embedding=stats["embedding_requests"], http=stats["http_requests"])
        dense_hits = [
            {"chunk_id": hit["chunk_id"], "rank": rank, "score": hit["score"]}
            for rank, hit in enumerate(dense, 1)
        ]
        by_id = {doc.metadata["chunk_id"]: doc for doc in documents}
        fused = reciprocal_rank_fusion(
            documents,
            {
                "bm25": [doc for doc, _, _ in lexical],
                "dense": [by_id[hit["chunk_id"]] for hit in dense],
            },
            rank_constant=RRF_CONSTANT,
            top_k=TOP_K,
        )
        fused_hits = [
            {
                "chunk_id": hit.document.metadata["chunk_id"],
                "rank": rank,
                "score": hit.score,
                "ranks": hit.ranks,
                "contributions": hit.contributions,
            }
            for rank, hit in enumerate(fused, 1)
        ]
    require(
        all(isfinite(hit["score"]) for hit in bm25_hits + dense_hits + fused_hits),
        "候选含非有限分数",
    )
    selected = {hit["chunk_id"] for hit in bm25_hits + dense_hits + fused_hits}
    return {
        "query": check["query"],
        "mode": mode,
        "tokens": tokens,
        "scope_count": len(records),
        "recall_k": recall_k,
        "top_k": TOP_K,
        "rrf_constant": RRF_CONSTANT,
        "request_counts": counts,
        # 保留每路全部8条的名次，前端先展示5条，可显式展开其余候选。
        "routes": {"bm25": bm25_hits, "dense": dense_hits, "rrf": fused_hits},
        "records": {
            row["chunk_id"]: {"text": row["text"], "metadata": row["metadata"]}
            for row in records
            if row["chunk_id"] in selected
        },
        "notice": NOTICE,
    }
