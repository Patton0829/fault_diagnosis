"""固定召回并集的云端重排对照；不重新检索、不补上下文、不生成回答。"""

from copy import deepcopy
from math import isclose, isfinite
import os

import httpx

from qa_platform.ingestion.embedding_plan import require
from qa_platform.providers.cloud_reranker import API_URL, MODEL_NAME, rerank_documents
from qa_platform.settings import ENV_PATH


MAX_CANDIDATES = 16  # 当前BM25与Dense各最多8块。
NOTICE = (
    "同一问题、同一召回并集、同一完整正文；RRF只看名次，Reranker联合判断问题与正文。"
    "不先截取RRF前5；上升或新进前5不代表更准确，分数不是概率。"
    "本步向云端发送问题和候选正文，不发送来源路径、元数据或向量。"
    "不重新编码、检索、补上下文或生成回答；不自动重试。"
)


def describe() -> dict:
    """仅返回固定计划；不读取密钥，不创建HTTP客户端。"""
    return {"model": MODEL_NAME, "url": API_URL, "max_candidates": MAX_CANDIDATES, "notice": NOTICE}


def load_api_key() -> str:
    """只在用户主动重排时读配置，不向页面返回或写入进程环境。"""
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not key:
        from dotenv import dotenv_values

        # 禁止插值，配置文件只作为数据读取，不展开其他环境变量。
        values = dotenv_values(ENV_PATH, encoding="utf-8", interpolate=False)
        key = (values.get("DASHSCOPE_API_KEY") or "").strip()
    require(bool(key), "云端重排密钥未配置")
    return key


def validate_pool(result: dict) -> list[dict]:
    """在读取密钥前验算本轮范围和RRF映射，杜绝截断或跨轮混用。"""
    require(result.get("mode") == "hybrid", "须先完成三路检索")
    require(isinstance(result.get("query"), str) and bool(result["query"].strip()), "问题缺失")
    routes, records = result["routes"], result["records"]
    require("reranker" not in routes, "该轮已经重排")
    pool = result["rrf_pool"]
    require(isinstance(pool, list) and 1 <= len(pool) <= MAX_CANDIDATES, "完整候选池大小无效")
    k, top_k = result["rrf_constant"], result["top_k"]
    require(type(k) is int and k > 0 and type(top_k) is int and top_k > 0, "排名参数无效")
    maps = {}
    for route in ("bm25", "dense"):
        hits = routes[route]
        require(isinstance(hits, list) and len(hits) <= 8, "召回数量无效")
        mapping = {}
        for rank, hit in enumerate(hits, 1):
            chunk_id = hit["chunk_id"]
            require(isinstance(chunk_id, str) and chunk_id not in mapping, "召回身份无效")
            require(type(hit["rank"]) is int and hit["rank"] == rank, "召回名次无效")
            mapping[chunk_id] = rank
        maps[route] = mapping
    union = set(maps["bm25"]) | set(maps["dense"])
    require(set(records) == union, "正文范围与召回并集不一致")
    require(len(pool) == len(union) and {hit["chunk_id"] for hit in pool} == union, "候选池有遗漏或重复")
    previous = float("inf")
    for rank, hit in enumerate(pool, 1):
        chunk_id = hit["chunk_id"]
        require(type(hit["rank"]) is int and hit["rank"] == rank, "RRF全池名次无效")
        expected_ranks = {route: mapping[chunk_id] for route, mapping in maps.items() if chunk_id in mapping}
        contributions = {route: 1 / (k + position) for route, position in expected_ranks.items()}
        require(hit["ranks"] == expected_ranks and hit["contributions"] == contributions, "RRF贡献映射无效")
        score = hit["score"]
        require(type(score) in (int, float) and isfinite(score), "RRF分数无效")
        require(isclose(score, sum(contributions.values()), rel_tol=0, abs_tol=1e-12) and score <= previous, "RRF排序或分数无效")
        previous = score
        text = records[chunk_id]["text"]
        require(isinstance(text, str) and bool(text.strip()), "候选原文为空")
    require(routes["rrf"] == pool[:top_k], "显示前K与完整池不一致")
    counts = result["request_counts"]
    require(all(type(counts[key]) is int and counts[key] >= 0 for key in ("embedding", "http", "reranker", "chat")), "请求计数无效")
    require(counts["reranker"] == 0 and counts["chat"] == 0, "不是原始召回轮次")
    return pool


def rerank(result: dict, *, client_factory=None, api_key_loader=None) -> dict:
    """一批完整正文一次请求；保留旧排序，只新增独立重排结果。"""
    pool = validate_pool(result)
    key = (api_key_loader or load_api_key)()
    require(isinstance(key, str) and bool(key.strip()), "云端重排密钥未配置")
    factory = client_factory or httpx.Client
    texts = [result["records"][hit["chunk_id"]]["text"] for hit in pool]
    with factory(
        headers={"Authorization": f"Bearer {key}"}, timeout=60.0,
        trust_env=False, follow_redirects=False,
    ) as client:
        ordered = rerank_documents(client, result["query"], texts, len(pool))
    # provider已检查范围、重复和分数；这里再次检查完整覆盖，明确全池合同。
    require(len(ordered) == len(pool) and {index for index, _ in ordered} == set(range(len(pool))), "重排结果未覆盖完整候选池")
    output = deepcopy(result)
    output["routes"]["reranker"] = [
        {
            "chunk_id": pool[index]["chunk_id"], "rank": rank, "score": score,
            "input_index": index, "rrf_rank": pool[index]["rank"],
            "rank_change": pool[index]["rank"] - rank,
        }
        for rank, (index, score) in enumerate(ordered, 1)
    ]
    output["request_counts"]["http"] += 1
    output["request_counts"]["reranker"] += 1
    output["rerank_comparison"] = {
        "candidate_count": len(pool), "final_k": result["top_k"], "model": MODEL_NAME,
        "input_order": [hit["chunk_id"] for hit in pool],
        "request_counts": {"embedding": 0, "http": 1, "reranker": 1, "chat": 0},
        "notice": NOTICE,
    }
    output["notice"] = NOTICE
    return output
