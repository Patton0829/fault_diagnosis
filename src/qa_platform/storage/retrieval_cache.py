"""短期缓存完整检索/重排的排名；不缓存问题、正文、向量或生成答案。

只有显式启用 Redis 的调用方才创建本对象。命中时仍从当前已校验快照取正文，
重新检查排名与引用合同；这不保证语义正确，也不证明服务端模型权重此刻未变。
"""

from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
from math import isfinite
from uuid import uuid4

from qa_platform.settings import PACKAGE_ROOT


TTL_SECONDS = 300
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
SCHEMA = "qa-retrieval-cache/v1"
PREFIX = "fault_diagnosis:retrieval:v1:"
NOTICE = (
    "Redis仅缓存检索与重排的块ID、排名和分数，最长300秒；不缓存问题、正文、向量或回答。"
    "命中仍生成新回答，追问改写也不会省略；缓存服务失败即停止，不隐式回退或重试。"
    "命中复用近期模型结果，不再次检查远端模型当前权重；快照/输入合同或相关代码变化会换键。"
)
CODE_FILES = (
    "retrieval/bm25.py", "retrieval/hybrid.py", "retrieval/kubernetes_hybrid.py",
    "retrieval/kubernetes_rerank.py", "storage/kubernetes_search.py",
    "providers/local_embedding.py", "providers/cloud_reranker.py",
    "rag/kubernetes_answer.py", "rag/quote_units.py", "storage/retrieval_cache.py",
)
RANKING_FIELDS = {
    "routes", "rrf_pool", "scope_count", "recall_k", "top_k", "rrf_constant",
    "rerank_comparison",
}


class CacheError(RuntimeError):
    """固定错误，不携带连接信息、查询、缓存值或服务原始异常。"""

    def __init__(self):
        super().__init__(
            "Redis检索缓存读取、写入或合同校验失败；本轮停止，未自动回退或重试。"
            "失败前模型请求可能已发生并产生费用。"
        )


def describe() -> dict:
    """纯配置描述；不导入 Redis SDK、不连接服务。"""
    return {
        "backend": "redis", "enabled": True, "ttl_seconds": TTL_SECONDS,
        "max_payload_bytes": MAX_PAYLOAD_BYTES, "notice": NOTICE,
    }


def validate_info(value: dict) -> dict:
    """供回答和持久化层共用的无敏感字段白名单。"""
    if (
        type(value) is not dict
        or set(value) != {"backend", "status", "ttl_seconds"}
        or value["backend"] != "redis"
        or type(value["status"]) is not str
        or value["status"] not in {"hit", "miss"}
        or type(value["ttl_seconds"]) is not int
        or value["ttl_seconds"] != TTL_SECONDS
    ):
        raise CacheError()
    return dict(value)


def _canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _check(condition: bool) -> None:
    if not condition:
        raise CacheError()


def _only(value, keys) -> None:
    _check(type(value) is dict and set(value) == set(keys))


def _score(value) -> None:
    _check(type(value) in (int, float) and isfinite(value))


def _unique_json(pairs):
    output = {}
    for key, value in pairs:
        _check(key not in output)
        output[key] = value
    return output


def _no_constant(_value):
    raise CacheError()


def _identity(check: dict) -> dict:
    """只在内存里构造合同；Redis 键只有最终 SHA256，不保存这些明文。"""
    from qa_platform.retrieval import kubernetes_hybrid as hybrid
    from qa_platform.retrieval import kubernetes_rerank as rerank

    query, plan, records = check["query"], check["plan"], check["records"]
    _check(type(query) is str and bool(query.strip()) and len(query) <= 2000)
    _check(type(records) is list and bool(records))
    ids = [row["chunk_id"] for row in records]
    _check(len(ids) == len(set(ids)))
    _check(all(row["metadata"]["chunk_id"] == row["chunk_id"] for row in records))
    _check(type(check["top_k"]) is int and check["top_k"] == min(hybrid.RECALL_K, len(records)))
    query_input = check.get("query_input")
    _check(query_input is None or type(query_input) is str)
    config = rerank.describe()
    return {
        "schema": SCHEMA,
        "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
        "query_input_sha256": sha256(query_input.encode("utf-8")).hexdigest() if query_input is not None else None,
        "embedding_base_url": check.get("base_url"),
        "document_snapshot": plan["document_snapshot_sha256"],
        "vector_snapshot": plan["vector_snapshot_sha256"],
        "selected_records_sha256": sha256(_canonical(records)).hexdigest(),
        "embedding": plan["embedding"],
        "filter": check["filter"], "dense_top_k": check["top_k"],
        "recall_k": hybrid.RECALL_K, "top_k": hybrid.TOP_K,
        "rrf_constant": hybrid.RRF_CONSTANT,
        "reranker": {"model": config["model"], "url": config["url"]},
        "code_sha256": {
            name: sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()
            for name in CODE_FILES
        },
        "packages": {name: version(name) for name in ("jieba", "rank-bm25")},
    }


def _validate_ranking(ranking: dict, check: dict) -> None:
    """严格白名单和映射检查；不把缓存内容当作可信 Python 对象反序列化。"""
    from qa_platform.retrieval import kubernetes_hybrid as hybrid
    from qa_platform.retrieval import kubernetes_rerank as rerank

    _only(ranking, RANKING_FIELDS)
    for name, expected in (
        ("scope_count", len(check["records"])), ("recall_k", check["top_k"]),
        ("top_k", hybrid.TOP_K), ("rrf_constant", hybrid.RRF_CONSTANT),
    ):
        _check(type(ranking[name]) is int and ranking[name] == expected)
    routes = ranking["routes"]
    _only(routes, ("bm25", "dense", "rrf", "reranker"))
    available = {row["chunk_id"] for row in check["records"]}
    for name, hits in routes.items():
        _check(type(hits) is list and len(hits) <= rerank.MAX_CANDIDATES)
        seen, previous = set(), float("inf")
        for rank, hit in enumerate(hits, 1):
            fields = {"chunk_id", "rank", "score"}
            if name == "rrf":
                fields |= {"ranks", "contributions"}
            if name == "reranker":
                fields |= {"input_index", "rrf_rank", "rank_change"}
            _only(hit, fields)
            chunk_id = hit["chunk_id"]
            _check(type(chunk_id) is str and chunk_id in available and chunk_id not in seen)
            _check(type(hit["rank"]) is int and hit["rank"] == rank)
            _score(hit["score"])
            _check(hit["score"] <= previous)
            if name == "dense":
                _check(-1.0001 <= hit["score"] <= 1.0001)
            if name == "rrf":
                _check(type(hit["ranks"]) is dict and type(hit["contributions"]) is dict)
                _check(all(type(v) is int for v in hit["ranks"].values()))
                for value in hit["contributions"].values():
                    _score(value)
            if name == "reranker":
                _check(all(type(hit[field]) is int for field in ("input_index", "rrf_rank", "rank_change")))
            seen.add(chunk_id)
            previous = hit["score"]
    _check(len(routes["bm25"]) <= check["top_k"])
    _check(len(routes["dense"]) == check["top_k"])
    pool = ranking["rrf_pool"]
    _check(type(pool) is list and 1 <= len(pool) <= rerank.MAX_CANDIDATES)
    # 全池与显示部分沿用相同字段白名单；详细 RRF 数学映射由 prepare_evidence 再核验。
    for hit in pool:
        _only(hit, ("chunk_id", "rank", "score", "ranks", "contributions"))
        _check(type(hit["ranks"]) is dict and type(hit["contributions"]) is dict)
        _check(type(hit["rank"]) is int)
        _check(all(type(value) is int for value in hit["ranks"].values()))
        _score(hit["score"])
        for value in hit["contributions"].values():
            _score(value)
    comparison = ranking["rerank_comparison"]
    _only(comparison, ("model", "candidate_count", "final_k", "input_order"))
    _check(comparison["model"] == rerank.describe()["model"])
    _check(type(comparison["candidate_count"]) is int and comparison["candidate_count"] == len(pool))
    _check(type(comparison["final_k"]) is int and comparison["final_k"] == hybrid.TOP_K)
    _check(type(comparison["input_order"]) is list)
    _check(all(type(value) is str for value in comparison["input_order"]))


def _ranking(result: dict) -> dict:
    routes = deepcopy(result["routes"])
    # 分词与命中词可能泄露问题内容，Redis 只存 ID、排名、分数及其映射。
    for hit in routes["bm25"]:
        hit.pop("matched_terms", None)
    return {
        **{name: deepcopy(result[name]) for name in RANKING_FIELDS - {"routes", "rerank_comparison"}},
        "routes": routes,
        "rerank_comparison": {
            name: deepcopy(result["rerank_comparison"][name])
            for name in ("model", "candidate_count", "final_k", "input_order")
        },
    }


def _restore(ranking: dict, check: dict) -> dict:
    from qa_platform.retrieval.bm25 import tokenize
    from qa_platform.retrieval import kubernetes_rerank as rerank

    output = deepcopy(ranking)
    tokens = list(dict.fromkeys(tokenize(check["query"])))
    ids = {hit["chunk_id"] for hit in output["rrf_pool"]}
    output.update(
        query=check["query"], mode="hybrid", experiment_id=uuid4().hex,
        tokens=tokens, notice=rerank.NOTICE,
        request_counts={"embedding": 0, "http": 0, "reranker": 0, "chat": 0},
        cache={"backend": "redis", "status": "hit", "ttl_seconds": TTL_SECONDS},
        records={
            row["chunk_id"]: {"text": row["text"], "metadata": deepcopy(row["metadata"])}
            for row in check["records"] if row["chunk_id"] in ids
        },
    )
    for hit in output["routes"]["bm25"]:
        terms = set(tokenize(output["records"][hit["chunk_id"]]["text"]))
        hit["matched_terms"] = [term for term in tokens if term in terms]
    output["rerank_comparison"].update(
        notice=rerank.NOTICE,
        request_counts={"embedding": 0, "http": 0, "reranker": 0, "chat": 0},
    )
    return output


class RetrievalCache:
    """固定本机 Redis DB0；首次 ping/get/put 才创建 SDK 客户端。"""

    def __init__(self, *, client_factory=None, port: int = 6379, password: str | None = None):
        if type(port) is not int or port not in {6379, 16379}:
            raise CacheError()
        if (password is not None and (not isinstance(password, str) or not password.strip())) or (
            port == 16379 and password is None
        ):
            raise CacheError()
        self._factory = client_factory
        self._connection = None
        self._port = port
        self._password = password

    def _client(self):
        if self._connection is None:
            from redis import Redis
            from redis.backoff import NoBackoff
            from redis.retry import Retry

            factory = self._factory or Redis
            options = dict(
                host="127.0.0.1", port=self._port, db=0,
                socket_timeout=1, socket_connect_timeout=1,
                decode_responses=False, retry=Retry(NoBackoff(), 0),
                retry_on_timeout=False, retry_on_error=[],
                health_check_interval=0,
            )
            if self._password is not None:
                options["password"] = self._password
            self._connection = factory(**options)
        return self._connection

    def key(self, check: dict) -> str:
        try:
            return PREFIX + sha256(_canonical(_identity(check))).hexdigest()
        except Exception:
            raise CacheError() from None

    def ping(self) -> None:
        try:
            _check(self._client().ping() is True)
        except Exception:
            raise CacheError() from None

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                raise CacheError() from None
            finally:
                self._connection = None

    def get(self, check: dict) -> dict | None:
        try:
            from qa_platform.rag.kubernetes_answer import prepare_evidence

            key = self.key(check)
            raw = self._client().get(key)
            if raw is None:
                return None
            _check(type(raw) is bytes and 0 < len(raw) <= MAX_PAYLOAD_BYTES)
            envelope = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_unique_json,
                parse_constant=_no_constant,
            )
            _only(envelope, ("schema_version", "contract_sha256", "ranking"))
            _check(envelope["schema_version"] == SCHEMA)
            _check(envelope["contract_sha256"] == key[len(PREFIX):])
            ranking = envelope["ranking"]
            _validate_ranking(ranking, check)
            result = _restore(ranking, check)
            prepare_evidence(result)
            return result
        except Exception:
            raise CacheError() from None

    def put(self, check: dict, result: dict) -> None:
        try:
            from qa_platform.rag.kubernetes_answer import prepare_evidence

            _check(result["query"] == check["query"] and "answer_generation" not in result)
            counts = result["request_counts"]
            _only(counts, ("embedding", "http", "reranker", "chat"))
            _check(all(type(value) is int for value in counts.values()))
            _check(counts == {"embedding": 1, "http": 4, "reranker": 1, "chat": 0})
            prepare_evidence(result)
            ranking = _ranking(result)
            _validate_ranking(ranking, check)
            restored = _restore(ranking, check)
            _check(result["records"] == restored["records"])
            key = self.key(check)
            raw = _canonical({
                "schema_version": SCHEMA, "contract_sha256": key[len(PREFIX):],
                "ranking": ranking,
            })
            _check(len(raw) <= MAX_PAYLOAD_BYTES)
            _check(self._client().set(key, raw, ex=TTL_SECONDS) is True)
        except Exception:
            raise CacheError() from None
