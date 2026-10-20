"""Redis retrieval-cache contracts, entirely offline with a tiny expiring fake.

The fake exercises the Redis wire boundary, not a second runtime cache backend.
No test reads local credentials, opens Redis/MySQL/Milvus, or calls a model.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
import socket
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
import pytest
import requests

pytest.importorskip("redis")  # 可选cache依赖未安装时，不影响默认无缓存路径的其他测试。

from qa_platform.rag import kubernetes_answer as answering
from qa_platform.retrieval import kubernetes_hybrid as hybrid
from qa_platform.retrieval import kubernetes_rerank as rerank
from qa_platform.storage import retrieval_cache as cache
from qa_platform import retrieval_lab as lab
from qa_platform.workflows import live_rag as live


QUERY = "SYNTHETIC_QUERY_MUST_NOT_CACHE dns question?"
BODY = "SYNTHETIC_BODY_MUST_NOT_CACHE: synthetic DNS source paragraph."
ZERO = {"embedding": 0, "http": 0, "reranker": 0, "chat": 0}
MODEL = "qwen3.8-flash"
HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cache tests must not contact services or read credentials")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    monkeypatch.setattr(answering, "load_chat_key", forbidden)
    monkeypatch.setattr(rerank, "load_api_key", forbidden)


class FakeRedis:
    """Only the used Redis methods, with an explicitly advanced monotonic clock."""

    def __init__(self):
        self.values = {}
        self.deadlines = {}
        self.calls = []
        self.clock = 0
        self.closed = False

    def get(self, key):
        self.calls.append(("get", key))
        if key in self.deadlines and self.clock >= self.deadlines[key]:
            self.values.pop(key, None)
            self.deadlines.pop(key, None)
        return self.values.get(key)

    def set(self, key, raw, *, ex):
        self.calls.append(("set", key, raw, ex))
        self.values[key] = raw
        self.deadlines[key] = self.clock + ex
        return True

    def ping(self):
        self.calls.append(("ping",))
        return True

    def close(self):
        self.calls.append(("close",))
        self.closed = True


def check_and_result(query=QUERY):
    lexical = [f"chunk-{i:02d}" for i in range(8)]
    dense = [f"chunk-{i:02d}" for i in range(5, 13)]
    records = [
        {"chunk_id": chunk_id, "text": BODY + " " + chunk_id,
         "metadata": {"chunk_id": chunk_id, "title": "Synthetic title",
                      "section_path": ["Synthetic section"],
                      "source_url": "https://example.invalid/source",
                      "review_status": "unreviewed_official_document",
                      "source_path": "/private/LOCAL_PATH_MUST_NOT_CACHE.md",
                      "is_test": False}}
        for chunk_id in sorted(set(lexical) | set(dense))
    ]
    check = {
        "query": query, "top_k": 8, "records": records,
        "filter": 'kb_id == "synthetic"', "index_dir": Path("/synthetic/index"),
        "plan": {
            "document_snapshot_sha256": "a" * 64,
            "vector_snapshot_sha256": "b" * 64,
            "embedding": {"model": "synthetic-embedding", "actual_dimension": 4,
                          "server_model_digest": "c" * 64,
                          "query_instruction": "Synthetic retrieval instruction",
                          "query_format": "Instruct: {instruction}\nQuery: {query}",
                          "document_mode": "body-only", "truncate": False,
                          "url": "http://127.0.0.1:11434/api/embed"},
        },
    }
    pool = []
    for row in records:
        chunk_id = row["chunk_id"]
        ranks = {name: ids.index(chunk_id) + 1 for name, ids in
                 (("bm25", lexical), ("dense", dense)) if chunk_id in ids}
        contributions = {name: 1 / (60 + rank) for name, rank in ranks.items()}
        pool.append({"chunk_id": chunk_id, "ranks": ranks,
                     "contributions": contributions, "score": sum(contributions.values())})
    pool.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(pool, 1):
        row["rank"] = rank
    ranking = []
    for rank, index in enumerate(reversed(range(len(pool))), 1):
        ranking.append({"chunk_id": pool[index]["chunk_id"], "rank": rank,
                        "score": 1 - rank / 100, "input_index": index,
                        "rrf_rank": index + 1, "rank_change": index + 1 - rank})
    result = {
        "query": query, "mode": "hybrid", "tokens": ["dns"],
        "scope_count": 13, "recall_k": 8, "top_k": 5, "rrf_constant": 60,
        "experiment_id": "a" * 32, "notice": "Synthetic only",
        "request_counts": {"embedding": 1, "http": 4, "reranker": 1, "chat": 0},
        "routes": {
            "bm25": [{"chunk_id": key, "rank": i, "score": 20 - i,
                      "matched_terms": ["dns"]} for i, key in enumerate(lexical, 1)],
            "dense": [{"chunk_id": key, "rank": i, "score": 1 - i / 100}
                      for i, key in enumerate(dense, 1)],
            "rrf": deepcopy(pool[:5]), "reranker": ranking,
        },
        "rrf_pool": pool,
        "records": {row["chunk_id"]: {"text": row["text"], "metadata": deepcopy(row["metadata"])}
                    for row in records},
        "rerank_comparison": {
            "model": rerank.MODEL_NAME, "candidate_count": len(pool), "final_k": 5,
            "input_order": [row["chunk_id"] for row in pool],
            "request_counts": {"embedding": 0, "http": 1, "reranker": 1, "chat": 0},
            "notice": rerank.NOTICE,
        },
    }
    return check, result


@pytest.fixture
def backend():
    fake = FakeRedis()
    factory = Mock(return_value=fake)
    instance = cache.RetrievalCache(client_factory=factory)
    yield instance, fake, factory
    instance.close()


def test_constructor_and_describe_are_lazy_zero_io():
    factory = Mock(side_effect=AssertionError("Plan created a Redis client"))
    instance = cache.RetrievalCache(client_factory=factory)
    plan = cache.describe()
    assert plan["backend"] == "redis" and plan["enabled"] is True
    assert plan["ttl_seconds"] == 300
    assert "不缓存" in plan["notice"] and "不隐式回退" in plan["notice"]
    instance.close()
    factory.assert_not_called()


def test_first_ping_uses_fixed_local_server_bounded_timeout_and_zero_retries(backend):
    instance, fake, factory = backend
    instance.ping()
    instance.ping()
    factory.assert_called_once()
    options = factory.call_args.kwargs
    assert options["host"] == "127.0.0.1" and options["port"] == 6379 and options["db"] == 0
    assert options["socket_timeout"] == options["socket_connect_timeout"] == 1
    assert options["decode_responses"] is False
    assert options["retry_on_timeout"] is False and options["retry_on_error"] == []
    assert options["retry"]._retries == 0
    assert fake.calls == [("ping",), ("ping",)]


def test_key_is_deterministic_opaque_and_requires_no_redis(backend):
    instance, _, factory = backend
    check, _ = check_and_result()
    key = instance.key(check)
    assert key == instance.key(deepcopy(check))
    assert re.fullmatch(re.escape(cache.PREFIX) + r"[0-9a-f]{64}", key)
    assert QUERY not in key and "synthetic" not in key
    factory.assert_not_called()


@pytest.mark.parametrize("part", ["query", "documents", "vectors", "model", "digest", "instruction", "endpoint", "query_input", "base_url", "filter", "body", "metadata", "scope"])
def test_key_isolates_every_frozen_retrieval_contract(backend, part):
    instance, _, _ = backend
    check, _ = check_and_result()
    baseline = instance.key(check)
    altered = deepcopy(check)
    if part == "query":
        altered["query"] += " different"
    elif part == "documents":
        altered["plan"]["document_snapshot_sha256"] = "d" * 64
    elif part == "vectors":
        altered["plan"]["vector_snapshot_sha256"] = "d" * 64
    elif part in {"model", "digest", "instruction", "endpoint"}:
        field = {"model": "model", "digest": "server_model_digest",
                 "instruction": "query_instruction", "endpoint": "url"}[part]
        altered["plan"]["embedding"][field] = "different"
    elif part in {"query_input", "base_url"}:
        altered[part] = "different"
    elif part == "filter":
        altered["filter"] += " and is_test == false"
    elif part == "body":
        altered["records"][0]["text"] += " different"
    elif part == "metadata":
        altered["records"][0]["metadata"]["review_status"] = "reviewed"
    else:
        altered["records"] = altered["records"][:-1]
    assert instance.key(altered) != baseline


@pytest.mark.parametrize("part", ["rrf_constant", "top_k", "recall_k", "rerank_model", "rerank_endpoint", "package_version", "source_code"])
def test_key_changes_with_algorithms_model_service_and_source_fingerprints(backend, monkeypatch, tmp_path, part):
    instance, _, _ = backend
    check, _ = check_and_result()
    baseline = instance.key(check)
    if part in {"rrf_constant", "top_k", "recall_k"}:
        name = {"rrf_constant": "RRF_CONSTANT", "top_k": "TOP_K", "recall_k": "RECALL_K"}[part]
        monkeypatch.setattr(hybrid, name, getattr(hybrid, name) - 1)
        if part == "recall_k":
            check["top_k"] -= 1
    elif part == "rerank_model":
        monkeypatch.setattr(rerank, "MODEL_NAME", "different-reranker")
    elif part == "rerank_endpoint":
        monkeypatch.setattr(rerank, "API_URL", "https://example.invalid/new-rerank")
    elif part == "package_version":
        monkeypatch.setattr(cache, "version", lambda name: "different-version")
    else:
        (tmp_path / "synthetic.py").write_text("synthetic source", encoding="utf-8")
        monkeypatch.setattr(cache, "PACKAGE_ROOT", tmp_path)
        monkeypatch.setattr(cache, "CODE_FILES", ("synthetic.py",))
    assert instance.key(check) != baseline


def test_put_only_stores_json_ranking_with_fixed_expiry_and_no_secrets_or_answer(backend):
    instance, fake, _ = backend
    check, result = check_and_result()
    original = deepcopy(result)
    instance.put(check, result)
    assert result == original
    assert len(fake.calls) == 1
    _, key, raw, expiry = fake.calls[0]
    assert key == instance.key(check) and expiry == 300 and isinstance(raw, bytes)
    envelope = json.loads(raw)
    assert set(envelope) == {"schema_version", "contract_sha256", "ranking"}
    assert envelope["schema_version"] == cache.SCHEMA
    assert envelope["contract_sha256"] == key[len(cache.PREFIX):]
    assert set(envelope["ranking"]) == cache.RANKING_FIELDS
    assert all("matched_terms" not in row for row in envelope["ranking"]["routes"]["bm25"])
    text = raw.decode()
    for forbidden in (QUERY, BODY, "LOCAL_PATH", "records", "tokens", "answer_generation", "embedding"):
        assert forbidden not in text


def test_hit_rehydrates_current_records_validates_evidence_and_zeroes_all_model_counts(backend):
    instance, fake, _ = backend
    check, result = check_and_result()
    instance.put(check, result)
    first = instance.get(check)
    second = instance.get(check)
    assert first["request_counts"] == second["request_counts"] == ZERO
    assert first["cache"] == {"backend": "redis", "status": "hit", "ttl_seconds": 300}
    assert first["experiment_id"] != second["experiment_id"] != result["experiment_id"]
    assert first["records"] == result["records"]
    assert first["routes"]["reranker"] == result["routes"]["reranker"]
    assert first["rrf_pool"] == result["rrf_pool"]
    assert first["rerank_comparison"]["request_counts"] == ZERO
    evidence = answering.prepare_evidence(first)
    assert len(evidence) == 5 and all(row["text"].startswith(BODY) for row in evidence.values())
    first_id = first["routes"]["reranker"][0]["chunk_id"]
    first["records"][first_id]["metadata"]["title"] = "modified"
    assert second["records"][first_id]["metadata"]["title"] == "Synthetic title"
    assert len([call for call in fake.calls if call[0] == "set"]) == 1


def test_expiration_is_not_extended_by_read_and_later_get_is_a_normal_miss(backend):
    instance, fake, _ = backend
    check, result = check_and_result()
    assert instance.get(check) is None
    instance.put(check, result)
    fake.clock = 299
    assert instance.get(check) is not None
    fake.clock = 300
    assert instance.get(check) is None
    assert len([row for row in fake.calls if row[0] == "set"]) == 1


@pytest.mark.parametrize("mutation", ["query", "already_answered", "wrong_counts", "bad_order", "unknown_record", "changed_original"])
def test_put_rejects_unvalidated_or_mismatched_snapshot_before_any_redis_write(backend, mutation):
    instance, fake, _ = backend
    check, result = check_and_result()
    if mutation == "query":
        result["query"] = "another query"
    elif mutation == "already_answered":
        result["answer_generation"] = {"answer": "should never cache"}
    elif mutation == "wrong_counts":
        result["request_counts"]["reranker"] = 0
    elif mutation == "bad_order":
        result["routes"]["reranker"].reverse()
    elif mutation == "unknown_record":
        result["routes"]["reranker"][0]["chunk_id"] = "not-in-current-snapshot"
    else:
        next(iter(result["records"].values()))["text"] = "Modified source not in current check."
    with pytest.raises(cache.CacheError):
        instance.put(check, result)
    assert not any(row[0] == "set" for row in fake.calls)


@pytest.mark.parametrize("poison", ["empty", "text_not_bytes", "pickle", "oversized", "duplicate_json_key", "nan", "schema", "contract", "unknown_field", "unknown_id", "bad_rrf", "truncated_rerank", "bad_mapping", "bool_rank"])
def test_corrupt_cache_value_is_a_sanitized_failure_not_a_miss_or_silent_retry(backend, poison):
    instance, fake, _ = backend
    check, result = check_and_result()
    instance.put(check, result)
    key = instance.key(check)
    envelope = json.loads(fake.values[key])
    if poison == "empty":
        raw = b""
    elif poison == "text_not_bytes":
        raw = "PRIVATE_BAD_CACHE"
    elif poison == "pickle":
        raw = b"\x80\x04PRIVATE_BAD_CACHE"
    elif poison == "oversized":
        raw = b"X" * (cache.MAX_PAYLOAD_BYTES + 1)
    elif poison == "duplicate_json_key":
        raw = b'{"schema_version":"PRIVATE_BAD_CACHE","schema_version":"x"}'
    elif poison == "nan":
        raw = b'{"PRIVATE_BAD_CACHE":NaN}'
    else:
        ranking = envelope["ranking"]
        if poison == "schema":
            envelope["schema_version"] = "PRIVATE_BAD_CACHE"
        elif poison == "contract":
            envelope["contract_sha256"] = "PRIVATE_BAD_CACHE"
        elif poison == "unknown_field":
            ranking["answer_generation"] = "PRIVATE_BAD_CACHE"
        elif poison == "unknown_id":
            ranking["routes"]["dense"][0]["chunk_id"] = "PRIVATE_BAD_CACHE"
        elif poison == "bad_rrf":
            ranking["rrf_pool"][0]["score"] += 1
        elif poison == "truncated_rerank":
            ranking["routes"]["reranker"].pop()
        elif poison == "bad_mapping":
            ranking["routes"]["reranker"][0]["input_index"] = 99
        else:
            ranking["routes"]["reranker"][0]["rank"] = True
        raw = json.dumps(envelope).encode()
    fake.values[key] = raw
    before = len(fake.calls)
    with pytest.raises(cache.CacheError) as caught:
        instance.get(check)
    assert "PRIVATE_BAD_CACHE" not in str(caught.value)
    assert "未自动回退或重试" in str(caught.value)
    assert [row[0] for row in fake.calls[before:]] == ["get"]


@pytest.mark.parametrize("operation", ["ping", "get", "set"])
def test_redis_transport_failure_is_sanitized_and_attempted_once(backend, monkeypatch, operation):
    instance, fake, _ = backend
    check, result = check_and_result()
    failure = Mock(side_effect=RuntimeError("PRIVATE_REDIS_CONNECTION_SECRET"))
    monkeypatch.setattr(fake, operation, failure)
    with pytest.raises(cache.CacheError) as caught:
        if operation == "ping":
            instance.ping()
        elif operation == "get":
            instance.get(check)
        else:
            instance.put(check, result)
    failure.assert_called_once()
    assert "PRIVATE_REDIS_CONNECTION_SECRET" not in str(caught.value)


@pytest.mark.parametrize("value", [None, {}, {"backend": "redis", "status": "hit", "ttl_seconds": True},
                                   {"backend": "redis", "status": "hit", "ttl_seconds": 299},
                                   {"backend": "redis", "status": "hit", "ttl_seconds": 300, "key": "secret"},
                                   {"backend": "redis", "status": "disabled", "ttl_seconds": 300}])
def test_public_cache_info_only_accepts_explicit_safe_whitelist(value):
    with pytest.raises(cache.CacheError):
        cache.validate_info(value)


@pytest.fixture
def pipeline(monkeypatch):
    """Actual graph/cache/evidence validation; only expensive providers are faked."""
    calls = []

    def prepare(index_dir, query):
        calls.append("prepare")
        check, _ = check_and_result(query)
        return check

    def compare(check, mode):
        assert mode == "hybrid"
        calls.append("retrieve")
        _, result = check_and_result(check["query"])
        del result["routes"]["reranker"]
        del result["rerank_comparison"]
        result["request_counts"] = {"embedding": 1, "http": 3, "reranker": 0, "chat": 0}
        return result

    def reranking(result):
        calls.append("rerank")
        _, output = check_and_result(result["query"])
        output["experiment_id"] = result["experiment_id"]
        return output

    def generate(result, model):
        calls.append("answer")
        sources = answering.prepare_evidence(result)
        output = deepcopy(result)
        output["answer_generation"] = {
            "model": model, "sources": sources,
            "answer": {"status": "partial", "claims": [{
                "text": f"Fresh synthetic response {calls.count('answer')}.",
                "evidence": [{"source_id": "S1", "quote_id": "Q1"}],
            }], "missing_info": ["Actual cluster information is still required."]},
        }
        output["request_counts"]["http"] += 1
        output["request_counts"]["chat"] += 1
        return output

    monkeypatch.setattr(live.hybrid, "prepare", prepare)
    monkeypatch.setattr(live.hybrid, "compare", compare)
    monkeypatch.setattr(live.reranking, "rerank", reranking)
    monkeypatch.setattr(live.answering, "generate", generate)
    return calls


def test_real_graph_miss_then_hit_skips_only_retrieval_and_reranking_not_generation(backend, pipeline, tmp_path):
    instance, fake, _ = backend
    first = live.run(QUERY, MODEL, tmp_path, retrieval_cache=instance)
    second = live.run(QUERY, MODEL, tmp_path, retrieval_cache=instance)
    assert first["cache"] == {"backend": "redis", "status": "miss", "ttl_seconds": 300}
    assert second["cache"] == {"backend": "redis", "status": "hit", "ttl_seconds": 300}
    assert first["request_counts"] == {"embedding": 1, "http": 5, "reranker": 1, "chat": 1}
    assert second["request_counts"] == {"embedding": 0, "http": 1, "reranker": 0, "chat": 1}
    assert first["sources"] == second["sources"]
    assert first["answer"] != second["answer"]  # Not an answer cache.
    assert pipeline == ["prepare", "retrieve", "rerank", "answer", "prepare", "answer"]
    assert first["steps"] == second["steps"] == ["retrieve", "rerank", "answer"]
    assert len([row for row in fake.calls if row[0] == "set"]) == 1
    assert "LOCAL_PATH" not in json.dumps(second)


def test_changing_answer_model_can_reuse_ranking_but_still_generates_selected_model(backend, pipeline, tmp_path):
    instance, _, _ = backend
    live.run(QUERY, MODEL, tmp_path, retrieval_cache=instance)
    second = live.run(QUERY, "glm", tmp_path, retrieval_cache=instance)
    assert second["model"] == "glm" and second["cache"]["status"] == "hit"
    assert pipeline.count("retrieve") == pipeline.count("rerank") == 1
    assert pipeline.count("answer") == 2


def test_unchanged_default_run_never_creates_or_uses_redis(pipeline, monkeypatch, tmp_path):
    forbidden = Mock(side_effect=AssertionError("Default path touched Redis"))
    monkeypatch.setattr(cache, "RetrievalCache", forbidden)
    first = live.run(QUERY, MODEL, tmp_path)
    second = live.run(QUERY, MODEL, tmp_path)
    assert "cache" not in first and "cache" not in second
    assert pipeline.count("retrieve") == pipeline.count("rerank") == pipeline.count("answer") == 2
    forbidden.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["get", "put"])
def test_cache_failure_stops_graph_once_without_generating_answer(backend, pipeline, monkeypatch, tmp_path, failure_stage):
    instance, fake, _ = backend
    failed = Mock(side_effect=RuntimeError("PRIVATE_REDIS_DETAIL"))
    monkeypatch.setattr(fake, "get" if failure_stage == "get" else "set", failed)
    with pytest.raises(live.RAGRunError) as caught:
        live.run(QUERY, MODEL, tmp_path, retrieval_cache=instance)
    assert "PRIVATE_REDIS_DETAIL" not in str(caught.value)
    assert caught.value.stage == ("retrieve" if failure_stage == "get" else "rerank")
    assert caught.value.completed_steps == ([] if failure_stage == "get" else ["retrieve"])
    assert pipeline == (["prepare"] if failure_stage == "get" else ["prepare", "retrieve", "rerank"])
    failed.assert_called_once()


def test_answer_failure_does_not_invalidate_successful_ranking_cache(backend, pipeline, monkeypatch, tmp_path):
    instance, _, _ = backend
    generate = live.answering.generate
    monkeypatch.setattr(live.answering, "generate", Mock(side_effect=ValueError("Invalid draft")))
    with pytest.raises(live.RAGRunError) as caught:
        live.run(QUERY, MODEL, tmp_path, retrieval_cache=instance)
    assert caught.value.stage == "answer"
    monkeypatch.setattr(live.answering, "generate", generate)
    result = live.run(QUERY, MODEL, tmp_path, retrieval_cache=instance)
    assert result["cache"]["status"] == "hit"
    assert pipeline == ["prepare", "retrieve", "rerank", "prepare", "answer"]


def test_cache_enabled_plan_is_zero_io_and_disabled_plan_stays_unchanged(backend, tmp_path):
    instance, _, factory = backend
    with TestClient(lab.create_app(tmp_path, retrieval_cache=instance, cache_backend="redis"),
                    base_url="http://127.0.0.1") as client:
        response = client.get("/api/qa/plan")
        assert response.status_code == 200 and response.json()["cache"] == cache.describe()
    with TestClient(lab.create_app(tmp_path), base_url="http://127.0.0.1") as client:
        assert "cache" not in client.get("/api/qa/plan").json()
    factory.assert_not_called()


@pytest.mark.parametrize("backend_name, injected", [("redis", False), ("none", True), ("unknown", False)])
def test_cache_backend_must_be_explicit_and_consistent(tmp_path, backend_name, injected):
    with pytest.raises(ValueError):
        lab.create_app(tmp_path, retrieval_cache=object() if injected else None, cache_backend=backend_name)


def test_agent_api_bypasses_cache_without_changing_legacy_three_argument_run(backend, monkeypatch, tmp_path):
    from qa_platform.workflows import agent_rag

    instance, _, factory = backend
    run = Mock(return_value={"synthetic_agent": True})
    monkeypatch.setattr(agent_rag, "run", run)
    with TestClient(lab.create_app(tmp_path, retrieval_cache=instance, cache_backend="redis"),
                    base_url="http://127.0.0.1") as client:
        response = client.post("/api/qa", headers=HEADERS,
                               json={"query": QUERY, "model": MODEL, "confirmed": True, "mode": "agent"})
    assert response.status_code == 200
    run.assert_called_once_with(QUERY, MODEL, tmp_path)
    factory.assert_not_called()


@pytest.mark.parametrize("endpoint", ["/api/qa", "/api/qa/stream"])
@pytest.mark.parametrize("failure_stage", ["get", "put"])
def test_failed_cache_never_commits_successful_question_or_exposes_draft(backend, pipeline, monkeypatch, tmp_path, endpoint, failure_stage):
    instance, fake, _ = backend
    failed = Mock(side_effect=RuntimeError("PRIVATE_REDIS_DETAIL"))
    monkeypatch.setattr(fake, "get" if failure_stage == "get" else "set", failed)
    store = Mock()
    with TestClient(lab.create_app(tmp_path, conversation_store=store, retrieval_cache=instance, cache_backend="redis"),
                    base_url="http://127.0.0.1") as client:
        response = client.post(endpoint, headers=HEADERS,
                               json={"query": QUERY, "model": MODEL, "confirmed": True, "conversation": True})
    assert response.status_code == (503 if endpoint == "/api/qa" else 200)
    if endpoint.endswith("/stream"):
        assert "event: error" in response.text and "event: result" not in response.text
    assert "PRIVATE_REDIS_DETAIL" not in response.text and "Fresh synthetic response" not in response.text
    assert "answer" not in pipeline
    store.commit.assert_not_called()
    failed.assert_called_once()


def test_api_mysql_keeps_cache_metadata_and_can_restore_it_after_cache_is_disabled(backend, pipeline, tmp_path):
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine
    from qa_platform.storage import sql_conversations as sql

    instance, _, _ = backend
    engine = create_engine("sqlite+pysqlite:///" + str(tmp_path / "synthetic.sqlite3"))
    sql.Base.metadata.create_all(engine)  # Explicit temporary test schema, not runtime fallback.
    store = sql.SQLConversationStore(engine)
    try:
        with TestClient(lab.create_app(tmp_path, conversation_store=store, storage_backend="mysql",
                                       retrieval_cache=instance, cache_backend="redis"),
                        base_url="http://127.0.0.1") as client:
            payload = {"query": QUERY, "model": MODEL, "confirmed": True, "conversation": True}
            assert client.post("/api/qa", headers=HEADERS, json=payload).json()["cache"]["status"] == "miss"
            result = client.post("/api/qa", headers=HEADERS, json=payload).json()
            assert result["cache"]["status"] == "hit"
            assert result["request_counts"] == {"embedding": 0, "http": 1, "reranker": 0, "chat": 1}
        before = list(pipeline)
        with TestClient(lab.create_app(tmp_path, conversation_store=store, storage_backend="mysql"),
                        base_url="http://127.0.0.1") as client:
            assert "cache" not in client.get("/api/qa/plan").json()
            restored = client.post("/api/qa/conversation/load", headers=HEADERS,
                                   json={"conversation_id": result["conversation"]["id"]}).json()["turns"][0]
        assert restored == result
        assert pipeline == before  # Restoring is zero model requests, regardless of current cache setting.
    finally:
        engine.dispose()


def test_followup_cache_hit_still_rewrites_history_and_counts_two_new_chat_requests(backend, pipeline, monkeypatch, tmp_path):
    from qa_platform.workflows import conversation

    instance, _, _ = backend
    contextualize = Mock(return_value=QUERY)
    monkeypatch.setattr(conversation, "contextualize", contextualize)
    with TestClient(lab.create_app(tmp_path, retrieval_cache=instance, cache_backend="redis"),
                    base_url="http://127.0.0.1") as client:
        payload = {"query": QUERY, "model": MODEL, "confirmed": True, "conversation": True}
        first = client.post("/api/qa", headers=HEADERS, json=payload).json()
        followup = {**payload, "query": "What next?", "conversation_id": first["conversation"]["id"]}
        response = client.post("/api/qa", headers=HEADERS, json=followup)
    assert response.status_code == 200
    result = response.json()
    assert result["query"] == "What next?" and result["conversation"]["effective_query"] == QUERY
    assert result["conversation"]["rewritten"] is True and result["conversation"]["turn"] == 2
    assert result["cache"]["status"] == "hit"
    assert result["request_counts"] == {"embedding": 0, "http": 2, "reranker": 0, "chat": 2}
    assert result["steps"] == ["contextualize", "retrieve", "rerank", "answer"]
    contextualize.assert_called_once()
    assert contextualize.call_args.args[0] == "What next?"
    assert contextualize.call_args.args[1][0]["query"] == QUERY
    assert pipeline.count("retrieve") == pipeline.count("rerank") == 1
    assert pipeline.count("answer") == 2


def test_server_refuses_to_start_when_explicit_redis_ping_fails(monkeypatch, capsys):
    import sys
    import uvicorn

    fake = Mock()
    fake.ping.side_effect = cache.CacheError()
    monkeypatch.setattr(cache, "RetrievalCache", Mock(return_value=fake))
    run = Mock(side_effect=AssertionError("Failed Redis started a server"))
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(sys, "argv", ["serve_retrieval_lab.py", "--cache", "redis"])
    assert lab.main() == 1
    run.assert_not_called()
    fake.ping.assert_called_once()
    fake.close.assert_called_once()
    assert "未回退" in capsys.readouterr().out


@pytest.mark.parametrize("inconsistency", ["hit_with_miss_counts", "miss_with_hit_counts"])
def test_api_rejects_inconsistent_success_cache_tag_and_counts_before_return_or_commit(
    backend, pipeline, monkeypatch, tmp_path, inconsistency,
):
    instance, _, _ = backend
    generate = live.answering.generate

    def contradict(result, model):
        output = generate(result, model)
        if inconsistency == "hit_with_miss_counts":
            output["cache"]["status"] = "hit"
        else:
            output["request_counts"] = {"embedding": 0, "http": 1, "reranker": 0, "chat": 1}
        return output

    monkeypatch.setattr(live.answering, "generate", contradict)
    store = Mock()
    with TestClient(lab.create_app(tmp_path, conversation_store=store,
                                   retrieval_cache=instance, cache_backend="redis"),
                    base_url="http://127.0.0.1") as client:
        response = client.post("/api/qa", headers=HEADERS,
                               json={"query": QUERY, "model": MODEL, "confirmed": True, "conversation": True})
    assert response.status_code == 503
    assert "Fresh synthetic response" not in response.text
    assert pipeline == ["prepare", "retrieve", "rerank", "answer"]
    store.commit.assert_not_called()


def test_sql_rejects_cached_followup_without_its_additional_contextualize_request_before_commit(
    backend, pipeline, tmp_path,
):
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import Session
    from qa_platform.storage import sql_conversations as sql

    instance, _, _ = backend
    live.run(QUERY, MODEL, tmp_path, retrieval_cache=instance)
    result = live.run(QUERY, MODEL, tmp_path, retrieval_cache=instance)
    assert result["cache"]["status"] == "hit"
    result["steps"].insert(0, "contextualize")
    # A follow-up must count rewriting plus generation, not generation alone.
    assert result["request_counts"]["chat"] == result["request_counts"]["http"] == 1
    engine = create_engine("sqlite+pysqlite:///" + str(tmp_path / "invalid-counts.sqlite3"))
    sql.Base.metadata.create_all(engine)
    try:
        store = sql.SQLConversationStore(engine)
        with pytest.raises(ValueError, match="缓存状态与持久化请求计数不一致"):
            store.commit(None, QUERY, QUERY, result)
        with Session(engine) as session:
            assert session.scalar(select(func.count()).select_from(sql.ConversationRecord)) == 0
            assert session.scalar(select(func.count()).select_from(sql.TurnRecord)) == 0
    finally:
        engine.dispose()
