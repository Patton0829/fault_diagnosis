"""Offline reranking contracts: frozen candidates, one request, explicit consent.

All documents and credentials are synthetic. MockTransport replaces the cloud;
the HTTP API runs in-process and no test may read a real .env or open a socket.
"""

import builtins
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
from pathlib import Path
import socket
from threading import Event
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
import pytest
import requests

from qa_platform import retrieval_lab as lab
from qa_platform.retrieval import kubernetes_rerank as reranking


HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline test attempted network or .env access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    original_open, original_io_open = builtins.open, io.open

    def guard_open(original):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                name = Path(file).name
                if name == ".env" or name.startswith(".env."):
                    forbidden()
            return original(file, *args, **kwargs)

        return guarded

    monkeypatch.setattr(builtins, "open", guard_open(original_open))
    monkeypatch.setattr(io, "open", guard_open(original_io_open))


def make_result(*, separate=False):
    """Thirteen candidates, three overlaps; or the maximum sixteen candidates."""
    lexical_ids = [f"synthetic-{position:02d}" for position in range(8)]
    dense_ids = [f"synthetic-{position:02d}" for position in range(8 if separate else 5, 16 if separate else 13)]
    lexical = [
        {"chunk_id": chunk_id, "rank": rank, "score": 20.0 - rank, "matched_terms": ["dns"]}
        for rank, chunk_id in enumerate(lexical_ids, 1)
    ]
    dense = [
        {"chunk_id": chunk_id, "rank": rank, "score": 0.9 - rank / 100}
        for rank, chunk_id in enumerate(dense_ids, 1)
    ]
    pool = []
    for chunk_id in sorted(set(lexical_ids) | set(dense_ids)):
        ranks = {}
        for name, route in (("bm25", lexical_ids), ("dense", dense_ids)):
            if chunk_id in route:
                ranks[name] = route.index(chunk_id) + 1
        contributions = {name: 1 / (60 + rank) for name, rank in ranks.items()}
        pool.append({"chunk_id": chunk_id, "score": sum(contributions.values()), "ranks": ranks, "contributions": contributions})
    pool.sort(key=lambda hit: hit["score"], reverse=True)
    for rank, hit in enumerate(pool, 1):
        hit["rank"] = rank
    return {
        "query": "Synthetic DNS question?",
        "mode": "hybrid",
        "tokens": ["synthetic", "dns"],
        "scope_count": 156,
        "recall_k": 8,
        "top_k": 5,
        "rrf_constant": 60,
        "request_counts": {"embedding": 1, "http": 3, "reranker": 0, "chat": 0},
        "routes": {"bm25": lexical, "dense": dense, "rrf": deepcopy(pool[:5])},
        "rrf_pool": pool,
        "records": {
            hit["chunk_id"]: {
                "text": f"Original body {hit['chunk_id']}\n```shell\nkubectl synthetic\n```",
                "metadata": {
                    "chunk_id": hit["chunk_id"],
                    "title": "MetadataTitleMustNotBeSent",
                    "source_url": "https://example.invalid/MetadataSourceMustNotBeSent",
                    "source_path": "/private/SyntheticPathMustNotBeSent.md",
                    "review_status": "unreviewed_official_document",
                    "nested": {"sections": ["SyntheticSection"]},
                },
            }
            for hit in pool
        },
        "notice": "Original synthetic notice",
    }


def mock_cloud(result, *, response_payload=None, status=200, error=None):
    sent, clients, options = [], [], []
    count = len(result["rrf_pool"])
    if response_payload is None:
        response_payload = {
            "output": {"results": [
                {"index": index, "relevance_score": (index + 1) / count}
                for index in range(count)
            ]}
        }

    def respond(request):
        sent.append(request)
        if error is not None:
            raise error
        return httpx.Response(status, json=response_payload)

    def factory(**kwargs):
        options.append(kwargs)
        client = httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    return factory, sent, clients, options


def test_describe_is_a_zero_io_plan(monkeypatch):
    key_loader = Mock(side_effect=AssertionError("Plan must not read credentials"))
    monkeypatch.setattr(reranking, "load_api_key", key_loader)
    plan = reranking.describe()
    assert plan["model"] == reranking.MODEL_NAME
    assert plan["url"] == reranking.API_URL
    assert plan["max_candidates"] == 16
    assert "云端" in plan["notice"] and "不自动重试" in plan["notice"]
    key_loader.assert_not_called()


@pytest.mark.parametrize("separate", [False, True])
def test_rerank_sends_all_original_bodies_once_and_preserves_every_baseline(separate):
    base = make_result(separate=separate)
    before = deepcopy(base)
    factory, sent, clients, options = mock_cloud(base)
    key_loader = Mock(return_value="synthetic-credential")
    result = reranking.rerank(base, client_factory=factory, api_key_loader=key_loader)
    assert base == before
    assert len(sent) == len(clients) == len(options) == 1
    assert clients[0].is_closed
    key_loader.assert_called_once_with()
    assert options[0] == {
        "headers": {"Authorization": "Bearer synthetic-credential"},
        "timeout": 60.0, "trust_env": False, "follow_redirects": False,
    }
    request = sent[0]
    assert request.method == "POST" and str(request.url) == reranking.API_URL
    payload = json.loads(request.content)
    pool = base["rrf_pool"]
    assert payload == {
        "model": reranking.MODEL_NAME,
        "input": {"query": base["query"], "documents": [base["records"][hit["chunk_id"]]["text"] for hit in pool]},
        "parameters": {"top_n": len(pool)},
    }
    assert len(payload["input"]["documents"]) > len(base["routes"]["rrf"])
    assert "MustNotBeSent" not in request.content.decode()
    assert "metadata" not in payload and "vector" not in payload
    for route in ("bm25", "dense", "rrf"):
        assert result["routes"][route] == base["routes"][route]
    assert result["rrf_pool"] == base["rrf_pool"]
    assert result["records"] == base["records"]
    assert result["request_counts"] == {"embedding": 1, "http": 4, "reranker": 1, "chat": 0}
    assert result["rerank_comparison"] == {
        "candidate_count": len(pool), "final_k": 5, "model": reranking.MODEL_NAME,
        "input_order": [hit["chunk_id"] for hit in pool],
        "request_counts": {"embedding": 0, "http": 1, "reranker": 1, "chat": 0},
        "notice": reranking.NOTICE,
    }
    assert len(result["routes"]["reranker"]) == len(pool)
    for rank, hit in enumerate(result["routes"]["reranker"], 1):
        index = len(pool) - rank
        assert hit == {
            "chunk_id": pool[index]["chunk_id"], "rank": rank,
            "score": (index + 1) / len(pool), "input_index": index,
            "rrf_rank": index + 1, "rank_change": index + 1 - rank,
        }
    assert result["routes"]["reranker"][0]["chunk_id"] not in {
        hit["chunk_id"] for hit in base["routes"]["rrf"]
    }
    # A caller mutating the returned nested data must not mutate its frozen input.
    first = pool[0]["chunk_id"]
    result["records"][first]["metadata"]["nested"]["sections"].append("changed")
    result["routes"]["bm25"][0]["matched_terms"].append("changed")
    assert base == before


@pytest.mark.parametrize("problem", [
    "bm25_only", "query", "truncated_pool", "duplicate_pool", "extra_record",
    "missing_record", "empty_body", "rank", "contribution", "score", "nan",
    "display_prefix", "duplicate_recall", "bad_counts", "already_reranked",
])
def test_invalid_frozen_candidate_contract_fails_before_key_or_http(problem):
    base = make_result()
    first = base["rrf_pool"][0]
    if problem == "bm25_only":
        base["mode"] = "bm25"
    elif problem == "query":
        base["query"] = "  "
    elif problem == "truncated_pool":
        base["rrf_pool"] = base["rrf_pool"][:5]
    elif problem == "duplicate_pool":
        base["rrf_pool"][-1] = deepcopy(first)
    elif problem == "extra_record":
        base["records"]["not_retrieved"] = {"text": "Injected body", "metadata": {}}
    elif problem == "missing_record":
        del base["records"][first["chunk_id"]]
    elif problem == "empty_body":
        base["records"][first["chunk_id"]]["text"] = " "
    elif problem == "rank":
        first["rank"] = True
    elif problem == "contribution":
        first["contributions"]["bm25"] = 0.5
    elif problem == "score":
        first["score"] = 0.9
    elif problem == "nan":
        first["score"] = float("nan")
    elif problem == "display_prefix":
        base["routes"]["rrf"].reverse()
    elif problem == "duplicate_recall":
        base["routes"]["bm25"][-1]["chunk_id"] = base["routes"]["bm25"][0]["chunk_id"]
    elif problem == "bad_counts":
        base["request_counts"]["http"] = True
    else:
        base["routes"]["reranker"] = []
    factory = Mock(side_effect=AssertionError("Invalid pool created an HTTP client"))
    key_loader = Mock(side_effect=AssertionError("Invalid pool read credentials"))
    with pytest.raises((ValueError, KeyError, TypeError)):
        reranking.rerank(base, client_factory=factory, api_key_loader=key_loader)
    factory.assert_not_called()
    key_loader.assert_not_called()


@pytest.mark.parametrize("credential", [None, "", "  ", False, 123])
def test_missing_or_invalid_credential_creates_no_http_client(credential):
    factory = Mock(side_effect=AssertionError("Missing credential created client"))
    with pytest.raises(ValueError):
        reranking.rerank(make_result(), client_factory=factory, api_key_loader=lambda: credential)
    factory.assert_not_called()


@pytest.mark.parametrize("problem", [
    "root", "code", "output", "results", "short", "extra", "item", "bool_index",
    "negative", "outside", "duplicate", "string_score", "bool_score", "nan", "infinity",
])
def test_malformed_cloud_responses_are_rejected_without_partial_results(problem):
    base = make_result()
    before = deepcopy(base)
    count = len(base["rrf_pool"])
    items = [{"index": index, "relevance_score": 0.1} for index in range(count)]
    payload = {"output": {"results": items}}
    if problem == "root":
        payload = []
    elif problem == "code":
        payload["code"] = "SyntheticServiceError"
    elif problem == "output":
        payload["output"] = []
    elif problem == "results":
        payload["output"]["results"] = {}
    elif problem == "short":
        items.pop()
    elif problem == "extra":
        items.append({"index": count, "relevance_score": 0.1})
    elif problem == "item":
        items[0] = []
    elif problem in {"bool_index", "negative", "outside", "duplicate"}:
        items[0]["index"] = {"bool_index": True, "negative": -1, "outside": count, "duplicate": 1}[problem]
    else:
        items[0]["relevance_score"] = {"string_score": "0.9", "bool_score": True, "nan": float("nan"), "infinity": float("inf")}[problem]
    # NaN/Infinity are deliberately encoded as raw malformed JSON numbers rather
    # than passing through httpx's strict json= encoder.
    sent, clients = [], []

    def respond(request):
        sent.append(request)
        return httpx.Response(200, content=json.dumps(payload), headers={"content-type": "application/json"})

    def factory(**kwargs):
        client = httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    with pytest.raises(ValueError):
        reranking.rerank(base, client_factory=factory, api_key_loader=lambda: "synthetic-key")
    assert len(sent) == 1 and clients[0].is_closed
    assert base == before


@pytest.mark.parametrize("failure", ["status", "timeout"])
def test_cloud_failure_closes_the_client_and_never_retries(failure):
    base = make_result()
    before = deepcopy(base)
    factory, sent, clients, _ = mock_cloud(
        base, status=503 if failure == "status" else 200,
        error=httpx.ReadTimeout("Synthetic timeout") if failure == "timeout" else None,
    )
    with pytest.raises(httpx.HTTPError):
        reranking.rerank(base, client_factory=factory, api_key_loader=lambda: "synthetic-key")
    assert len(sent) == 1 and clients[0].is_closed
    assert base == before


@pytest.fixture
def api(tmp_path, monkeypatch):
    check = {"synthetic": True}
    prepare = Mock(return_value=check)
    compare = Mock(return_value=make_result())
    describe = Mock(return_value={"synthetic_plan": True})
    rerank = Mock()
    monkeypatch.setattr(lab.hybrid, "prepare", prepare)
    monkeypatch.setattr(lab.hybrid, "describe", describe)
    monkeypatch.setattr(lab.hybrid, "compare", compare)
    monkeypatch.setattr(lab.reranking, "rerank", rerank)
    page = tmp_path / "synthetic-page.html"
    page.write_text("<!doctype html><title>Synthetic rerank page</title>", encoding="utf-8")
    monkeypatch.setattr(lab, "PAGE", page)
    with TestClient(lab.create_app(tmp_path), base_url="http://127.0.0.1") as client:
        yield client, prepare, compare, describe, rerank


def start_experiment(client):
    response = client.post("/api/compare", headers=HEADERS, json={"query": "DNS", "mode": "hybrid"})
    assert response.status_code == 200
    return response.json()


def add_synthetic_rerank(result):
    result["routes"]["reranker"] = [{"chunk_id": result["rrf_pool"][-1]["chunk_id"], "rank": 1, "score": 0.9}]
    result["request_counts"]["http"] += 1
    result["request_counts"]["reranker"] += 1
    return result


def test_api_page_static_and_plan_do_not_rerank_or_read_credentials(api):
    client, _, compare, _, rerank = api
    for path in ("/", "/assets/retrieval_export.js", "/api/plan"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "access-control-allow-origin" not in response.headers
    assert client.get("/api/plan").json()["reranker"]["max_candidates"] == 16
    compare.assert_not_called()
    rerank.assert_not_called()


def test_api_freezes_server_candidates_and_reuses_success_without_more_requests(api):
    client, _, compare, _, rerank = api
    base = start_experiment(client)
    token = base["experiment_id"]
    assert len(token) == 32
    original = deepcopy(compare.return_value)
    # Mutating the object returned by a producer after comparison cannot change
    # the server's frozen snapshot.
    compare.return_value["query"] = "Changed producer-owned question"
    rerank.side_effect = add_synthetic_rerank
    first = client.post("/api/rerank", headers=HEADERS, json={"experiment_id": token})
    second = client.post("/api/rerank", headers=HEADERS, json={"experiment_id": token})
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["query"] == original["query"]
    assert first.json()["records"] == original["records"]
    assert first.json()["experiment_id"] == token
    rerank.assert_called_once()
    argument = rerank.call_args.args[0]
    assert argument["rrf_pool"] == original["rrf_pool"]
    assert argument["records"] == original["records"]
    assert first.json()["request_counts"] == {"embedding": 1, "http": 4, "reranker": 1, "chat": 0}


@pytest.mark.parametrize("payload", [
    {}, {"experiment_id": ""}, {"experiment_id": " "}, {"experiment_id": "a" * 65},
    {"experiment_id": 123}, {"experiment_id": True},
    {"experiment_id": "a", "query": "Injected question"},
    {"experiment_id": "a", "documents": ["Injected body"]},
    {"experiment_id": "a", "model": "Injected model"}, [], None,
])
def test_rerank_api_rejects_invalid_or_extra_fields_before_cloud(api, payload):
    client, prepare, compare, _, rerank = api
    response = client.post("/api/rerank", headers={**HEADERS, "Content-Type": "application/json"}, content=json.dumps(payload))
    assert response.status_code == 422
    prepare.assert_not_called()
    compare.assert_not_called()
    rerank.assert_not_called()


@pytest.mark.parametrize("headers,content,status", [
    ({"Content-Type": "application/json"}, b"{}", 403),
    ({**HEADERS, "Origin": "https://evil.invalid", "Content-Type": "application/json"}, b"{}", 403),
    ({**HEADERS, "Origin": "null", "Content-Type": "application/json"}, b"{}", 403),
    ({**HEADERS, "X-Retrieval-Lab": "0", "Content-Type": "application/json"}, b"{}", 403),
    ({**HEADERS, "Content-Type": "text/plain"}, b"{}", 415),
    ({**HEADERS, "Content-Type": "application/json"}, b"{invalid", 422),
    ({**HEADERS, "Content-Type": "application/json"}, b"\xff", 422),
    ({**HEADERS, "Content-Type": "application/json"}, b" " * 16385, 413),
])
def test_rerank_api_preserves_same_origin_json_and_size_guards(api, headers, content, status):
    client, _, _, _, rerank = api
    response = client.post("/api/rerank", headers=headers, content=content)
    assert response.status_code == status
    assert set(response.json()) == {"detail"}
    rerank.assert_not_called()


def test_rerank_api_requires_a_current_hybrid_snapshot(api):
    client, _, _, _, rerank = api
    assert client.post("/api/rerank", headers=HEADERS, json={"experiment_id": "unknown"}).status_code == 409
    base = start_experiment(client)
    assert client.post("/api/rerank", headers=HEADERS, json={"experiment_id": "different"}).status_code == 409
    response = client.post("/api/compare", headers=HEADERS, json={"query": "DNS", "mode": "bm25"})
    assert response.status_code == 200 and "experiment_id" not in response.json()
    assert client.post("/api/rerank", headers=HEADERS, json={"experiment_id": base["experiment_id"]}).status_code == 409
    rerank.assert_not_called()


def test_a_new_failed_comparison_also_invalidates_the_old_token(api):
    client, _, compare, _, rerank = api
    token = start_experiment(client)["experiment_id"]
    compare.side_effect = ValueError("Synthetic comparison failure")
    assert client.post("/api/compare", headers=HEADERS, json={"query": "Next", "mode": "hybrid"}).status_code == 503
    assert client.post("/api/rerank", headers=HEADERS, json={"experiment_id": token}).status_code == 409
    rerank.assert_not_called()


def test_cloud_api_failure_keeps_unmodified_base_for_explicit_manual_retry(api, capsys):
    client, _, _, _, rerank = api
    base = start_experiment(client)
    received = []

    def fail_then_succeed(result):
        received.append(deepcopy(result))
        if len(received) == 1:
            result["records"].clear()
            result["query"] = "Mutated failed worker"
            raise ValueError("PRIVATE_SYNTHETIC_KEY_AND_RESPONSE")
        return add_synthetic_rerank(result)

    rerank.side_effect = fail_then_succeed
    payload = {"experiment_id": base["experiment_id"]}
    failed = client.post("/api/rerank", headers=HEADERS, json=payload)
    assert failed.status_code == 503 and set(failed.json()) == {"detail"}
    assert "PRIVATE_SYNTHETIC" not in failed.text
    assert "PRIVATE_SYNTHETIC" not in capsys.readouterr().out
    assert rerank.call_count == 1
    recovered = client.post("/api/rerank", headers=HEADERS, json=payload)
    assert recovered.status_code == 200
    assert received == [base, base]
    assert recovered.json()["records"] == base["records"]
    assert rerank.call_count == 2


def test_compare_and_rerank_share_one_lock_and_cannot_duplicate_a_cloud_request(api):
    client, _, compare, _, rerank = api
    base = start_experiment(client)
    entered, release = Event(), Event()

    def delayed(result):
        entered.set()
        assert release.wait(5), "Synthetic reranking worker was not released"
        return add_synthetic_rerank(result)

    rerank.side_effect = delayed
    payload = {"experiment_id": base["experiment_id"]}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/api/rerank", headers=HEADERS, json=payload)
        try:
            assert entered.wait(5), "Synthetic reranking did not start"
            assert client.post("/api/rerank", headers=HEADERS, json=payload).status_code == 409
            assert client.post("/api/compare", headers=HEADERS, json={"query": "Next", "mode": "hybrid"}).status_code == 409
            assert rerank.call_count == compare.call_count == 1
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
    assert client.post("/api/rerank", headers=HEADERS, json=payload).status_code == 200
    rerank.assert_called_once()
