"""Offline contracts for the Kubernetes comparison and its local-only HTTP API.

Every scope, vector, database client and model response is synthetic. TestClient
uses its in-process transport; network transports and .env reads are forbidden.
"""

import builtins
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import io
import json
from pathlib import Path
import socket
from threading import Event
from unittest.mock import Mock, patch

import dotenv
from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest
import requests

from qa_platform import retrieval_lab as lab
from qa_platform.retrieval import bm25
from qa_platform.retrieval import kubernetes_hybrid as hybrid
from qa_platform.storage import kubernetes_milvus as build


HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}
TEXTS = [
    "DNS resolves service names",
    "kube-proxy forwards service traffic",
    "DNS resolver selects endpoints",
    "scheduler assigns pods to nodes",
    "persistent volumes store application files",
    "container images have registry references",
    "deployment controls replica counts",
    "secrets provide application configuration",
    "node conditions report resource pressure",
    "ingress routes external requests",
]


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


def make_check(directory, query="DNS", *, count=10, database=True):
    if database:
        (directory / build.DATABASE).mkdir()
    records = []
    for position, text in enumerate(TEXTS[:count]):
        chunk_id = f"synthetic-{position}"
        records.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "title": "TitleOnlyNeedle",
                    "sections": [["SectionOnlyNeedle"]],
                    "source_url": "https://example.invalid/SourceOnlyNeedle",
                    "parent_id": f"parent-{position}",
                    "synthetic": True,
                },
            }
        )
    vectors = np.asarray(
        [[1.0, position / 20, 0.0] for position in range(count)], dtype="<f4"
    )
    return {
        "index_dir": directory,
        "query": query,
        "records": records,
        "top_k": min(hybrid.RECALL_K, count),
        "filter": "synthetic_scope == true",
        "plan": {"documents": records, "document_count": count},
        "bundle": {"vectors": vectors.tolist()},
        "vectors": vectors,
    }


def mocked_dense(check, *, failure=None, mismatch=False, dense=None):
    """Patch all storage/model operations, retaining their order for assertions."""
    events = []
    client = Mock()
    client.close.side_effect = lambda: events.append("close")
    query = np.asarray([1.0, 0.0, 0.0], dtype="<f4")
    if dense is None:
        dense = [
            {"chunk_id": row["chunk_id"], "score": 0.9 - position / 100}
            for position, row in enumerate(check["records"][1:9])
        ]

    def operation(name, value):
        def run(*args, **kwargs):
            events.append(name)
            if failure == name:
                raise ValueError(f"synthetic {name} failure")
            return value

        return run

    matrix = check["vectors"].copy()
    if mismatch:
        matrix[0, 0] += 1
    mocks = {
        "read_index": Mock(side_effect=operation("read", matrix)),
        "encode_query": Mock(
            side_effect=operation(
                "encode", (query, {"embedding_requests": 1, "http_requests": 3})
            )
        ),
        "search_milvus": Mock(side_effect=operation("search", dense)),
        "validate_top_k": Mock(side_effect=operation("validate", None)),
    }
    stack = ExitStack()
    for name, mocked in mocks.items():
        stack.enter_context(patch.object(hybrid.search, name, mocked))
    factory = Mock(side_effect=lambda **kwargs: (events.append("client"), client)[1])
    return stack, factory, client, mocks, events, query


def test_prepare_only_delegates_to_offline_preparation(tmp_path, monkeypatch):
    expected = {"synthetic": True}
    prepare = Mock(return_value=expected)
    monkeypatch.setattr(hybrid.search, "prepare_search", prepare)
    assert hybrid.prepare(tmp_path, "synthetic query") is expected
    prepare.assert_called_once_with(tmp_path, "synthetic query", top_k=8)


def test_bm25_uses_body_only_and_never_creates_model_or_database(tmp_path, monkeypatch):
    check = make_check(tmp_path)
    forbidden = Mock(side_effect=AssertionError("BM25 must remain entirely offline"))
    for name in ("read_index", "encode_query", "search_milvus", "validate_top_k"):
        monkeypatch.setattr(hybrid.search, name, forbidden)
    result = hybrid.compare(check, "bm25", session_factory=forbidden, client_factory=forbidden)
    assert result["request_counts"] == {"embedding": 0, "http": 0, "reranker": 0, "chat": 0}
    assert result["scope_count"] == 10
    assert {hit["chunk_id"] for hit in result["routes"]["bm25"]} == {
        "synthetic-0", "synthetic-2"
    }
    assert result["routes"]["dense"] == result["routes"]["rrf"] == []
    for row in result["records"].values():
        assert row["text"] in TEXTS
        assert "TitleOnlyNeedle" not in row["text"]
    forbidden.assert_not_called()


@pytest.mark.parametrize("query", ["TitleOnlyNeedle", "SectionOnlyNeedle", "SourceOnlyNeedle"])
def test_metadata_never_leaks_into_bm25_body(tmp_path, query):
    result = hybrid.compare(make_check(tmp_path, query), "bm25")
    assert result["routes"] == {"bm25": [], "dense": [], "rrf": []}
    assert result["records"] == {}


def test_kubernetes_english_tokenizer_keeps_the_existing_baseline(tmp_path):
    query = "CoreDNS kube-system DNS DNS foo.bar.svc:53 10.96.0.10 /etc/resolv.conf"
    assert bm25.tokenize(query) == [
        "coredns", "kube-system", "dns", "dns", "foo", "bar", "svc", "53",
        "10", "96", "0", "10", "etc", "resolv", "conf",
    ]
    result = hybrid.compare(make_check(tmp_path, query), "bm25")
    assert result["tokens"] == list(dict.fromkeys(bm25.tokenize(query)))
    assert "kube-system" in result["tokens"]
    assert "foo.bar.svc" not in result["tokens"]


@pytest.mark.parametrize("query", ["！！！ ... /", "的 了 吗", "NoBodyContainsThisTerm"])
def test_empty_tokens_or_no_lexical_match_returns_an_empty_result(tmp_path, query):
    result = hybrid.compare(make_check(tmp_path, query), "bm25")
    assert result["records"] == {}
    assert result["routes"] == {"bm25": [], "dense": [], "rrf": []}
    assert all(value == 0 for value in result["request_counts"].values())


@pytest.mark.parametrize("problem", ["empty", "duplicate", "metadata", "recall_contract", "mode"])
def test_invalid_scope_is_rejected_before_encoding_or_client_creation(tmp_path, problem):
    check = make_check(tmp_path)
    mode = "hybrid"
    if problem == "empty":
        check["records"] = []
    elif problem == "duplicate":
        check["records"][1] = deepcopy(check["records"][0])
    elif problem == "metadata":
        check["records"][0]["metadata"]["chunk_id"] = "wrong-identity"
    elif problem == "recall_contract":
        check["top_k"] = 5
    else:
        mode = "unknown"
    stack, factory, client, mocks, events, _ = mocked_dense(check)
    with stack, pytest.raises(ValueError):
        hybrid.compare(check, mode, client_factory=factory)
    assert events == []
    factory.assert_not_called()
    mocks["encode_query"].assert_not_called()


def test_hybrid_encodes_once_reuses_common_scope_and_closes_client(tmp_path):
    check = make_check(tmp_path)
    stack, factory, client, mocks, events, query = mocked_dense(check)
    session_factory = Mock(side_effect=AssertionError("mock encoder owns its transport"))
    with stack:
        result = hybrid.compare(
            check, "hybrid", client_factory=factory, session_factory=session_factory
        )
    assert events == ["client", "read", "encode", "search", "validate", "close"]
    factory.assert_called_once_with(uri=str(tmp_path / build.DATABASE), timeout=30)
    mocks["read_index"].assert_called_once_with(client, check["plan"])
    mocks["encode_query"].assert_called_once_with(check, session_factory=session_factory)
    args = mocks["search_milvus"].call_args.args
    assert args[0] is client and args[1] is check["records"] and args[2] is query
    assert args[3:] == (8, check["filter"])
    validation = mocks["validate_top_k"].call_args.args
    assert validation[1] is check["records"] and validation[2] is check["vectors"]
    assert validation[3] is query and validation[4] == 8
    client.close.assert_called_once_with()
    session_factory.assert_not_called()
    assert result["request_counts"] == {"embedding": 1, "http": 3, "reranker": 0, "chat": 0}
    assert len(result["routes"]["dense"]) == 8
    assert len(result["routes"]["rrf"]) == 5
    union = {hit["chunk_id"] for route in result["routes"].values() for hit in route}
    assert set(result["records"]) == union
    assert "synthetic-9" not in result["records"]


@pytest.mark.parametrize("failure", ["read", "encode", "search", "validate", "frozen_vectors"])
def test_hybrid_closes_on_failure_and_validates_frozen_vectors_before_encoding(tmp_path, failure):
    check = make_check(tmp_path)
    stack, factory, client, mocks, events, _ = mocked_dense(
        check, failure=failure, mismatch=failure == "frozen_vectors"
    )
    with stack, pytest.raises(ValueError):
        hybrid.compare(check, "hybrid", client_factory=factory)
    assert events[-1] == "close"
    client.close.assert_called_once_with()
    if failure in {"read", "frozen_vectors"}:
        mocks["encode_query"].assert_not_called()
    else:
        mocks["encode_query"].assert_called_once()


@pytest.mark.parametrize("database_kind", ["missing", "symlink"])
def test_invalid_database_is_rejected_before_client_or_encoder(tmp_path, database_kind):
    check = make_check(tmp_path, database=False)
    if database_kind == "symlink":
        target = tmp_path / "synthetic-directory"
        target.mkdir()
        (tmp_path / build.DATABASE).symlink_to(target, target_is_directory=True)
    stack, factory, _, mocks, events, _ = mocked_dense(check)
    with stack, pytest.raises(ValueError):
        hybrid.compare(check, "hybrid", client_factory=factory)
    assert events == []
    factory.assert_not_called()
    mocks["encode_query"].assert_not_called()


def test_rrf_uses_only_rank_contributions_and_candidate_union(tmp_path, monkeypatch):
    check = make_check(tmp_path, count=6)
    scores = [12345.0, -9876.0]

    class SyntheticBM25:
        def __init__(self, documents):
            self.documents = documents
            assert [doc.page_content for doc in documents] == TEXTS[:6]

        def search(self, tokens, recall_k):
            assert tokens == ["dns"] and recall_k == 6
            return [(self.documents[i], scores[i], ["dns"]) for i in range(2)]

    monkeypatch.setattr(bm25, "BM25Index", SyntheticBM25)
    dense = [
        {"chunk_id": "synthetic-1", "score": 0.99},
        {"chunk_id": "synthetic-2", "score": 0.01},
    ]
    stack, factory, _, _, _, _ = mocked_dense(check, dense=dense)
    with stack:
        first = hybrid.compare(check, "hybrid", client_factory=factory)
        scores[:] = [-0.001, -0.002]
        dense[0]["score"], dense[1]["score"] = 0.1, -0.9
        second = hybrid.compare(check, "hybrid", client_factory=factory)
    assert first["routes"]["rrf"] == second["routes"]["rrf"]
    hits = {hit["chunk_id"]: hit for hit in first["routes"]["rrf"]}
    assert hits["synthetic-1"]["ranks"] == {"bm25": 2, "dense": 1}
    assert hits["synthetic-1"]["contributions"] == {"bm25": 1 / 62, "dense": 1 / 61}
    assert hits["synthetic-1"]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert hits["synthetic-0"]["score"] == pytest.approx(1 / 61)
    assert hits["synthetic-2"]["score"] == pytest.approx(1 / 62)
    assert set(first["records"]) == {"synthetic-0", "synthetic-1", "synthetic-2"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    check = make_check(tmp_path)
    prepare = Mock(return_value=check)
    compare = Mock(return_value={"routes": {}, "records": {}, "synthetic": "fresh"})
    monkeypatch.setattr(lab.hybrid, "prepare", prepare)
    monkeypatch.setattr(lab.hybrid, "compare", compare)
    page = tmp_path / "synthetic-page.html"
    page.write_text("<!doctype html><title>Synthetic learning page</title>", encoding="utf-8")
    monkeypatch.setattr(lab, "PAGE", page)
    with TestClient(lab.create_app(tmp_path), base_url="http://127.0.0.1") as client:
        yield client, prepare, compare, check


def test_get_page_and_plan_make_no_comparison_and_send_security_headers(api):
    client, prepare, compare, check = api
    for path in ("/", "/api/plan"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert "access-control-allow-origin" not in response.headers
    prepare.assert_called_once_with(check["index_dir"])
    compare.assert_not_called()
    assert client.get("/api/plan").json()["model_requests"] == {
        "embedding": 1, "metadata": 2, "reranker": 0, "chat": 0
    }


@pytest.mark.parametrize(
    "headers,content,status",
    [
        ({"Content-Type": "application/json"}, b'{"query":"DNS","mode":"bm25"}', 403),
        ({**HEADERS, "X-Retrieval-Lab": "0", "Content-Type": "application/json"}, b"{}", 403),
        ({**HEADERS, "Origin": "https://evil.invalid", "Content-Type": "application/json"}, b"{}", 403),
        ({**HEADERS, "Origin": "null", "Content-Type": "application/json"}, b"{}", 403),
        ({**HEADERS, "Origin": "http://localhost", "Content-Type": "application/json"}, b"{}", 403),
        ({**HEADERS, "Content-Type": "text/plain"}, b"{}", 415),
        ({**HEADERS, "Content-Type": "application/x-www-form-urlencoded"}, b"query=DNS&mode=bm25", 415),
        (HEADERS, b"{}", 415),
        ({**HEADERS, "Content-Type": "application/json"}, b"{invalid", 422),
        ({**HEADERS, "Content-Type": "application/json"}, b"\xff", 422),
        ({**HEADERS, "Content-Type": "application/json"}, b" " * 16385, 413),
    ],
)
def test_api_rejects_bad_transport_requests_before_preparation_or_comparison(api, headers, content, status):
    client, prepare, compare, _ = api
    response = client.post("/api/compare", headers=headers, content=content)
    assert response.status_code == status
    assert "records" not in response.json() and "routes" not in response.json()
    prepare.assert_not_called()
    compare.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {}, {"query": "DNS"}, {"mode": "bm25"},
        {"query": "", "mode": "bm25"}, {"query": "   ", "mode": "bm25"},
        {"query": "D" * 2001, "mode": "bm25"},
        {"query": 123, "mode": "bm25"}, {"query": True, "mode": "bm25"},
        {"query": "DNS", "mode": "dense"}, {"query": "DNS", "mode": True},
        {"query": "DNS", "mode": "hybrid", "top_k": 1},
        {"query": "DNS", "mode": "bm25", "index_dir": "/other"},
        ["DNS", "bm25"], None,
    ],
)
def test_api_rejects_invalid_or_extra_payload_fields_without_comparison(api, payload):
    client, prepare, compare, _ = api
    response = client.post(
        "/api/compare", headers={**HEADERS, "Content-Type": "application/json"},
        content=json.dumps(payload),
    )
    assert response.status_code == 422
    prepare.assert_not_called()
    compare.assert_not_called()


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_api_success_trims_query_and_runs_the_selected_mode_once(api, mode):
    client, prepare, compare, check = api
    response = client.post("/api/compare", headers=HEADERS, json={"query": "  DNS  ", "mode": mode})
    assert response.status_code == 200
    assert response.json() == compare.return_value
    prepare.assert_called_once_with(check["index_dir"], "DNS")
    compare.assert_called_once_with(check, mode)


def test_api_accepts_same_origin_json_charset_and_no_origin_for_local_clients(api):
    client, _, compare, _ = api
    response = client.post(
        "/api/compare",
        headers={"X-Retrieval-Lab": "1", "Content-Type": "application/json; charset=utf-8"},
        content=b'{"query":"DNS","mode":"bm25"}',
    )
    assert response.status_code == 200
    compare.assert_called_once()


def test_api_failure_never_returns_partial_or_previous_results_and_releases_lock(api):
    client, _, compare, _ = api
    compare.side_effect = [
        {"records": {"old-result": {"text": "previous body"}}, "routes": {}},
        ValueError("PRIVATE model response and environment details"),
        {"records": {"new-result": {"text": "fresh body"}}, "routes": {}},
    ]
    payload = {"query": "DNS", "mode": "hybrid"}
    first = client.post("/api/compare", headers=HEADERS, json=payload)
    failed = client.post("/api/compare", headers=HEADERS, json=payload)
    final = client.post("/api/compare", headers=HEADERS, json=payload)
    assert first.status_code == final.status_code == 200
    assert failed.status_code == 503
    assert set(failed.json()) == {"detail"}
    assert "PRIVATE" not in failed.text and "old-result" not in failed.text
    assert set(final.json()["records"]) == {"new-result"}
    assert compare.call_count == 3


def test_api_rejects_overlapping_comparisons_without_a_second_model_request(api):
    client, _, compare, _ = api
    entered, release = Event(), Event()

    def delayed_comparison(*args):
        entered.set()
        assert release.wait(5), "Synthetic comparison was not released"
        return {"records": {}, "routes": {}}

    compare.side_effect = delayed_comparison
    payload = {"query": "DNS", "mode": "hybrid"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/api/compare", headers=HEADERS, json=payload)
        try:
            assert entered.wait(5), "Synthetic comparison did not start"
            second = client.post("/api/compare", headers=HEADERS, json=payload)
            assert second.status_code == 409
            assert compare.call_count == 1
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200


def test_api_rejects_an_untrusted_host_before_comparison(api):
    client, prepare, compare, _ = api
    response = client.post(
        "/api/compare", headers={**HEADERS, "Host": "evil.invalid"},
        json={"query": "DNS", "mode": "bm25"},
    )
    assert response.status_code == 400
    prepare.assert_not_called()
    compare.assert_not_called()


def test_plan_failure_does_not_leak_environment_or_run_comparison(api):
    client, prepare, compare, _ = api
    prepare.side_effect = OSError("PRIVATE local path")
    response = client.get("/api/plan")
    assert response.status_code == 503
    assert "PRIVATE" not in response.text
    compare.assert_not_called()
