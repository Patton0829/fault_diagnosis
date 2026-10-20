"""Offline evidence-answer contracts; every document and credential is synthetic.

Cloud traffic uses MockTransport, the API runs in-process, and real network or
.env access is forbidden. Successful output still does not prove entailment.
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
from langchain_core.documents import Document
from openai import OpenAI
import pytest
import requests

from qa_platform import retrieval_lab as lab
from qa_platform.rag import answering as legacy
from qa_platform.rag import kubernetes_answer as answering
from qa_platform.retrieval.hybrid import RRFHit
from qa_platform.retrieval.pipeline import RerankedHit


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
                if Path(file).name == ".env" or Path(file).name.startswith(".env."):
                    forbidden()
            return original(file, *args, **kwargs)
        return guarded

    monkeypatch.setattr(builtins, "open", guard_open(original_open))
    monkeypatch.setattr(io, "open", guard_open(original_io_open))


def make_result(*, top_k=5):
    lexical_ids = [f"synthetic-{i:02d}" for i in range(8)]
    dense_ids = [f"synthetic-{i:02d}" for i in range(5, 13)]
    lexical = [{"chunk_id": value, "rank": rank, "score": 20 - rank, "matched_terms": ["dns"]}
               for rank, value in enumerate(lexical_ids, 1)]
    dense = [{"chunk_id": value, "rank": rank, "score": 0.9 - rank / 100}
             for rank, value in enumerate(dense_ids, 1)]
    pool = []
    for chunk_id in sorted(set(lexical_ids) | set(dense_ids)):
        ranks = {route: ids.index(chunk_id) + 1
                 for route, ids in (("bm25", lexical_ids), ("dense", dense_ids))
                 if chunk_id in ids}
        contributions = {route: 1 / (60 + rank) for route, rank in ranks.items()}
        pool.append({"chunk_id": chunk_id, "score": sum(contributions.values()),
                     "ranks": ranks, "contributions": contributions})
    pool.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(pool, 1):
        row["rank"] = rank
    reranked = [
        {"chunk_id": row["chunk_id"], "rank": rank, "score": 1 - rank / 100,
         "input_index": len(pool) - rank, "rrf_rank": row["rank"],
         "rank_change": row["rank"] - rank}
        for rank, row in enumerate(reversed(pool), 1)
    ]
    return {
        "query": "Synthetic DNS question?", "mode": "hybrid", "tokens": ["dns"],
        "scope_count": 156, "recall_k": 8, "top_k": top_k, "rrf_constant": 60,
        "experiment_id": "a" * 32,
        "request_counts": {"embedding": 1, "http": 4, "reranker": 1, "chat": 0},
        "routes": {"bm25": lexical, "dense": dense, "rrf": deepcopy(pool[:top_k]),
                   "reranker": reranked},
        "rrf_pool": pool,
        "records": {
            row["chunk_id"]: {
                "text": (f"First body paragraph {row['chunk_id']}.\n\n"
                         "## Synthetic section heading\n\n"
                         f"Second evidence line {row['chunk_id']}.\n"
                         "```shell\nkubectl synthetic\n```"),
                "metadata": {
                    "chunk_id": row["chunk_id"], "title": "TitleMustNotBeSent",
                    "source_path": "/private/PathMustNotBeSent.md",
                    "source_url": "https://example.invalid/SourceMustNotBeSent",
                    "review_status": "unreviewed_official_document",
                    "nested": {"sections": ["SectionMustNotBeSent"]},
                    "vector": ["VectorMustNotBeSent"],
                },
            }
            for row in pool
        },
        "rerank_comparison": {
            "candidate_count": len(pool), "final_k": top_k,
            "model": "synthetic-reranker",
            "input_order": [row["chunk_id"] for row in pool],
            "request_counts": {"embedding": 0, "http": 1, "reranker": 1, "chat": 0},
            "notice": "Synthetic reranking notice",
        },
        "notice": "Original synthetic notice",
    }


def valid_answer(**overrides):
    return {"status": "partial", "claims": [
        {"text": "A synthetic conditional explanation, not a verified cluster result.",
         "evidence": [{"source_id": "S1", "quote_id": "Q1"}]}
    ], "missing_info": ["Actual cluster version and namespace are not known."], **overrides}


def mock_chat(*, payload=None, raw=None, finish="stop", status=200, error=None):
    """Use the real SDK against an in-memory SSE transport."""
    sent, clients, options, sdk_options = [], [], [], []
    if raw is None:
        raw = json.dumps(valid_answer() if payload is None else payload)

    def respond(request):
        sent.append(request)
        if error is not None:
            raise error
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "PRIVATE_RESPONSE_MUST_NOT_LEAK", "type": "synthetic"}})
        chunks = [
            {"id": "synthetic", "choices": [{"index": 0, "delta": {"reasoning_content": "PRIVATE_REASONING_MUST_NOT_LEAK"}, "finish_reason": None}]},
            {"id": "synthetic", "choices": [{"index": 0, "delta": {"content": raw[:len(raw)//2]}, "finish_reason": None}]},
            {"id": "synthetic", "choices": [{"index": 0, "delta": {"content": raw[len(raw)//2:]}, "finish_reason": finish}]},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    def http_factory(**kwargs):
        options.append(kwargs)
        client = httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    def sdk_factory(**kwargs):
        sdk_options.append(kwargs)
        return OpenAI(**kwargs)

    return {"http_client_factory": http_factory, "client_factory": sdk_factory}, sent, clients, options, sdk_options


def test_describe_and_evidence_preview_are_zero_io_and_do_not_load_keys(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Pure plan or preview created cloud resources"))
    monkeypatch.setattr(answering, "load_chat_key", forbidden)
    monkeypatch.setattr(answering, "OpenAI", forbidden)
    monkeypatch.setattr(answering.httpx, "Client", forbidden)
    plan = answering.describe()
    assert plan["default_model"] == "qwen3.8-flash"
    assert plan["max_sources"] == 5 and plan["max_tokens"] == 4096
    assert {row["id"] for row in plan["models"]} == set(answering.MODEL_CONFIGS)
    glm = next(row for row in plan["models"] if row["id"] == "glm")
    assert glm["base_url"].startswith("http://") and "明文" in glm["transport_notice"]
    assert "不自动重试" in plan["notice"] and "不发送本机路径或向量" in plan["notice"]
    assert len(answering.prepare_evidence(make_result())) == 5
    forbidden.assert_not_called()


@pytest.mark.parametrize("top_k,expected", [(2, 2), (5, 5), (8, 5)])
def test_evidence_uses_rerank_prefix_not_rrf_prefix_and_preserves_originals(top_k, expected):
    base = make_result(top_k=top_k)
    before = deepcopy(base)
    sources = answering.prepare_evidence(base)
    assert list(sources) == [f"S{i}" for i in range(1, expected + 1)]
    assert [row["chunk_id"] for row in sources.values()] == [row["chunk_id"] for row in base["routes"]["reranker"][:expected]]
    assert sources["S1"]["chunk_id"] != base["routes"]["rrf"][0]["chunk_id"]
    for source in sources.values():
        original = base["records"][source["chunk_id"]]
        assert source["text"] == original["text"] and source["metadata"] == original["metadata"]
        assert source["quotes"]["Q1"] == original["text"].splitlines()[0]
        assert all(quote in source["text"] for quote in source["quotes"].values())
        assert not any(quote.startswith(("#", "```")) for quote in source["quotes"].values())
        assert "document_mode" not in source["metadata"] and "is_test" not in source["metadata"]
    sources["S1"]["metadata"]["nested"]["sections"].append("Changed preview")
    assert base == before


def test_body_only_retains_first_paragraph_but_legacy_title_prefix_is_unchanged():
    def hit(text, metadata):
        return RerankedHit(RRFHit(Document(page_content=text, metadata=metadata), 0.1, {}, {}), 0, 0.9)

    text = "Added legacy title prefix\n\nOriginal body must remain.\nSecond body line remains."
    old_quotes = legacy.build_quotes(hit(text, {}))
    assert list(old_quotes.values()) == ["Original body must remain.", "Second body line remains."]
    body_quotes = legacy.build_quotes(hit(text, {"document_mode": "body-only"}))
    assert body_quotes["Q1"] == "Added legacy title prefix"
    only_paragraph = "A genuine first paragraph.\n\n## Heading is not evidence\n```shell\nsynthetic-command\n```"
    quotes = legacy.build_quotes(hit(only_paragraph, {"document_mode": "body-only"}))
    assert list(quotes.values()) == ["A genuine first paragraph.", "synthetic-command"]


def test_default_glm_auth_is_placeholder_without_env_or_qwen_key(monkeypatch):
    loader = Mock(return_value="synthetic-qwen-key")
    monkeypatch.setattr(answering, "load_api_key", loader)
    assert answering.load_chat_key("glm") == "not-needed"
    loader.assert_not_called()
    assert answering.load_chat_key("qwen3.8-flash") == "synthetic-qwen-key"
    loader.assert_called_once_with()
    with pytest.raises(ValueError):
        answering.load_chat_key("not-allowed")


@pytest.mark.parametrize("model", ["qwen3.8-flash", "qwen3.8-max", "glm"])
def test_generation_sends_only_five_frozen_sources_once_and_validates_answer(model):
    base = make_result()
    before = deepcopy(base)
    first_id = base["routes"]["reranker"][0]["chunk_id"]
    base["records"][first_id]["metadata"]["is_test"] = True
    before = deepcopy(base)
    kwargs, sent, clients, options, sdk_options = mock_chat()
    key_loader = Mock(return_value="synthetic-credential")
    result = answering.generate(base, model, api_key_loader=key_loader, **kwargs)
    assert base == before and len(sent) == 1 and clients[0].is_closed
    key_loader.assert_called_once_with(model)
    assert options == [{"trust_env": False, "follow_redirects": False, "timeout": 60.0}]
    config = answering.MODEL_CONFIGS[model]
    assert sdk_options[0] == {"api_key": "not-needed" if model == "glm" else "synthetic-credential",
                              "base_url": config["base_url"], "timeout": 60.0,
                              "organization": "", "project": "",
                              "max_retries": 0, "http_client": clients[0]}
    request = sent[0]
    assert request.method == "POST" and str(request.url) == config["base_url"] + "/chat/completions"
    assert request.headers["authorization"] == "Bearer " + ("not-needed" if model == "glm" else "synthetic-credential")
    payload = json.loads(request.content)
    assert payload["model"] == model and payload["stream"] is True and payload["max_tokens"] == 4096
    if model.startswith("qwen"):
        assert payload["enable_thinking"] is True
    messages = payload["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert answering.SYSTEM_SUFFIX in messages[0]["content"]
    assert "版本" in messages[0]["content"] and "不会执行" in answering.NOTICE
    user = json.loads(messages[1]["content"])
    assert user["question"] == base["query"] and len(user["evidence"]) == 5
    assert user["evidence"][0]["is_test"] is True
    assert all(row["is_test"] is False for row in user["evidence"][1:])
    assert all(row["review_status"] == "unreviewed_official_document" for row in user["evidence"])
    for source, hit in zip(user["evidence"], base["routes"]["reranker"][:5]):
        assert set(source) == {"id", "entry_id", "is_test", "review_status", "content", "quotes"}
        assert source["content"] == base["records"][hit["chunk_id"]]["text"]
        assert source["quotes"]["Q1"] == source["content"].splitlines()[0]
    assert "MustNotBeSent" not in request.content.decode()
    assert "rrf_rank" not in request.content.decode() and "vector" not in user
    assert result["request_counts"] == {"embedding": 1, "http": 5, "reranker": 1, "chat": 1}
    generated = result["answer_generation"]
    assert generated["model"] == model and generated["max_tokens"] == 4096
    assert generated["evidence_ids"] == [row["chunk_id"] for row in base["routes"]["reranker"][:5]]
    assert generated["answer"] == valid_answer()
    assert generated["sources"] == answering.prepare_evidence(base)
    assert generated["request_counts"] == {"embedding": 0, "http": 1, "reranker": 0, "chat": 1}
    assert "PRIVATE_REASONING" not in json.dumps(result)
    assert "raw" not in generated and "reasoning" not in generated
    result["records"][first_id]["metadata"]["nested"]["sections"].append("Changed output")
    assert base == before


@pytest.mark.parametrize("problem", [
    "no_rerank", "wrong_mode", "already_answered", "empty_token", "short_rerank",
    "duplicate_index", "wrong_identity", "wrong_rank", "wrong_change", "ascending_score",
    "nan_score", "input_order", "candidate_count", "final_k", "metadata_identity",
    "bad_test_flag", "no_quotable_text", "bad_chat_count", "bad_rerank_count",
])
def test_invalid_evidence_contract_fails_before_any_key_or_client(problem):
    base = make_result()
    first = base["routes"]["reranker"][0]
    if problem == "no_rerank":
        del base["routes"]["reranker"]
    elif problem == "wrong_mode":
        base["mode"] = "bm25"
    elif problem == "already_answered":
        base["answer_generation"] = {}
    elif problem == "empty_token":
        base["experiment_id"] = " "
    elif problem == "short_rerank":
        base["routes"]["reranker"].pop()
    elif problem == "duplicate_index":
        first["input_index"] = base["routes"]["reranker"][1]["input_index"]
    elif problem == "wrong_identity":
        first["chunk_id"] = "foreign-source"
    elif problem == "wrong_rank":
        first["rank"] = True
    elif problem == "wrong_change":
        first["rank_change"] = 100
    elif problem == "ascending_score":
        base["routes"]["reranker"][1]["score"] = 2.0
    elif problem == "nan_score":
        first["score"] = float("nan")
    elif problem == "input_order":
        base["rerank_comparison"]["input_order"].reverse()
    elif problem == "candidate_count":
        base["rerank_comparison"]["candidate_count"] = 5
    elif problem == "final_k":
        base["rerank_comparison"]["final_k"] = 3
    elif problem == "metadata_identity":
        base["records"][first["chunk_id"]]["metadata"]["chunk_id"] = "different"
    elif problem == "bad_test_flag":
        base["records"][first["chunk_id"]]["metadata"]["is_test"] = "false"
    elif problem == "no_quotable_text":
        base["records"][first["chunk_id"]]["text"] = "## Only heading\n```shell\n```"
    elif problem == "bad_chat_count":
        base["request_counts"]["chat"] = 1
    else:
        base["request_counts"]["reranker"] = 0
    forbidden = Mock(side_effect=AssertionError("Invalid evidence touched cloud resources"))
    with pytest.raises((ValueError, KeyError, TypeError)):
        answering.generate(base, api_key_loader=forbidden, client_factory=forbidden,
                           http_client_factory=forbidden)
    forbidden.assert_not_called()


@pytest.mark.parametrize("model", ["unknown", "", "https://injected.invalid/model"])
def test_disallowed_model_fails_before_credentials(model):
    forbidden = Mock(side_effect=AssertionError("Unknown model accessed a key"))
    with pytest.raises(ValueError):
        answering.generate(make_result(), model, api_key_loader=forbidden)
    forbidden.assert_not_called()


@pytest.mark.parametrize("key", [None, "", " ", False, 123])
def test_invalid_auth_creates_no_clients(key):
    forbidden = Mock(side_effect=AssertionError("Missing key created a client"))
    with pytest.raises(ValueError):
        answering.generate(make_result(), api_key_loader=lambda model: key,
                           client_factory=forbidden, http_client_factory=forbidden)
    forbidden.assert_not_called()


@pytest.mark.parametrize("problem", ["json", "source", "quote", "missing_field", "extra_field", "empty_claim", "bad_refusal", "truncated", "no_stop", "empty"])
def test_bad_or_truncated_output_never_returns_partial_raw_or_reasoning(problem):
    base, payload, raw, finish = make_result(), valid_answer(), None, "stop"
    if problem == "json":
        raw = "PRIVATE_RAW_MUST_NOT_LEAK not JSON"
    elif problem in {"source", "quote"}:
        payload["claims"][0]["evidence"][0]["source_id" if problem == "source" else "quote_id"] = "S99" if problem == "source" else "Q999"
    elif problem == "missing_field":
        del payload["missing_info"]
    elif problem == "extra_field":
        payload["PRIVATE_FIELD_MUST_NOT_LEAK"] = "PRIVATE_RAW_MUST_NOT_LEAK"
    elif problem == "empty_claim":
        payload["claims"] = []
    elif problem == "bad_refusal":
        payload["status"] = "insufficient_evidence"
    elif problem == "truncated":
        raw, finish = '{"PRIVATE_RAW_MUST_NOT_LEAK":', "length"
    elif problem == "no_stop":
        finish = None
    else:
        raw = ""
    before = deepcopy(base)
    kwargs, sent, clients, _, _ = mock_chat(payload=payload, raw=raw, finish=finish)
    with pytest.raises((legacy.AnswerValidationError, legacy.TruncatedAnswerError)) as caught:
        answering.generate(base, api_key_loader=lambda model: "synthetic-key", **kwargs)
    assert len(sent) == 1 and clients[0].is_closed and base == before
    assert "PRIVATE_" not in str(caught.value)
    assert "PRIVATE_" not in json.dumps(caught.value.diagnostics)
    assert "raw" not in caught.value.diagnostics and "reasoning" not in caught.value.diagnostics


def test_valid_refusal_is_preserved_without_invented_claims():
    payload = valid_answer(status="insufficient_evidence", claims=[], missing_info=["No relevant evidence is available."])
    kwargs, sent, _, _, _ = mock_chat(payload=payload)
    result = answering.generate(make_result(), api_key_loader=lambda model: "synthetic-key", **kwargs)
    assert result["answer_generation"]["answer"] == payload and len(sent) == 1


@pytest.mark.parametrize("failure", ["status", "timeout"])
def test_sdk_failures_close_transport_and_never_retry(failure):
    base = make_result()
    before = deepcopy(base)
    kwargs, sent, clients, _, _ = mock_chat(status=503 if failure == "status" else 200,
                                          error=httpx.ReadTimeout("synthetic-timeout") if failure == "timeout" else None)
    with pytest.raises(Exception):
        answering.generate(base, api_key_loader=lambda model: "synthetic-key", **kwargs)
    assert len(sent) == 1 and clients[0].is_closed and base == before


@pytest.fixture
def api(tmp_path, monkeypatch):
    frozen = make_result()
    baseline = deepcopy(frozen)
    del baseline["routes"]["reranker"]
    del baseline["rerank_comparison"]
    del baseline["experiment_id"]
    baseline["request_counts"] = {"embedding": 1, "http": 3, "reranker": 0, "chat": 0}
    compare = Mock(return_value=baseline)

    def fake_rerank(result):
        result["routes"]["reranker"] = deepcopy(frozen["routes"]["reranker"])
        result["rerank_comparison"] = deepcopy(frozen["rerank_comparison"])
        result["request_counts"] = deepcopy(frozen["request_counts"])
        return result

    rerank, generate = Mock(side_effect=fake_rerank), Mock()
    monkeypatch.setattr(lab.hybrid, "prepare", Mock(return_value={"synthetic": True}))
    monkeypatch.setattr(lab.hybrid, "describe", Mock(return_value={"synthetic_plan": True}))
    monkeypatch.setattr(lab.hybrid, "compare", compare)
    monkeypatch.setattr(lab.reranking, "rerank", rerank)
    monkeypatch.setattr(lab.answering, "generate", generate)
    page = tmp_path / "synthetic-page.html"
    page.write_text("<!doctype html><title>Synthetic answer page</title>", encoding="utf-8")
    monkeypatch.setattr(lab, "PAGE", page)
    with TestClient(lab.create_app(tmp_path), base_url="http://127.0.0.1") as client:
        yield client, compare, rerank, generate


def start_experiment(client, *, rerank=True):
    response = client.post("/api/compare", headers=HEADERS, json={"query": "DNS", "mode": "hybrid"})
    assert response.status_code == 200
    result = response.json()
    if rerank:
        response = client.post("/api/rerank", headers=HEADERS, json={"experiment_id": result["experiment_id"]})
        assert response.status_code == 200
        result = response.json()
    return result


def fake_generation(result, model):
    result = deepcopy(result)
    result["answer_generation"] = {"model": model, "answer": valid_answer(),
                                   "sources": {}, "request_counts": {"embedding": 0, "http": 1, "reranker": 0, "chat": 1}}
    result["request_counts"]["http"] += 1
    result["request_counts"]["chat"] += 1
    return result


def test_api_plan_and_preview_do_not_generate_or_load_env(api):
    client, compare, rerank, generate = api
    for path in ("/", "/api/plan", "/assets/retrieval_export.js"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "access-control-allow-origin" not in response.headers
    assert client.get("/api/plan").json()["answer"]["default_model"] == "qwen3.8-flash"
    compare.assert_not_called()
    rerank.assert_not_called()
    generate.assert_not_called()


def test_answer_api_uses_frozen_rerank_and_reuses_same_model_success(api):
    client, compare, rerank, generate = api
    base = start_experiment(client)
    generate.side_effect = fake_generation
    compare.return_value["records"].clear()
    payload = {"experiment_id": base["experiment_id"], "model": "qwen3.8-flash"}
    first = client.post("/api/answer", headers=HEADERS, json=payload)
    second = client.post("/api/answer", headers=HEADERS, json=payload)
    assert first.status_code == second.status_code == 200 and first.json() == second.json()
    assert first.json()["records"] == base["records"]
    assert first.json()["request_counts"] == {"embedding": 1, "http": 5, "reranker": 1, "chat": 1}
    generate.assert_called_once_with(base, "qwen3.8-flash")
    assert compare.call_count == rerank.call_count == 1
    changed = client.post("/api/answer", headers=HEADERS, json={**payload, "model": "glm"})
    assert changed.status_code == 409 and generate.call_count == 1
    # Repeating rerank after answering also returns the complete saved stage.
    rerun = client.post("/api/rerank", headers=HEADERS, json={"experiment_id": base["experiment_id"]})
    assert rerun.status_code == 200 and rerun.json() == first.json()


@pytest.mark.parametrize("payload", [
    {}, {"experiment_id": "a"}, {"model": "glm"},
    {"experiment_id": "", "model": "glm"}, {"experiment_id": " ", "model": "glm"},
    {"experiment_id": "a" * 65, "model": "glm"}, {"experiment_id": 1, "model": "glm"},
    {"experiment_id": True, "model": "glm"}, {"experiment_id": "a", "model": "unknown"},
    {"experiment_id": "a", "model": ""}, {"experiment_id": "a", "model": True},
    {"experiment_id": "a", "model": "glm", "query": "Injected question"},
    {"experiment_id": "a", "model": "glm", "sources": ["Injected evidence"]},
    {"experiment_id": "a", "model": "glm", "base_url": "https://evil.invalid"},
    {"experiment_id": "a", "model": "glm", "api_key": "Injected key"}, [], None,
])
def test_answer_api_strictly_rejects_missing_extra_or_disallowed_fields(api, payload):
    client, compare, rerank, generate = api
    response = client.post("/api/answer", headers={**HEADERS, "Content-Type": "application/json"}, content=json.dumps(payload))
    assert response.status_code == 422
    compare.assert_not_called()
    rerank.assert_not_called()
    generate.assert_not_called()


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
def test_answer_api_requires_same_origin_header_json_and_limited_body(api, headers, content, status):
    client, _, _, generate = api
    response = client.post("/api/answer", headers=headers, content=content)
    assert response.status_code == status and set(response.json()) == {"detail"}
    generate.assert_not_called()


def test_answer_api_requires_current_reranked_snapshot(api):
    client, _, _, generate = api
    payload = {"experiment_id": "unknown", "model": "glm"}
    assert client.post("/api/answer", headers=HEADERS, json=payload).status_code == 409
    base = start_experiment(client, rerank=False)
    assert client.post("/api/answer", headers=HEADERS, json={**payload, "experiment_id": base["experiment_id"]}).status_code == 409
    latest = start_experiment(client)
    assert client.post("/api/answer", headers=HEADERS, json={**payload, "experiment_id": base["experiment_id"]}).status_code == 409
    client.post("/api/compare", headers=HEADERS, json={"query": "New question", "mode": "bm25"})
    assert client.post("/api/answer", headers=HEADERS, json={**payload, "experiment_id": latest["experiment_id"]}).status_code == 409
    generate.assert_not_called()


def test_new_failed_compare_invalidates_answer_token(api):
    client, compare, _, generate = api
    base = start_experiment(client)
    compare.side_effect = ValueError("Synthetic failure")
    assert client.post("/api/compare", headers=HEADERS, json={"query": "Next", "mode": "hybrid"}).status_code == 503
    assert client.post("/api/answer", headers=HEADERS, json={"experiment_id": base["experiment_id"], "model": "glm"}).status_code == 409
    generate.assert_not_called()


def test_failed_answer_is_sanitized_preserves_snapshot_and_requires_explicit_retry(api, capsys):
    client, _, _, generate = api
    base, received = start_experiment(client), []

    def fail_then_succeed(result, model):
        received.append(deepcopy(result))
        if len(received) == 1:
            result["records"].clear()
            result["query"] = "Worker mutation"
            raise ValueError("PRIVATE_RAW_RESPONSE_AND_KEY_MUST_NOT_LEAK")
        return fake_generation(result, model)

    generate.side_effect = fail_then_succeed
    payload = {"experiment_id": base["experiment_id"], "model": "glm"}
    failed = client.post("/api/answer", headers=HEADERS, json=payload)
    assert failed.status_code == 503 and set(failed.json()) == {"detail"}
    assert "PRIVATE_" not in failed.text and "PRIVATE_" not in capsys.readouterr().out
    assert "answer_generation" not in failed.json() and generate.call_count == 1
    retry = client.post("/api/answer", headers=HEADERS, json=payload)
    assert retry.status_code == 200 and received == [base, base]
    assert retry.json()["records"] == base["records"] and generate.call_count == 2


def test_generation_shares_lock_with_compare_and_rerank_to_avoid_duplicate_requests(api):
    client, compare, rerank, generate = api
    base = start_experiment(client)
    entered, release = Event(), Event()

    def delayed(result, model):
        entered.set()
        assert release.wait(5), "Synthetic generation was not released"
        return fake_generation(result, model)

    generate.side_effect = delayed
    payload = {"experiment_id": base["experiment_id"], "model": "glm"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/api/answer", headers=HEADERS, json=payload)
        try:
            assert entered.wait(5), "Synthetic generation did not start"
            assert client.post("/api/answer", headers=HEADERS, json=payload).status_code == 409
            assert client.post("/api/rerank", headers=HEADERS, json={"experiment_id": base["experiment_id"]}).status_code == 409
            assert client.post("/api/compare", headers=HEADERS, json={"query": "Next", "mode": "hybrid"}).status_code == 409
            assert generate.call_count == compare.call_count == rerank.call_count == 1
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
    assert client.post("/api/answer", headers=HEADERS, json=payload).status_code == 200
    generate.assert_called_once()
