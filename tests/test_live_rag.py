"""Offline tests for the real-function LangGraph entry point.

The graph really runs, while business provider boundaries use synthetic data or
in-memory HTTP transports. No credentials, Milvus instance or network is used.
"""

import builtins
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import socket
from threading import Event
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
from langgraph.graph.state import CompiledStateGraph
from langsmith.run_helpers import get_tracing_context
from openai import OpenAI
import pytest
import requests

from qa_platform import retrieval_lab as lab
from qa_platform.workflows import live_rag as live


HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}
QUERY = "Synthetic DNS question?"
MODEL = "qwen3.8-flash"
COUNTS = {"embedding": 1, "http": 5, "reranker": 1, "chat": 1}


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live RAG test attempted network or .env access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    monkeypatch.setattr(live.answering, "load_chat_key", forbidden)
    monkeypatch.setattr(live.reranking, "load_api_key", forbidden)
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


@pytest.fixture
def client(tmp_path):
    with TestClient(lab.create_app(tmp_path / "missing-index"), base_url="http://127.0.0.1") as active:
        yield active


def retrieval_result(query=QUERY):
    lexical, dense = [f"c{i}" for i in range(8)], [f"c{i}" for i in range(5, 13)]
    pool = []
    for chunk_id in sorted(set(lexical) | set(dense)):
        ranks = {name: ids.index(chunk_id) + 1 for name, ids in (("bm25", lexical), ("dense", dense)) if chunk_id in ids}
        contributions = {name: 1 / (60 + rank) for name, rank in ranks.items()}
        pool.append({"chunk_id": chunk_id, "ranks": ranks, "contributions": contributions, "score": sum(contributions.values())})
    pool.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(pool, 1):
        row["rank"] = rank
    return {
        "query": query, "mode": "hybrid", "top_k": 5, "rrf_constant": 60,
        "scope_count": 156, "recall_k": 8, "tokens": ["dns"],
        "request_counts": {"embedding": 1, "http": 3, "reranker": 0, "chat": 0},
        "routes": {
            "bm25": [{"chunk_id": value, "rank": rank, "score": 20 - rank, "matched_terms": ["dns"]} for rank, value in enumerate(lexical, 1)],
            "dense": [{"chunk_id": value, "rank": rank, "score": 1 - rank / 100} for rank, value in enumerate(dense, 1)],
            "rrf": deepcopy(pool[:5]),
        },
        "rrf_pool": pool,
        "records": {
            row["chunk_id"]: {
                "text": f"Synthetic original paragraph for {row['chunk_id']}.\n\nAnother synthetic paragraph.",
                "metadata": {
                    "chunk_id": row["chunk_id"], "title": "Synthetic title",
                    "section_path": ["Synthetic section"],
                    "source_url": "https://example.invalid/source",
                    "review_status": "unreviewed_official_document", "kubernetes_version": "unknown",
                    "source_path": "/private/LOCAL_PATH_MUST_NOT_LEAK.md",
                    "vector": ["VECTOR_MUST_NOT_LEAK"], "extra": "EXTRA_METADATA_MUST_NOT_LEAK",
                },
            } for row in pool
        },
        "notice": "Synthetic only",
    }


def final_result(query=QUERY, model=MODEL):
    result = retrieval_result(query)
    first_id = result["rrf_pool"][-1]["chunk_id"]
    row = result["records"][first_id]
    result["request_counts"] = dict(COUNTS)
    source = {"chunk_id": first_id, **deepcopy(row), "quotes": {"Q1": row["text"].splitlines()[0]}}
    result["answer_generation"] = {
        "model": model,
        "answer": {"status": "partial", "claims": [
            {"text": "Synthetic conditional explanation.", "evidence": [{"source_id": "S1", "quote_id": "Q1"}]},
        ], "missing_info": ["Actual cluster details remain unknown."]},
        "sources": {"S1": source, "S2": deepcopy(source)},
    }
    return result


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = []
    check = {"synthetic_check": True}
    final = final_result()

    def prepare(index_dir, query):
        calls.append(("prepare", index_dir, query, get_tracing_context()["enabled"]))
        return check

    def compare(actual_check, mode):
        assert actual_check is check and mode == "hybrid"
        calls.append(("retrieve", mode))
        return retrieval_result(calls[0][2])

    def rerank(result):
        assert re.fullmatch(r"[0-9a-f]{32}", result["experiment_id"])
        assert len(result["rrf_pool"]) == 13 and len(result["routes"]["rrf"]) == 5
        calls.append(("rerank", result))
        return {**result, "reranked_sentinel": True}

    def answer(result, model):
        assert result["reranked_sentinel"] is True
        calls.append(("answer", result, model))
        final["query"] = result["query"]
        final["answer_generation"]["model"] = model
        return deepcopy(final)

    for module, name, function in (
        (live.hybrid, "prepare", prepare), (live.hybrid, "compare", compare),
        (live.reranking, "rerank", rerank), (live.answering, "generate", answer),
    ):
        monkeypatch.setattr(module, name, function)
    return {"calls": calls, "check": check, "final": final}


def test_graph_registers_real_three_stage_structure_without_running_business(monkeypatch, tmp_path):
    forbidden = Mock(side_effect=AssertionError("Building a graph must not execute its nodes"))
    for module, name in ((live.hybrid, "prepare"), (live.hybrid, "compare"), (live.reranking, "rerank"), (live.answering, "generate")):
        monkeypatch.setattr(module, name, forbidden)
    graph = live.build_graph(tmp_path)
    assert isinstance(graph, CompiledStateGraph)
    assert graph.checkpointer is graph.cache is graph.store is None
    assert all(node.retry_policy is None for node in graph.nodes.values())
    structure = graph.get_graph()
    assert set(structure.nodes) == {"__start__", "retrieve", "rerank", "answer", "__end__"}
    assert {(edge.source, edge.target) for edge in structure.edges} == {
        ("__start__", "retrieve"), ("retrieve", "rerank"), ("rerank", "answer"), ("answer", "__end__"),
    }
    assert not any(edge.conditional for edge in structure.edges)
    forbidden.assert_not_called()


def test_state_is_small_and_result_replaces_previous_node_result():
    assert live.LiveRAGState.__annotations__ == {"query": str, "model": str, "result": dict}
    assert not live.LiveRAGState.__required_keys__  # Initial input has no result yet.


def test_real_graph_stream_runs_each_existing_business_function_once(fake_pipeline, tmp_path):
    graph = live.build_graph(tmp_path)
    updates = list(graph.stream({"query": QUERY, "model": MODEL}, stream_mode="updates"))
    assert [next(iter(update)) for update in updates] == list(live.PIPELINE)
    assert all(set(next(iter(update.values()))) == {"result"} for update in updates)
    assert fake_pipeline["calls"][0][:3] == ("prepare", tmp_path, QUERY)
    assert [row[0] for row in fake_pipeline["calls"]] == ["prepare", "retrieve", "rerank", "answer"]
    assert updates[0]["retrieve"]["result"]["rrf_pool"] == updates[1]["rerank"]["result"]["rrf_pool"]
    assert "reranked_sentinel" not in updates[0]["retrieve"]["result"]
    assert updates[-1]["answer"]["result"] == fake_pipeline["final"]


@pytest.mark.parametrize("model", [MODEL, "qwen3.8-max", "glm"])
def test_run_uses_selected_model_and_returns_only_validated_public_sources(fake_pipeline, tmp_path, model):
    result = live.run("  " + QUERY + "  ", model, tmp_path)
    assert set(result) == {"schema_version", "query", "model", "answer", "sources", "request_counts", "steps", "elapsed_seconds"}
    assert result["schema_version"] == "rag-live/v1" and result["model"] == model
    assert result["query"] == QUERY and result["steps"] == list(live.PIPELINE)
    assert result["request_counts"] == COUNTS and result["elapsed_seconds"] >= 0
    assert set(result["sources"]) == {"S1"}  # S2 was selected but not cited.
    source = result["sources"]["S1"]
    assert set(source) == {"chunk_id", "text", "quotes", "metadata"}
    assert set(source["metadata"]) == {"title", "section_path", "source_url", "review_status", "kubernetes_version"}
    assert source["quotes"]["Q1"] in source["text"]
    assert "MUST_NOT_LEAK" not in json.dumps(result)
    assert fake_pipeline["calls"][0][3] is False


def test_run_disables_environment_enabled_langsmith_tracing(fake_pipeline, monkeypatch, tmp_path):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    live.run(QUERY, MODEL, tmp_path)
    assert fake_pipeline["calls"][0][3] is False


@pytest.mark.parametrize("query,model", [("", MODEL), (" \n ", MODEL), ("x" * 2001, MODEL), (None, MODEL), (123, MODEL), (QUERY, "unknown")])
def test_invalid_run_input_is_rejected_before_graph_construction(query, model, monkeypatch, tmp_path):
    build = Mock(side_effect=AssertionError("Invalid input reached graph"))
    monkeypatch.setattr(live, "build_graph", build)
    with pytest.raises(ValueError):
        live.run(query, model, tmp_path)
    build.assert_not_called()


@pytest.mark.parametrize("stage,completed", [("retrieve", []), ("rerank", ["retrieve"]), ("answer", ["retrieve", "rerank"])])
def test_stage_failures_stop_immediately_without_retry_or_private_error_leak(fake_pipeline, monkeypatch, tmp_path, stage, completed):
    failed = Mock(side_effect=RuntimeError("PRIVATE_FAILURE_body_secret_path"))
    module, name = {"retrieve": (live.hybrid, "compare"), "rerank": (live.reranking, "rerank"), "answer": (live.answering, "generate")}[stage]
    monkeypatch.setattr(module, name, failed)
    with pytest.raises(live.RAGRunError) as exc:
        live.run(QUERY, MODEL, tmp_path)
    assert exc.value.stage == stage and exc.value.completed_steps == completed
    assert "PRIVATE_FAILURE" not in str(exc.value)
    assert "不自动重试" in str(exc.value) and "费用" in str(exc.value)
    failed.assert_called_once()
    called = [row[0] for row in fake_pipeline["calls"]]
    assert "answer" not in called
    if stage == "retrieve":
        assert called == ["prepare"]
    if stage == "rerank":
        assert called == ["prepare", "retrieve"]


@pytest.mark.parametrize("mutation", ["unknown_source", "unknown_quote", "mismatched_quote", "wrong_model", "extra_answer_field", "empty_claims", "partial_no_gap", "refusal_with_claims"])
def test_public_validation_is_fail_closed_before_returning_answer(fake_pipeline, tmp_path, mutation):
    generation = fake_pipeline["final"]["answer_generation"]
    answer = generation["answer"]
    if mutation == "unknown_source":
        answer["claims"][0]["evidence"][0]["source_id"] = "S99"
    elif mutation == "unknown_quote":
        answer["claims"][0]["evidence"][0]["quote_id"] = "Q99"
    elif mutation == "mismatched_quote":
        generation["sources"]["S1"]["quotes"]["Q1"] = "PRIVATE_UNVALIDATED_RESPONSE"
    elif mutation == "wrong_model":
        # The stub intentionally respects the selected model, so override it.
        original = live.answering.generate

        def mismatch(result, model):
            returned = original(result, model)
            returned["answer_generation"]["model"] = "wrong-model"
            return returned
        # Scoped mutation, restored by the fixture's monkeypatch teardown.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(live.answering, "generate", mismatch)
            with pytest.raises(live.RAGRunError) as exc:
                live.run(QUERY, MODEL, tmp_path)
        assert exc.value.stage == "answer"
        return
    elif mutation == "extra_answer_field":
        answer["private_draft"] = "PRIVATE_UNVALIDATED_RESPONSE"
    elif mutation == "empty_claims":
        answer["claims"] = []
    elif mutation == "partial_no_gap":
        answer["missing_info"] = []
    else:
        answer["status"] = "insufficient_evidence"
    with pytest.raises(live.RAGRunError) as exc:
        live.run(QUERY, MODEL, tmp_path)
    assert exc.value.stage == "answer"
    assert "PRIVATE_UNVALIDATED_RESPONSE" not in str(exc.value)


def test_evidence_insufficiency_is_valid_refusal_not_an_invented_answer(fake_pipeline, tmp_path):
    fake_pipeline["final"]["answer_generation"]["answer"] = {
        "status": "insufficient_evidence", "claims": [], "missing_info": ["Relevant source material is missing."],
    }
    result = live.run(QUERY, MODEL, tmp_path)
    assert result["answer"]["status"] == "insufficient_evidence"
    assert result["sources"] == {} and result["steps"] == list(live.PIPELINE)


def test_reused_real_reranker_and_answer_functions_receive_full_pool_and_five_sources(monkeypatch, tmp_path):
    base, sent = retrieval_result(), []
    monkeypatch.setattr(live.hybrid, "prepare", lambda path, query: {"query": query})
    monkeypatch.setattr(live.hybrid, "compare", lambda check, mode: deepcopy(base))
    original_rerank, original_answer = live.reranking.rerank, live.answering.generate

    def cloud(request):
        sent.append(request)
        payload = json.loads(request.content)
        if "documents" in payload.get("input", {}):
            count = len(payload["input"]["documents"])
            return httpx.Response(200, json={"output": {"results": [
                {"index": index, "relevance_score": (index + 1) / count} for index in range(count)
            ]}})
        raw = json.dumps(final_result()["answer_generation"]["answer"])
        chunks = [{"id": "synthetic", "choices": [{"index": 0, "delta": {"content": raw}, "finish_reason": "stop"}]}]
        body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    def transport(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(cloud), **kwargs)

    monkeypatch.setattr(live.reranking, "rerank", lambda result: original_rerank(result, client_factory=transport, api_key_loader=lambda: "synthetic"))
    monkeypatch.setattr(live.answering, "generate", lambda result, model: original_answer(result, model, http_client_factory=transport, client_factory=OpenAI, api_key_loader=lambda selected: "synthetic"))
    result = live.run(QUERY, MODEL, tmp_path)
    assert len(sent) == 2 and result["request_counts"] == COUNTS
    rerank_payload, answer_payload = [json.loads(request.content) for request in sent]
    assert len(rerank_payload["input"]["documents"]) == rerank_payload["parameters"]["top_n"] == 13
    assert rerank_payload["input"]["documents"] == [base["records"][row["chunk_id"]]["text"] for row in base["rrf_pool"]]
    evidence = json.loads(answer_payload["messages"][1]["content"])["evidence"]
    assert len(evidence) == 5
    assert evidence[0]["content"] == base["records"][base["rrf_pool"][-1]["chunk_id"]]["text"]
    assert "MUST_NOT_LEAK" not in b"".join(request.content for request in sent).decode()
    assert result["sources"]["S1"]["chunk_id"] == base["rrf_pool"][-1]["chunk_id"]


def test_describe_and_get_plan_never_prepare_index_or_build_graph(client, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Read-only plan started business work"))
    for module, name in ((live, "build_graph"), (live, "run"), (live.hybrid, "prepare"), (live.hybrid, "compare"), (live.reranking, "rerank"), (live.answering, "generate")):
        monkeypatch.setattr(module, name, forbidden)
    plan = live.describe()
    assert plan["schema_version"] == "rag-live-plan/v1"
    assert plan["pipeline"] == list(live.PIPELINE)
    assert plan["default_model"] == MODEL
    assert {row["id"] for row in plan["models"]} == set(live.answering.MODEL_CONFIGS)
    assert set(plan["reranker"]) == {"model", "url"}
    assert "不自动重试" in plan["notice"] and "不外发LangSmith" in plan["notice"]
    response = client.get("/api/qa/plan")
    assert response.status_code == 200 and response.json() == plan
    assert response.headers["cache-control"] == "no-store"
    forbidden.assert_not_called()


def test_qa_page_is_static_and_does_not_start_models(client, monkeypatch, tmp_path):
    static = tmp_path / "qa.html"
    static.write_text("<title>Synthetic page</title>", encoding="utf-8")
    monkeypatch.setattr(lab, "QA_PAGE", static)
    forbidden = Mock(side_effect=AssertionError("Opening page ran graph"))
    monkeypatch.setattr(live, "run", forbidden)
    response = client.get("/qa")
    assert response.status_code == 200 and "Synthetic page" in response.text
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    forbidden.assert_not_called()


def test_api_uses_actual_graph_and_only_its_public_result(client, fake_pipeline):
    response = client.post("/api/qa", headers=HEADERS, json={"query": QUERY, "model": MODEL, "confirmed": True})
    assert response.status_code == 200
    assert response.json()["schema_version"] == "rag-live/v1"
    assert response.json()["steps"] == list(live.PIPELINE)
    assert "MUST_NOT_LEAK" not in response.text
    assert [row[0] for row in fake_pipeline["calls"]] == ["prepare", "retrieve", "rerank", "answer"]


@pytest.mark.parametrize("updates", [
    {"query": ""}, {"query": "  "}, {"query": "q" * 2001}, {"query": 1},
    {"model": "unknown"}, {"model": 1}, {"confirmed": False}, {"confirmed": 1},
    {"confirmed": "true"}, {"confirmed": None}, {"index_dir": "/private"},
    {"evidence": [{"text": "Injected arbitrary body"}]}, {"api_key": "injected"},
])
def test_invalid_api_payload_never_calls_graph(client, monkeypatch, updates):
    run = Mock(side_effect=AssertionError("Invalid payload reached graph"))
    monkeypatch.setattr(live, "run", run)
    payload = {"query": QUERY, "model": MODEL, "confirmed": True, **updates}
    assert client.post("/api/qa", headers=HEADERS, json=payload).status_code == 422
    run.assert_not_called()


@pytest.mark.parametrize("missing", ["query", "model", "confirmed"])
def test_missing_required_api_field_is_rejected(client, monkeypatch, missing):
    run = Mock(side_effect=AssertionError("Missing required field reached graph"))
    monkeypatch.setattr(live, "run", run)
    payload = {"query": QUERY, "model": MODEL, "confirmed": True}
    del payload[missing]
    assert client.post("/api/qa", headers=HEADERS, json=payload).status_code == 422
    run.assert_not_called()


@pytest.mark.parametrize("headers,status", [
    ({}, 403), ({"X-Retrieval-Lab": "0"}, 403),
    ({"X-Retrieval-Lab": "1", "Origin": "https://evil.invalid"}, 403),
    ({"X-Retrieval-Lab": "1", "Origin": "http://localhost"}, 403),
    ({**HEADERS, "Content-Type": "text/plain"}, 415),
])
def test_qa_endpoint_requires_same_origin_custom_header_and_json(client, monkeypatch, headers, status):
    run = Mock(side_effect=AssertionError("CSRF or invalid content type reached graph"))
    monkeypatch.setattr(live, "run", run)
    assert client.post("/api/qa", headers=headers, json={"query": QUERY, "model": MODEL, "confirmed": True}).status_code == status
    run.assert_not_called()


def test_qa_rejects_oversized_and_malformed_bodies_without_execution(client, monkeypatch):
    run = Mock(side_effect=AssertionError("Invalid body reached graph"))
    monkeypatch.setattr(live, "run", run)
    headers = {**HEADERS, "Content-Type": "application/json"}
    assert client.post("/api/qa", headers=headers, content=b"x" * 16385).status_code == 413
    assert client.post("/api/qa", headers=headers, content=b"{").status_code == 422
    run.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError("PRIVATE_ERROR_BODY_API_KEY"), live.RAGRunError("answer", ["retrieve", "rerank"])])
def test_api_failure_sanitizes_error_and_releases_shared_lock(client, monkeypatch, error):
    run = Mock(side_effect=[error, {"synthetic": "next call"}])
    monkeypatch.setattr(live, "run", run)
    payload = {"query": QUERY, "model": MODEL, "confirmed": True}
    failed = client.post("/api/qa", headers=HEADERS, json=payload)
    assert failed.status_code == 503 and "PRIVATE_ERROR" not in failed.text
    assert "不自动重试" in failed.json()["detail"] and "费用" in failed.json()["detail"]
    assert run.call_count == 1
    succeeded = client.post("/api/qa", headers=HEADERS, json=payload)
    assert succeeded.status_code == 200 and run.call_count == 2


def test_api_shared_lock_blocks_other_operations_without_starting_them(client, monkeypatch):
    entered, release = Event(), Event()

    def waiting(*args):
        entered.set()
        assert release.wait(5)
        return {"synthetic": "done"}

    run = Mock(side_effect=waiting)
    monkeypatch.setattr(live, "run", run)
    payload = {"query": QUERY, "model": MODEL, "confirmed": True}
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.post, "/api/qa", headers=HEADERS, json=payload)
        try:
            assert entered.wait(5)
            assert client.post("/api/qa", headers=HEADERS, json=payload).status_code == 409
            assert client.post("/api/compare", headers=HEADERS, json={"query": QUERY, "mode": "bm25"}).status_code == 409
            assert client.post("/api/workflow/approval/start", headers=HEADERS, json={}).status_code == 409
            assert client.get("/api/workflow/approval/state").status_code == 409
            assert run.call_count == 1
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
    assert client.get("/api/workflow/approval/state").status_code == 200


def test_automatic_qa_does_not_replace_existing_experiment_snapshot(client, monkeypatch):
    base = retrieval_result()
    monkeypatch.setattr(live.hybrid, "prepare", lambda *args: {})
    monkeypatch.setattr(live.hybrid, "compare", lambda *args: deepcopy(base))
    compared = client.post("/api/compare", headers=HEADERS, json={"query": QUERY, "mode": "hybrid"})
    identity = compared.json()["experiment_id"]
    assert compared.status_code == 200
    monkeypatch.setattr(live, "run", lambda *args: {"synthetic": "independent QA"})
    response = client.post("/api/qa", headers=HEADERS, json={"query": "Other question", "model": MODEL, "confirmed": True})
    assert response.status_code == 200
    seen = []

    def rerank(result):
        seen.append(result)
        return {**result, "routes": {**result["routes"], "reranker": []}}

    monkeypatch.setattr(live.reranking, "rerank", rerank)
    response = client.post("/api/rerank", headers=HEADERS, json={"experiment_id": identity})
    assert response.status_code == 200
    assert seen[0]["query"] == QUERY and seen[0]["experiment_id"] == identity


def test_automatic_qa_does_not_modify_paused_approval_lesson(client, monkeypatch):
    paused = client.post("/api/workflow/approval/start", headers=HEADERS, json={}).json()
    monkeypatch.setattr(live, "run", lambda *args: {"synthetic": "independent QA"})
    assert client.post("/api/qa", headers=HEADERS, json={"query": QUERY, "model": MODEL, "confirmed": True}).status_code == 200
    after = client.get("/api/workflow/approval/state").json()
    assert after == paused
