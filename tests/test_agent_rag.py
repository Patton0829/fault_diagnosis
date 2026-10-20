"""Offline integration tests for the bounded native-tool LangGraph workflow.

The graph and OpenAI protocol actually run; search results are synthetic and all
HTTP uses MockTransport. No credentials, model server or Milvus file is touched.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import socket
from threading import Event
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph
from langsmith.run_helpers import get_tracing_context
from openai import OpenAI
import pytest
import requests

from qa_platform import retrieval_lab as lab
from qa_platform.rag.answering import GroundedAnswer
from qa_platform.retrieval.hybrid import RRFHit
from qa_platform.retrieval.pipeline import RerankedHit
from qa_platform.workflows import agent_rag as agent
from qa_platform.workflows import live_rag as single


MODEL = "qwen3.8-flash"
QUERY = "Synthetic DNS and namespace question?"
SECRET = "PRIVATE_TOKEN_PATH_VECTOR_MUST_NOT_LEAK"
HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Agent tests must not read credentials or use external I/O")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    monkeypatch.setattr(agent, "load_chat_key", forbidden)
    monkeypatch.setattr(agent.answering, "load_chat_key", forbidden)


def native_call(query="Synthetic first search", call_id="call-first"):
    return {
        "id": "synthetic-completion", "object": "chat.completion", "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": "search_knowledge", "arguments": json.dumps({"query": query}),
            }}],
        }}],
    }


def stop_response(content="PRIVATE_UNVALIDATED_PLANNER_TEXT"):
    result = native_call()
    result["choices"][0] = {"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": content, "tool_calls": None,
    }}
    return result


def synthetic_round(query, ids, *, contract="same-contract"):
    records, sources = {}, {}
    for position, chunk_id in enumerate(ids):
        text = f"Synthetic source paragraph for {chunk_id}.\n\nConditional explanation only."
        metadata = {
            "chunk_id": chunk_id, "title": "Synthetic DNS source", "section_path": ["Synthetic"],
            "source_url": "https://example.invalid/source", "review_status": "unreviewed_official_document",
            "kubernetes_version": "unknown", "is_test": False, "source_path": SECRET,
            "vector": [SECRET], "document_mode": "body-only", "quote_strategy": "paragraph-v1",
        }
        records[chunk_id] = {"text": text, "metadata": deepcopy(metadata)}
        candidate = RRFHit(Document(page_content=text, metadata=metadata), 0.02, {"dense": 1}, {"dense": 0.02})
        sources[f"S{position + 1}"] = RerankedHit(candidate, position, 0.9 - position / 100)
    return {
        "contract": contract, "sources": sources,
        "result": {"query": query, "records": records,
                   "request_counts": {"embedding": 1, "http": 4, "reranker": 1, "chat": 0}},
    }


@pytest.fixture
def business(monkeypatch):
    calls = {"search": [], "answer": [], "tracing": []}

    def search(index_dir, query, *, expected_contract=None):
        calls["search"].append((index_dir, query, expected_contract))
        calls["tracing"].append(get_tracing_context()["enabled"])
        prefix = "a" if len(calls["search"]) == 1 else "b"
        return synthetic_round(query, [prefix + str(number) for number in range(5)])

    def answer(query, sources, model):
        calls["answer"].append((query, deepcopy(sources), model))
        calls["tracing"].append(get_tracing_context()["enabled"])
        return GroundedAnswer(status="partial", claims=[
            {"text": "Synthetic conditional explanation.", "evidence": [
                {"source_id": source_id, "quote_id": "Q1"} for source_id in list(sources)[:2]
            ]},
        ], missing_info=["Actual cluster evidence is still required."])

    monkeypatch.setattr(agent.knowledge, "search_knowledge", search)
    monkeypatch.setattr(agent.answering, "call_answer", answer)
    return calls


def run_with_responses(tmp_path, responses, **kwargs):
    sent, http_options, sdk_options = [], [], []

    def handler(request):
        sent.append(request)
        if len(sent) > len(responses):
            raise AssertionError("Unexpected additional planner request")
        response = responses[len(sent) - 1]
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, httpx.Response) else httpx.Response(200, json=response)

    def transport(**options):
        http_options.append(options)
        return httpx.Client(transport=httpx.MockTransport(handler), **options)

    def sdk(**options):
        sdk_options.append(options)
        return OpenAI(**options)

    result = agent.run(QUERY, MODEL, tmp_path, http_client_factory=transport,
                       client_factory=sdk, api_key_loader=lambda selected: SECRET, **kwargs)
    return result, sent, http_options, sdk_options


def test_graph_has_real_bounded_conditional_edges_without_running_nodes(monkeypatch, tmp_path):
    forbidden = Mock(side_effect=AssertionError("Constructing graph must not execute business work"))
    monkeypatch.setattr(agent.knowledge, "search_knowledge", forbidden)
    monkeypatch.setattr(agent.answering, "call_answer", forbidden)
    graph = agent.build_graph(tmp_path, forbidden)
    assert isinstance(graph, CompiledStateGraph)
    assert graph.checkpointer is graph.cache is graph.store is None
    assert all(node.retry_policy is None for node in graph.nodes.values())
    structure = graph.get_graph()
    assert set(structure.nodes) == {"__start__", "plan", "search", "answer", "__end__"}
    assert {(edge.source, edge.target, edge.conditional) for edge in structure.edges} == {
        ("__start__", "plan", False), ("plan", "search", True), ("plan", "answer", True),
        ("search", "plan", True), ("search", "answer", True), ("answer", "__end__", False),
    }
    forbidden.assert_not_called()


def test_two_searches_preserve_both_rounds_and_finish_at_budget(business, tmp_path):
    result, sent, _, _ = run_with_responses(tmp_path, [native_call(), native_call("Synthetic second search", "call-second")])
    assert result["schema_version"] == "rag-live/v1" and result["mode"] == "agent"
    assert result["steps"] == ["plan", "search", "plan", "search", "answer"]
    assert result["agent"]["planner_calls"] == len(sent) == 2
    assert result["agent"]["max_tool_calls"] == 2
    assert result["agent"]["stop_reason"] == "tool_budget_reached"
    assert result["request_counts"] == {"embedding": 2, "http": 11, "reranker": 2, "chat": 3}
    assert [row[2] for row in business["search"]] == [None, "same-contract"]
    selected = business["answer"][0][1]
    assert [hit.candidate.document.metadata["chunk_id"] for hit in selected.values()] == ["a0", "b0", "a1", "b1", "a2"]
    assert [row["chunk_id"] for row in result["sources"].values()] == ["a0", "b0"]
    assert all(set(call) == {"id", "name", "query", "evidence_ids"} for call in result["agent"]["tool_calls"])
    assert SECRET not in json.dumps(result)
    assert result["elapsed_seconds"] >= 0


def test_one_search_then_native_stop_generates_once_and_ignores_planner_text(business, tmp_path):
    result, sent, _, _ = run_with_responses(tmp_path, [native_call(), stop_response()])
    assert result["steps"] == ["plan", "search", "plan", "answer"]
    assert len(business["search"]) == len(business["answer"]) == 1
    assert result["agent"]["stop_reason"] == "model_finished"
    assert result["request_counts"] == {"embedding": 1, "http": 7, "reranker": 1, "chat": 3}
    assert len(sent) == 2 and "PRIVATE_UNVALIDATED_PLANNER_TEXT" not in json.dumps(result)


def test_stop_without_native_call_refuses_locally_without_final_generation(business, tmp_path):
    result, sent, _, _ = run_with_responses(tmp_path, [stop_response('{"tool_calls": [{"name": "search_knowledge"}]}')])
    assert result["steps"] == ["plan", "answer"]
    assert result["answer"]["status"] == "insufficient_evidence"
    assert result["answer"]["claims"] == [] and result["sources"] == {}
    assert result["agent"]["tool_calls"] == [] and result["agent"]["planner_calls"] == len(sent) == 1
    assert result["request_counts"] == {"embedding": 0, "http": 1, "reranker": 0, "chat": 1}
    assert business["search"] == business["answer"] == []


def test_native_protocol_uses_call_ids_tools_and_only_whitelisted_evidence(business, tmp_path):
    _, sent, http_options, sdk_options = run_with_responses(tmp_path, [native_call(), stop_response()])
    first, second = [json.loads(row.content) for row in sent]
    assert first["messages"][1] == {"role": "user", "content": QUERY}
    assert first["tools"] == second["tools"] == agent.knowledge.TOOLS
    assert first["tool_choice"] == second["tool_choice"] == "auto"
    assert first["stream"] is second["stream"] is False
    assert first["enable_thinking"] is second["enable_thinking"] is False
    assert first["max_tokens"] == second["max_tokens"] == agent.MAX_PLANNER_TOKENS
    assert second["messages"][:-2] == first["messages"]
    assistant, tool = second["messages"][-2:]
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"] == "call-first"
    assert tool["role"] == "tool"
    evidence = json.loads(tool["content"])["evidence"]
    assert len(evidence) == 5
    assert set(evidence[0]) == {"chunk_id", "text", "quotes", "is_test", "review_status"}
    assert SECRET not in b"".join(row.content for row in sent).decode()
    assert http_options == [{"trust_env": False, "follow_redirects": False, "timeout": 60.0}]
    assert sdk_options[0]["max_retries"] == 0
    assert sdk_options[0]["organization"] == sdk_options[0]["project"] == ""


@pytest.mark.parametrize("failure", [
    "multiple_choices", "multiple_calls", "unknown_tool", "bad_json", "extra_argument",
    "integer_query", "empty_query", "long_query", "empty_id", "control_id", "long_id", "truncated",
])
def test_invalid_native_protocol_stops_before_search_and_never_retries(business, tmp_path, failure):
    response = native_call()
    choice, message = response["choices"][0], response["choices"][0]["message"]
    call = message["tool_calls"][0]
    if failure == "multiple_choices":
        response["choices"].append(deepcopy(choice))
    elif failure == "multiple_calls":
        message["tool_calls"].append(deepcopy(call))
    elif failure == "unknown_tool":
        call["function"]["name"] = "run_shell_" + SECRET
    elif failure == "bad_json":
        call["function"]["arguments"] = SECRET
    elif failure in {"extra_argument", "integer_query", "empty_query", "long_query"}:
        args = {"query": "Valid synthetic query"}
        if failure == "extra_argument":
            args["index_dir"] = SECRET
        else:
            args["query"] = {"integer_query": 123, "empty_query": "  ", "long_query": "x" * 2001}[failure]
        call["function"]["arguments"] = json.dumps(args)
    elif failure in {"empty_id", "control_id", "long_id"}:
        call["id"] = {"empty_id": " ", "control_id": "call\n" + SECRET, "long_id": "x" * 257}[failure]
    else:
        choice["finish_reason"] = "length"
    with pytest.raises(agent.AgentRunError) as caught:
        run_with_responses(tmp_path, [response])
    assert caught.value.stage == "plan" and caught.value.completed_steps == []
    assert SECRET not in str(caught.value) and caught.value.__cause__ is None
    assert "不自动重试" in str(caught.value)
    assert business["search"] == business["answer"] == []


@pytest.mark.parametrize("failure", ["same_id", "same_query", "normalized_same_query"])
def test_second_call_must_have_unique_id_and_different_query(business, tmp_path, failure):
    second = native_call("Synthetic second search", "call-second")
    if failure == "same_id":
        second["choices"][0]["message"]["tool_calls"][0]["id"] = "call-first"
    else:
        query = "Synthetic first search" if failure == "same_query" else " SYNTHETIC   FIRST search "
        second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"query": query})
    with pytest.raises(agent.AgentRunError) as caught:
        run_with_responses(tmp_path, [native_call(), second])
    assert caught.value.stage == "plan" and caught.value.completed_steps == ["plan", "search"]
    assert len(business["search"]) == 1 and business["answer"] == []


@pytest.mark.parametrize("stage", ["plan", "search", "answer"])
def test_stage_failure_is_sanitized_and_no_fallback_occurs(business, monkeypatch, tmp_path, stage):
    failed = Mock(side_effect=RuntimeError(SECRET))
    responses = [native_call(), stop_response()]
    if stage == "plan":
        responses = [httpx.Response(500, json={"error": {"message": SECRET}})]
    else:
        monkeypatch.setattr(agent.knowledge if stage == "search" else agent.answering,
                            "search_knowledge" if stage == "search" else "call_answer", failed)
    with pytest.raises(agent.AgentRunError) as caught:
        run_with_responses(tmp_path, responses)
    assert caught.value.stage == stage
    assert SECRET not in str(caught.value) and caught.value.__cause__ is None
    assert "不自动重试" in str(caught.value) and "费用" in str(caught.value)
    if stage != "plan":
        failed.assert_called_once()


def test_invalid_final_reference_fails_closed(business, monkeypatch, tmp_path):
    monkeypatch.setattr(agent.answering, "call_answer", lambda *args: GroundedAnswer(
        status="answered", claims=[{"text": SECRET, "evidence": [{"source_id": "S99", "quote_id": "Q1"}]}], missing_info=[]))
    with pytest.raises(agent.AgentRunError) as caught:
        run_with_responses(tmp_path, [native_call(), stop_response()])
    assert caught.value.stage == "answer" and SECRET not in str(caught.value)


@pytest.mark.parametrize("invalid_reference", [False, True])
def test_final_generation_reuses_real_sse_json_and_quote_validation(monkeypatch, tmp_path, invalid_reference):
    original_answer = agent.answering.call_answer
    monkeypatch.setenv("OPENAI_ORG_ID", "ENV_ORG_MUST_NOT_SEND")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ENV_PROJECT_MUST_NOT_SEND")
    monkeypatch.setattr(agent.knowledge, "search_knowledge", lambda path, query, **kwargs: synthetic_round(query, ["a0", "a1"]))
    sent = []

    def handler(request):
        sent.append(request)
        raw = json.dumps({"status": "partial", "claims": [{
            "text": "Synthetic conditional explanation.",
            "evidence": [{"source_id": "S1", "quote_id": "Q99" if invalid_reference else "Q1"}],
        }], "missing_info": ["Actual cluster evidence is required."]})
        chunk = {"id": "synthetic", "choices": [{"index": 0, "delta": {"content": raw}, "finish_reason": "stop"}]}
        return httpx.Response(200, content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})

    def transport(**options):
        return httpx.Client(transport=httpx.MockTransport(handler), **options)

    def answer(query, sources, model):
        return original_answer(query, sources, model, http_client_factory=transport,
                               client_factory=OpenAI, api_key_loader=lambda selected: SECRET)

    monkeypatch.setattr(agent.answering, "call_answer", answer)
    if invalid_reference:
        with pytest.raises(agent.AgentRunError) as caught:
            run_with_responses(tmp_path, [native_call(), stop_response()])
        assert caught.value.stage == "answer"
    else:
        result, _, _, _ = run_with_responses(tmp_path, [native_call(), stop_response()])
        assert result["sources"]["S1"]["chunk_id"] == "a0"
        assert result["answer"]["status"] == "partial"
    assert len(sent) == 1
    payload = json.loads(sent[0].content)
    assert payload["model"] == MODEL and payload["stream"] is True
    evidence = json.loads(payload["messages"][1]["content"])["evidence"]
    assert len(evidence) == 2 and evidence[0]["id"] == "S1"
    assert SECRET not in sent[0].content.decode()
    assert "ENV_ORG_MUST_NOT_SEND" not in str(sent[0].headers)
    assert "ENV_PROJECT_MUST_NOT_SEND" not in str(sent[0].headers)


@pytest.mark.parametrize("mutation", ["text", "metadata"])
def test_cross_round_same_chunk_identity_must_not_change_before_answer(business, monkeypatch, tmp_path, mutation):
    rounds = [synthetic_round("First", ["shared", "a1"]), synthetic_round("Second", ["shared", "b1"])]
    if mutation == "text":
        rounds[1]["result"]["records"]["shared"]["text"] += SECRET
        rounds[1]["sources"]["S1"].candidate.document.page_content += SECRET
    else:
        rounds[1]["result"]["records"]["shared"]["metadata"]["title"] = SECRET
        rounds[1]["sources"]["S1"].candidate.document.metadata["title"] = SECRET
    monkeypatch.setattr(agent.knowledge, "search_knowledge", lambda *args, **kwargs: rounds.pop(0))
    with pytest.raises(agent.AgentRunError) as caught:
        run_with_responses(tmp_path, [native_call(), native_call("Synthetic second search", "call-second")])
    assert caught.value.stage == "answer" and SECRET not in str(caught.value)
    assert business["answer"] == []


def test_each_run_has_fresh_protocol_state_and_no_persistent_session(business, tmp_path):
    first, _, _, _ = run_with_responses(tmp_path, [native_call(), stop_response()])
    second, _, _, _ = run_with_responses(tmp_path, [native_call(), stop_response()])
    for result in (first, second):
        assert result["agent"]["planner_calls"] == 2
        assert len(result["agent"]["tool_calls"]) == 1
        assert result["agent"]["tool_calls"][0]["id"] == "call-first"
        assert result["request_counts"] == {"embedding": 1, "http": 7, "reranker": 1, "chat": 3}


def test_environment_enabled_tracing_is_disabled_for_search_and_answer(business, monkeypatch, tmp_path):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    run_with_responses(tmp_path, [native_call(), stop_response()])
    assert business["tracing"] == [False, False]


@pytest.mark.parametrize("query,model", [("", MODEL), ("  ", MODEL), ("x" * 2001, MODEL), (None, MODEL), (QUERY, "qwen3.8-max"), (QUERY, "glm")])
def test_invalid_input_is_rejected_before_credentials_or_clients(query, model, tmp_path):
    forbidden = Mock(side_effect=AssertionError("Invalid input reached provider"))
    with pytest.raises(ValueError):
        agent.run(query, model, tmp_path, client_factory=forbidden, http_client_factory=forbidden, api_key_loader=forbidden)
    forbidden.assert_not_called()


@pytest.fixture
def api(tmp_path):
    with TestClient(lab.create_app(tmp_path / "missing-index"), base_url="http://127.0.0.1") as client:
        yield client


def test_plan_is_pure_config_and_exposes_bounded_agent(api, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("GET plan must not start graph or provider"))
    monkeypatch.setattr(agent, "run", forbidden)
    monkeypatch.setattr(agent, "build_graph", forbidden)
    monkeypatch.setattr(agent.knowledge, "search_knowledge", forbidden)
    response = api.get("/api/qa/plan")
    assert response.status_code == 200
    assert response.json()["agent"]["enabled_models"] == [MODEL]
    assert response.json()["agent"]["max_tool_calls"] == 2
    assert response.json()["agent"]["max_request_counts"] == agent.MAX_REQUEST_COUNTS
    forbidden.assert_not_called()


@pytest.mark.parametrize("mode", [None, "single", "agent"])
def test_api_dispatches_explicit_agent_without_changing_single_default(api, monkeypatch, mode):
    ordinary, bounded = Mock(return_value={"route": "single"}), Mock(return_value={"route": "agent"})
    monkeypatch.setattr(single, "run", ordinary)
    monkeypatch.setattr(agent, "run", bounded)
    payload = {"query": QUERY, "model": MODEL, "confirmed": True}
    if mode is not None:
        payload["mode"] = mode
    response = api.post("/api/qa", headers=HEADERS, json=payload)
    assert response.status_code == 200 and response.json() == {"route": "agent" if mode == "agent" else "single"}
    assert bounded.call_count == (mode == "agent")
    assert ordinary.call_count == (mode != "agent")


@pytest.mark.parametrize("updates", [{"model": "glm"}, {"model": "qwen3.8-max"}, {"mode": "autonomous"}, {"mode": 1}, {"confirmed": False}, {"query": "  "}, {"tools": []}, {"api_key": SECRET}])
def test_agent_api_rejects_unsupported_or_unconfirmed_payload_before_execution(api, monkeypatch, updates):
    forbidden = Mock(side_effect=AssertionError("Invalid agent payload reached graph"))
    monkeypatch.setattr(agent, "run", forbidden)
    monkeypatch.setattr(single, "run", forbidden)
    response = api.post("/api/qa", headers=HEADERS,
                        json={"query": QUERY, "model": MODEL, "mode": "agent", "confirmed": True, **updates})
    assert response.status_code == 422 and SECRET not in response.text
    forbidden.assert_not_called()


def test_agent_api_sanitizes_failure_releases_lock_and_never_falls_back(api, monkeypatch):
    ordinary = Mock(side_effect=AssertionError("Agent failures must not fall back to single"))
    bounded = Mock(side_effect=[agent.AgentRunError("search", ["plan"]), {"synthetic": "next request"}])
    monkeypatch.setattr(single, "run", ordinary)
    monkeypatch.setattr(agent, "run", bounded)
    payload = {"query": QUERY, "model": MODEL, "mode": "agent", "confirmed": True}
    failed = api.post("/api/qa", headers=HEADERS, json=payload)
    assert failed.status_code == 503 and "不自动重试" in failed.text
    assert bounded.call_count == 1
    assert api.post("/api/qa", headers=HEADERS, json=payload).status_code == 200
    assert bounded.call_count == 2
    ordinary.assert_not_called()


def test_agent_shares_existing_busy_lock_with_single_and_teaching_routes(api, monkeypatch):
    entered, release = Event(), Event()

    def waiting(*args):
        entered.set()
        assert release.wait(5)
        return {"synthetic": "done"}

    bounded, ordinary = Mock(side_effect=waiting), Mock(side_effect=AssertionError("Busy request must not run"))
    monkeypatch.setattr(agent, "run", bounded)
    monkeypatch.setattr(single, "run", ordinary)
    payload = {"query": QUERY, "model": MODEL, "mode": "agent", "confirmed": True}
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(api.post, "/api/qa", headers=HEADERS, json=payload)
        try:
            assert entered.wait(5)
            assert api.post("/api/qa", headers=HEADERS, json={**payload, "mode": "single"}).status_code == 409
            assert api.post("/api/qa", headers=HEADERS, json=payload).status_code == 409
            assert api.post("/api/workflow/approval/start", headers=HEADERS, json={}).status_code == 409
            assert bounded.call_count == 1
            ordinary.assert_not_called()
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
