"""Short-term conversation tests: real store/SDK protocol, synthetic RAG results."""

from copy import deepcopy
import json
import re
import socket
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
from langsmith.run_helpers import get_tracing_context
from openai import OpenAI
import pytest

from qa_platform import retrieval_lab as lab
from qa_platform.workflows import agent_rag, conversation as convo, live_rag


MODEL = "qwen3.8-flash"
QUERY = "Synthetic DNS question?"
FOLLOWUP = "And if only that namespace fails?"
EFFECTIVE = "Synthetic DNS failure across namespaces?"
PRIVATE = "PRIVATE_SOURCE_PATH_VECTOR_TOKEN_MUST_NOT_LEAK"
HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Conversation tests must not read keys or contact external services")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    monkeypatch.setattr(convo, "load_chat_key", forbidden)


def result(query=QUERY):
    text = "Synthetic conditional source paragraph."
    return {
        "schema_version": "rag-live/v1", "query": query, "model": MODEL,
        "answer": {"status": "partial", "claims": [{
            "text": "Synthetic conditional explanation.",
            "evidence": [{"source_id": "S1", "quote_id": "Q1"}],
        }], "missing_info": ["Actual cluster information is required."]},
        "sources": {"S1": {"chunk_id": "synthetic-source", "text": text,
                             "quotes": {"Q1": text}, "metadata": {"source_path": PRIVATE, "vector": [PRIVATE]}}},
        "request_counts": {"embedding": 1, "http": 5, "reranker": 1, "chat": 1},
        "steps": ["retrieve", "rerank", "answer"], "elapsed_seconds": 0.1,
    }


def stored_history():
    store = convo.ConversationStore()
    identity = store.commit(None, QUERY, QUERY, result())["id"]
    return store.get(identity)


def test_store_only_keeps_validated_answer_text_and_returns_deep_copies():
    store = convo.ConversationStore()
    source = result()
    meta = store.commit(None, QUERY, QUERY, source)
    assert re.fullmatch(r"[0-9a-f]{32}", meta["id"])
    assert meta == {"id": meta["id"], "turn": 1, "history_turns": 0, "retained_turns": 1,
                    "effective_query": QUERY, "rewritten": False}
    history = store.get(meta["id"])
    assert set(history[0]) == {"query", "effective_query", "answer"}
    assert history[0]["answer"]["claims"] == ["Synthetic conditional explanation."]
    assert PRIVATE not in json.dumps(history) and "sources" not in history[0]
    source["answer"]["claims"][0]["text"] = "changed input"
    history[0]["answer"]["claims"][0] = "changed output"
    assert store.get(meta["id"])[0]["answer"]["claims"] == ["Synthetic conditional explanation."]


def test_store_bounds_history_to_three_but_retains_absolute_turn_count():
    store = convo.ConversationStore()
    identity = None
    for index in range(5):
        query = f"Synthetic question {index}"
        meta = store.commit(identity, query, query, result(query))
        identity = meta["id"]
    assert meta["turn"] == 5 and meta["history_turns"] == meta["retained_turns"] == 3
    assert [turn["query"] for turn in store.get(identity)] == [f"Synthetic question {index}" for index in (2, 3, 4)]


def test_escaped_answer_is_summarized_within_limit_without_mutating_original():
    store = convo.ConversationStore()
    source = result()
    claim = source["answer"]["claims"][0]
    source["answer"]["claims"] = [{**deepcopy(claim), "text": "\\" * 300} for _ in range(3)]
    source["answer"]["missing_info"] = ["\\" * 240 for _ in range(3)]
    original = deepcopy(source)
    identity = store.commit(None, QUERY, QUERY, source)["id"]
    summary = store.get(identity)[0]["answer"]
    encoded = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    assert len(encoded) <= convo.MAX_SUMMARY_CHARS
    assert "…" in encoded and source == original


def test_store_evicts_oldest_created_not_most_recently_accessed_session():
    store = convo.ConversationStore()
    ids = [store.commit(None, QUERY, QUERY, result())["id"] for _ in range(convo.MAX_SESSIONS)]
    store.get(ids[0])
    store.commit(ids[0], QUERY, QUERY, result())
    latest = store.commit(None, QUERY, QUERY, result())["id"]
    with pytest.raises(KeyError):
        store.get(ids[0])
    assert store.get(ids[1]) and store.get(latest)


@pytest.mark.parametrize("mutation", ["query", "missing_quote", "bad_quote", "invalid_status"])
def test_invalid_or_uncited_result_is_not_committed(mutation):
    store = convo.ConversationStore()
    identity = store.commit(None, QUERY, QUERY, result())["id"]
    before = store.get(identity)
    invalid = result()
    if mutation == "query":
        invalid["query"] = PRIVATE
    elif mutation == "missing_quote":
        invalid["sources"]["S1"]["quotes"] = {}
    elif mutation == "bad_quote":
        invalid["sources"]["S1"]["quotes"]["Q1"] = PRIVATE
    else:
        invalid["answer"]["claims"] = []
    with pytest.raises((ValueError, KeyError)):
        store.commit(identity, QUERY, QUERY, invalid)
    assert store.get(identity) == before


def test_reset_removes_only_selected_session_and_restart_discards_memory():
    store = convo.ConversationStore()
    first = store.commit(None, QUERY, QUERY, result())["id"]
    other = store.commit(None, QUERY, QUERY, result())["id"]
    store.clear(first)
    with pytest.raises(KeyError):
        store.get(first)
    assert store.get(other)
    with pytest.raises(KeyError):
        convo.ConversationStore().get(other)


def completion(content=None):
    return {"id": "synthetic", "object": "chat.completion", "created": 1, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content or json.dumps({"query": EFFECTIVE}),
            }}]}


def contextualize(response, *, query=FOLLOWUP, model=MODEL):
    sent, http_options, sdk_options, trace = [], [], [], []

    def handler(request):
        sent.append(request)
        trace.append(get_tracing_context()["enabled"])
        return response if isinstance(response, httpx.Response) else httpx.Response(200, json=response)

    def transport(**kwargs):
        http_options.append(kwargs)
        return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)

    def client(**kwargs):
        sdk_options.append(kwargs)
        return OpenAI(**kwargs)

    rewritten = convo.contextualize(query, stored_history(), model, client_factory=client,
                                    http_client_factory=transport, api_key_loader=lambda selected: PRIVATE)
    return rewritten, sent, http_options, sdk_options, trace


def test_contextualize_makes_one_strict_request_without_private_evidence(monkeypatch):
    monkeypatch.setenv("OPENAI_ORG_ID", "ENV_ORG_MUST_NOT_SEND")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ENV_PROJECT_MUST_NOT_SEND")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    rewritten, sent, http_options, sdk_options, trace = contextualize(completion())
    assert rewritten == EFFECTIVE and len(sent) == 1 and trace == [False]
    payload = json.loads(sent[0].content)
    assert payload["model"] == MODEL and payload["stream"] is False and payload["enable_thinking"] is False
    assert "tools" not in payload
    user = json.loads(payload["messages"][1]["content"])
    assert user["query"] == FOLLOWUP and len(user["history"]) == 1
    assert PRIVATE not in sent[0].content.decode()
    assert "ENV_ORG_MUST_NOT_SEND" not in str(sent[0].headers)
    assert "ENV_PROJECT_MUST_NOT_SEND" not in str(sent[0].headers)
    assert http_options == [{"trust_env": False, "follow_redirects": False, "timeout": 60.0}]
    assert sdk_options[0]["organization"] == sdk_options[0]["project"] == ""
    assert sdk_options[0]["max_retries"] == 0


def test_new_topic_can_keep_original_query_without_forced_history_content():
    query = "Synthetic independent storage question?"
    rewritten, sent, *_ = contextualize(completion(json.dumps({"query": query})), query=query)
    assert rewritten == query and len(sent) == 1


def test_glm_context_never_receives_cloud_credentials_or_qwen_thinking_options():
    _, sent, _, sdk_options, _ = contextualize(completion(), model="glm")
    assert len(sent) == 1 and sdk_options[0]["api_key"] == "not-needed"
    assert "enable_thinking" not in json.loads(sent[0].content)
    assert sent[0].headers["authorization"] == "Bearer not-needed"


@pytest.mark.parametrize("failure", ["invalid_json", "extra_field", "wrong_type", "empty_query", "truncated", "tool_call", "multiple_choices", "http"])
def test_context_failures_are_sanitized_without_retry_or_fallback(failure):
    response = completion()
    choice = response["choices"][0]
    if failure == "invalid_json":
        choice["message"]["content"] = PRIVATE
    elif failure == "extra_field":
        choice["message"]["content"] = json.dumps({"query": EFFECTIVE, "answer": PRIVATE})
    elif failure == "wrong_type":
        choice["message"]["content"] = '{"query": 3}'
    elif failure == "empty_query":
        choice["message"]["content"] = '{"query": "  "}'
    elif failure == "truncated":
        choice["finish_reason"] = "length"
    elif failure == "tool_call":
        choice["message"]["tool_calls"] = [{"id": "bad-call", "type": "function", "function": {"name": "search_knowledge", "arguments": "{}"}}]
    elif failure == "multiple_choices":
        response["choices"].append(deepcopy(choice))
    else:
        response = httpx.Response(500, json={"error": {"message": PRIVATE}})
    with pytest.raises(convo.ContextError) as caught:
        contextualize(response)
    assert PRIVATE not in str(caught.value) and caught.value.__cause__ is None
    assert "不自动重试" in str(caught.value)


@pytest.fixture
def api(tmp_path):
    with TestClient(lab.create_app(tmp_path / "missing-index"), base_url="http://127.0.0.1") as client:
        yield client


@pytest.fixture
def pipeline(monkeypatch):
    calls = {"run": [], "context": []}

    def run(query, model, index_dir):
        calls["run"].append((query, model, index_dir))
        return result(query)

    def context(query, history, model):
        calls["context"].append((query, deepcopy(history), model))
        return EFFECTIVE

    monkeypatch.setattr(live_rag, "run", run)
    monkeypatch.setattr(convo, "contextualize", context)
    return calls


def ask(api, query=QUERY, **extra):
    return api.post("/api/qa", headers=HEADERS,
                    json={"query": query, "model": MODEL, "confirmed": True, **extra})


def test_default_old_request_does_not_start_or_rewrite_conversation(api, pipeline):
    response = ask(api)
    assert response.status_code == 200 and response.json() == result()
    assert pipeline["context"] == []


def test_first_turn_and_followup_run_retrieval_with_effective_query_but_display_original(api, pipeline):
    first = ask(api, conversation=True)
    assert first.status_code == 200
    meta = first.json()["conversation"]
    assert meta["turn"] == 1 and meta["history_turns"] == 0 and meta["rewritten"] is False
    assert pipeline["context"] == [] and first.json()["request_counts"] == result()["request_counts"]
    second = ask(api, FOLLOWUP, conversation=True, conversation_id=meta["id"])
    assert second.status_code == 200
    data = second.json()
    assert data["query"] == FOLLOWUP and pipeline["run"][-1][0] == EFFECTIVE
    assert data["conversation"] == {"id": meta["id"], "turn": 2, "history_turns": 1,
                                    "retained_turns": 2, "effective_query": EFFECTIVE, "rewritten": True}
    assert data["steps"] == ["contextualize", "retrieve", "rerank", "answer"]
    assert data["request_counts"] == {"embedding": 1, "http": 6, "reranker": 1, "chat": 2}
    history = pipeline["context"][0][1]
    assert len(history) == 1 and history[0]["query"] == QUERY and PRIVATE not in json.dumps(history)


def test_agent_followup_adds_one_rewrite_request_without_changing_tool_budget(api, pipeline, monkeypatch):
    def run(query, model, index_dir):
        data = result(query)
        data.update(mode="agent", steps=["plan", "search", "plan", "search", "answer"],
                    request_counts={"embedding": 2, "http": 11, "reranker": 2, "chat": 3},
                    agent={"max_tool_calls": 2, "planner_calls": 2, "tool_calls": [], "stop_reason": "tool_budget_reached"})
        return data

    monkeypatch.setattr(agent_rag, "run", run)
    first = ask(api, conversation=True, mode="agent").json()
    second = ask(api, FOLLOWUP, conversation=True, mode="agent", conversation_id=first["conversation"]["id"])
    assert second.status_code == 200
    data = second.json()
    assert data["request_counts"] == {"embedding": 2, "http": 12, "reranker": 2, "chat": 4}
    assert data["steps"] == ["contextualize", "plan", "search", "plan", "search", "answer"]
    assert data["agent"]["max_tool_calls"] == 2 and data["conversation"]["turn"] == 2


def test_separate_conversations_and_apps_have_isolated_histories(api, pipeline, tmp_path):
    first = ask(api, "First synthetic topic", conversation=True).json()["conversation"]["id"]
    other = ask(api, "Other synthetic topic", conversation=True).json()["conversation"]["id"]
    assert first != other
    assert ask(api, FOLLOWUP, conversation=True, conversation_id=first).status_code == 200
    assert pipeline["context"][-1][1][0]["query"] == "First synthetic topic"
    with TestClient(lab.create_app(tmp_path), base_url="http://127.0.0.1") as restarted:
        assert ask(restarted, FOLLOWUP, conversation=True, conversation_id=other).status_code == 410


@pytest.mark.parametrize("stage", ["context", "run", "mismatched_result"])
def test_failed_followup_does_not_commit_history_or_fall_back(api, pipeline, monkeypatch, stage):
    identity = ask(api, conversation=True).json()["conversation"]["id"]
    context, run = convo.contextualize, live_rag.run
    failed = Mock(side_effect=convo.ContextError() if stage == "context" else RuntimeError(PRIVATE))
    if stage == "context":
        monkeypatch.setattr(convo, "contextualize", failed)
    elif stage == "run":
        monkeypatch.setattr(live_rag, "run", failed)
    else:
        monkeypatch.setattr(live_rag, "run", lambda *args: result("Other mismatched question"))
    response = ask(api, FOLLOWUP, conversation=True, conversation_id=identity)
    assert response.status_code == 503 and PRIVATE not in response.text
    if stage == "context":
        assert len(pipeline["run"]) == 1
    monkeypatch.setattr(convo, "contextualize", context)
    monkeypatch.setattr(live_rag, "run", run)
    retried = ask(api, FOLLOWUP, conversation=True, conversation_id=identity)
    assert retried.status_code == 200 and retried.json()["conversation"]["turn"] == 2
    assert len(pipeline["context"][-1][1]) == 1


@pytest.mark.parametrize("extra", [{"conversation_id": "a" * 32}, {"conversation": 1}, {"conversation": True, "conversation_id": "bad-id"}])
def test_invalid_conversation_request_stops_before_model(api, pipeline, extra):
    assert ask(api, **extra).status_code == 422
    assert pipeline["run"] == pipeline["context"] == []


def test_unknown_session_returns_gone_without_rewriting_or_retrieving(api, pipeline):
    assert ask(api, FOLLOWUP, conversation=True, conversation_id="a" * 32).status_code == 410
    assert pipeline["run"] == pipeline["context"] == []


def test_reset_is_explicit_same_origin_idempotent_and_zero_model(api, pipeline):
    identity = ask(api, conversation=True).json()["conversation"]["id"]
    payload = {"conversation_id": identity}
    assert api.post("/api/qa/conversation/reset", headers={}, json=payload).status_code == 403
    assert api.post("/api/qa/conversation/reset", headers=HEADERS, json=payload).status_code == 200
    assert api.post("/api/qa/conversation/reset", headers=HEADERS, json=payload).status_code == 200
    assert ask(api, FOLLOWUP, conversation=True, conversation_id=identity).status_code == 410
    assert len(pipeline["run"]) == 1 and pipeline["context"] == []


def test_plan_is_pure_config_and_exposes_short_term_limits(api, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Opening plan must not load history or models"))
    monkeypatch.setattr(convo, "contextualize", forbidden)
    monkeypatch.setattr(live_rag, "run", forbidden)
    monkeypatch.setattr(agent_rag, "run", forbidden)
    response = api.get("/api/qa/plan")
    assert response.status_code == 200
    assert response.json()["conversation"]["max_history_turns"] == 3
    assert response.json()["conversation"]["max_sessions"] == 16
    assert response.json()["conversation"]["extra_chat_requests"] == 1
    forbidden.assert_not_called()
