"""Offline SSE integration: real ASGI delivery, bounded worker and synthetic RAG."""

import asyncio
from copy import deepcopy
import json
import socket
from threading import Event
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
import pytest

from qa_platform import retrieval_lab as lab, web_stream
from qa_platform.workflows import agent_rag, conversation, live_rag, progress


MODEL = "qwen3.8-flash"
QUERY = "Synthetic DNS question?"
PRIVATE = "PRIVATE_UNVALIDATED_BODY_PATH_KEY"
HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}


@pytest.fixture(autouse=True)
def no_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SSE tests must not read credentials or contact model services")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)


def public_result(query=QUERY):
    text = "Synthetic conditional source paragraph."
    return {
        "schema_version": "rag-live/v1", "query": query, "model": MODEL,
        "answer": {"status": "partial", "claims": [{
            "text": "Synthetic conditional explanation.",
            "evidence": [{"source_id": "S1", "quote_id": "Q1"}],
        }], "missing_info": ["Actual cluster information is required."]},
        "sources": {"S1": {"chunk_id": "synthetic-source", "text": text,
                            "quotes": {"Q1": text}, "metadata": {"title": "Synthetic source"}}},
        "request_counts": {"embedding": 1, "http": 5, "reranker": 1, "chat": 1},
        "steps": ["retrieve", "rerank", "answer"], "elapsed_seconds": 0.1,
    }


def payload(**updates):
    return {"query": QUERY, "model": MODEL, "confirmed": True, **updates}


def events(text):
    parsed = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        event, parts = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                parts.append(line.partition(":")[2].lstrip())
        if event is not None:
            parsed.append((event, json.loads("\n".join(parts))))
    return parsed


@pytest.fixture
def app(tmp_path):
    return lab.create_app(tmp_path / "missing-index")


@pytest.fixture
def api(app):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


@pytest.fixture
def fake_run(monkeypatch):
    calls = []

    def run(query, model, index_dir):
        calls.append((query, model, index_dir))
        for stage in ("retrieve", "rerank", "answer"):
            progress.report(stage, "started")
            progress.report(stage, "completed")
        return public_result(query)

    monkeypatch.setattr(live_rag, "run", run)
    return calls


def test_stream_sends_only_safe_progress_then_one_complete_result(api, fake_run):
    response = api.post("/api/qa/stream", headers=HEADERS, json=payload())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "no-store" in response.headers["cache-control"]
    messages = events(response.text)
    assert messages[-1] == ("result", public_result())
    assert [kind for kind, _ in messages].count("result") == 1
    assert not any(kind == "error" for kind, _ in messages)
    assert [data for kind, data in messages if kind == "progress"] == [
        {"schema_version": "qa-progress/v1", "stage": stage, "status": status}
        for stage in ("retrieve", "rerank", "answer") for status in ("started", "completed")
    ]
    assert all(set(data) == {"schema_version", "stage", "status"} for kind, data in messages if kind == "progress")
    assert len(fake_run) == 1


def test_real_three_node_graph_reports_actual_stage_order_without_model_io(api, monkeypatch):
    output = public_result()
    monkeypatch.setattr(live_rag.hybrid, "prepare", lambda path, query: {"query": query})
    monkeypatch.setattr(live_rag.hybrid, "compare", lambda check, mode: {"query": check["query"]})
    monkeypatch.setattr(live_rag.reranking, "rerank", lambda result: result)
    monkeypatch.setattr(live_rag.answering, "generate", lambda result, model: {
        "query": result["query"], "request_counts": output["request_counts"],
        "answer_generation": {"model": model, "answer": output["answer"], "sources": output["sources"]},
    })
    response = api.post("/api/qa/stream", headers=HEADERS, json=payload())
    messages = events(response.text)
    assert [data["stage"] + ":" + data["status"] for kind, data in messages if kind == "progress"] == [
        "retrieve:started", "retrieve:completed", "rerank:started", "rerank:completed", "answer:started", "answer:completed",
    ]
    assert messages[-1][0] == "result" and messages[-1][1]["steps"] == ["retrieve", "rerank", "answer"]


def test_stream_followup_retains_original_question_and_adds_one_request(api, fake_run, monkeypatch):
    first = api.post("/api/qa/stream", headers=HEADERS, json=payload(conversation=True))
    identity = events(first.text)[-1][1]["conversation"]["id"]

    def contextualize(query, history, model):
        assert len(history) == 1 and history[0]["query"] == QUERY
        return "Synthetic complete namespace question?"

    monkeypatch.setattr(conversation, "contextualize", contextualize)
    second = api.post("/api/qa/stream", headers=HEADERS,
                      json=payload(query="What about that namespace?", conversation=True, conversation_id=identity))
    messages = events(second.text)
    data = messages[-1][1]
    assert messages[-1][0] == "result" and data["query"] == "What about that namespace?"
    assert data["conversation"]["turn"] == 2 and data["conversation"]["rewritten"] is True
    assert data["steps"] == ["contextualize", "retrieve", "rerank", "answer"]
    assert data["request_counts"] == {"embedding": 1, "http": 6, "reranker": 1, "chat": 2}
    assert fake_run[-1][0] == "Synthetic complete namespace question?"


def test_stream_agent_returns_original_tool_counts_without_extra_model_requests(api, monkeypatch):
    calls = []

    def run(query, model, index_dir):
        calls.append(query)
        progress.report("plan", "started")
        progress.report("plan", "completed")
        data = public_result(query)
        data.update(mode="agent", steps=["plan", "search", "plan", "search", "answer"],
                    request_counts={"embedding": 2, "http": 11, "reranker": 2, "chat": 3},
                    agent={"max_tool_calls": 2, "planner_calls": 2, "tool_calls": [], "stop_reason": "tool_budget_reached"})
        return data

    monkeypatch.setattr(agent_rag, "run", run)
    response = api.post("/api/qa/stream", headers=HEADERS, json=payload(mode="agent"))
    data = events(response.text)[-1][1]
    assert data["mode"] == "agent" and data["request_counts"] == {"embedding": 2, "http": 11, "reranker": 2, "chat": 3}
    assert calls == [QUERY]


def test_worker_failure_emits_error_without_result_and_releases_lock(api, monkeypatch):
    run = Mock(side_effect=[RuntimeError(PRIVATE), public_result()])
    monkeypatch.setattr(live_rag, "run", run)
    first = api.post("/api/qa/stream", headers=HEADERS, json=payload())
    messages = events(first.text)
    assert messages[-1][0] == "error" and not any(kind == "result" for kind, _ in messages)
    assert set(messages[-1][1]) == {"detail"}
    assert PRIVATE not in first.text and "不自动重试" in first.text
    assert run.call_count == 1
    assert api.post("/api/qa", headers=HEADERS, json=payload()).status_code == 200
    assert run.call_count == 2


@pytest.mark.parametrize("headers,updates,status", [
    ({}, {}, 403),
    ({**HEADERS, "Origin": "https://evil.invalid"}, {}, 403),
    ({**HEADERS, "Content-Type": "text/plain"}, {}, 415),
    (HEADERS, {"confirmed": False}, 422),
    (HEADERS, {"query": "  "}, 422),
    (HEADERS, {"mode": "agent", "model": "glm"}, 422),
    (HEADERS, {"conversation_id": "a" * 32}, 422),
    (HEADERS, {"conversation": True, "conversation_id": "a" * 32}, 410),
    (HEADERS, {"api_key": PRIVATE}, 422),
])
def test_stream_guards_reject_before_starting_or_sending_sse(api, monkeypatch, headers, updates, status):
    forbidden = Mock(side_effect=AssertionError("Invalid request started a worker"))
    monkeypatch.setattr(live_rag, "run", forbidden)
    monkeypatch.setattr(agent_rag, "run", forbidden)
    monkeypatch.setattr(conversation, "contextualize", forbidden)
    response = api.post("/api/qa/stream", headers=headers, json=payload(**updates))
    assert response.status_code == status and "text/event-stream" not in response.headers.get("content-type", "")
    assert PRIVATE not in response.text
    forbidden.assert_not_called()


def test_stream_rejects_large_or_malformed_body_before_response_headers(api, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Invalid body started a worker"))
    monkeypatch.setattr(live_rag, "run", forbidden)
    headers = {**HEADERS, "Content-Type": "application/json"}
    assert api.post("/api/qa/stream", headers=headers, content=b"x" * 16385).status_code == 413
    assert api.post("/api/qa/stream", headers=headers, content=b"{").status_code == 422
    forbidden.assert_not_called()


def test_plan_describes_stream_without_starting_worker(api, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Plan must not execute a graph"))
    monkeypatch.setattr(live_rag, "run", forbidden)
    response = api.get("/api/qa/plan")
    assert response.status_code == 200
    assert response.json()["stream"] == {"url": "/api/qa/stream", "protocol": "qa-progress/v1", "heartbeat_seconds": 10}
    forbidden.assert_not_called()


def test_progress_scope_rejects_arbitrary_body_and_cooperatively_cancels():
    delivered, cancelled = [], Event()
    progress.report("retrieve", "started")  # Existing JSON route has no listener.
    with progress.progress_scope(delivered.append, cancelled):
        progress.report("retrieve", "started")
        with pytest.raises(ValueError):
            progress.report(PRIVATE, "completed")
        cancelled.set()
        with pytest.raises(progress.RunCancelled):
            progress.report("retrieve", "completed")
    assert delivered == [{"schema_version": "qa-progress/v1", "stage": "retrieve", "status": "started"}]
    progress.report("answer", "started")  # Context is restored despite cancellation.


def test_sink_cancellation_prevents_starting_stage_work():
    cancelled, later_actions = Event(), []
    with progress.progress_scope(lambda event: cancelled.set(), cancelled):
        with pytest.raises(progress.RunCancelled):
            progress.report("retrieve", "started")
            later_actions.append("would start a model call")
    assert later_actions == []


@pytest.mark.parametrize("failure", ["constructor", "thread_start"])
def test_stream_startup_failure_releases_lock_once_without_work(monkeypatch, failure):
    work, release = Mock(), Mock()
    if failure == "constructor":
        monkeypatch.setattr(web_stream.StreamingResponse, "__init__", Mock(side_effect=RuntimeError(PRIVATE)))
    else:
        monkeypatch.setattr(web_stream.Thread, "start", Mock(side_effect=RuntimeError(PRIVATE)))

    async def exercise():
        with pytest.raises(RuntimeError):
            web_stream.stream_response(work, release, lambda exc: "sanitized")

    asyncio.run(exercise())
    release.assert_called_once_with()
    work.assert_not_called()


def test_bridge_construction_failure_is_sanitized_before_sse_and_lock_is_released(api, fake_run):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(web_stream.StreamingResponse, "__init__", Mock(side_effect=RuntimeError(PRIVATE)))
        response = api.post("/api/qa/stream", headers=HEADERS, json=payload())
    assert response.status_code == 503 and PRIVATE not in response.text
    assert "text/event-stream" not in response.headers.get("content-type", "")
    assert fake_run == []
    assert api.post("/api/qa", headers=HEADERS, json=payload()).status_code == 200
    assert len(fake_run) == 1


async def asgi_request(app, path, data):
    """Start an actual ASGI request without TestClient's full-body buffering."""
    received = asyncio.Queue()
    await received.put({"type": "http.request", "body": json.dumps(data).encode(), "more_body": False})
    messages, changed = [], asyncio.Event()

    async def send(message):
        messages.append(deepcopy(message))
        changed.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "http_version": "1.1", "scheme": "http", "method": "POST", "path": path,
             "raw_path": path.encode(), "root_path": "", "query_string": b"",
             "server": ("127.0.0.1", 80), "client": ("testclient", 123), "state": {},
             "headers": [(b"host", b"127.0.0.1"), (b"origin", b"http://127.0.0.1"),
                         (b"x-retrieval-lab", b"1"), (b"content-type", b"application/json")]}
    task = asyncio.create_task(app(scope, received.get, send))
    return task, received, messages, changed


async def wait_for_body(messages, changed, needle):
    async def waiting():
        while needle not in b"".join(message.get("body", b"") for message in messages):
            changed.clear()
            await changed.wait()
    await asyncio.wait_for(waiting(), 3)


def test_asgi_delivers_progress_before_worker_completes_and_keeps_busy_lock(app, monkeypatch):
    release = Event()

    def blocked(*args):
        progress.report("retrieve", "started")
        assert release.wait(5)
        progress.report("retrieve", "completed")
        return public_result()

    monkeypatch.setattr(live_rag, "run", blocked)

    async def exercise():
        task, _, messages, changed = await asgi_request(app, "/api/qa/stream", payload())
        try:
            await wait_for_body(messages, changed, b'"started"')
            assert not task.done()
            body = b"".join(message.get("body", b"") for message in messages)
            assert b"event: result" not in body
            reset, _, replies, _ = await asgi_request(app, "/api/qa/conversation/reset", {"conversation_id": "a" * 32})
            await asyncio.wait_for(reset, 3)
            assert next(message["status"] for message in replies if message["type"] == "http.response.start") == 409
        finally:
            release.set()
            await asyncio.wait_for(task, 5)
        assert events(b"".join(message.get("body", b"") for message in messages).decode())[-1][0] == "result"

    asyncio.run(exercise())


def test_disconnect_stops_at_next_boundary_and_does_not_unlock_inflight_worker(app, monkeypatch):
    release, exited = Event(), Event()
    later_actions = []
    commit = Mock(side_effect=AssertionError("Disconnected execution must not commit a turn"))
    monkeypatch.setattr(conversation.ConversationStore, "commit", commit)

    def blocked(*args):
        try:
            progress.report("retrieve", "started")
            assert release.wait(5)
            progress.report("retrieve", "completed")
            later_actions.append("would start next model")
            return public_result()
        finally:
            exited.set()

    monkeypatch.setattr(live_rag, "run", blocked)

    async def exercise():
        task, received, messages, changed = await asgi_request(app, "/api/qa/stream", payload(conversation=True))
        try:
            await wait_for_body(messages, changed, b'"started"')
            await received.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, 3)
            assert not exited.is_set()
            reset, _, replies, _ = await asgi_request(app, "/api/qa/conversation/reset", {"conversation_id": "a" * 32})
            await asyncio.wait_for(reset, 3)
            assert next(message["status"] for message in replies if message["type"] == "http.response.start") == 409
        finally:
            release.set()
        assert await asyncio.to_thread(exited.wait, 3)
        assert later_actions == []
        commit.assert_not_called()
        assert not any(kind == "result" for kind, _ in events(b"".join(message.get("body", b"") for message in messages).decode()))
        # The in-flight synchronous request cannot be killed, but the worker's
        # finally must eventually release the same lock after boundary cancel.
        async def wait_for_release():
            while True:
                reset, _, replies, _ = await asgi_request(app, "/api/qa/conversation/reset", {"conversation_id": "a" * 32})
                await asyncio.wait_for(reset, 1)
                status = next(message["status"] for message in replies if message["type"] == "http.response.start")
                if status == 200:
                    return
                assert status == 409
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait_for_release(), 3)

    asyncio.run(exercise())


def test_idle_heartbeat_is_comment_only_and_does_not_execute_another_request(app, monkeypatch):
    release, calls = Event(), []
    monkeypatch.setattr(web_stream, "HEARTBEAT_SECONDS", 0.01)

    def blocked(*args):
        calls.append("run")
        assert release.wait(5)
        return public_result()

    monkeypatch.setattr(live_rag, "run", blocked)

    async def exercise():
        task, _, messages, changed = await asgi_request(app, "/api/qa/stream", payload())
        try:
            await wait_for_body(messages, changed, b": keep-alive\n\n")
            assert not task.done() and calls == ["run"]
            assert events(b"".join(message.get("body", b"") for message in messages).decode()) == []
        finally:
            release.set()
            await asyncio.wait_for(task, 5)
        assert calls == ["run"]
        assert events(b"".join(message.get("body", b"") for message in messages).decode())[-1][0] == "result"

    asyncio.run(exercise())
