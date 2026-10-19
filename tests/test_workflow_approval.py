"""Real interrupt/checkpoint/resume tests for the zero-model approval lesson.

These checks distinguish a paused graph from a completed graph whose answer was
merely hidden in the browser. They deliberately forbid business-model IO.
"""

import builtins
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from importlib.metadata import PackageNotFoundError
import inspect
import io
import json
from pathlib import Path
import re
import socket
import sys
from threading import Event
from textwrap import dedent
from types import ModuleType
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langsmith import Client
import openai
import pytest
import requests

from qa_platform import retrieval_lab as lab
from qa_platform.workflows import approval, replay


HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}
BASE = "/api/workflow/approval"
PREFIX = ["retrieve", "fuse", "rerank", "prepare_evidence"]
NODE_IDS = [*PREFIX, "review", "generate", "cancel", "validate"]
ZERO = {"embedding": 0, "http": 0, "reranker": 0, "chat": 0}
WRITES = {
    "retrieve": {"bm25", "dense"}, "fuse": {"rrf_pool"},
    "rerank": {"reranked"}, "prepare_evidence": {"evidence"},
    "review": {"approved"}, "generate": {"draft"},
    "cancel": {"cancelled"}, "validate": {"answer"},
}


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Approval lesson attempted network, .env or real RAG access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(lab.answering, "OpenAI", forbidden)
    for module_name in ("pymilvus", "milvus_lite"):
        blocked = ModuleType(module_name)
        blocked.MilvusClient = forbidden
        monkeypatch.setitem(sys.modules, module_name, blocked)
    for module, names in (
        (lab.hybrid, ("prepare", "compare")),
        (lab.reranking, ("rerank",)),
        (lab.answering, ("generate", "load_chat_key")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    original_open, original_io_open = builtins.open, io.open

    def guarded_open(original):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                name = Path(file).name
                if name == ".env" or name.startswith(".env."):
                    forbidden()
            return original(file, *args, **kwargs)
        return guarded

    monkeypatch.setattr(builtins, "open", guarded_open(original_open))
    monkeypatch.setattr(io, "open", guarded_open(original_io_open))


@pytest.fixture
def client(tmp_path):
    with TestClient(lab.create_app(tmp_path / "missing-index"), base_url="http://127.0.0.1") as active:
        yield active


def assert_events(snapshot, expected):
    assert [event["node"] for event in snapshot["events"]] == expected
    assert [event["step"] for event in snapshot["events"]] == list(range(1, len(expected) + 1))
    state = {"query": replay.load_fixture()["query"]}
    for event in snapshot["events"]:
        assert event["before"] == state
        assert set(event["update"]) == WRITES[event["node"]]
        state.update(deepcopy(event["update"]))
        assert event["after"] == state
    assert snapshot["values"] == state
    assert snapshot["request_counts"] == ZERO


def assert_paused(snapshot):
    assert snapshot["phase"] == "paused"
    assert re.fullmatch(r"[0-9a-f]{32}", snapshot["session_id"])
    assert snapshot["checkpoint"]["thread_id"] == snapshot["session_id"]
    assert snapshot["checkpoint"]["checkpoint_id"]
    assert snapshot["checkpoint"]["next"] == ["review"]
    assert snapshot["interrupt"]["id"]
    assert snapshot["interrupt"]["value"]["kind"] == "review_evidence"
    assert snapshot["interrupt"]["value"]["evidence_count"] == 2
    assert snapshot["interrupt"]["value"]["source_ids"] == ["S1", "S2"]
    assert snapshot["interrupt"]["value"]["question"]
    assert len(snapshot["values"]["evidence"]) == 2
    assert not {"approved", "cancelled", "draft", "answer"} & snapshot["values"].keys()
    assert_events(snapshot, PREFIX)
    assert snapshot["run_counts"] == {
        **{name: {"entered": 1, "completed": 1} for name in PREFIX},
        "review": {"entered": 1, "completed": 0},
        **{name: {"entered": 0, "completed": 0} for name in ("generate", "cancel", "validate")},
    }


def test_idle_description_is_actual_source_only_and_does_not_construct_graph(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Idle inspection must not build or execute a graph"))
    for name in ("make_nodes", "build_graph", "_drive"):
        wrapped = Mock(side_effect=forbidden)
        wrapped.__wrapped__ = getattr(approval, name)
        monkeypatch.setattr(approval, name, wrapped)
    idle = approval.ApprovalLesson().current()
    assert idle["schema_version"] == "rag-workflow-approval/v1"
    assert idle["available"] is True and idle["langgraph_version"]
    assert idle["fixture"]["origin"] == "synthetic"
    assert idle["phase"] == "idle"
    for key in ("session_id", "graph", "checkpoint", "interrupt"):
        assert idle[key] is None
    assert idle["values"] == {} and idle["events"] == []
    assert [row["id"] for row in idle["nodes"]] == NODE_IDS
    assert all(value == {"entered": 0, "completed": 0} for value in idle["run_counts"].values())
    assert idle["request_counts"] == ZERO
    for phrase in ("InMemorySaver", "重启", "从确认节点开头", "不读.env", "不连接Milvus", "不自动重试"):
        assert phrase in idle["notice"]
    forbidden.assert_not_called()


def test_dependency_unavailable_is_reported_without_installing_or_faking_run(monkeypatch):
    def unavailable(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr(approval, "version", unavailable)
    plan = approval.describe()
    assert plan["available"] is False
    assert plan["langgraph_version"] is None
    assert plan["phase"] == "idle"


def test_state_extends_previous_type_and_factory_preserves_real_shared_functions():
    assert approval.ApprovalState.__annotations__ == {
        **replay.ReplayState.__annotations__, "approved": bool, "cancelled": bool,
    }
    assert not approval.ApprovalState.__required_keys__
    nodes = approval.make_nodes(replay.load_fixture())
    assert list(nodes) == NODE_IDS
    assert nodes["review"] is approval.review_evidence
    assert nodes["cancel"] is approval.cancel
    for name in [*PREFIX, "generate", "validate"]:
        assert nodes[name].__module__ == replay.__name__
        assert nodes[name].__name__ == name


def test_real_compiled_graph_uses_provided_memory_saver_and_conditional_edges():
    saver = InMemorySaver()
    graph = approval.build_graph(approval.make_nodes(replay.load_fixture()), saver)
    assert isinstance(graph, CompiledStateGraph)
    assert graph.checkpointer is saver
    assert graph.cache is graph.store is None
    assert all(node.retry_policy is None for node in graph.nodes.values())
    structure = graph.get_graph()
    assert set(structure.nodes) == {"__start__", "__end__", *NODE_IDS}
    assert {(edge.source, edge.target) for edge in structure.edges} == {
        ("__start__", "retrieve"), ("retrieve", "fuse"), ("fuse", "rerank"),
        ("rerank", "prepare_evidence"), ("prepare_evidence", "review"),
        ("review", "generate"), ("review", "cancel"), ("generate", "validate"),
        ("validate", "__end__"), ("cancel", "__end__"),
    }
    assert {(edge.source, edge.target) for edge in structure.edges if edge.conditional} == {
        ("review", "generate"), ("review", "cancel"),
    }


@pytest.mark.parametrize("approved,phase,tail", [
    (True, "completed", ["review", "generate", "validate"]),
    (False, "cancelled", ["review", "cancel"]),
])
def test_real_interrupt_leaves_no_answer_and_resume_reenters_review_only(approved, phase, tail):
    lesson = approval.ApprovalLesson()
    paused = lesson.start()
    assert_paused(paused)
    assert len(paused["graph"]["nodes"]) == len(paused["graph"]["edges"]) == 10
    active = lesson._session
    saver, graph, config = active.graph.checkpointer, active.graph, active.config
    checkpoint = graph.get_state(config)
    assert isinstance(saver, InMemorySaver)
    assert checkpoint.next == ("review",)
    assert checkpoint.values == paused["values"]
    assert checkpoint.interrupts[0].id == paused["interrupt"]["id"]
    final = lesson.resume(paused["session_id"], approved)
    assert lesson._session is active
    assert active.graph is graph and graph.checkpointer is saver and active.config is config
    assert final["session_id"] == paused["session_id"]
    assert final["phase"] == phase
    assert final["interrupt"] is None
    assert final["checkpoint"]["next"] == []
    assert final["checkpoint"]["checkpoint_id"] != paused["checkpoint"]["checkpoint_id"]
    assert graph.get_state(config).values == final["values"]
    assert graph.get_state(config).next == ()
    assert_events(final, [*PREFIX, *tail])
    assert final["values"]["approved"] is approved
    for name in PREFIX:
        assert final["run_counts"][name] == {"entered": 1, "completed": 1}
    assert final["run_counts"]["review"] == {"entered": 2, "completed": 1}
    skipped = "cancel" if approved else "generate"
    assert final["run_counts"][skipped] == {"entered": 0, "completed": 0}
    if approved:
        assert final["values"]["answer"]["status"] == "partial"
        assert final["values"]["draft"] == final["values"]["answer"]
        assert "cancelled" not in final["values"]
    else:
        assert final["values"]["cancelled"] is True
        assert not {"draft", "answer"} & final["values"].keys()
        assert final["run_counts"]["validate"] == {"entered": 0, "completed": 0}


def test_count_report_matches_actual_function_entries_not_synthetic_step_counts(monkeypatch):
    original = approval.make_nodes
    entered, completed = [], []

    def observed_make(fixture):
        functions = original(fixture)
        for name, function in list(functions.items()):
            @wraps(function)
            def observed(state, _name=name, _function=function):
                entered.append(_name)
                update = _function(state)
                completed.append(_name)
                return update
            functions[name] = observed
        return functions

    monkeypatch.setattr(approval, "make_nodes", observed_make)
    lesson = approval.ApprovalLesson()
    paused = lesson.start()
    assert entered == [*PREFIX, "review"]
    assert completed == PREFIX
    final = lesson.resume(paused["session_id"], True)
    assert entered == [*PREFIX, "review", "review", "generate", "validate"]
    assert completed == [*PREFIX, "review", "generate", "validate"]
    for name in NODE_IDS:
        assert final["run_counts"][name] == {"entered": entered.count(name), "completed": completed.count(name)}


def test_resume_supplies_command_not_query_and_reuses_exact_graph_config(monkeypatch):
    original = approval._drive
    inputs, sessions = [], []

    def observed(session, graph_input):
        inputs.append(graph_input)
        sessions.append(session)
        return original(session, graph_input)

    monkeypatch.setattr(approval, "_drive", observed)
    lesson = approval.ApprovalLesson()
    paused = lesson.start()
    lesson.resume(paused["session_id"], True)
    assert inputs[0] == {"query": replay.load_fixture()["query"]}
    assert isinstance(inputs[1], Command) and inputs[1].resume is True
    assert sessions[0] is sessions[1]


@pytest.mark.parametrize("invalid", [1, 0, "true", "false", None, [], {}])
def test_direct_resume_requires_strict_bool_without_consuming_paused_session(invalid):
    lesson = approval.ApprovalLesson()
    paused = lesson.start()
    with pytest.raises(ValueError, match="布尔"):
        lesson.resume(paused["session_id"], invalid)
    assert lesson.current() == paused
    assert lesson.resume(paused["session_id"], False)["phase"] == "cancelled"


def test_stale_repeated_or_missing_session_cannot_resume():
    lesson = approval.ApprovalLesson()
    with pytest.raises(approval.ApprovalConflict):
        lesson.resume("a" * 32, True)
    first = lesson.start()
    second = lesson.start()
    assert first["session_id"] != second["session_id"]
    with pytest.raises(approval.ApprovalConflict):
        lesson.resume(first["session_id"], True)
    assert lesson.current() == second
    lesson.resume(second["session_id"], True)
    with pytest.raises(approval.ApprovalConflict):
        lesson.resume(second["session_id"], True)
    assert approval.ApprovalLesson().current()["phase"] == "idle"


def test_current_is_deeply_isolated_and_does_not_drive_nodes(monkeypatch):
    lesson = approval.ApprovalLesson()
    paused = lesson.start()
    forbidden = Mock(side_effect=AssertionError("Reading a snapshot must not execute the graph"))
    monkeypatch.setattr(approval, "_drive", forbidden)
    monkeypatch.setattr(lesson._session.graph, "stream", forbidden)
    monkeypatch.setattr(lesson._session.graph, "get_state", forbidden)
    outside = lesson.current()
    outside["values"]["evidence"]["S1"]["text"] = "outside mutation"
    outside["events"][0]["after"]["query"] = "outside mutation"
    outside["run_counts"]["review"]["entered"] = 999
    outside["checkpoint"]["next"].clear()
    outside["interrupt"]["value"]["source_ids"].clear()
    outside["definitions"]["review"]["code"] = "outside mutation"
    assert lesson.current() == paused
    forbidden.assert_not_called()


@pytest.mark.parametrize("when", ["start", "resume"])
def test_failure_invalidates_session_without_retry_fallback_or_reusing_old_pause(monkeypatch, when):
    lesson = approval.ApprovalLesson()
    first = lesson.start()
    failure = Mock(side_effect=ValueError("synthetic failure"))
    monkeypatch.setattr(approval, "_drive", failure)
    with pytest.raises(ValueError, match="synthetic failure"):
        if when == "start":
            lesson.start()
        else:
            lesson.resume(first["session_id"], True)
    failure.assert_called_once()
    assert lesson.current()["phase"] == "idle"
    with pytest.raises(approval.ApprovalConflict):
        lesson.resume(first["session_id"], True)


@pytest.mark.parametrize("broken_node", ["fuse", "generate"])
def test_real_node_failure_stops_the_graph_and_invalidates_the_session(monkeypatch, broken_node):
    original = approval.make_nodes
    entered = []

    def broken_make(fixture):
        nodes = original(fixture)
        for name, function in list(nodes.items()):
            @wraps(function)
            def observed(state, _name=name, _function=function):
                entered.append(_name)
                if _name == broken_node:
                    raise ValueError("synthetic node failure")
                return _function(state)
            nodes[name] = observed
        return nodes

    monkeypatch.setattr(approval, "make_nodes", broken_make)
    lesson = approval.ApprovalLesson()
    if broken_node == "fuse":
        with pytest.raises(ValueError, match="synthetic node failure"):
            lesson.start()
        assert entered == ["retrieve", "fuse"]
    else:
        paused = lesson.start()
        with pytest.raises(ValueError, match="synthetic node failure"):
            lesson.resume(paused["session_id"], True)
        assert entered == [*PREFIX, "review", "review", "generate"]
        with pytest.raises(approval.ApprovalConflict):
            lesson.resume(paused["session_id"], True)
    assert lesson.current()["phase"] == "idle"


@pytest.mark.parametrize("state", [{}, {"evidence": {}}, {"evidence": []}, {"evidence": None}])
def test_review_invalid_evidence_does_not_become_a_successful_pause(state):
    with pytest.raises((ValueError, KeyError)):
        approval.review_evidence(state)


@pytest.mark.parametrize("decision,target", [(True, "generate"), (False, "cancel")])
def test_review_returns_resume_bool_then_router_selects_only_one_branch(monkeypatch, decision, target):
    import langgraph.types
    fake_interrupt = Mock(return_value=decision)
    monkeypatch.setattr(langgraph.types, "interrupt", fake_interrupt)
    state = {"evidence": {"S1": {"text": "synthetic"}}}
    before = deepcopy(state)
    update = approval.review_evidence(state)
    assert state == before and update == {"approved": decision}
    assert approval.route_after_review(update) == target
    assert fake_interrupt.call_count == 1
    assert fake_interrupt.call_args.args[0]["evidence_count"] == 1


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_review_and_router_reject_non_boolean_decisions(monkeypatch, invalid):
    import langgraph.types
    monkeypatch.setattr(langgraph.types, "interrupt", Mock(return_value=invalid))
    with pytest.raises(ValueError):
        approval.review_evidence({"evidence": {"S1": {}}})
    with pytest.raises(ValueError):
        approval.route_after_review({"approved": invalid})


def test_cancel_writes_only_cancel_flag_not_an_insufficient_evidence_answer():
    state = {"approved": False, "evidence": {"S1": {}}}
    before = deepcopy(state)
    assert approval.cancel(state) == {"cancelled": True}
    assert state == before
    with pytest.raises(ValueError):
        approval.cancel({"approved": True})


def test_source_records_match_real_whitelisted_functions_and_lines():
    definitions = approval.describe()["definitions"]
    objects = {
        "state": approval.ApprovalState, "review": approval.review_evidence,
        "router": approval.route_after_review, "graph_builder": approval.build_graph,
        "start": approval.ApprovalLesson.start, "resume": approval.ApprovalLesson.resume,
    }
    assert set(definitions) == set(objects)
    for key, function in objects.items():
        lines, first = inspect.getsourcelines(function)
        assert definitions[key] == {
            "file": "src/qa_platform/workflows/approval.py", "name": function.__name__,
            "code": dedent("".join(lines)).rstrip("\n"),
            "start_line": first, "end_line": first + len(lines) - 1,
        }
    assert "interrupt(" in definitions["review"]["code"]
    assert "checkpointer=checkpointer" in definitions["graph_builder"]["code"]
    assert "InMemorySaver()" in definitions["start"]["code"]
    assert "Command(resume=approved)" in definitions["resume"]["code"]


@pytest.mark.parametrize("target", [Path, json.loads, replay.ReplayState, lambda state: {}, "/tmp/private.py"])
def test_source_inspection_refuses_arbitrary_paths_and_external_code(target):
    with pytest.raises(ValueError):
        approval._source_record(target)


def test_langsmith_disabled_for_both_start_and_resume_despite_environment(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "SYNTHETIC-NEVER-SEND")
    original = approval.tracing_context
    observed = []

    @contextmanager
    def checked_context(**kwargs):
        observed.append(kwargs)
        assert kwargs.get("enabled") is False
        with original(**kwargs):
            yield

    monkeypatch.setattr(approval, "tracing_context", checked_context)
    forbidden = Mock(side_effect=AssertionError("Tracing attempted external send"))
    for name in ("create_run", "update_run", "batch_ingest_runs"):
        monkeypatch.setattr(Client, name, forbidden)
    lesson = approval.ApprovalLesson()
    paused = lesson.start()
    lesson.resume(paused["session_id"], True)
    assert len(observed) == 2
    forbidden.assert_not_called()


def test_api_page_and_read_only_state_work_without_a_real_index(client):
    for path in ("/workflow/approval", f"{BASE}/state"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get(f"{BASE}/state").json()["phase"] == "idle"


@pytest.mark.parametrize("approved,phase", [(True, "completed"), (False, "cancelled")])
def test_api_roundtrip_restores_paused_snapshot_and_consumes_decision_once(client, approved, phase):
    started = client.post(f"{BASE}/start", headers=HEADERS, json={})
    assert started.status_code == 200
    paused = started.json()
    assert_paused(paused)
    assert client.get(f"{BASE}/state").json() == paused
    response = client.post(f"{BASE}/resume", headers=HEADERS,
                           json={"session_id": paused["session_id"], "approved": approved})
    assert response.status_code == 200
    assert response.json()["phase"] == phase
    assert client.get(f"{BASE}/state").json() == response.json()
    repeated = client.post(f"{BASE}/resume", headers=HEADERS,
                           json={"session_id": paused["session_id"], "approved": approved})
    assert repeated.status_code == 409


@pytest.mark.parametrize("payload", [{"query": "new"}, {"scenario": "empty_evidence"}, {"path": "/tmp/private"}, [], None, True])
def test_api_start_requires_empty_object_and_cannot_upload_or_change_fixture(client, monkeypatch, payload):
    start = Mock(side_effect=AssertionError("Bad start body must not execute"))
    start.__wrapped__ = approval.ApprovalLesson.start
    monkeypatch.setattr(approval.ApprovalLesson, "start", start)
    response = client.post(f"{BASE}/start", headers={**HEADERS, "Content-Type": "application/json"},
                           content=json.dumps(payload))
    assert response.status_code == 422
    start.assert_not_called()


@pytest.mark.parametrize("payload", [
    {}, {"session_id": "a" * 32}, {"session_id": "a" * 32, "approved": 1},
    {"session_id": "a" * 32, "approved": "true"}, {"session_id": "a" * 32, "approved": None},
    {"session_id": "a" * 31, "approved": True}, {"session_id": "A" * 32, "approved": True},
    {"session_id": "/tmp/private", "approved": True},
    {"session_id": "a" * 32, "approved": True, "query": "new"}, [], None,
])
def test_api_resume_requires_strict_boolean_session_id_and_no_extra_fields(client, monkeypatch, payload):
    resume = Mock(side_effect=AssertionError("Bad resume body must not execute"))
    resume.__wrapped__ = approval.ApprovalLesson.resume
    monkeypatch.setattr(approval.ApprovalLesson, "resume", resume)
    response = client.post(f"{BASE}/resume", headers={**HEADERS, "Content-Type": "application/json"},
                           content=json.dumps(payload))
    assert response.status_code == 422
    resume.assert_not_called()


@pytest.mark.parametrize("suffix", ["start", "resume"])
@pytest.mark.parametrize("headers,content,status", [
    ({}, "{}", 403),
    ({"X-Retrieval-Lab": "1", "Origin": "https://example.invalid", "Content-Type": "application/json"}, "{}", 403),
    ({**HEADERS, "Content-Type": "text/plain"}, "{}", 415),
    ({**HEADERS, "Content-Type": "application/json"}, "{", 422),
    ({**HEADERS, "Content-Type": "application/json"}, "{" + " " * 16385 + "}", 413),
])
def test_api_reuses_origin_json_and_size_guards(client, suffix, headers, content, status):
    assert client.post(f"{BASE}/{suffix}", headers=headers, content=content).status_code == status


def test_api_stale_session_conflict_and_per_app_session_isolation(client, tmp_path):
    first = client.post(f"{BASE}/start", headers=HEADERS, json={}).json()
    second = client.post(f"{BASE}/start", headers=HEADERS, json={}).json()
    assert first["session_id"] != second["session_id"]
    assert client.post(f"{BASE}/resume", headers=HEADERS,
                       json={"session_id": first["session_id"], "approved": True}).status_code == 409
    assert client.get(f"{BASE}/state").json() == second
    with TestClient(lab.create_app(tmp_path / "another-missing-index"), base_url="http://127.0.0.1") as fresh:
        assert fresh.get(f"{BASE}/state").json()["phase"] == "idle"
        assert fresh.post(f"{BASE}/resume", headers=HEADERS,
                          json={"session_id": second["session_id"], "approved": True}).status_code == 409


@pytest.mark.parametrize("when", ["start", "resume"])
def test_api_execution_failure_is_sanitized_no_retry_and_lock_released(client, monkeypatch, capsys, when):
    paused = client.post(f"{BASE}/start", headers=HEADERS, json={}).json()
    original = getattr(approval.ApprovalLesson, when)
    failed = Mock(side_effect=ValueError("PRIVATE_APPROVAL_DATA_NEVER_DISPLAY"))
    failed.__wrapped__ = original
    monkeypatch.setattr(approval.ApprovalLesson, when, failed)
    payload = {} if when == "start" else {"session_id": paused["session_id"], "approved": True}
    response = client.post(f"{BASE}/{when}", headers=HEADERS, json=payload)
    assert response.status_code == 503
    assert "PRIVATE_APPROVAL_DATA_NEVER_DISPLAY" not in response.text
    assert "PRIVATE_APPROVAL_DATA_NEVER_DISPLAY" not in capsys.readouterr().out
    failed.assert_called_once()
    monkeypatch.setattr(approval.ApprovalLesson, when, original)
    assert client.post(f"{BASE}/start", headers=HEADERS, json={}).status_code == 200


def test_approval_uses_shared_busy_lock_for_read_start_resume_and_real_rag(client, monkeypatch):
    entered, release = Event(), Event()
    original = approval.ApprovalLesson.start

    @wraps(original)
    def blocked(self):
        entered.set()
        assert release.wait(5)
        return original(self)

    monkeypatch.setattr(approval.ApprovalLesson, "start", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(client.post, f"{BASE}/start", headers=HEADERS, json={})
        try:
            assert entered.wait(5)
            assert client.get(f"{BASE}/state").status_code == 409
            for path, payload in (
                (f"{BASE}/start", {}),
                (f"{BASE}/resume", {"session_id": "a" * 32, "approved": True}),
                ("/api/workflow/branching/replay", {"scenario": "with_evidence"}),
                ("/api/workflow/replay", {}),
                ("/api/compare", {"query": "new", "mode": "hybrid"}),
            ):
                assert client.post(path, headers=HEADERS, json=payload).status_code == 409
            assert client.get("/api/workflow/branching/plan").status_code == 200
        finally:
            release.set()
        assert task.result(timeout=5).status_code == 200


def test_approval_preserves_existing_real_rag_snapshot(client, monkeypatch):
    baseline = {"query": "existing synthetic question", "mode": "hybrid", "routes": {}, "records": {}}
    monkeypatch.setattr(lab.hybrid, "prepare", Mock(return_value={"synthetic": True}))
    monkeypatch.setattr(lab.hybrid, "compare", Mock(return_value=deepcopy(baseline)))
    existing = client.post("/api/compare", headers=HEADERS,
                           json={"query": "existing", "mode": "hybrid"}).json()
    paused = client.post(f"{BASE}/start", headers=HEADERS, json={}).json()
    assert client.post(f"{BASE}/resume", headers=HEADERS,
                       json={"session_id": paused["session_id"], "approved": False}).status_code == 200
    rerank = Mock(side_effect=lambda frozen: {**frozen, "sentinel": "original snapshot retained"})
    monkeypatch.setattr(lab.reranking, "rerank", rerank)
    after = client.post("/api/rerank", headers=HEADERS, json={"experiment_id": existing["experiment_id"]})
    assert after.status_code == 200
    assert after.json()["sentinel"] == "original snapshot retained"
    assert rerank.call_args.args[0] == existing
