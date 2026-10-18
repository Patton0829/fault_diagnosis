"""Offline Python/LangGraph teaching comparison; no real providers are permitted.

Only the packaged synthetic fixture is replayed. These tests validate scheduling,
state and API boundaries, not retrieval quality or semantic answer entailment.
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
import socket
import sys
from threading import Event
from textwrap import dedent
from types import ModuleType
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
from langsmith import Client
import openai
import pytest
import requests

from qa_platform import retrieval_lab as lab
from qa_platform.workflows import replay


# workflow是可选依赖；只安装基础项目的环境应明确跳过，而不是收集测试时报导入错误。
CompiledStateGraph = pytest.importorskip("langgraph.graph.state").CompiledStateGraph


HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}
NODE_IDS = ["retrieve", "fuse", "rerank", "prepare_evidence", "generate", "validate"]
WRITES = {
    "retrieve": {"bm25", "dense"}, "fuse": {"rrf_pool"}, "rerank": {"reranked"},
    "prepare_evidence": {"evidence"}, "generate": {"draft"}, "validate": {"answer"},
}
ZERO_REQUESTS = {"embedding": 0, "http": 0, "reranker": 0, "chat": 0}


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay attempted network, .env or real RAG provider access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(lab.answering, "OpenAI", forbidden)
    # Optional Milvus need not be installed to prove this feature never uses it.
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


def test_plan_and_fixture_are_synthetic_and_independent():
    fixture = replay.load_fixture()
    assert fixture["origin"] == "synthetic"
    assert len(fixture["documents"]) == 3
    assert len({row["chunk_id"] for row in fixture["documents"]}) == 3
    assert all(document.metadata["is_test"] is True for document in replay._documents(fixture))
    fixture["documents"][0]["text"] = "Only this local copy was changed"
    assert replay.load_fixture()["documents"][0]["text"] != fixture["documents"][0]["text"]
    plan = replay.describe()
    assert plan["available"] is True
    assert isinstance(plan["langgraph_version"], str)
    assert plan["fixture"]["origin"] == "synthetic"
    assert [row["id"] for row in plan["nodes"]] == NODE_IDS
    assert plan["request_counts"] == ZERO_REQUESTS
    for term in ("合成", "不读.env", "不连接Milvus", "不外发LangSmith", "普通Python"):
        assert term in plan["notice"]


def test_plan_reports_missing_dependency_without_pretending_graph_success(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(replay, "version", missing)
    plan = replay.describe()
    assert plan["available"] is False
    assert plan["langgraph_version"] is None
    assert "缺少依赖" in plan["dependency_message"]


def assert_route(route, query):
    assert [event["node"] for event in route["events"]] == NODE_IDS
    assert [event["step"] for event in route["events"]] == list(range(1, 7))
    state = {"query": query}
    for event in route["events"]:
        assert event["before"] == state
        assert set(event["update"]) == WRITES[event["node"]]
        state.update(deepcopy(event["update"]))
        assert event["after"] == state
    assert route["final_state"] == state


def test_both_routes_run_the_same_state_updates_and_answer():
    fixture = replay.load_fixture()
    before = deepcopy(fixture)
    result = replay.compare_replay()
    assert result["schema_version"] == replay.SCHEMA
    assert result["plain"]["engine"] == "python"
    assert result["langgraph"]["engine"] == "langgraph"
    assert_route(result["plain"], fixture["query"])
    assert_route(result["langgraph"], fixture["query"])
    assert result["plain"]["events"] == result["langgraph"]["events"]
    assert result["plain"]["final_state"] == result["langgraph"]["final_state"]
    assert result["comparison"] == {
        "same_steps": True, "same_updates": True,
        "same_final_state": True, "same_answer": True,
    }
    assert result["request_counts"] == ZERO_REQUESTS
    assert fixture == before
    assert not any(key in result for key in ("performance", "speedup", "accuracy"))


def test_rrf_is_recomputed_while_reranker_and_draft_are_replayed():
    fixture = replay.load_fixture()
    state = replay.run_plain(fixture)["final_state"]
    fused = {row["chunk_id"]: row for row in state["rrf_pool"]}
    assert fused["demo-namespace"]["score"] == pytest.approx(2 / 62)
    assert fused["demo-overview"]["score"] == pytest.approx(1 / 61)
    assert fused["demo-dns-config"]["score"] == pytest.approx(1 / 61)
    assert state["rrf_pool"][0]["chunk_id"] == "demo-namespace"
    assert [row["chunk_id"] for row in state["reranked"]] == [
        "demo-namespace", "demo-dns-config", "demo-overview",
    ]
    assert state["draft"] == fixture["answer"]
    assert state["answer"] == fixture["answer"]
    quotes = state["evidence"]["S2"]["quotes"]
    assert "\n`/etc/resolv.conf`" in quotes["Q1"]


def test_each_node_returns_partial_update_without_mutating_state_or_fixture():
    fixture = replay.load_fixture()
    original = deepcopy(fixture)
    nodes = replay.make_nodes(fixture)
    state = {"query": fixture["query"]}
    for node_id in NODE_IDS:
        before = deepcopy(state)
        update = nodes[node_id](state)
        assert state == before
        assert set(update) == WRITES[node_id]
        state.update(deepcopy(update))
    assert fixture == original
    fixture["bm25"][0]["score"] = -999
    fixture["answer"]["claims"][0]["text"] = "changed outside frozen node closure"
    assert nodes["retrieve"]({"query": original["query"]})["bm25"] == original["bm25"]
    assert nodes["generate"]({"evidence": {"S1": {}}})["draft"] == original["answer"]


def test_returned_route_event_states_do_not_alias_each_other():
    result = replay.compare_replay()
    original_graph = deepcopy(result["langgraph"])
    plain = result["plain"]
    later_before = deepcopy(plain["events"][1]["before"])
    final_before = deepcopy(plain["final_state"])
    plain["events"][0]["after"]["bm25"][0]["score"] = -777
    assert plain["events"][1]["before"] == later_before
    assert plain["final_state"] == final_before
    assert result["langgraph"] == original_graph
    plain["final_state"]["answer"]["claims"][0]["text"] = "local mutation"
    assert result["langgraph"] == original_graph
    assert replay.compare_replay()["comparison"]["same_final_state"] is True


def test_graph_is_real_compiled_stategraph_with_explicit_chain():
    fixture = replay.load_fixture()
    graph = replay.build_graph(fixture)
    assert isinstance(graph, CompiledStateGraph)
    assert graph.checkpointer is None
    assert graph.cache is None
    assert graph.store is None
    assert all(node.retry_policy is None for node in graph.nodes.values())
    structure = graph.get_graph()
    assert set(structure.nodes) == {"__start__", "__end__", *NODE_IDS}
    expected_edges = list(zip(["__start__", *NODE_IDS], [*NODE_IDS, "__end__"]))
    assert {(edge.source, edge.target) for edge in structure.edges} == set(expected_edges)
    result = replay.compare_replay()
    assert {row["id"] for row in result["graph"]["nodes"]} == set(structure.nodes)
    assert {(row["source"], row["target"]) for row in result["graph"]["edges"]} == set(expected_edges)


def test_graph_stream_emits_actual_updates_and_framework_merged_values():
    fixture = replay.load_fixture()
    graph = replay.build_graph(fixture)
    stream = list(graph.stream({"query": fixture["query"]}, stream_mode=["updates", "values"]))
    assert [kind for kind, _ in stream] == ["values", *[kind for _ in NODE_IDS for kind in ("updates", "values")]]
    updates = [data for kind, data in stream if kind == "updates"]
    assert [next(iter(update)) for update in updates] == NODE_IDS
    values = [data for kind, data in stream if kind == "values"]
    assert values[0] == {"query": fixture["query"]}
    assert values[-1] == replay.run_langgraph(fixture)["final_state"]


def test_compare_calls_each_same_node_function_twice(monkeypatch):
    original_make = replay.make_nodes
    calls = []

    def observed_make(fixture):
        nodes = original_make(fixture)
        observed = {}
        for name, function in nodes.items():
            @wraps(function)
            def observed_node(state, _name=name, _function=function):
                calls.append((_name, deepcopy(state)))
                return _function(state)
            observed[name] = observed_node
        return observed

    make_spy = Mock(side_effect=observed_make)
    make_spy.__wrapped__ = original_make
    monkeypatch.setattr(replay, "make_nodes", make_spy)
    result = replay.compare_replay()
    assert make_spy.call_count == 1
    assert [name for name, _ in calls] == NODE_IDS * 2
    assert calls[:6] == calls[6:]
    assert result["comparison"]["same_updates"] is True


def assert_source_record(entry, function):
    function = inspect.unwrap(function)
    lines, start = inspect.getsourcelines(function)
    assert entry == {
        "name": function.__name__,
        "code": dedent("".join(lines)).rstrip("\n"),
        "start_line": start,
        "end_line": start + len(lines) - 1,
    }
    source = Path(inspect.getsourcefile(replay.ReplayState)).read_text(encoding="utf-8").splitlines(keepends=True)
    assert entry["code"] == dedent("".join(source[start - 1:entry["end_line"]])).rstrip("\n")


def test_definitions_report_actual_state_factory_shared_setup_and_graph_source():
    definitions = replay.describe_definitions()
    assert definitions["schema_version"] == "workflow-definitions/v1"
    assert definitions["source_file"] == "src/qa_platform/workflows/replay.py"
    state = dict(definitions["state"])
    state.pop("fields")
    assert_source_record(state, replay.ReplayState)
    assert_source_record(definitions["factory"], replay.make_nodes)
    assert_source_record(definitions["setup"], replay._execute_shared)
    assert_source_record(definitions["graph_builder"], replay.build_graph)
    assert "shared_nodes = make_nodes(fixture)" in definitions["setup"]["code"]
    assert "nodes=shared_nodes" in definitions["setup"]["code"]
    for term in ("TypedDict", "total=False", "Python字典", "不接受任意文件路径"):
        assert term in definitions["notice"]
    # This contract exposes selected definitions, never a dump of the whole module.
    assert "def load_fixture(" not in definitions["factory"]["code"]
    assert "def describe_definitions(" not in definitions["graph_builder"]["code"]
    assert "module_source" not in definitions and "source" not in definitions


def test_definitions_state_fields_exactly_match_types_and_declared_read_write_edges():
    fields = replay.describe_definitions()["state"]["fields"]
    assert [row["name"] for row in fields] == [
        "query", "bm25", "dense", "rrf_pool", "reranked", "evidence", "draft", "answer",
    ]
    expected_types = {
        "query": "str", "bm25": "list[dict]", "dense": "list[dict]", "rrf_pool": "list[dict]",
        "reranked": "list[dict]", "evidence": "dict", "draft": "dict", "answer": "dict",
    }
    assert replay.ReplayState.__total__ is False
    assert replay.ReplayState.__required_keys__ == frozenset()
    assert replay.ReplayState.__optional_keys__ == frozenset(expected_types)
    for field in fields:
        name = field["name"]
        assert field == {
            "name": name, "type": expected_types[name], "required": False,
            "read_by": [row["id"] for row in replay.NODES if name in row["reads"]],
            "written_by": [row["id"] for row in replay.NODES if name in row["writes"]],
        }
    assert fields[0]["read_by"] == ["retrieve"]
    assert fields[0]["written_by"] == []  # Initial input, not a synthetic writer node.
    assert fields[-1]["read_by"] == []


@pytest.mark.parametrize("node_id", NODE_IDS)
def test_definitions_node_code_is_the_actual_registered_closure(node_id):
    fixture = replay.load_fixture()
    nodes = replay.make_nodes(fixture)
    definitions = replay.describe_definitions(fixture, nodes)
    assert list(definitions["nodes"]) == NODE_IDS
    assert nodes[node_id].__name__ == node_id
    assert_source_record(definitions["nodes"][node_id], nodes[node_id])


def test_plan_extracts_source_without_running_nodes_graph_or_sdk(monkeypatch):
    original_make = replay.make_nodes
    forbidden = Mock(side_effect=AssertionError("Metadata must not run graph, nodes or provider SDK"))

    def observed_make(fixture):
        nodes = original_make(fixture)
        observed = {}
        for name, function in nodes.items():
            @wraps(function)
            def no_execution(*args, **kwargs):
                return forbidden(*args, **kwargs)
            observed[name] = no_execution
        return observed

    make_spy = Mock(side_effect=observed_make)
    make_spy.__wrapped__ = original_make
    monkeypatch.setattr(replay, "make_nodes", make_spy)
    for name in ("build_graph", "_execute_shared", "_run_plain", "_run_langgraph"):
        wrapped = Mock(side_effect=forbidden)
        wrapped.__wrapped__ = getattr(replay, name)
        monkeypatch.setattr(replay, name, wrapped)
    plan = replay.describe()
    assert plan["definitions"]["nodes"]["retrieve"]["name"] == "retrieve"
    assert plan["request_counts"] == ZERO_REQUESTS
    assert make_spy.call_count == 1
    forbidden.assert_not_called()


def test_shared_execution_passes_the_same_registry_object_to_plain_and_graph(monkeypatch):
    original_plain, original_graph = replay._run_plain, replay.build_graph
    registries = []

    @wraps(original_plain)
    def observed_plain(fixture, nodes):
        registries.append(nodes)
        return original_plain(fixture, nodes)

    @wraps(original_graph)
    def observed_graph(fixture, *, nodes=None):
        registries.append(nodes)
        return original_graph(fixture, nodes=nodes)

    monkeypatch.setattr(replay, "_run_plain", observed_plain)
    monkeypatch.setattr(replay, "build_graph", observed_graph)
    result = replay.compare_replay()
    assert len(registries) == 2
    assert registries[0] is registries[1]
    assert all(callable(function) for function in registries[0].values())
    assert result["definitions"] == replay.describe_definitions(nodes=registries[0])
    assert result["plain"]["events"] == result["langgraph"]["events"]
    assert result["request_counts"] == ZERO_REQUESTS


def test_describing_definitions_keeps_fixture_nodes_and_future_results_independent():
    fixture = replay.load_fixture()
    before = deepcopy(fixture)
    nodes = replay.make_nodes(fixture)
    definitions = replay.describe_definitions(fixture, nodes)
    definitions["state"]["fields"][0]["read_by"].append("invented")
    definitions["nodes"]["retrieve"]["code"] = "not actual source"
    assert fixture == before
    fresh = replay.describe_definitions(fixture, nodes)
    assert fresh["state"]["fields"][0]["read_by"] == ["retrieve"]
    assert_source_record(fresh["nodes"]["retrieve"], nodes["retrieve"])
    assert replay.compare_replay()["comparison"]["same_answer"] is True


@pytest.mark.parametrize("target", [Path, json.loads, lambda: "external definition", "/tmp/private.py"])
def test_source_extraction_rejects_external_objects_and_arbitrary_paths(target):
    with pytest.raises(ValueError, match="本回放模块"):
        replay._source_record(target)


@pytest.mark.parametrize("mutation", ["missing", "reordered", "extra"])
def test_definitions_reject_registry_drift_without_execution(mutation):
    nodes = replay.make_nodes(replay.load_fixture())
    if mutation == "missing":
        nodes.pop("validate")
    elif mutation == "reordered":
        nodes = dict(reversed(list(nodes.items())))
    else:
        nodes["external"] = lambda state: {}
    with pytest.raises(ValueError, match="节点注册表"):
        replay.describe_definitions(nodes=nodes)


def test_definitions_reject_swapped_functions_even_when_registry_names_are_correct():
    nodes = replay.make_nodes(replay.load_fixture())
    nodes["retrieve"], nodes["fuse"] = nodes["fuse"], nodes["retrieve"]
    assert list(nodes) == NODE_IDS
    with pytest.raises(ValueError, match="注册名称与对应函数不一致"):
        replay.describe_definitions(nodes=nodes)


def test_definitions_do_not_accept_a_browser_supplied_source_path():
    with pytest.raises(TypeError):
        replay.describe_definitions(path="/tmp/private.py")


def test_plan_and_replay_api_include_the_same_actual_definitions_without_new_io(client):
    plan = client.get("/api/workflow/plan")
    assert plan.status_code == 200
    result = client.post("/api/workflow/replay", headers=HEADERS, json={})
    assert result.status_code == 200
    assert plan.json()["definitions"] == result.json()["definitions"] == replay.describe_definitions()
    assert plan.json()["request_counts"] == result.json()["request_counts"] == ZERO_REQUESTS
    assert_route(result.json()["plain"], replay.load_fixture()["query"])
    assert_route(result.json()["langgraph"], replay.load_fixture()["query"])


@pytest.mark.parametrize("runner", [replay.run_plain, replay.run_langgraph])
def test_error_stops_at_failing_node_without_retry_or_later_execution(monkeypatch, runner):
    calls = []
    original_make = replay.make_nodes

    def broken_make(fixture):
        nodes = original_make(fixture)
        for name, function in list(nodes.items()):
            def observed(state, _name=name, _function=function):
                calls.append(_name)
                if _name == "fuse":
                    raise ValueError("synthetic fuse failure")
                return _function(state)
            nodes[name] = observed
        return nodes

    monkeypatch.setattr(replay, "make_nodes", broken_make)
    with pytest.raises(ValueError, match="synthetic fuse failure"):
        runner(replay.load_fixture())
    assert calls == ["retrieve", "fuse"]


@pytest.mark.parametrize("runner", [replay.run_plain, replay.run_langgraph])
def test_replayed_invalid_citation_is_actually_rejected(runner):
    fixture = replay.load_fixture()
    fixture["answer"]["claims"][0]["evidence"][0]["quote_id"] = "Q999"
    with pytest.raises(ValueError):
        runner(fixture)


@pytest.mark.parametrize("mutation", ["origin", "query", "duplicate", "unknown_hit", "nonfinite"])
def test_bad_fixture_fails_without_real_io(mutation):
    fixture = replay.load_fixture()
    if mutation == "origin":
        fixture["origin"] = "real"
    elif mutation == "query":
        fixture["query"] = ""
    elif mutation == "duplicate":
        fixture["documents"][1]["chunk_id"] = fixture["documents"][0]["chunk_id"]
    elif mutation == "unknown_hit":
        fixture["bm25"][0]["chunk_id"] = "unknown-document"
    else:
        fixture["dense"][0]["score"] = float("nan")
    with pytest.raises(ValueError):
        replay.run_plain(fixture)


def test_enabled_langsmith_environment_is_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "SYNTHETIC-NEVER-SEND")
    original_context = replay.tracing_context
    seen = []

    @contextmanager
    def observed_context(**kwargs):
        seen.append(kwargs)
        assert kwargs.get("enabled") is False
        with original_context(**kwargs):
            yield

    forbidden = Mock(side_effect=AssertionError("LangSmith attempted to send replay data"))
    for name in ("create_run", "update_run", "batch_ingest_runs"):
        monkeypatch.setattr(Client, name, forbidden)
    monkeypatch.setattr(replay, "tracing_context", observed_context)
    fixture = replay.load_fixture()
    replay.run_plain(fixture)
    replay.run_langgraph(fixture)
    assert replay.compare_replay()["comparison"]["same_answer"] is True
    assert len(seen) >= 4
    forbidden.assert_not_called()


@pytest.fixture
def client(tmp_path):
    with TestClient(lab.create_app(tmp_path / "definitely-missing-index"), base_url="http://127.0.0.1") as active:
        yield active


def test_workflow_plan_and_page_do_not_require_existing_index(client):
    for path in ("/workflow", "/api/workflow/plan"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
    plan = client.get("/api/workflow/plan").json()
    assert plan["fixture"]["origin"] == "synthetic"
    assert plan["request_counts"] == ZERO_REQUESTS


def test_api_replay_runs_without_retrieval_snapshot_or_model_requests(client):
    response = client.post("/api/workflow/replay", headers=HEADERS, json={})
    assert response.status_code == 200
    result = response.json()
    assert all(result["comparison"].values())
    assert result["request_counts"] == ZERO_REQUESTS
    assert result["fixture"]["origin"] == "synthetic"


@pytest.mark.parametrize("payload", [
    {"query": "Do not use old result as new query"}, {"path": "/tmp/private.json"},
    {"url": "https://example.invalid"}, {"experiment_id": "a" * 32},
    {"model": "glm"}, {"fixture": {}}, [], None, "text", 1,
])
def test_api_replay_rejects_every_nonempty_or_nonobject_body(client, payload):
    response = client.post("/api/workflow/replay", headers={**HEADERS, "Content-Type": "application/json"},
                           content=json.dumps(payload))
    assert response.status_code == 422


@pytest.mark.parametrize("headers,content,status", [
    ({}, "{}", 403),
    ({"X-Retrieval-Lab": "1", "Origin": "https://example.invalid", "Content-Type": "application/json"}, "{}", 403),
    ({**HEADERS, "Content-Type": "text/plain"}, "{}", 415),
    ({**HEADERS, "Content-Type": "application/json"}, "{", 422),
    ({**HEADERS, "Content-Type": "application/json"}, "{" + " " * 16385 + "}", 413),
])
def test_api_replay_reuses_origin_type_json_and_size_guards(client, headers, content, status):
    response = client.post("/api/workflow/replay", headers=headers, content=content)
    assert response.status_code == status


def test_api_failure_is_fixed_sanitized_and_lock_is_released(client, monkeypatch, capsys):
    real_compare = replay.compare_replay
    failed = Mock(side_effect=ValueError("PRIVATE_FIXTURE_BODY_MUST_NOT_LEAK"))
    monkeypatch.setattr(replay, "compare_replay", failed)
    response = client.post("/api/workflow/replay", headers=HEADERS, json={})
    assert response.status_code == 503
    assert "PRIVATE_FIXTURE_BODY_MUST_NOT_LEAK" not in response.text
    assert "PRIVATE_FIXTURE_BODY_MUST_NOT_LEAK" not in capsys.readouterr().out
    assert failed.call_count == 1
    monkeypatch.setattr(replay, "compare_replay", real_compare)
    assert client.post("/api/workflow/replay", headers=HEADERS, json={}).status_code == 200


def test_replay_holds_same_busy_lock_as_real_workflow(client, monkeypatch):
    entered, release = Event(), Event()
    real_compare = replay.compare_replay

    def blocked_compare():
        entered.set()
        assert release.wait(5)
        return real_compare()

    monkeypatch.setattr(replay, "compare_replay", blocked_compare)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(client.post, "/api/workflow/replay", headers=HEADERS, json={})
        try:
            assert entered.wait(5)
            for path, payload in (
                ("/api/workflow/replay", {}),
                ("/api/compare", {"query": "new", "mode": "hybrid"}),
                ("/api/rerank", {"experiment_id": "a" * 32}),
                ("/api/quotes", {"experiment_id": "a" * 32}),
                ("/api/answer", {"experiment_id": "a" * 32, "model": "glm"}),
            ):
                assert client.post(path, headers=HEADERS, json=payload).status_code == 409
        finally:
            release.set()
        assert task.result(timeout=5).status_code == 200


def test_real_retrieval_busy_lock_also_blocks_replay(client, monkeypatch):
    entered, release = Event(), Event()

    def blocked_compare(check, mode):
        entered.set()
        assert release.wait(5)
        return {"mode": mode, "routes": {}}

    monkeypatch.setattr(lab.hybrid, "prepare", Mock(return_value={"synthetic": True}))
    monkeypatch.setattr(lab.hybrid, "compare", blocked_compare)
    replay_spy = Mock(side_effect=AssertionError("Busy replay must not execute"))
    monkeypatch.setattr(replay, "compare_replay", replay_spy)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(client.post, "/api/compare", headers=HEADERS,
                           json={"query": "existing", "mode": "hybrid"})
        try:
            assert entered.wait(5)
            assert client.post("/api/workflow/replay", headers=HEADERS, json={}).status_code == 409
            # Its purely local plan remains readable while execution is busy.
            assert client.get("/api/workflow/plan").status_code == 200
        finally:
            release.set()
        assert task.result(timeout=5).status_code == 200
    replay_spy.assert_not_called()


def test_replay_does_not_replace_existing_real_retrieval_snapshot(client, monkeypatch):
    baseline = {"query": "existing synthetic question", "mode": "hybrid", "routes": {}, "records": {}}
    monkeypatch.setattr(lab.hybrid, "prepare", Mock(return_value={"synthetic": True}))
    monkeypatch.setattr(lab.hybrid, "compare", Mock(return_value=deepcopy(baseline)))
    start = client.post("/api/compare", headers=HEADERS, json={"query": "existing", "mode": "hybrid"})
    assert start.status_code == 200
    existing = start.json()
    assert client.post("/api/workflow/replay", headers=HEADERS, json={}).status_code == 200
    rerank_spy = Mock(side_effect=lambda frozen: {**frozen, "sentinel": "original snapshot retained"})
    monkeypatch.setattr(lab.reranking, "rerank", rerank_spy)
    after = client.post("/api/rerank", headers=HEADERS, json={"experiment_id": existing["experiment_id"]})
    assert after.status_code == 200
    assert after.json()["query"] == baseline["query"]
    assert after.json()["sentinel"] == "original snapshot retained"
    assert rerank_spy.call_args.args[0] == existing
