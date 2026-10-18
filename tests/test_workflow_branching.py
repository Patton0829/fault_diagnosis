"""Conditional Python/LangGraph lesson; all examples stay offline and synthetic.

These tests prove real branch selection, state propagation and API isolation.
They do not treat a nonempty evidence dictionary as a relevance or quality test.
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
from qa_platform.workflows import branching, replay


CompiledStateGraph = pytest.importorskip("langgraph.graph.state").CompiledStateGraph

HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}
SCENARIOS = ("with_evidence", "empty_evidence")
PREFIX = ["retrieve", "fuse", "rerank", "prepare_evidence"]
NODE_IDS = [*PREFIX, "generate", "refuse", "validate"]
WRITES = {
    "retrieve": {"bm25", "dense"}, "fuse": {"rrf_pool"}, "rerank": {"reranked"},
    "prepare_evidence": {"evidence"}, "generate": {"draft"}, "refuse": {"draft"},
    "validate": {"answer"},
}
ZERO_REQUESTS = {"embedding": 0, "http": 0, "reranker": 0, "chat": 0}
API = "/api/workflow/branching/replay"


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Branching lesson attempted network, .env or real RAG access")

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


def selected_branch(scenario):
    return "generate" if scenario == "with_evidence" else "refuse"


def executed_nodes(scenario):
    return [*PREFIX, selected_branch(scenario), "validate"]


def assert_route(route, fixture, scenario):
    assert [event["node"] for event in route["events"]] == executed_nodes(scenario)
    assert [event["step"] for event in route["events"]] == list(range(1, 7))
    state = {"query": fixture["query"]}
    for event in route["events"]:
        assert event["before"] == state
        assert set(event["update"]) == WRITES[event["node"]]
        state.update(deepcopy(event["update"]))
        assert event["after"] == state
    assert route["final_state"] == state
    assert len(route["decisions"]) == 1
    decision = route["decisions"][0]
    assert decision["after_node"] == "prepare_evidence"
    assert decision["target"] == selected_branch(scenario)
    assert decision["evidence_count"] == (2 if scenario == "with_evidence" else 0)
    assert isinstance(decision["reason"], str) and decision["reason"]
    # The router returns a destination, not a hidden state-writing node.
    assert set(state) == set(replay.ReplayState.__annotations__)


def test_fixture_scenarios_change_only_the_two_recall_lists():
    normal = branching.load_fixture("with_evidence")
    empty = branching.load_fixture("empty_evidence")
    assert normal == replay.load_fixture()
    assert normal["origin"] == empty["origin"] == "synthetic"
    assert empty["bm25"] == empty["dense"] == []
    assert {key: value for key, value in empty.items() if key not in ("bm25", "dense")} == {
        key: value for key, value in normal.items() if key not in ("bm25", "dense")
    }
    empty["documents"][0]["text"] = "mutated outside fixture"
    assert branching.load_fixture("empty_evidence")["documents"] == normal["documents"]
    assert [row["id"] for row in branching.SCENARIOS] == list(SCENARIOS)


@pytest.mark.parametrize("scenario", ["unknown", "", None, 1, True, [], {}, "/tmp/fixture.json"])
def test_unknown_scenario_is_rejected_before_reading_fixture(monkeypatch, scenario):
    fixture_reader = Mock(side_effect=AssertionError("Invalid scenario must fail before fixture IO"))
    monkeypatch.setattr(replay, "load_fixture", fixture_reader)
    for function in (branching.load_fixture, branching.compare_replay):
        with pytest.raises(ValueError, match="场景"):
            function(scenario)
    fixture_reader.assert_not_called()


def test_plan_reports_available_dependency_and_zero_request_boundaries():
    plan = branching.describe()
    assert plan["schema_version"] == branching.SCHEMA == "rag-workflow-branching/v1"
    assert plan["available"] is True
    assert isinstance(plan["langgraph_version"], str)
    assert plan["fixture"]["origin"] == "synthetic"
    assert [row["id"] for row in plan["nodes"]] == NODE_IDS
    assert [row["id"] for row in plan["scenarios"]] == list(SCENARIOS)
    assert plan["request_counts"] == ZERO_REQUESTS
    for phrase in ("非空证据", "不是证据质量", "普通Python", "if/else", "不读.env", "不连接Milvus", "不外发LangSmith"):
        assert phrase in plan["notice"]


def test_plan_extracts_source_without_executing_router_nodes_or_graph(monkeypatch):
    original_make = branching.make_nodes
    forbidden = Mock(side_effect=AssertionError("Plan is source inspection, not execution"))

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
    monkeypatch.setattr(branching, "make_nodes", make_spy)
    for name in ("route_after_evidence", "_run_plain", "build_graph", "_run_langgraph"):
        wrapped = Mock(side_effect=forbidden)
        wrapped.__wrapped__ = getattr(branching, name)
        monkeypatch.setattr(branching, name, wrapped)
    plan = branching.describe()
    assert plan["definitions"]["router"]["name"] == "route_after_evidence"
    assert plan["request_counts"] == ZERO_REQUESTS
    assert make_spy.call_count == 1
    forbidden.assert_not_called()


def test_plan_reports_missing_langgraph_without_faking_success(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(branching, "version", missing)
    plan = branching.describe()
    assert plan["available"] is False
    assert plan["langgraph_version"] is None
    assert "缺少依赖" in plan["dependency_message"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_both_schedulers_execute_the_same_selected_six_nodes_and_decision(scenario):
    fixture = branching.load_fixture(scenario)
    before = deepcopy(fixture)
    result = branching.compare_replay(scenario)
    assert result["scenario"]["id"] == scenario
    assert result["plain"]["engine"] == "python"
    assert result["langgraph"]["engine"] == "langgraph"
    assert_route(result["plain"], fixture, scenario)
    assert_route(result["langgraph"], fixture, scenario)
    assert result["plain"]["events"] == result["langgraph"]["events"]
    assert result["comparison"] == {
        "same_steps": True, "same_updates": True, "same_final_state": True,
        "same_answer": True, "same_decisions": True,
    }
    assert result["request_counts"] == ZERO_REQUESTS
    assert fixture == before
    assert not any(key in result for key in ("accuracy", "performance", "speedup"))


def test_empty_path_propagates_empty_candidates_and_validates_a_fixed_refusal():
    result = branching.compare_replay("empty_evidence")
    state = result["langgraph"]["final_state"]
    for key in ("bm25", "dense", "rrf_pool", "reranked"):
        assert state[key] == []
    assert state["evidence"] == {}
    assert state["draft"] == state["answer"]
    assert state["answer"]["status"] == "insufficient_evidence"
    assert state["answer"]["claims"] == []
    assert state["answer"]["missing_info"]
    assert result["langgraph"]["events"][-1]["node"] == "validate"


@pytest.mark.parametrize("evidence,expected", [
    ({}, "refuse"),
    ({"S1": {}}, "generate"),
    ({"S1": {"score": 0, "text": "unrelated synthetic text"}}, "generate"),
])
def test_router_only_uses_dictionary_emptiness_not_quality_or_scores(evidence, expected):
    state = {"evidence": deepcopy(evidence), "query": "unrelated question"}
    before = deepcopy(state)
    assert branching.route_after_evidence(state) == expected
    assert state == before


@pytest.mark.parametrize("state", [{}, {"evidence": None}, {"evidence": []}, {"evidence": ""}])
def test_malformed_evidence_is_an_error_not_a_normal_refusal(state):
    with pytest.raises((ValueError, KeyError)):
        branching.route_after_evidence(state)


def test_refusal_is_pure_fresh_and_cannot_hide_a_nonempty_evidence_error():
    state = {"evidence": {}}
    before = deepcopy(state)
    first = branching.refuse(state)
    assert state == before
    assert set(first) == {"draft"}
    first["draft"]["missing_info"].append("outside mutation")
    assert "outside mutation" not in branching.refuse(state)["draft"]["missing_info"]
    with pytest.raises(ValueError):
        branching.refuse({"evidence": {"S1": {}}})


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_decision_log_records_exactly_one_actual_router_call_per_scheduler(monkeypatch, scenario):
    original_router = branching.route_after_evidence
    observed = []

    @wraps(original_router)
    def tracked_router(state):
        target = original_router(state)
        observed.append((target, deepcopy(state["evidence"])))
        return target

    monkeypatch.setattr(branching, "route_after_evidence", tracked_router)
    result = branching.compare_replay(scenario)
    assert len(observed) == 2
    assert observed[0] == observed[1]
    for name, (target, evidence) in zip(("plain", "langgraph"), observed):
        assert result[name]["decisions"][0]["target"] == target
        assert result[name]["decisions"][0]["evidence_count"] == len(evidence)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_compare_reuses_one_registry_and_calls_only_the_selected_branch(monkeypatch, scenario):
    original_make = branching.make_nodes
    calls, registries = [], []

    def observed_make(fixture):
        nodes = original_make(fixture)
        observed = {}
        for name, function in nodes.items():
            @wraps(function)
            def observed_node(state, _name=name, _function=function):
                calls.append((_name, deepcopy(state)))
                return _function(state)
            observed[name] = observed_node
        registries.append(observed)
        return observed

    make_spy = Mock(side_effect=observed_make)
    make_spy.__wrapped__ = original_make
    monkeypatch.setattr(branching, "make_nodes", make_spy)
    original_plain, original_graph = branching._run_plain, branching.build_graph
    used_registries = []

    @wraps(original_plain)
    def observed_plain(fixture, nodes):
        used_registries.append(nodes)
        return original_plain(fixture, nodes)

    @wraps(original_graph)
    def observed_graph(fixture, *, nodes=None, router=None):
        used_registries.append(nodes)
        return original_graph(fixture, nodes=nodes, router=router)

    monkeypatch.setattr(branching, "_run_plain", observed_plain)
    monkeypatch.setattr(branching, "build_graph", observed_graph)
    result = branching.compare_replay(scenario)
    assert make_spy.call_count == 1
    assert used_registries[0] is used_registries[1] is registries[0]
    assert [name for name, _ in calls] == executed_nodes(scenario) * 2
    assert calls[:6] == calls[6:]
    assert result["comparison"]["same_decisions"] is True


def test_factory_reuses_the_original_six_node_definitions_and_adds_refuse():
    assert branching.ReplayState is replay.ReplayState
    nodes = branching.make_nodes(branching.load_fixture())
    assert list(nodes) == NODE_IDS
    for name in [*PREFIX, "generate", "validate"]:
        assert nodes[name].__module__ == replay.__name__
        assert nodes[name].__name__ == name
    assert nodes["refuse"] is branching.refuse


def test_real_graph_contains_two_conditional_edges_without_checkpoint_cache_or_retry():
    graph = branching.build_graph(branching.load_fixture())
    assert isinstance(graph, CompiledStateGraph)
    assert graph.checkpointer is graph.cache is graph.store is None
    assert all(node.retry_policy is None for node in graph.nodes.values())
    structure = graph.get_graph()
    assert set(structure.nodes) == {"__start__", "__end__", *NODE_IDS}
    expected = {
        ("__start__", "retrieve"), ("retrieve", "fuse"), ("fuse", "rerank"),
        ("rerank", "prepare_evidence"), ("prepare_evidence", "generate"),
        ("prepare_evidence", "refuse"), ("generate", "validate"),
        ("refuse", "validate"), ("validate", "__end__"),
    }
    assert {(edge.source, edge.target) for edge in structure.edges} == expected
    assert {(edge.source, edge.target) for edge in structure.edges if edge.conditional} == {
        ("prepare_evidence", "generate"), ("prepare_evidence", "refuse"),
    }
    result = branching.compare_replay()
    assert len(result["graph"]["nodes"]) == len(result["graph"]["edges"]) == 9
    assert {(row["source"], row["target"], row["conditional"]) for row in result["graph"]["edges"]} == {
        (edge.source, edge.target, edge.conditional) for edge in structure.edges
    }


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_actual_graph_stream_skips_unselected_branch_and_emits_framework_values(scenario):
    fixture = branching.load_fixture(scenario)
    nodes = branching.make_nodes(fixture)
    skipped = "refuse" if scenario == "with_evidence" else "generate"
    nodes[skipped] = Mock(side_effect=AssertionError("Unselected branch was executed"))
    router = Mock(wraps=branching.route_after_evidence)
    graph = branching.build_graph(fixture, nodes=nodes, router=router)
    stream = list(graph.stream({"query": fixture["query"]}, stream_mode=["updates", "values"]))
    assert [kind for kind, _ in stream] == ["values", *[kind for _ in range(6) for kind in ("updates", "values")]]
    updates = [data for kind, data in stream if kind == "updates"]
    assert [next(iter(update)) for update in updates] == executed_nodes(scenario)
    values = [data for kind, data in stream if kind == "values"]
    assert values[0] == {"query": fixture["query"]}
    assert values[-1] == branching.run_langgraph(fixture)["final_state"]
    nodes[skipped].assert_not_called()
    assert router.call_count == 1
    assert router.call_args.args[0]["evidence"] == values[4]["evidence"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_fixture_and_returned_events_remain_deeply_independent(scenario):
    fixture = branching.load_fixture(scenario)
    original_fixture = deepcopy(fixture)
    nodes = branching.make_nodes(fixture)
    state = {"query": fixture["query"]}
    for node_id in executed_nodes(scenario):
        before = deepcopy(state)
        update = nodes[node_id](state)
        assert state == before
        state.update(deepcopy(update))
    assert fixture == original_fixture
    fixture["documents"][0]["text"] = "outside mutation"
    assert nodes["retrieve"]({"query": fixture["query"]})["bm25"] == original_fixture["bm25"]
    result = branching.compare_replay(scenario)
    original_graph = deepcopy(result["langgraph"])
    original_final = deepcopy(result["plain"]["final_state"])
    result["plain"]["events"][0]["after"]["query"] = "local mutation"
    assert result["plain"]["events"][1]["before"]["query"] == original_fixture["query"]
    assert result["plain"]["final_state"] == original_final
    result["plain"]["decisions"][0]["target"] = "other"
    assert result["langgraph"] == original_graph
    assert branching.compare_replay(scenario)["comparison"]["same_answer"] is True


@pytest.mark.parametrize("runner", [branching.run_plain, branching.run_langgraph])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_node_failure_stops_without_refusal_fallback_retry_or_later_execution(monkeypatch, runner, scenario):
    original_make = branching.make_nodes
    calls = []

    def broken_make(fixture):
        nodes = original_make(fixture)
        for name, function in list(nodes.items()):
            @wraps(function)
            def observed(state, _name=name, _function=function):
                calls.append(_name)
                if _name == "fuse":
                    raise ValueError("synthetic fusion failure")
                return _function(state)
            nodes[name] = observed
        return nodes

    monkeypatch.setattr(branching, "make_nodes", broken_make)
    with pytest.raises(ValueError, match="synthetic fusion failure"):
        runner(branching.load_fixture(scenario))
    assert calls == ["retrieve", "fuse"]


@pytest.mark.parametrize("runner", [branching.run_plain, branching.run_langgraph])
def test_nonempty_path_still_runs_real_citation_validation(runner):
    fixture = branching.load_fixture()
    fixture["answer"]["claims"][0]["evidence"][0]["quote_id"] = "Q999"
    with pytest.raises(ValueError):
        runner(fixture)


def assert_source_record(record, function):
    function = inspect.unwrap(function)
    lines, start = inspect.getsourcelines(function)
    source_path = Path(inspect.getsourcefile(function))
    assert record == {
        "file": f"src/qa_platform/workflows/{source_path.name}",
        "name": function.__name__, "code": dedent("".join(lines)).rstrip("\n"),
        "start_line": start, "end_line": start + len(lines) - 1,
    }
    saved_lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert record["code"] == dedent("".join(saved_lines[start - 1:record["end_line"]])).rstrip("\n")


def test_definitions_expose_real_whitelisted_state_router_factory_and_graph_code():
    fixture = branching.load_fixture()
    nodes = branching.make_nodes(fixture)
    definitions = branching.describe_definitions(fixture, nodes)
    expected = {
        "state": replay.ReplayState, "router": branching.route_after_evidence,
        "factory": branching.make_nodes, "plain": branching._run_plain,
        "graph_builder": branching.build_graph,
    }
    for key, function in expected.items():
        assert_source_record(definitions[key], function)
    assert list(definitions["nodes"]) == NODE_IDS
    for name, function in nodes.items():
        assert_source_record(definitions["nodes"][name], function)
    assert "add_conditional_edges" in definitions["graph_builder"]["code"]
    assert "if target == \"generate\"" in definitions["plain"]["code"]
    assert definitions["state"]["file"].endswith("/replay.py")
    assert definitions["router"]["file"].endswith("/branching.py")
    definitions["nodes"]["generate"]["code"] = "outside mutation"
    assert_source_record(branching.describe_definitions(fixture, nodes)["nodes"]["generate"], nodes["generate"])


@pytest.mark.parametrize("target", [Path, json.loads, lambda state: "generate", "/tmp/private.py"])
def test_source_extraction_rejects_external_functions_and_arbitrary_paths(target):
    with pytest.raises(ValueError):
        branching._source_record(target)


@pytest.mark.parametrize("mutation", ["missing", "reordered", "extra", "swapped"])
def test_definition_registry_rejects_drift_or_wrong_function_bindings(mutation):
    nodes = branching.make_nodes(branching.load_fixture())
    if mutation == "missing":
        nodes.pop("validate")
    elif mutation == "reordered":
        nodes = dict(reversed(list(nodes.items())))
    elif mutation == "extra":
        nodes["external"] = lambda state: {}
    else:
        nodes["generate"], nodes["refuse"] = nodes["refuse"], nodes["generate"]
    with pytest.raises(ValueError):
        branching.describe_definitions(nodes=nodes)


def test_definitions_have_no_browser_supplied_source_path_parameter():
    with pytest.raises(TypeError):
        branching.describe_definitions(path="/tmp/private.py")


def test_langsmith_remains_disabled_even_when_enabled_in_user_environment(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "SYNTHETIC-NEVER-SEND")
    original_context = branching.tracing_context
    observed = []

    @contextmanager
    def checked_context(**kwargs):
        observed.append(kwargs)
        assert kwargs.get("enabled") is False
        with original_context(**kwargs):
            yield

    monkeypatch.setattr(branching, "tracing_context", checked_context)
    forbidden = Mock(side_effect=AssertionError("LangSmith attempted to send lesson data"))
    for name in ("create_run", "update_run", "batch_ingest_runs"):
        monkeypatch.setattr(Client, name, forbidden)
    for scenario in SCENARIOS:
        fixture = branching.load_fixture(scenario)
        branching.run_plain(fixture)
        branching.run_langgraph(fixture)
        assert branching.compare_replay(scenario)["comparison"]["same_answer"] is True
    assert len(observed) >= 6
    forbidden.assert_not_called()


def test_page_and_plan_work_without_real_index_and_reuse_security_headers(client):
    for path in ("/workflow/branching", "/api/workflow/branching/plan"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
    plan = client.get("/api/workflow/branching/plan").json()
    assert plan["fixture"]["origin"] == "synthetic"
    assert plan["request_counts"] == ZERO_REQUESTS
    assert plan["definitions"] == branching.describe_definitions()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_api_runs_each_fixed_scenario_without_real_rag_snapshot(client, scenario):
    response = client.post(API, headers=HEADERS, json={"scenario": scenario})
    assert response.status_code == 200
    result = response.json()
    assert result["schema_version"] == branching.SCHEMA
    assert result["scenario"]["id"] == scenario
    assert result["request_counts"] == ZERO_REQUESTS
    assert all(result["comparison"].values())
    assert_route(result["plain"], branching.load_fixture(scenario), scenario)
    assert_route(result["langgraph"], branching.load_fixture(scenario), scenario)


@pytest.mark.parametrize("payload", [
    {}, {"scenario": "unknown"}, {"scenario": None}, {"scenario": True},
    {"scenario": ["with_evidence"]}, {"scenario": "with_evidence", "query": "new"},
    {"scenario": "with_evidence", "path": "/tmp/private.json"},
    {"scenario": "with_evidence", "fixture": {}},
    {"scenario": "with_evidence", "model": "glm"}, [], None, "text", 1,
])
def test_api_strictly_requires_fixed_scenario_and_forbids_extra_fields(client, monkeypatch, payload):
    compare = Mock(side_effect=AssertionError("Malformed request must not execute lesson"))
    monkeypatch.setattr(branching, "compare_replay", compare)
    response = client.post(API, headers={**HEADERS, "Content-Type": "application/json"}, content=json.dumps(payload))
    assert response.status_code == 422
    compare.assert_not_called()


@pytest.mark.parametrize("headers,content,status", [
    ({}, '{"scenario":"with_evidence"}', 403),
    ({"X-Retrieval-Lab": "1", "Origin": "https://example.invalid", "Content-Type": "application/json"},
     '{"scenario":"with_evidence"}', 403),
    ({**HEADERS, "Content-Type": "text/plain"}, '{"scenario":"with_evidence"}', 415),
    ({**HEADERS, "Content-Type": "application/json"}, "{", 422),
    ({**HEADERS, "Content-Type": "application/json"}, "{" + " " * 16385 + "}", 413),
])
def test_api_reuses_origin_type_json_and_body_size_guards(client, headers, content, status):
    response = client.post(API, headers=headers, content=content)
    assert response.status_code == status


def test_api_failure_is_sanitized_has_no_retry_or_fallback_and_releases_lock(client, monkeypatch, capsys):
    original = branching.compare_replay
    failed = Mock(side_effect=ValueError("PRIVATE_FIXTURE_MUST_NOT_LEAK"))
    monkeypatch.setattr(branching, "compare_replay", failed)
    response = client.post(API, headers=HEADERS, json={"scenario": "empty_evidence"})
    assert response.status_code == 503
    assert "PRIVATE_FIXTURE_MUST_NOT_LEAK" not in response.text
    assert "PRIVATE_FIXTURE_MUST_NOT_LEAK" not in capsys.readouterr().out
    failed.assert_called_once_with("empty_evidence")
    monkeypatch.setattr(branching, "compare_replay", original)
    assert client.post(API, headers=HEADERS, json={"scenario": "with_evidence"}).status_code == 200
    assert client.post("/api/workflow/replay", headers=HEADERS, json={}).status_code == 200


def test_branching_holds_the_shared_busy_lock_but_plans_remain_readable(client, monkeypatch):
    entered, release = Event(), Event()
    original = branching.compare_replay

    def blocked(scenario):
        entered.set()
        assert release.wait(5)
        return original(scenario)

    monkeypatch.setattr(branching, "compare_replay", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(client.post, API, headers=HEADERS, json={"scenario": "with_evidence"})
        try:
            assert entered.wait(5)
            for path, payload in (
                (API, {"scenario": "empty_evidence"}),
                ("/api/workflow/replay", {}),
                ("/api/compare", {"query": "new", "mode": "hybrid"}),
                ("/api/rerank", {"experiment_id": "a" * 32}),
                ("/api/quotes", {"experiment_id": "a" * 32}),
                ("/api/answer", {"experiment_id": "a" * 32, "model": "glm"}),
            ):
                assert client.post(path, headers=HEADERS, json=payload).status_code == 409
            assert client.get("/api/workflow/branching/plan").status_code == 200
            assert client.get("/api/workflow/plan").status_code == 200
        finally:
            release.set()
        assert task.result(timeout=5).status_code == 200


def test_original_lesson_busy_lock_blocks_branching_without_executing_it(client, monkeypatch):
    entered, release = Event(), Event()
    original = replay.compare_replay

    def blocked():
        entered.set()
        assert release.wait(5)
        return original()

    monkeypatch.setattr(replay, "compare_replay", blocked)
    branch_spy = Mock(side_effect=AssertionError("Busy conditional lesson must not execute"))
    monkeypatch.setattr(branching, "compare_replay", branch_spy)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(client.post, "/api/workflow/replay", headers=HEADERS, json={})
        try:
            assert entered.wait(5)
            assert client.post(API, headers=HEADERS, json={"scenario": "empty_evidence"}).status_code == 409
        finally:
            release.set()
        assert task.result(timeout=5).status_code == 200
    branch_spy.assert_not_called()


def test_real_retrieval_busy_lock_blocks_branching_without_executing_it(client, monkeypatch):
    entered, release = Event(), Event()

    def blocked(check, mode):
        entered.set()
        assert release.wait(5)
        return {"mode": mode, "routes": {}}

    monkeypatch.setattr(lab.hybrid, "prepare", Mock(return_value={"synthetic": True}))
    monkeypatch.setattr(lab.hybrid, "compare", blocked)
    branch_spy = Mock(side_effect=AssertionError("Busy lesson must not execute"))
    monkeypatch.setattr(branching, "compare_replay", branch_spy)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(client.post, "/api/compare", headers=HEADERS,
                           json={"query": "existing", "mode": "hybrid"})
        try:
            assert entered.wait(5)
            assert client.post(API, headers=HEADERS, json={"scenario": "empty_evidence"}).status_code == 409
            assert client.get("/api/workflow/branching/plan").status_code == 200
        finally:
            release.set()
        assert task.result(timeout=5).status_code == 200
    branch_spy.assert_not_called()


def test_branching_preserves_real_retrieval_snapshot_and_original_lesson(client, monkeypatch):
    baseline = {"query": "existing synthetic question", "mode": "hybrid", "routes": {}, "records": {}}
    monkeypatch.setattr(lab.hybrid, "prepare", Mock(return_value={"synthetic": True}))
    monkeypatch.setattr(lab.hybrid, "compare", Mock(return_value=deepcopy(baseline)))
    response = client.post("/api/compare", headers=HEADERS, json={"query": "existing", "mode": "hybrid"})
    assert response.status_code == 200
    existing = response.json()
    for scenario in SCENARIOS:
        assert client.post(API, headers=HEADERS, json={"scenario": scenario}).status_code == 200
    assert client.post("/api/workflow/replay", headers=HEADERS, json={}).status_code == 200
    rerank = Mock(side_effect=lambda frozen: {**frozen, "sentinel": "original snapshot retained"})
    monkeypatch.setattr(lab.reranking, "rerank", rerank)
    after = client.post("/api/rerank", headers=HEADERS, json={"experiment_id": existing["experiment_id"]})
    assert after.status_code == 200
    assert after.json()["query"] == baseline["query"]
    assert after.json()["sentinel"] == "original snapshot retained"
    assert rerank.call_args.args[0] == existing
