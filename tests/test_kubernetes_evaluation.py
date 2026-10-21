"""Offline regression checks for the Kubernetes evaluation observer and CLI.

The graph can be synthetic or the real compiled LangGraph with provider functions
replaced. These tests never connect to a model, Milvus, Redis, or MySQL.
"""

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import io
import json
from pathlib import Path
import socket
from unittest.mock import Mock

import dotenv
import httpx
import pytest
import requests

from qa_platform.rag import kubernetes_evaluation as evaluation
from qa_platform.workflows import live_rag as live


MODEL = "qwen3.8-flash"
CASE = evaluation.EvalCase(
    "synthetic", "Synthetic question?", ("target",), ("answered", "partial"), (),
    ("Manually assess whether the quote supports the claim.",),
)
PRIVATE = "PRIVATE_FAILURE_SECRET_AND_PATH_MUST_NOT_LEAK"
COUNTS = {"embedding": 1, "http": 5, "reranker": 1, "chat": 1}


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation test attempted external IO or credential access")

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
                if name == ".env" or name.endswith(".env") or name.startswith(".env."):
                    forbidden()
            return original(file, *args, **kwargs)
        return guarded

    monkeypatch.setattr(builtins, "open", guard_open(original_open))
    monkeypatch.setattr(io, "open", guard_open(original_io_open))


def stages(case=CASE, *, answer=None):
    """Small complete observer payloads, including a real locatable citation."""
    retrieved = {
        "query": case.question,
        "routes": {
            "bm25": [{"chunk_id": "other"}, {"chunk_id": "target"}],
            "dense": [{"chunk_id": "target"}, {"chunk_id": "other"}],
            "rrf": [{"chunk_id": "other"}, {"chunk_id": "target"}],
        },
        "request_counts": {"embedding": 1, "http": 3, "reranker": 0, "chat": 0},
    }
    reranked = deepcopy(retrieved)
    reranked["routes"]["reranker"] = [{"chunk_id": "target"}, {"chunk_id": "other"}]
    reranked["request_counts"].update(http=4, reranker=1)
    final = deepcopy(reranked)
    final["request_counts"] = dict(COUNTS)
    final["answer_generation"] = {
        "model": MODEL,
        "answer": answer or {
            "status": "partial", "claims": [{
                "text": "Synthetic supported claim.",
                "evidence": [{"source_id": "S1", "quote_id": "Q1"}],
            }], "missing_info": ["Actual environment remains unverified."],
        },
        "sources": {
            "S1": {
                "chunk_id": "target", "text": "Synthetic original quote.",
                "quotes": {"Q1": "Synthetic original quote."},
                "metadata": {"title": "Synthetic source", "section_path": ["Section"],
                             "source_url": "https://example.invalid/synthetic", "review_status": "unreviewed",
                             "kubernetes_version": "unknown", "source_path": PRIVATE, "vector": [PRIVATE]},
            },
        },
    }
    return [{name: {"result": result}}
            for name, result in zip(live.PIPELINE, (retrieved, reranked, final))]


class FakeGraph:
    def __init__(self, updates, *, failure=None):
        self.updates, self.failure, self.calls = updates, failure, []

    def stream(self, initial, *, config, stream_mode):
        self.calls.append((deepcopy(initial), deepcopy(config), stream_mode))
        yield from deepcopy(self.updates)
        if self.failure:
            raise self.failure


def execute(tmp_path, *, updates=None, case=CASE, failure=None):
    graph = FakeGraph(stages(case) if updates is None else updates, failure=failure)
    factory = Mock(return_value=graph)
    row = evaluation.evaluate_case(case, tmp_path, MODEL, graph_factory=factory)
    factory.assert_called_once_with(tmp_path)  # No Redis or conversation store injected.
    assert len(graph.calls) == 1  # No automatic retry, even on a late failure.
    assert graph.calls[0] == ({"query": case.question, "model": MODEL},
                              {"callbacks": [], "recursion_limit": 10}, "updates")
    return row


def test_cases_are_three_frozen_development_examples_not_an_accuracy_dataset():
    assert [case.case_id for case in evaluation.CASES] == ["namespace", "dns_config", "out_of_scope"]
    assert all(case.review_points for case in evaluation.CASES)
    assert len(evaluation.CASES[0].relevant_chunk_ids) == 2
    assert evaluation.CASES[-1].relevant_chunk_ids == ()
    with pytest.raises(FrozenInstanceError):
        CASE.question = "changed"


@pytest.mark.parametrize("ids,rank,hit,rr", [
    (["target", "other"], 1, True, 1.0),
    (["a", "b", "c", "d", "target", "f"], 5, True, 0.2),
    (["a", "b", "c", "d", "e", "target"], 6, False, 0.0),
    (["a", "b"], None, False, 0.0),
    ([], None, False, 0.0),
])
def test_ranking_metrics_are_truncated_at_k_but_keep_full_first_rank(ids, rank, hit, rr):
    assert evaluation.ranking_metrics(ids, ("target",)) == {
        "top_chunk_ids": ids[:5], "first_relevant_rank": rank,
        "hit_at_k": hit, "reciprocal_rank_at_k": rr, "completed": True,
    }


def test_multiple_partial_relevant_labels_count_only_first_hit():
    result = evaluation.ranking_metrics(["other", "second", "first"], ("first", "second"), k=2)
    assert result["first_relevant_rank"] == 2
    assert result["reciprocal_rank_at_k"] == 0.5
    assert result["top_chunk_ids"] == ["other", "second"]


def test_unanswerable_case_has_no_retrieval_denominator():
    metric = evaluation.ranking_metrics(["other"], ())
    assert metric["hit_at_k"] is metric["reciprocal_rank_at_k"] is None


@pytest.mark.parametrize("ids,k", [
    (["a", "a"], 5), ([""], 5), ([None], 5), ([123], 5), ("target", 5),
    (["target"], 0), (["target"], -1), (["target"], True), (["target"], 1.5),
])
def test_invalid_rankings_are_rejected(ids, k):
    with pytest.raises(ValueError):
        evaluation.ranking_metrics(ids, ("target",), k)


def test_success_captures_four_routes_and_only_validated_public_answer(tmp_path):
    row = execute(tmp_path)
    assert row["technical_status"] == "completed"
    assert row["automatic_checks_pass"] is True
    assert set(row["retrieval"]) == set(evaluation.ROUTES)
    assert row["retrieval"]["bm25"]["reciprocal_rank_at_k"] == 0.5
    assert row["retrieval"]["reranker"]["reciprocal_rank_at_k"] == 1.0
    assert row["checks"] == {
        "expected_status_match": True, "citation_locators_valid": True,
        "cites_known_relevant_chunk": True, "forbidden_literals_found": [], "refusal_rule_pass": None,
    }
    assert row["manual_review"]["status"] == "pending"
    assert CASE.review_points[0] in row["manual_review"]["points"]
    assert row["error"] is None
    assert row["elapsed_seconds"] >= 0
    assert row["completed_request_counts"] == COUNTS
    assert PRIVATE not in json.dumps(row)


@pytest.mark.parametrize("completed", [0, 1, 2])
def test_late_execution_failure_preserves_prior_rankings_and_not_draft(tmp_path, completed):
    row = execute(tmp_path, updates=stages()[:completed], failure=RuntimeError(PRIVATE))
    assert row["technical_status"] == "failed"
    assert row["automatic_checks_pass"] is False
    assert row["answer"] is None and row["sources"] == {} and row["checks"] is None
    assert row["error"]["stage"] == live.PIPELINE[completed]
    assert row["error"]["kind"] == "execution_error"
    assert row["failed_request_counts_unknown"] is True
    assert row["manual_review"]["status"] == "pending"
    assert row["retrieval"]["bm25"]["completed"] is (completed >= 1)
    assert row["retrieval"]["reranker"]["completed"] is (completed >= 2)
    assert PRIVATE not in json.dumps(row)


def test_failure_before_graph_stream_is_also_sanitized_without_retry(tmp_path):
    factory = Mock(side_effect=RuntimeError(PRIVATE))
    row = evaluation.evaluate_case(CASE, tmp_path, MODEL, graph_factory=factory)
    factory.assert_called_once_with(tmp_path)
    assert row["technical_status"] == "failed"
    assert row["error"]["stage"] == "retrieve"
    assert PRIVATE not in json.dumps(row)


@pytest.mark.parametrize("mutation", ["order", "extra", "incomplete", "query", "negative_count", "bool_count", "duplicate"])
def test_malformed_graph_events_fail_closed(tmp_path, mutation):
    updates = stages()
    if mutation == "order":
        updates = [updates[1], updates[0], updates[2]]
    elif mutation == "extra":
        updates.append(deepcopy(updates[-1]))
    elif mutation == "incomplete":
        updates = updates[:2]
    elif mutation == "query":
        updates[1]["rerank"]["result"]["query"] = PRIVATE
    elif mutation in {"negative_count", "bool_count"}:
        updates[0]["retrieve"]["result"]["request_counts"]["embedding"] = -1 if mutation == "negative_count" else True
    elif mutation == "duplicate":
        updates[0]["retrieve"]["result"]["routes"]["bm25"].append({"chunk_id": "target"})
    row = execute(tmp_path, updates=updates)
    assert row["technical_status"] == "failed" and row["automatic_checks_pass"] is False
    assert row["answer"] is None and row["sources"] == {}
    assert row["error"]["kind"] == "validation_error"
    assert PRIVATE not in json.dumps(row)


@pytest.mark.parametrize("stage", list(live.PIPELINE))
@pytest.mark.parametrize("mutation", ["zero_embedding", "http", "extra_count", "cache"])
def test_each_stage_requires_exact_fresh_pipeline_counts_without_cache(tmp_path, stage, mutation):
    updates = stages()
    result = updates[list(live.PIPELINE).index(stage)][stage]["result"]
    if mutation == "zero_embedding":
        result["request_counts"]["embedding"] = 0
    elif mutation == "http":
        result["request_counts"]["http"] += 1
    elif mutation == "extra_count":
        result["request_counts"]["extra"] = 1
    else:
        result["cache"] = {"backend": "redis", "status": "hit", "ttl_seconds": 300}
    row = execute(tmp_path, updates=updates)
    assert row["technical_status"] == "failed" and row["automatic_checks_pass"] is False
    assert row["error"]["stage"] == stage
    assert row["answer"] is None and row["sources"] == {}


@pytest.mark.parametrize("mutation", ["model", "missing_quote", "wrong_quote", "extra_field"])
def test_final_public_validation_rejects_unvalidated_draft_and_keeps_retrieval(tmp_path, mutation):
    updates = stages()
    generation = updates[-1]["answer"]["result"]["answer_generation"]
    if mutation == "model":
        generation["model"] = "wrong-model"
    elif mutation == "missing_quote":
        generation["sources"]["S1"]["quotes"] = {}
    elif mutation == "wrong_quote":
        generation["sources"]["S1"]["quotes"]["Q1"] = PRIVATE
    else:
        generation["answer"][PRIVATE] = PRIVATE
    row = execute(tmp_path, updates=updates)
    assert row["technical_status"] == "failed" and row["answer"] is None
    assert row["retrieval"]["reranker"]["hit_at_k"] is True
    assert PRIVATE not in json.dumps(row)


def test_expected_status_mismatch_is_technical_completion_not_automatic_pass(tmp_path):
    case = replace(CASE, expected_statuses=("answered",))
    row = execute(tmp_path, case=case)
    assert row["technical_status"] == "completed"
    assert row["checks"]["expected_status_match"] is False
    assert row["automatic_checks_pass"] is False
    assert row["manual_review"]["status"] == "pending"


@pytest.mark.parametrize("leak", ["", "lOnDoN", "伦敦"])
def test_closed_book_refusal_checks_free_text_including_missing_info(tmp_path, leak):
    case = evaluation.CASES[-1]
    answer = {"status": "insufficient_evidence", "claims": [],
              "missing_info": ["No geographic evidence is present." + leak]}
    row = execute(tmp_path, case=case, updates=stages(case, answer=answer))
    assert row["technical_status"] == "completed"
    assert row["checks"]["refusal_rule_pass"] is (not bool(leak))
    assert row["automatic_checks_pass"] is (not bool(leak))
    assert row["checks"]["citation_locators_valid"] is None
    assert row["sources"] == {}
    assert all(metric["hit_at_k"] is None for metric in row["retrieval"].values())
    assert row["manual_review"]["status"] == "pending"


def test_summarize_keeps_failed_positive_cases_and_excludes_refusal(tmp_path):
    good = execute(tmp_path)
    failed_generation = execute(tmp_path, updates=stages()[:2], failure=RuntimeError(PRIVATE))
    failed_retrieval = execute(tmp_path, updates=[], failure=RuntimeError(PRIVATE))
    refusal_case = evaluation.CASES[-1]
    refusal = execute(tmp_path, case=refusal_case, updates=stages(refusal_case, answer={
        "status": "insufficient_evidence", "claims": [], "missing_info": ["No relevant evidence."],
    }))
    summary = evaluation.summarize([good, failed_generation, failed_retrieval, refusal])
    assert summary["total"] == 4 and summary["completed"] == summary["failed"] == 2
    assert summary["automatic_checks_passed"] == 2 and summary["manual_review_pending"] == 4
    assert summary["retrieval"]["reranker"] == {
        "denominator": 3, "hits": 2, "hit_at_k": 2 / 3, "mrr_at_k": 2 / 3,
    }
    assert summary["retrieval"]["bm25"]["mrr_at_k"] == 1 / 3


def test_summarize_no_positive_cases_uses_none_instead_of_perfect_score(tmp_path):
    case = evaluation.CASES[-1]
    row = execute(tmp_path, case=case, updates=[], failure=RuntimeError(PRIVATE))
    summary = evaluation.summarize([row])
    assert all(metric == {"denominator": 0, "hits": 0, "hit_at_k": None, "mrr_at_k": None}
               for metric in summary["retrieval"].values())
    assert evaluation.summarize([])["total"] == 0


def synthetic_prepared(tmp_path, cases=None):
    return {"cases": list(cases or [CASE]), "check": {
        "index_dir": tmp_path / "synthetic-index",
        "plan": {"document_snapshot_sha256": evaluation.DOCUMENT_SNAPSHOT,
                 "vector_snapshot_sha256": "synthetic-vector-snapshot",
                 "embedding": {"model": "synthetic-embedding"},
                 "documents": [{"snapshot_commit": evaluation.SNAPSHOT_COMMIT}]},
        "records": [{"chunk_id": chunk_id, "text": excerpt}
                    for chunk_id, excerpt in evaluation.LABEL_EXCERPTS.items()],
    }}


def test_prepare_only_checks_snapshot_and_selected_labels_without_running_graph(monkeypatch, tmp_path):
    prepared = synthetic_prepared(tmp_path)
    prepare = Mock(return_value=prepared["check"])
    build = Mock(side_effect=AssertionError("Offline preflight built the graph"))
    monkeypatch.setattr(live.hybrid, "prepare", prepare)
    monkeypatch.setattr(live, "build_graph", build)
    result = evaluation.prepare_suite(tmp_path, ["out_of_scope", "namespace"])
    assert [case.case_id for case in result["cases"]] == ["namespace", "out_of_scope"]
    prepare.assert_called_once_with(tmp_path, evaluation.CASES[0].question)
    build.assert_not_called()


@pytest.mark.parametrize("mutation", ["document_snapshot", "commit", "excerpt"])
def test_prepare_rejects_stale_or_changed_gold_without_model_requests(monkeypatch, tmp_path, mutation):
    check = synthetic_prepared(tmp_path)["check"]
    if mutation == "document_snapshot":
        check["plan"]["document_snapshot_sha256"] = "different"
    elif mutation == "commit":
        check["plan"]["documents"][0]["snapshot_commit"] = "different"
    else:
        check["records"][0]["text"] = "Changed original paragraph"
    monkeypatch.setattr(live.hybrid, "prepare", Mock(return_value=check))
    with pytest.raises(ValueError):
        evaluation.prepare_suite(tmp_path)


@pytest.mark.parametrize("selected", [[], ["not-a-case"]])
def test_prepare_invalid_selection_does_not_even_inspect_index(monkeypatch, tmp_path, selected):
    prepare = Mock(side_effect=AssertionError("Invalid selection inspected index"))
    monkeypatch.setattr(live.hybrid, "prepare", prepare)
    with pytest.raises(ValueError):
        evaluation.prepare_suite(tmp_path, selected)
    prepare.assert_not_called()


def test_default_cli_plan_does_not_reserve_output_or_run_graph(monkeypatch, tmp_path, capsys):
    prepared = synthetic_prepared(tmp_path)
    prepare = Mock(return_value=prepared)
    run = Mock(side_effect=AssertionError("Pure plan ran evaluation"))
    monkeypatch.setattr(evaluation, "prepare_suite", prepare)
    monkeypatch.setattr(evaluation, "run_suite", run)
    destination = tmp_path / "must-not-exist"
    assert evaluation.main(["--index-dir", str(tmp_path), "--output-dir", str(destination)]) == 0
    prepare.assert_called_once_with(tmp_path, None)
    run.assert_not_called()
    assert not destination.exists()
    assert "未读密钥" in capsys.readouterr().out


def test_cli_run_checks_exclusive_index_before_output_or_model(monkeypatch, tmp_path, capsys):
    from qa_platform import deployment

    monkeypatch.setattr(evaluation, "prepare_suite", Mock(return_value=synthetic_prepared(tmp_path)))
    guard = Mock(side_effect=RuntimeError(PRIVATE))
    run = Mock(side_effect=AssertionError("Busy app reached evaluation"))
    monkeypatch.setattr(deployment, "require_exclusive_app", guard)
    monkeypatch.setattr(evaluation, "run_suite", run)
    destination = tmp_path / "not-reserved"
    assert evaluation.main(["--run", "--output-dir", str(destination)]) == 1
    guard.assert_called_once_with()
    run.assert_not_called()
    assert not destination.exists()
    assert PRIVATE not in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_cli_never_overwrites_existing_output_or_calls_models(monkeypatch, tmp_path, kind):
    from qa_platform import deployment

    output = tmp_path / "existing"
    if kind == "directory":
        output.mkdir()
        (output / "sentinel").write_text("preserve", encoding="utf-8")
    elif kind == "file":
        output.write_text("preserve", encoding="utf-8")
    else:
        output.symlink_to(tmp_path / "not-created-target", target_is_directory=True)
    monkeypatch.setattr(evaluation, "prepare_suite", Mock(return_value=synthetic_prepared(tmp_path)))
    monkeypatch.setattr(deployment, "require_exclusive_app", lambda: None)
    run = Mock(side_effect=AssertionError("Existing output reached models"))
    monkeypatch.setattr(evaluation, "run_suite", run)
    assert evaluation.main(["--run", "--output-dir", str(output)]) == 1
    run.assert_not_called()
    if kind == "directory":
        assert (output / "sentinel").read_text(encoding="utf-8") == "preserve"
    elif kind == "file":
        assert output.read_text(encoding="utf-8") == "preserve"
    else:
        assert output.is_symlink() and not output.resolve().exists()


@pytest.mark.parametrize("failed", [False, True])
def test_cli_writes_both_offline_reports_once_and_returns_technical_status(monkeypatch, tmp_path, failed):
    from qa_platform import deployment

    prepared = synthetic_prepared(tmp_path)
    row = execute(tmp_path, updates=[] if failed else None,
                  failure=RuntimeError(PRIVATE) if failed else None)
    monkeypatch.setattr(evaluation, "evaluate_case", Mock(return_value=row))
    report = evaluation.run_suite(prepared, MODEL)
    monkeypatch.setattr(evaluation, "prepare_suite", Mock(return_value=prepared))
    run = Mock(return_value=report)
    monkeypatch.setattr(evaluation, "run_suite", run)
    guard = Mock()
    monkeypatch.setattr(deployment, "require_exclusive_app", guard)
    output = tmp_path / "new-report"
    assert evaluation.main(["--run", "--output-dir", str(output)]) == int(failed)
    guard.assert_called_once_with()
    run.assert_called_once_with(prepared, MODEL)
    assert set(path.name for path in output.iterdir()) == {"report.json", "report.html"}
    assert output.stat().st_mode & 0o777 == 0o700
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["summary"] == report["summary"]
    assert "人工审核" in (output / "report.html").read_text(encoding="utf-8")
    assert PRIVATE not in (output / "report.json").read_text(encoding="utf-8")


def test_run_suite_report_records_contract_and_never_marks_human_review_complete(monkeypatch, tmp_path):
    row = execute(tmp_path)
    evaluate = Mock(return_value=row)
    monkeypatch.setattr(evaluation, "evaluate_case", evaluate)
    report = evaluation.run_suite(synthetic_prepared(tmp_path), MODEL)
    evaluate.assert_called_once_with(CASE, tmp_path / "synthetic-index", MODEL)
    assert report["schema_version"] == "k8s-eval/v1"
    assert report["cache"] == "disabled" and report["conversation_storage"] == "not_used"
    assert report["cases"][0]["manual_review"]["status"] == "pending"
    assert report["summary"]["manual_review_pending"] == 1
    assert report["embedding_model"] == "synthetic-embedding"
    assert all(len(digest) == 64 for digest in report["source_versions"].values())
    assert report["cases"][0]["case"] == asdict(CASE)


def test_default_factory_runs_actual_graph_with_tracing_off_and_no_cache(monkeypatch, tmp_path):
    from langgraph.graph.state import CompiledStateGraph
    from langsmith.run_helpers import get_tracing_context

    updates = stages()
    calls = []

    def prepare(index_dir, question):
        calls.append(("prepare", index_dir, question, get_tracing_context()["enabled"]))
        return {"synthetic": True}

    def retrieve(check, mode):
        calls.append(("retrieve", mode))
        return deepcopy(updates[0]["retrieve"]["result"])

    def rerank(result):
        calls.append(("rerank", result["query"]))
        return deepcopy(updates[1]["rerank"]["result"])

    def answer(result, model):
        calls.append(("answer", model))
        return deepcopy(updates[2]["answer"]["result"])

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setattr(live.hybrid, "prepare", prepare)
    monkeypatch.setattr(live.hybrid, "compare", retrieve)
    monkeypatch.setattr(live.reranking, "rerank", rerank)
    monkeypatch.setattr(live.answering, "generate", answer)
    original = live.build_graph
    built = []

    def build(index_dir):
        graph = original(index_dir)
        assert isinstance(graph, CompiledStateGraph)
        assert graph.checkpointer is graph.cache is graph.store is None
        assert all(node.retry_policy is None for node in graph.nodes.values())
        built.append(graph)
        return graph

    monkeypatch.setattr(live, "build_graph", build)
    row = evaluation.evaluate_case(CASE, tmp_path, MODEL)
    assert row["technical_status"] == "completed" and len(built) == 1
    assert [call[0] for call in calls] == ["prepare", "retrieve", "rerank", "answer"]
    assert calls[0] == ("prepare", tmp_path, CASE.question, False)
