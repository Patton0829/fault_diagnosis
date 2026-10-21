"""Explicit expanded-index serving and safe page summaries; all services are fake."""

from copy import deepcopy
from pathlib import Path
import socket
import sys
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
import pytest
import requests
import uvicorn

from qa_platform import deployment
from qa_platform import retrieval_lab as lab
from qa_platform.storage import kubernetes_search as search
from qa_platform.storage import sql_conversations


SUMMARY = {
    "index_name": "kubernetes_full_preflight_v1",
    "parent_count": 755,
    "document_count": 1920,
    "dimension": 4096,
    "notice": "资料尚未逐条审核，包含参考数据页。历史可能来自旧索引，新问题使用当前知识库。",
}


@pytest.fixture(autouse=True)
def no_real_services_or_secrets(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Expanded-index tests must not contact services or read credentials"
        )

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    monkeypatch.setattr(deployment, "read_config", forbidden)


def fake_server(monkeypatch):
    events = []
    engine, cache, store, app = Mock(), Mock(), object(), object()
    exclusive = Mock(side_effect=lambda: events.append("exclusive"))
    describe = Mock(
        side_effect=lambda _: events.append("describe") or deepcopy(SUMMARY)
    )
    open_services = Mock(
        side_effect=lambda _: events.append("services") or (engine, cache)
    )
    create = Mock(side_effect=lambda **_: events.append("create") or app)
    run = Mock(side_effect=lambda *_args, **_kwargs: events.append("run"))
    make_store = Mock(return_value=store)
    monkeypatch.setattr(deployment, "require_exclusive_app", exclusive)
    monkeypatch.setattr(search, "describe_index", describe)
    monkeypatch.setattr(deployment, "open_services", open_services)
    monkeypatch.setattr(lab, "create_app", create)
    monkeypatch.setattr(sql_conversations, "SQLConversationStore", make_store)
    monkeypatch.setattr(uvicorn, "run", run)
    return {
        "events": events,
        "engine": engine,
        "cache": cache,
        "store": store,
        "app": app,
        "exclusive": exclusive,
        "describe": describe,
        "open_services": open_services,
        "create": create,
        "run": run,
        "make_store": make_store,
    }


def test_serve_checks_exclusive_then_offline_index_then_services(monkeypatch, tmp_path):
    fake = fake_server(monkeypatch)
    local_dir = tmp_path / "secrets-not-read"
    index_dir = tmp_path / "index-not-opened"
    deployment.serve(local_dir, index_dir=index_dir)
    assert fake["events"] == ["exclusive", "describe", "services", "create", "run"]
    fake["describe"].assert_called_once_with(index_dir.absolute())
    fake["open_services"].assert_called_once_with(local_dir)
    fake["make_store"].assert_called_once_with(fake["engine"])
    fake["create"].assert_called_once_with(
        index_dir=index_dir.absolute(),
        knowledge_summary=SUMMARY,
        conversation_store=fake["store"],
        storage_backend="mysql",
        retrieval_cache=fake["cache"],
        cache_backend="redis",
        deployment_mode=True,
    )
    fake["run"].assert_called_once_with(
        fake["app"], host="127.0.0.1", port=8767, access_log=False
    )
    fake["engine"].dispose.assert_called_once_with()
    fake["cache"].close.assert_called_once_with()


def test_default_serve_preserves_original_create_app_signature(monkeypatch, tmp_path):
    fake = fake_server(monkeypatch)
    deployment.serve(tmp_path)
    assert fake["events"] == ["exclusive", "services", "create", "run"]
    fake["describe"].assert_not_called()
    assert "index_dir" not in fake["create"].call_args.kwargs
    assert "knowledge_summary" not in fake["create"].call_args.kwargs


def test_exclusivity_failure_precedes_even_index_and_service_inspection(
    monkeypatch, tmp_path
):
    fake = fake_server(monkeypatch)
    fake["exclusive"].side_effect = deployment.DeploymentError("synthetic active app")
    with pytest.raises(deployment.DeploymentError):
        deployment.serve(tmp_path, index_dir=tmp_path / "index")
    fake["describe"].assert_not_called()
    fake["open_services"].assert_not_called()
    fake["create"].assert_not_called()


@pytest.mark.parametrize("error", [ValueError, KeyError, TypeError, OSError])
def test_bad_index_fails_before_secrets_services_and_never_falls_back(
    monkeypatch, tmp_path, error
):
    fake = fake_server(monkeypatch)
    fake["describe"].side_effect = error("SYNTHETIC_PRIVATE_PATH_MUST_NOT_PRINT")
    with pytest.raises(deployment.DeploymentError) as caught:
        deployment.serve(tmp_path, index_dir=tmp_path / "index")
    assert "SYNTHETIC_PRIVATE_PATH_MUST_NOT_PRINT" not in str(caught.value)
    fake["describe"].assert_called_once()
    fake["open_services"].assert_not_called()
    fake["create"].assert_not_called()
    fake["run"].assert_not_called()


@pytest.mark.parametrize("stage", ["create", "run"])
def test_later_startup_failure_closes_only_opened_services(
    monkeypatch, tmp_path, stage
):
    fake = fake_server(monkeypatch)
    fake[stage].side_effect = RuntimeError("synthetic failure")
    with pytest.raises(RuntimeError):
        deployment.serve(tmp_path, index_dir=tmp_path / "index")
    fake["engine"].dispose.assert_called_once_with()
    fake["cache"].close.assert_called_once_with()


@pytest.mark.parametrize("action", [None, "prepare", "up", "check", "down"])
def test_index_cli_option_only_allowed_for_serve(monkeypatch, action):
    forbidden = Mock(
        side_effect=AssertionError("Invalid CLI must not dispatch any action")
    )
    for name in ("prepare", "compose", "check", "serve", "open_services"):
        monkeypatch.setattr(deployment, name, forbidden)
    argv = ["serve_compose.py", "--index-dir", "synthetic-index"]
    if action is not None:
        argv.append("--" + action)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        deployment.main()
    assert caught.value.code == 2
    forbidden.assert_not_called()


def test_explicit_serve_cli_passes_selected_index_only(monkeypatch):
    serve = Mock()
    monkeypatch.setattr(deployment, "serve", serve)
    monkeypatch.setattr(
        sys, "argv", ["serve_compose.py", "--serve", "--index-dir", "synthetic-index"]
    )
    assert deployment.main() == 0
    serve.assert_called_once_with(index_dir=Path("synthetic-index"))


def test_default_cli_is_still_zero_configuration_and_zero_clients(monkeypatch, capsys):
    forbidden = Mock(
        side_effect=AssertionError("Plan must not inspect index or services")
    )
    for name in ("prepare", "compose", "check", "serve", "open_services"):
        monkeypatch.setattr(deployment, name, forbidden)
    monkeypatch.setattr(search, "describe_index", forbidden)
    monkeypatch.setattr(sys, "argv", ["serve_compose.py"])
    assert deployment.main() == 0
    assert "计划完成" in capsys.readouterr().out
    forbidden.assert_not_called()


def test_qa_plan_exposes_scope_without_retrieving_or_model_requests(
    monkeypatch, tmp_path
):
    forbidden = Mock(
        side_effect=AssertionError("GET plan must never run RAG or inspect source")
    )
    monkeypatch.setattr(lab.hybrid, "prepare", forbidden)
    monkeypatch.setattr(lab.hybrid, "compare", forbidden)
    monkeypatch.setattr(lab.reranking, "rerank", forbidden)
    monkeypatch.setattr(lab.answering, "generate", forbidden)
    monkeypatch.setattr(search, "load_ready", forbidden)
    monkeypatch.setattr(search, "describe_index", forbidden)
    private_index = tmp_path / "PRIVATE_INDEX_PATH_MUST_NOT_LEAK"
    summary = deepcopy(SUMMARY)
    app = lab.create_app(private_index, knowledge_summary=summary)
    summary["document_count"] = 1  # Caller changes cannot mutate captured summary.
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/qa/plan")
        assert response.status_code == 200
        payload = response.json()
        assert payload["knowledge"] == SUMMARY
        assert "PRIVATE_INDEX_PATH_MUST_NOT_LEAK" not in response.text
        assert str(tmp_path) not in response.text
        payload["knowledge"]["document_count"] = 1
        assert client.get("/api/qa/plan").json()["knowledge"] == SUMMARY
    forbidden.assert_not_called()


def test_qa_plan_without_selected_scope_remains_backwards_compatible(tmp_path):
    with TestClient(
        lab.create_app(tmp_path / "nonexistent"), base_url="http://127.0.0.1"
    ) as client:
        payload = client.get("/api/qa/plan").json()
    assert "knowledge" not in payload
    assert "storage" in payload and "conversation" in payload


@pytest.mark.parametrize(
    "mutation",
    [
        {"document_count": True},
        {"document_count": 0},
        {"document_count": 5001},
        {"document_count": "1920"},
        {"parent_count": False},
        {"parent_count": 0},
        {"parent_count": 1921},
        {"parent_count": "755"},
        {"dimension": True},
        {"dimension": 1},
        {"dimension": 8193},
        {"dimension": "4096"},
        {"index_name": "../private/path"},
        {"index_name": "<script>alert(1)</script>"},
        {"index_name": "bad label"},
        {"index_name": ""},
        {"index_name": "x" * 81},
        {"notice": ""},
        {"notice": "x" * 401},
        {"source_path": "/private/not-public"},
    ],
)
def test_qa_summary_rejects_invalid_types_limits_labels_or_private_extras(
    tmp_path, mutation
):
    with pytest.raises(ValueError):
        lab.create_app(tmp_path / "index", knowledge_summary={**SUMMARY, **mutation})


def test_describe_index_only_reads_verified_scope_and_never_exposes_source_path(
    monkeypatch, tmp_path
):
    records = [{"parent_id": f"parent-{ordinal % 755}"} for ordinal in range(1920)]
    plan = {
        "document_count": 1920,
        "embedding": {"actual_dimension": 4096},
        "documents": records,
    }
    load = Mock(return_value=(plan, {}))
    monkeypatch.setattr(search, "load_ready", load)
    index_dir = tmp_path / "PRIVATE_PATH_MUST_NOT_LEAK" / "kubernetes_full_preflight_v1"
    summary = search.describe_index(index_dir)
    load.assert_called_once_with(index_dir.absolute())
    assert summary["document_count"] == 1920
    assert summary["parent_count"] == 755
    assert summary["dimension"] == 4096
    assert summary["index_name"] == "kubernetes_full_preflight_v1"
    assert "PRIVATE_PATH_MUST_NOT_LEAK" not in str(summary)
