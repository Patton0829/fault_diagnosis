"""Offline export regression entry point, including the fixed JS resource route."""

import builtins
import io
from pathlib import Path
import shutil
import socket
import subprocess
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
import pytest
import requests

from qa_platform import retrieval_lab as lab


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_node_export_contracts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable; install Node to run offline retrieval export tests")
    completed = subprocess.run(
        [node, "--test", str(PROJECT_ROOT / "tests" / "retrieval_export.test.cjs")],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_fixed_export_asset_is_javascript_and_never_prepares_or_compares(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Static export asset attempted network, .env, or retrieval")

    prepare = Mock(side_effect=forbidden)
    compare = Mock(side_effect=forbidden)
    monkeypatch.setattr(lab.hybrid, "prepare", prepare)
    monkeypatch.setattr(lab.hybrid, "compare", compare)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)

    def guarded_open(original):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                name = Path(file).name
                if name == ".env" or name.startswith(".env."):
                    forbidden()
            return original(file, *args, **kwargs)

        return guarded

    monkeypatch.setattr(builtins, "open", guarded_open(builtins.open))
    monkeypatch.setattr(io, "open", guarded_open(io.open))
    with TestClient(lab.create_app(tmp_path), base_url="http://127.0.0.1") as client:
        response = client.get("/assets/retrieval_export.js")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/javascript")
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"
        assert response.text == (PROJECT_ROOT / "web" / "retrieval_export.js").read_text(encoding="utf-8")
        assert "buildReport" in response.text
        # This is a single explicit resource, not a catch-all path to local files.
        assert client.get("/assets/unknown.js").status_code == 404
        assert client.get("/assets/retrieval_export.js/extra").status_code == 404
    prepare.assert_not_called()
    compare.assert_not_called()
