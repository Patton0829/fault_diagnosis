"""Run the optional knowledge-scope label's offline browser contracts."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_node_knowledge_status_contracts():
    root = Path(__file__).resolve().parents[1]
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    completed = subprocess.run(
        [node, "--test", str(root / "tests" / "knowledge_status.test.cjs")],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
