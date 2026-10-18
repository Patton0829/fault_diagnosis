"""Run the conditional-workflow page's offline Node.js interaction tests."""

from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_node_branching_contracts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable; install Node to run offline branching UI tests")
    completed = subprocess.run(
        [node, "--test", str(PROJECT_ROOT / "tests" / "branching_lab.test.cjs")],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
