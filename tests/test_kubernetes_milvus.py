"""Offline Kubernetes cache import checks; all databases are temporary.

No test reads .env or contacts a model. The integration case launches the real
Milvus Lite SDK in two fresh processes with synthetic vectors only.
"""

from contextlib import redirect_stdout
from hashlib import sha256
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

cache = importlib.import_module("qa_platform.storage.embedding_cache")
module = importlib.import_module("qa_platform.storage.kubernetes_milvus")
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
MODEL = "offline-k8s-embedding"
BASE_URL = "http://127.0.0.1:11434"


def synthetic_job(directory, *, count=3):
    """Create the cache job contract without existing corpus or guide files."""
    directory = Path(directory)
    profile = {
        "model": MODEL,
        "endpoint": BASE_URL + "/api/embed",
        "documented_dimension": 4,
        "truncate": False,
        "keep_alive": "5m",
    }
    rows, chunks = [], {}
    for index in range(count):
        chunk_id = f"offline-k8s-{index}"
        text = f"合成 Pod DNS 正文 {index}\n\n```sh\nkubectl get pods\n```"
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "input_sha256": sha256(text.encode()).hexdigest(),
                "embedding_input_id": cache.fingerprint(
                    cache.canonical([chunk_id, text])
                ),
                "input_chars": len(text),
                "title": f"合成 Kubernetes 文档 {index}",
            }
        )
        chunks[chunk_id] = {
            "id": chunk_id,
            "page_content": text,
            "metadata": {
                "chunk_id": chunk_id,
                "kb_id": "kubernetes_zh",
                "parent_id": f"parent-{index}",
                "snapshot_commit": "c" * 40,
                "review_status": "upstream_unreviewed",
                "title": f"合成 Kubernetes 文档 {index}",
                "source_url": f"https://example.invalid/k8s/{index}",
                "section_path": ["故障排查", "DNS"],
                "source_line_ranges": [[4, 9], [15, 17]],
                "nested_provenance": {"license": "CC-BY-4.0", "test": True},
            },
        }
    return {
        "plan": {
            "profile": profile,
            "manifest": {"synthetic": True, "files": {}},
        },
        "scope": {
            "corpus_sha256": "c" * 64,
            "input_snapshot_sha256": "d" * 64,
            "input_contract_sha256": "e" * 64,
            "profile": profile,
            "policy": cache.POLICY_VERSION,
            "mode": "body-only",
            "dtype": "<f4",
            "dimension": 4,
            "expected_count": count,
        },
        "rows": rows,
        "chunks": chunks,
        "by_id": {row["chunk_id"]: row for row in rows},
        "ordinal": {row["chunk_id"]: index for index, row in enumerate(rows)},
        "corpus_path": directory.parent / "synthetic-corpus",
        "cache_dir": directory,
        "path": directory / cache.DATABASE_NAME,
        "base_url": BASE_URL,
        "batch_size": 8,
    }


def populate(job, *, complete=True, count=None):
    job["cache_dir"].mkdir()
    cache.create_cache(job, DIGEST)
    count = len(job["rows"]) if count is None else count
    matrix = np.asarray(
        [[1.0, index * 0.25, -0.5, 0.125] for index in range(count)], dtype="<f4"
    )
    with cache.connect_cache(job, writable=True) as connection:
        if count:
            cache.commit_batch(
                connection, job, job["rows"][:count], matrix, DIGEST, 17, 0.125
            )
        if complete:
            cache.finalize_cache(connection, job)
    return matrix


class FakeResponse:
    status_code = 200

    def __init__(self, digest):
        self.digest = digest
        self.closed = False

    def json(self):
        return {"models": [{"name": MODEL, "digest": self.digest}]}

    def close(self):
        self.closed = True


class TagsSession:
    """Only a single metadata GET is supported; POST raises immediately."""

    def __init__(self, digest=DIGEST):
        self.digest = digest
        self.trust_env = True
        self.calls = []
        self.response = FakeResponse(digest)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def request(self, method, url, *, json, timeout, allow_redirects):
        assert self.trust_env is False
        assert method == "GET" and url == BASE_URL + "/api/tags"
        assert json is None and timeout == (5, 15)
        assert allow_redirects is False
        assert not self.calls
        self.calls.append((method, url))
        return self.response


class KubernetesMilvusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="k8s-milvus-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.job = synthetic_job(self.root / "cache")

    def prepare(self):
        with patch.object(cache, "prepare_job", return_value=self.job):
            return module.prepare_import(self.job["corpus_path"])

    def minimal_plan(self):
        return {
            "milvus_packages": module.VERIFIED_VERSIONS,
            "embedding": {
                **self.job["plan"]["profile"],
                "server_model_digest": DIGEST,
            },
            "document_count": 3,
            "cache_path": str(self.job["path"]),
            "corpus_path": str(self.job["corpus_path"]),
            "document_snapshot_sha256": "d" * 64,
            "vector_snapshot_sha256": "e" * 64,
        }

    def test_missing_cache_is_rejected_without_creating_directory(self):
        with self.assertRaisesRegex(ValueError, "缓存不存在"):
            module.read_cache(self.job)
        self.assertFalse(self.job["cache_dir"].exists())

    def test_partial_cache_is_rejected_without_mutation(self):
        populate(self.job, complete=False, count=1)
        before = sha256(self.job["path"].read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "尚未完成"):
            module.read_cache(self.job)
        self.assertEqual(before, sha256(self.job["path"].read_bytes()).hexdigest())

    def test_full_rows_without_complete_marker_are_not_ready(self):
        populate(self.job, complete=False)
        with self.assertRaisesRegex(ValueError, "尚未完成"):
            module.read_cache(self.job)

    def test_full_body_metadata_and_float32_vector_hashes_are_preserved(self):
        matrix = populate(self.job)
        before = sha256(self.job["path"].read_bytes()).hexdigest()
        bundle = module.read_cache(self.job)
        self.assertEqual(bundle["snapshot"]["status"], "encoded_complete")
        for ordinal, (record, vector) in enumerate(
            zip(bundle["records"], bundle["vectors"])
        ):
            chunk = self.job["chunks"][record["chunk_id"]]
            self.assertEqual(record["text"], chunk["page_content"])
            self.assertEqual(record["metadata"], chunk["metadata"])
            for field in module.SCALARS:
                self.assertEqual(record[field], chunk["metadata"][field])
            self.assertEqual(vector.tobytes(), matrix[ordinal].tobytes())
            self.assertEqual(
                bundle["hashes"][record["chunk_id"]],
                sha256(matrix[ordinal].tobytes()).hexdigest(),
            )
        self.assertEqual(before, sha256(self.job["path"].read_bytes()).hexdigest())

    def test_non_kubernetes_metadata_is_rejected(self):
        self.job["chunks"]["offline-k8s-0"]["metadata"]["kb_id"] = "test"
        populate(self.job)
        with self.assertRaisesRegex(ValueError, "只允许独立Kubernetes"):
            module.read_cache(self.job)

    def test_utf8_text_bytes_not_character_count_determine_limit(self):
        text = "汉" * (module.TEXT_BYTES // 3 + 1)
        self.assertLess(len(text), module.TEXT_BYTES)
        self.job["chunks"]["offline-k8s-0"]["page_content"] = text
        populate(self.job)
        with self.assertRaisesRegex(ValueError, "正文为空或超过"):
            module.read_cache(self.job)

    def test_full_metadata_limit_does_not_silently_drop_source_fields(self):
        self.job["chunks"]["offline-k8s-0"]["metadata"]["long_source"] = (
            "中" * module.METADATA_BYTES
        )
        populate(self.job)
        with self.assertRaisesRegex(ValueError, "完整元数据超过"):
            module.read_cache(self.job)

    def test_overlong_scalar_and_invalid_commit_are_rejected(self):
        for field, value, expected in (
            ("parent_id", "a" * 65, "parent_id"),
            ("snapshot_commit", "g" * 40, "commit"),
        ):
            with self.subTest(field=field):
                job = synthetic_job(self.root / field)
                job["chunks"]["offline-k8s-0"]["metadata"][field] = value
                populate(job)
                with self.assertRaisesRegex(ValueError, expected):
                    module.read_cache(job)

    def test_prepare_import_binds_original_cache_contract_without_sdk(self):
        populate(self.job)
        with patch.object(cache, "prepare_job", return_value=self.job) as prepare:
            plan, bundle = module.prepare_import(
                self.job["corpus_path"], self.job["cache_dir"]
            )
        prepare.assert_called_once_with(
            self.job["corpus_path"], cache_dir=self.job["cache_dir"]
        )
        self.assertEqual(plan["document_count"], 3)
        self.assertEqual(plan["documents"], bundle["records"])
        self.assertEqual(plan["vector_sha256"], bundle["hashes"])
        self.assertEqual(plan["embedding"]["server_model_digest"], DIGEST)
        self.assertEqual(plan["embedding"]["actual_dimension"], 4)
        self.assertEqual(plan["cache_scope"], self.job["scope"])
        self.assertEqual(
            plan["document_snapshot_sha256"],
            cache.fingerprint(cache.canonical(bundle["records"])),
        )

    def test_digest_check_is_one_get_without_environment_proxy_or_post(self):
        session = TagsSession()
        result = module.check_live_digest(
            self.minimal_plan(), session_factory=lambda: session
        )
        self.assertEqual(session.calls, [("GET", BASE_URL + "/api/tags")])
        self.assertTrue(session.response.closed)
        self.assertTrue(session.closed)
        self.assertEqual(result["http_requests"], 1)
        self.assertEqual(result["observed_digest"], DIGEST)

    def test_changed_live_digest_rejects_after_one_get(self):
        session = TagsSession(OTHER_DIGEST)
        with self.assertRaisesRegex(ValueError, "digest与缓存不同"):
            module.check_live_digest(
                self.minimal_plan(), session_factory=lambda: session
            )
        self.assertEqual(len(session.calls), 1)

    def test_existing_target_refused_before_http(self):
        target = self.root / "existing-index"
        target.mkdir()
        sentinel = target / "keep.txt"
        sentinel.write_text("existing bytes", encoding="utf-8")
        with patch.object(module, "check_live_digest") as check:
            with self.assertRaisesRegex(ValueError, "索引目标已存在"):
                module.run_import(self.minimal_plan(), target)
        check.assert_not_called()
        self.assertEqual(sentinel.read_text(), "existing bytes")
        self.assertEqual(list(target.iterdir()), [sentinel])

    def test_wrong_dependency_versions_refused_before_http_or_directory(self):
        target = self.root / "wrong-version"
        plan = self.minimal_plan()
        plan["milvus_packages"] = {"pymilvus": "0.0", "milvus-lite": "0.0"}
        with patch.object(module, "check_live_digest") as check:
            with self.assertRaisesRegex(ValueError, "版本不符"):
                module.run_import(plan, target)
        check.assert_not_called()
        self.assertFalse(target.exists())

    def test_targets_inside_source_cache_or_corpus_refused_before_http(self):
        plan = self.minimal_plan()
        targets = [
            self.job["cache_dir"] / "bad-index",
            self.job["corpus_path"] / "nested" / "bad-index",
        ]
        for target in targets:
            with self.subTest(target=target):
                with patch.object(module, "check_live_digest") as check:
                    with self.assertRaisesRegex(ValueError, "缓存|语料"):
                        module.run_import(plan, target)
                check.assert_not_called()
                self.assertFalse(target.exists())

    def test_verify_subprocess_failure_keeps_evidence_and_publishes_no_ready(self):
        target = self.root / "failed-verification"
        results = [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 1),
        ]
        observed = {"observed_digest": DIGEST, "http_requests": 1}
        with (
            patch.object(module, "check_live_digest", return_value=observed) as check,
            patch.object(module.subprocess, "run", side_effect=results) as run,
        ):
            with self.assertRaisesRegex(ValueError, "verify阶段失败"):
                module.run_import(self.minimal_plan(), target)
        check.assert_called_once()
        self.assertEqual(run.call_count, 2)
        phases = [call.args[0][-3] for call in run.call_args_list]
        self.assertEqual(phases, ["write", "verify"])
        self.assertTrue((target / "build_plan.json").is_file())
        self.assertTrue((target / "model_observation.json").is_file())
        self.assertFalse((target / "manifest.json").exists())

    def test_mismatched_success_receipt_still_cannot_publish_ready(self):
        target = self.root / "bad-receipt"
        plan = self.minimal_plan()

        def fake_subprocess(argv, **_):
            if argv[-3] == "write":
                module.save_new_json(target / "build_result.json", {"writer_pid": 111})
            if argv[-3] == "verify":
                receipt = {
                    "status": "verified",
                    "reader_pid": 222,
                    "writer_pid": 111,
                    "document_count": 99,
                    "max_cosine_difference": 0.0,
                    "document_snapshot_sha256": plan["document_snapshot_sha256"],
                    "vector_snapshot_sha256": plan["vector_snapshot_sha256"],
                }
                module.save_new_json(target / "verification.json", receipt)
            return subprocess.CompletedProcess(argv, 0)

        with (
            patch.object(
                module, "check_live_digest", return_value={"http_requests": 1}
            ),
            patch.object(module.subprocess, "run", side_effect=fake_subprocess),
        ):
            with self.assertRaisesRegex(ValueError, "跨进程验证回执"):
                module.run_import(plan, target)
        self.assertFalse((target / "manifest.json").exists())
        self.assertTrue((target / "verification.json").is_file())

    def test_timeout_or_interrupt_keeps_failed_target_without_ready(self):
        for label, failure in (
            ("timeout", subprocess.TimeoutExpired("synthetic verifier", 180)),
            ("interrupt", KeyboardInterrupt()),
        ):
            with self.subTest(label=label):
                target = self.root / f"failed-{label}"
                plan = self.minimal_plan()
                results = [subprocess.CompletedProcess([], 0), failure]
                with (
                    patch.object(
                        module,
                        "check_live_digest",
                        return_value={"observed_digest": DIGEST, "http_requests": 1},
                    ),
                    patch.object(module.subprocess, "run", side_effect=results),
                ):
                    with self.assertRaises(type(failure)):
                        module.run_import(plan, target)
                self.assertTrue((target / "build_plan.json").is_file())
                self.assertTrue((target / "model_observation.json").is_file())
                self.assertFalse((target / "manifest.json").exists())

    def test_only_matching_cross_process_receipts_publish_ready(self):
        target = self.root / "matching-receipts"
        plan = self.minimal_plan()
        plan["documents"] = [{"synthetic": "not duplicated into manifest"}]
        receipt = {
            "status": "verified",
            "reader_pid": 222,
            "writer_pid": 111,
            "document_count": 3,
            "max_cosine_difference": 1e-7,
            "document_snapshot_sha256": plan["document_snapshot_sha256"],
            "vector_snapshot_sha256": plan["vector_snapshot_sha256"],
        }
        observation = {"observed_digest": DIGEST, "http_requests": 1}

        def fake_subprocess(argv, **_):
            if argv[-3] == "write":
                module.save_new_json(target / "build_result.json", {"writer_pid": 111})
            if argv[-3] == "verify":
                module.save_new_json(target / "verification.json", receipt)
            self.assertFalse((target / "manifest.json").exists())
            return subprocess.CompletedProcess(argv, 0)

        with (
            patch.object(
                module, "check_live_digest", return_value=observation
            ) as check,
            patch.object(module.subprocess, "run", side_effect=fake_subprocess) as run,
        ):
            module.run_import(plan, target)
        check.assert_called_once_with(plan)
        self.assertEqual(run.call_count, 2)
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(manifest["verification"], receipt)
        self.assertEqual(manifest["model_observation"], observation)
        self.assertNotIn("documents", manifest)
        self.assertEqual(
            manifest["vector_snapshot_sha256"], plan["vector_snapshot_sha256"]
        )

    def test_default_main_plan_does_not_run_or_create_target(self):
        populate(self.job)
        plan, bundle = self.prepare()
        target = self.root / "not-created"
        output = io.StringIO()
        with (
            patch.object(
                sys, "argv", ["build_k8s_milvus.py", "--index-dir", str(target)]
            ),
            patch.object(module, "prepare_import", return_value=(plan, bundle)),
            patch.object(module, "check_live_digest") as check,
            patch.object(module, "run_import") as run,
            redirect_stdout(output),
        ):
            self.assertEqual(module.main(), 0)
        check.assert_not_called()
        run.assert_not_called()
        self.assertFalse(target.exists())
        self.assertIn("实际HTTP=0", output.getvalue())
        self.assertIn("需1次文档Embedding", output.getvalue())

    def test_default_main_in_fresh_process_does_not_import_sdk_or_requests(self):
        populate(self.job)
        plan, _ = self.prepare()
        path = self.root / "synthetic-plan.json"
        path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        target = self.root / "fresh-process-no-target"
        program = r"""
import builtins, importlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def guard(name, *args, **kwargs):
    assert not name.startswith(("pymilvus", "milvus_lite", "requests", "dotenv")), name
    return original_import(name, *args, **kwargs)
builtins.__import__ = guard
m = importlib.import_module("qa_platform.storage.kubernetes_milvus")
plan = json.loads(pathlib.Path(sys.argv[2]).read_text())
target = pathlib.Path(sys.argv[3])
m.prepare_import = lambda *args: (plan, {})
sys.argv = ["build_k8s_milvus.py", "--index-dir", str(target)]
assert m.main() == 0
assert not target.exists()
assert not any(name.startswith(("pymilvus", "milvus_lite", "requests", "dotenv")) for name in sys.modules)
"""
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                program,
                str(SOURCE_ROOT),
                str(path),
                str(target),
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(target.exists())

    @unittest.skipUnless(
        module.installed_versions() == module.VERIFIED_VERSIONS,
        "requires the project's verified Milvus Lite dependency versions",
    )
    def test_real_milvus_lite_preserves_synthetic_rows_across_processes(self):
        matrix = populate(self.job)
        plan, bundle = self.prepare()
        target = self.root / "real-lite-synthetic"
        target.mkdir()
        payload = {
            "plan": plan,
            "records": bundle["records"],
            "vectors": matrix.tolist(),
        }
        payload_path = self.root / "integration-fixture.json"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        program = r"""
import importlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
m = importlib.import_module("qa_platform.storage.kubernetes_milvus")
fixture = json.loads(pathlib.Path(sys.argv[2]).read_text())
plan = fixture["plan"]
bundle = {"records": fixture["records"], "vectors": np.asarray(fixture["vectors"], dtype="<f4")}
m.load_build = lambda _: (plan, bundle)
def no_http(*args, **kwargs):
    raise AssertionError("Model HTTP is forbidden in this integration test")
m.http_json = no_http
getattr(m, sys.argv[4] + "_index")(pathlib.Path(sys.argv[3]))
"""
        processes = []
        for phase in ("write", "verify"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    program,
                    str(SOURCE_ROOT),
                    str(payload_path),
                    str(target),
                    phase,
                ],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            processes.append(result)
        written = json.loads((target / "build_result.json").read_text())
        verified = json.loads((target / "verification.json").read_text())
        self.assertEqual(written["document_count"], 3)
        self.assertEqual(verified["document_count"], 3)
        self.assertEqual(verified["status"], "verified")
        self.assertNotEqual(verified["writer_pid"], verified["reader_pid"])
        self.assertEqual(
            verified["vector_snapshot_sha256"], plan["vector_snapshot_sha256"]
        )
        self.assertLessEqual(verified["max_cosine_difference"], 1e-5)
        self.assertFalse((target / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
