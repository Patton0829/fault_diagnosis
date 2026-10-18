"""Offline regression checks for the src migration and preserved corpus formats.

Run with pytest or ``python -B -m unittest discover -s tests``. Existing corpora
are only read; all synthetic bundles and deliberate changes use temporary files.
"""

from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

MODULES = (
    "qa_platform.settings",
    "qa_platform.providers.chat",
    "qa_platform.providers.cloud_embedding",
    "qa_platform.providers.cloud_reranker",
    "qa_platform.providers.local_embedding",
    "qa_platform.ingestion.railway",
    "qa_platform.ingestion.kubernetes_cleaning",
    "qa_platform.ingestion.kubernetes_chunking",
    "qa_platform.ingestion.kubernetes_cleanup",
    "qa_platform.ingestion.kubernetes_corpus",
    "qa_platform.ingestion.embedding_plan",
    "qa_platform.retrieval.bm25",
    "qa_platform.retrieval.dense",
    "qa_platform.retrieval.hybrid",
    "qa_platform.retrieval.pipeline",
    "qa_platform.retrieval.context_expansion",
    "qa_platform.storage.milvus_smoke",
    "qa_platform.storage.milvus_build",
    "qa_platform.storage.milvus_search",
    "qa_platform.storage.embedding_cache",
    "qa_platform.storage.kubernetes_milvus",
    "qa_platform.storage.kubernetes_search",
    "qa_platform.rag.answering",
    "qa_platform.rag.evaluation",
    "qa_platform.rag.milvus_demo",
    "qa_platform.web_app",
)
WRAPPERS = (
    "check_chat.py",
    "check_embedding.py",
    "check_reranker.py",
    "check_chunking.py",
    "check_bm25.py",
    "check_dense.py",
    "check_hybrid.py",
    "check_retrieval.py",
    "check_rag.py",
    "check_eval.py",
    "check_milvus.py",
    "build_milvus.py",
    "check_milvus_search.py",
    "check_milvus_rag.py",
    "check_k8s_documents.py",
    "check_k8s_chunking.py",
    "check_k8s_chunk_cleanup.py",
    "prepare_k8s_corpus.py",
    "check_k8s_embedding_plan.py",
    "check_local_embedding.py",
    "serve_data_lab.py",
    "build_k8s_embeddings.py",
    "build_k8s_milvus.py",
    "check_k8s_search.py",
    "check_k8s_context.py",
)
SNAPSHOT_HASHES = {
    "check_k8s_documents.py": "a531e08d98b0cfac1d1e7a1c1d013259302f2d6d2646688de6365c4271cc9330",
    "check_k8s_chunking.py": "6dfe08402002cf3157705470a783924a9a9e481132e88b7de342f7ad1e3daba0",
    "check_k8s_chunk_cleanup.py": "969b592f5f45a2ef9c5105c344bf81e5043e9504b0eca3a7d9dcfc000fc5351d",
    "prepare_k8s_corpus.py": "8944b26185dcbdf86fe5aa53b2b1959edb2d074236d4959f32309b1f975acfcc",
}
LEGACY_CORPUS = PROJECT_ROOT / "data/k8s_corpora/troubleshooting_v2"


def directory_hashes(directory):
    return {
        path.relative_to(directory).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def corpus_module():
    return importlib.import_module("qa_platform.ingestion.kubernetes_corpus")


def synthetic_batch():
    """One in-memory document with the same public bundle contract as ingestion."""
    corpus = corpus_module()
    text = "# 离线测试\n\n排查 DNS 服务。\n\n```yaml\nkind: Service\n```\n"
    commit = "a" * 40
    size = 40
    metadata = {
        "title": "离线测试",
        "source_url": "https://example.invalid/offline-test",
        "snapshot_commit": commit,
        "raw_sha256": corpus.digest(text.encode("utf-8")),
        "cleaned_sha256": corpus.digest(text.encode("utf-8")),
        "cleaner_version": corpus.CLEANER_VERSION,
        "corpus_relative_path": "tasks/debug/offline-test.md",
    }
    parent, chunks, audit = corpus.pack_page(
        {"cleaned": text, "metadata": metadata}, size
    )
    record = {
        "path": metadata["corpus_relative_path"],
        "status": "prepared",
        "messages": [],
        "raw_sha256": metadata["raw_sha256"],
        "parent_id": parent["id"],
        "chunks": len(chunks),
        **corpus.chunk_stats(chunks),
    }
    report = {
        "paths": [record["path"]],
        "records": [record],
        "summary": {"prepared": 1},
        "selected_files": 1,
        "prepared_documents": 1,
        "candidate_chunks": len(chunks),
        "source_commit": commit,
        "target_chars": size,
        **corpus.chunk_stats(chunks),
    }
    source = {
        "source": {"commit_sha": commit},
        "license": {"spdx_id": "CC-BY-4.0", "attribution": "Synthetic fixture"},
    }
    return {
        "parents": [parent],
        "chunks": chunks,
        "audits": [audit],
        "report": report,
        "source": source,
        "source_sha256": corpus.digest(corpus.json_text(source).encode("utf-8")),
        "license_bytes": b"Synthetic offline test fixture.\n",
        "config_sha256": "0" * 64,
        "commit": commit,
    }, size


class RepositoryLayoutTests(unittest.TestCase):
    def test_settings_resolve_from_package_location(self):
        settings = importlib.import_module("qa_platform.settings")
        self.assertEqual(settings.PROJECT_ROOT, PROJECT_ROOT)
        self.assertEqual(settings.PACKAGE_ROOT, SOURCE_ROOT / "qa_platform")
        self.assertEqual(settings.ENV_PATH, PROJECT_ROOT / "models/.env")
        self.assertEqual(settings.MODEL_DOCS_DIR, PROJECT_ROOT / "docs/model_apis")
        self.assertEqual(settings.KNOWLEDGE_DIR, PROJECT_ROOT / "knowledge")
        self.assertEqual(settings.DATA_DIR, PROJECT_ROOT / "data")
        self.assertEqual(settings.WEB_PAGE, PROJECT_ROOT / "web/data_lab.html")

    def test_application_imports_without_network_clients_or_env_reads(self):
        # A fresh isolated interpreter catches import side effects even if pytest
        # already imported the same modules. Credentials are never loaded.
        code = textwrap.dedent("""\
            from contextlib import ExitStack
            import builtins
            import importlib
            import io
            import json
            import os
            from pathlib import Path
            import socket
            import subprocess
            import sys
            import tempfile
            from unittest.mock import patch
            import dotenv
            import httpx
            import openai
            import requests

            project = Path(sys.argv[1]).resolve()
            sys.path.insert(0, str(project / 'src'))
            tempfile.tempdir = os.getcwd()
            original_open = builtins.open
            original_io_open = io.open
            original_os_open = os.open

            def deny(*args, **kwargs):
                raise AssertionError('Import attempted network, client setup, or subprocess')

            def check_path(file, writing):
                if isinstance(file, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(file))
                    if path.name == '.env':
                        raise AssertionError('Import attempted to read .env')
                    if writing and path.resolve().is_relative_to(project):
                        raise AssertionError('Import attempted to write project files')

            def guarded_open(file, mode='r', *args, **kwargs):
                check_path(file, any(flag in mode for flag in 'wax+'))
                return original_open(file, mode, *args, **kwargs)

            def guarded_io_open(file, mode='r', *args, **kwargs):
                check_path(file, any(flag in mode for flag in 'wax+'))
                return original_io_open(file, mode, *args, **kwargs)

            def guarded_os_open(file, flags, *args, **kwargs):
                check_path(file, bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
                return original_os_open(file, flags, *args, **kwargs)

            with ExitStack() as guards:
                for target in ('socket.socket.connect', 'socket.socket.connect_ex',
                               'socket.socket.bind', 'socket.socket.sendto',
                               'socket.create_connection', 'subprocess.Popen',
                               'dotenv.load_dotenv', 'httpx.Client.__init__',
                               'httpx.AsyncClient.__init__', 'openai.OpenAI.__init__',
                               'openai.AsyncOpenAI.__init__', 'requests.Session.__init__'):
                    guards.enter_context(patch(target, side_effect=deny))
                guards.enter_context(patch('builtins.open', guarded_open))
                guards.enter_context(patch('io.open', guarded_io_open))
                guards.enter_context(patch('os.open', guarded_os_open))
                modules = json.loads(sys.argv[2])
                for name in modules:
                    importlib.import_module(name)
            print(json.dumps(modules))
            """)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    code,
                    str(PROJECT_ROOT),
                    json.dumps(MODULES),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), list(MODULES))

    def test_all_compatibility_scripts_accept_help_from_another_directory(self):
        self.assertEqual(len(WRAPPERS), 25)
        with tempfile.TemporaryDirectory() as directory:
            for name in WRAPPERS:
                with self.subTest(script=name):
                    script = PROJECT_ROOT / "scripts" / name
                    self.assertTrue(script.is_file(), str(script))
                    result = subprocess.run(
                        [sys.executable, "-I", "-B", str(script), "--help"],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                        env={**os.environ, "TMPDIR": directory},
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn("usage:", result.stdout.lower())

    def test_historical_rule_sources_retain_original_bytes(self):
        legacy = importlib.import_module("qa_platform.ingestion.legacy_verifier")
        self.assertEqual(legacy.SNAPSHOT_HASHES, SNAPSHOT_HASHES)
        for name, expected in SNAPSHOT_HASHES.items():
            with self.subTest(snapshot=name):
                self.assertEqual(
                    sha256((legacy.SNAPSHOT_DIR / name).read_bytes()).hexdigest(),
                    expected,
                )
        legacy.verify_snapshot()

    def test_changed_historical_source_is_rejected_before_execution(self):
        legacy = importlib.import_module("qa_platform.ingestion.legacy_verifier")
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            shutil.copytree(legacy.SNAPSHOT_DIR, snapshot)
            changed = snapshot / "check_k8s_documents.py"
            changed.write_bytes(changed.read_bytes() + b"\n# deliberate test change\n")
            bundle = Path(directory) / "corpus"
            bundle.mkdir()
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "builder_version": "k8s-prepare-v1",
                        "rule_sha256": SNAPSHOT_HASHES,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(legacy, "SNAPSHOT_DIR", snapshot):
                with patch.object(legacy.subprocess, "run") as child:
                    with self.assertRaises(ValueError):
                        legacy.verify_legacy_directory(bundle)
                    child.assert_not_called()

    @unittest.skipUnless(
        (LEGACY_CORPUS / "manifest.json").is_file(), "v2 corpus absent"
    )
    def test_existing_v2_verifies_without_changing_any_file(self):
        before = directory_hashes(LEGACY_CORPUS)
        manifest = corpus_module().verify_directory(LEGACY_CORPUS)
        self.assertEqual(manifest["builder_version"], "k8s-prepare-v1")
        self.assertEqual(manifest["prepared_documents"], 11)
        self.assertEqual(manifest["candidate_chunks"], 156)
        self.assertEqual(directory_hashes(LEGACY_CORPUS), before)

    @unittest.skipUnless(
        (LEGACY_CORPUS / "manifest.json").is_file()
        and (PROJECT_ROOT / "knowledge/kubernetes_zh/upstream/hugo.toml").is_file(),
        "v2 corpus or upstream snapshot absent",
    )
    def test_current_processing_preserves_v2_content_after_layout_migration(self):
        corpus = corpus_module()
        batch = corpus.prepare_scope(800)
        self.assertEqual(len(batch["parents"]), 11)
        self.assertEqual(len(batch["chunks"]), 156)
        for name, key in (("documents.jsonl", "parents"), ("chunks.jsonl", "chunks")):
            expected = corpus.read_jsonl(LEGACY_CORPUS / name)
            # 新构建器明确记录新版本；仅允许这一项预期留痕发生改变。
            for record in expected:
                record["metadata"]["builder_version"] = corpus.BUILDER_VERSION
            self.assertEqual(corpus.normalized(batch[key]), expected)
        self.assertEqual(
            batch["audits"], corpus.read_jsonl(LEGACY_CORPUS / "audit.jsonl")
        )

    @unittest.skipUnless(
        (LEGACY_CORPUS / "manifest.json").is_file(), "v2 corpus absent"
    )
    def test_changed_v2_payload_and_rule_fingerprint_are_rejected(self):
        for change in ("payload", "rule_fingerprint"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle = Path(directory) / "corpus"
                shutil.copytree(LEGACY_CORPUS, bundle)
                if change == "payload":
                    path = bundle / "chunks.jsonl"
                    path.write_bytes(path.read_bytes() + b"\n")
                else:
                    path = bundle / "manifest.json"
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    manifest["rule_sha256"]["check_k8s_documents.py"] = "0" * 64
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    corpus_module().verify_directory(bundle)

    def test_new_bundle_roundtrip_and_existing_destination_refusal(self):
        corpus = corpus_module()
        batch, size = synthetic_batch()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-corpus"
            manifest = corpus.write_bundle(batch, output, size)
            before = directory_hashes(output)
            self.assertEqual(corpus.verify_directory(output), manifest)
            self.assertEqual(manifest["builder_version"], corpus.BUILDER_VERSION)
            self.assertEqual(manifest["prepared_documents"], 1)
            self.assertEqual(manifest["candidate_chunks"], len(batch["chunks"]))
            self.assertIs(manifest["ready_for_embedding"], False)
            self.assertIs(manifest["token_limit_checked"], False)
            self.assertEqual(manifest["model_api_calls"], 0)
            self.assertFalse((output / ".manifest.pending.json").exists())
            with self.assertRaises(FileExistsError):
                corpus.write_bundle(batch, output, size)
            self.assertEqual(directory_hashes(output), before)

    def test_new_bundle_detects_payload_and_rehashed_semantic_changes(self):
        corpus = corpus_module()
        for rehash in (False, True):
            with (
                self.subTest(rehash=rehash),
                tempfile.TemporaryDirectory() as directory,
            ):
                batch, size = synthetic_batch()
                output = Path(directory) / "new-corpus"
                manifest = corpus.write_bundle(batch, output, size)
                path = output / "audit.jsonl"
                audit = json.loads(path.read_text(encoding="utf-8"))
                audit["kept"][0]["start"] += 1
                data = (corpus.json_text(audit) + "\n").encode("utf-8")
                path.write_bytes(data)
                if rehash:
                    manifest["files"][path.name] = {
                        "sha256": corpus.digest(data),
                        "bytes": len(data),
                    }
                    (output / "manifest.json").write_text(
                        corpus.json_text(manifest), encoding="utf-8"
                    )
                with self.assertRaises(ValueError):
                    corpus.verify_directory(output)


if __name__ == "__main__":
    unittest.main()
