"""Offline cache checkpoints and corruption checks using only temporary SQLite files.

Every HTTP response is synthetic. These tests neither load a corpus nor inspect
environment files, and cannot contact a model server.
"""

from contextlib import closing
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

local_embedding = importlib.import_module("qa_platform.providers.local_embedding")
local_endpoint_base = local_embedding.local_endpoint_base
model_digest = local_embedding.model_digest
cache = importlib.import_module("qa_platform.storage.embedding_cache")


DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
MODEL = "offline-test-embedding"
BASE_URL = "http://127.0.0.1:11434"


def synthetic_job(directory, *, count=5, batch_size=2, dimension=2):
    """The public job contract, without reading any existing project data."""
    profile = {
        "model": MODEL,
        "endpoint": BASE_URL + "/api/embed",
        "documented_dimension": dimension,
        "truncate": False,
        "keep_alive": "5m",
    }
    rows, chunks = [], {}
    for index in range(count):
        chunk_id = f"chunk-{index:03d}"
        text = f"离线合成正文 {index}：排查 Pod DNS。"
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "input_sha256": sha256(text.encode("utf-8")).hexdigest(),
                "embedding_input_id": cache.fingerprint(
                    cache.canonical([chunk_id, text])
                ),
                "input_chars": len(text),
                "title": f"离线文档 {index}",
            }
        )
        chunks[chunk_id] = {
            "id": chunk_id,
            "text": text,
            "metadata": {
                "title": f"离线文档 {index}",
                "source_url": f"https://example.invalid/{index}",
                "section_path": ["故障排查", "DNS"],
                "ordinal_in_parent": index,
            },
        }
    scope = {
        "corpus_sha256": "c" * 64,
        "input_snapshot_sha256": "d" * 64,
        "input_contract_sha256": "e" * 64,
        "profile": profile,
        "policy": cache.POLICY_VERSION,
        "mode": "body-only",
        "dtype": "<f4",
        "dimension": dimension,
        "expected_count": count,
    }
    directory = Path(directory)
    return {
        "plan": {
            "profile": profile,
            "manifest": {
                "files": {"chunks.jsonl": {"sha256": "f" * 64}},
                "synthetic": True,
            },
        },
        "scope": scope,
        "rows": rows,
        "chunks": chunks,
        "by_id": {row["chunk_id"]: row for row in rows},
        "ordinal": {row["chunk_id"]: index for index, row in enumerate(rows)},
        "corpus_path": directory.parent / "synthetic-corpus-never-created",
        "cache_dir": directory,
        "path": directory / cache.DATABASE_NAME,
        "base_url": BASE_URL,
        "batch_size": batch_size,
    }


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.closed = False

    def json(self):
        return self.payload

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, *, digests=None, fail_post=None, tokens=17, dimension=2):
        self.digests = list(digests or [DIGEST])
        self.fail_post = fail_post
        self.tokens = tokens
        self.dimension = dimension
        self.trust_env = True
        self.calls = []
        self.responses = []
        self.get_count = 0
        self.post_count = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def request(self, method, url, *, json, timeout, allow_redirects):
        assert self.trust_env is False, "environment proxy use must be disabled"
        assert allow_redirects is False, "redirects must never be followed"
        assert isinstance(timeout, tuple) and timeout[0] == 5
        self.calls.append((method, url, deepcopy(json)))
        if method == "GET":
            assert url == BASE_URL + "/api/tags"
            assert json is None
            digest = self.digests[min(self.get_count, len(self.digests) - 1)]
            self.get_count += 1
            response = FakeResponse({"models": [{"name": MODEL, "digest": digest}]})
        elif method == "POST":
            assert url == BASE_URL + "/api/embed"
            assert json["model"] == MODEL
            assert json["truncate"] is False
            assert isinstance(json["input"], list) and json["input"]
            self.post_count += 1
            if self.post_count == self.fail_post:
                response = FakeResponse({"error": "synthetic failure"}, 500)
            else:
                response = FakeResponse(
                    {
                        "model": MODEL,
                        "embeddings": [
                            [1.25] + [-0.5] * (self.dimension - 1)
                            for _ in json["input"]
                        ],
                        "prompt_eval_count": self.tokens,
                    }
                )
        else:
            raise AssertionError(f"Unexpected fake HTTP method: {method}")
        self.responses.append(response)
        return response


class SessionFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sessions = []

    def __call__(self):
        session = FakeSession(**self.kwargs)
        self.sessions.append(session)
        return session

    @property
    def calls(self):
        return [call for session in self.sessions for call in session.calls]

    @property
    def post_inputs(self):
        return [body["input"] for method, _, body in self.calls if method == "POST"]


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="embedding-cache-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.job = synthetic_job(self.root / "cache")

    def populated(self, *, max_batches=None, job=None):
        job = job or self.job
        factory = SessionFactory()
        result = cache.run_job(job, max_batches=max_batches, session_factory=factory)
        return result, factory

    def edit_header(self, update, job=None):
        job = job or self.job
        with closing(sqlite3.connect(job["path"])) as connection:
            header = json.loads(
                connection.execute("SELECT header_json FROM metadata").fetchone()[0]
            )
            update(header)
            connection.execute(
                "UPDATE metadata SET header_json=?", (cache.canonical(header),)
            )
            connection.commit()

    def execute(self, sql, parameters=(), job=None):
        with closing(sqlite3.connect((job or self.job)["path"])) as connection:
            connection.execute(sql, parameters)
            connection.commit()

    def test_inspect_uncreated_cache_never_creates_directory(self):
        snapshot = cache.inspect_cache(self.job)
        self.assertEqual(snapshot["status"], "not_created")
        self.assertEqual(snapshot["done"], 0)
        self.assertEqual(snapshot["missing"], self.job["rows"])
        self.assertFalse(self.job["cache_dir"].exists())
        with self.assertRaisesRegex(ValueError, "缓存尚不存在"):
            cache.inspect_cache(self.job, required=True)
        self.assertFalse(self.job["cache_dir"].exists())

    def test_resume_requests_only_missing_rows_and_commits_three_http_per_batch(self):
        first, first_factory = self.populated(max_batches=1)
        self.assertEqual(
            first,
            {
                "embedding_requests": 1,
                "http_requests": 3,
                "status": "building",
                "done": 2,
                "newly_encoded": 2,
            },
        )
        self.assertEqual(
            [call[0] for call in first_factory.calls], ["GET", "POST", "GET"]
        )
        snapshot = cache.inspect_cache(self.job)
        self.assertEqual(snapshot["batches"], 1)
        self.assertEqual(snapshot["missing"], self.job["rows"][2:])
        second, factory = self.populated()
        self.assertEqual(second["status"], "encoded_complete")
        self.assertEqual(second["newly_encoded"], 3)
        self.assertEqual(second["embedding_requests"], 2)
        self.assertEqual(second["http_requests"], 6)
        self.assertEqual(
            factory.post_inputs,
            [
                [row["text"] for row in self.job["rows"][2:4]],
                [self.job["rows"][4]["text"]],
            ],
        )
        self.assertEqual(
            [call[0] for call in factory.calls], ["GET", "POST", "GET"] * 2
        )
        self.assertTrue(
            all(response.closed for response in factory.sessions[0].responses)
        )
        self.assertTrue(factory.sessions[0].closed)

    def test_complete_rerun_does_not_even_create_session(self):
        self.populated()
        with patch.object(cache, "show_plan"):
            result = cache.run_job(
                self.job,
                session_factory=lambda: self.fail("completed cache created a session"),
            )
        self.assertEqual(result["http_requests"], 0)
        self.assertEqual(result["embedding_requests"], 0)
        self.assertEqual(result["newly_encoded"], 0)
        self.assertEqual(result["done"], 5)
        self.assertEqual(result["status"], "encoded_complete")

    def test_complete_vectors_with_building_status_finalize_without_http(self):
        self.populated()
        self.edit_header(lambda header: header.update(status="building"))
        result = cache.run_job(
            self.job,
            session_factory=lambda: self.fail("finalization created HTTP session"),
        )
        self.assertEqual(result["status"], "encoded_complete")
        self.assertEqual(result["http_requests"], 0)

    def test_second_batch_http_failure_preserves_first_committed_batch(self):
        factory = SessionFactory(fail_post=2)
        with self.assertRaisesRegex(ValueError, "HTTP 500"):
            cache.run_job(self.job, session_factory=factory)
        snapshot = cache.inspect_cache(self.job)
        self.assertEqual((snapshot["done"], snapshot["batches"]), (2, 1))
        self.assertEqual(snapshot["missing"], self.job["rows"][2:])
        self.assertEqual(len(factory.calls), 5)
        result, resumed = self.populated()
        self.assertEqual(result["newly_encoded"], 3)
        self.assertEqual(
            resumed.post_inputs[0], [row["text"] for row in self.job["rows"][2:4]]
        )

    def test_post_request_digest_drift_does_not_commit_current_batch(self):
        factory = SessionFactory(digests=[DIGEST, OTHER_DIGEST])
        with self.assertRaisesRegex(ValueError, "批次后模型digest已变化"):
            cache.run_job(self.job, session_factory=factory)
        snapshot = cache.inspect_cache(self.job)
        self.assertEqual((snapshot["done"], snapshot["batches"]), (0, 0))
        self.assertEqual(snapshot["status"], "building")
        self.assertEqual(len(factory.calls), 3)

    def test_resume_digest_mismatch_rejects_before_post(self):
        self.populated(max_batches=1)
        before = self.job["path"].read_bytes()
        factory = SessionFactory(digests=[OTHER_DIGEST])
        with self.assertRaisesRegex(ValueError, "服务器模型digest与缓存不同"):
            cache.run_job(self.job, session_factory=factory)
        self.assertEqual([call[0] for call in factory.calls], ["GET"])
        self.assertEqual(self.job["path"].read_bytes(), before)
        self.assertEqual(cache.inspect_cache(self.job)["done"], 2)

    def test_vector_bytes_tampering_is_rejected(self):
        self.populated(max_batches=1)
        self.execute(
            "UPDATE vectors SET vector=? WHERE ordinal=0", (struct.pack("<ff", 2, 3),)
        )
        with self.assertRaisesRegex(ValueError, "向量SHA256不一致"):
            cache.inspect_cache(self.job)

    def test_rehashed_zero_vector_is_rejected(self):
        self.populated(max_batches=1)
        blob = struct.pack("<ff", 0, 0)
        self.execute(
            "UPDATE vectors SET vector=?,vector_sha=? WHERE ordinal=0",
            (blob, sha256(blob).hexdigest()),
        )
        with self.assertRaisesRegex(ValueError, "非有限或为零"):
            cache.inspect_cache(self.job)

    def test_original_text_and_full_metadata_tampering_are_rejected(self):
        for field in ("text", "metadata"):
            with self.subTest(field=field):
                job = synthetic_job(self.root / field)
                self.populated(max_batches=1, job=job)
                record = deepcopy(job["chunks"]["chunk-000"])
                if field == "text":
                    record["text"] += "被改写"
                else:
                    record["metadata"]["source_url"] = "https://example.invalid/changed"
                self.execute(
                    "UPDATE vectors SET record_json=? WHERE ordinal=0",
                    (cache.canonical(record),),
                    job=job,
                )
                with self.assertRaisesRegex(ValueError, "原文或完整元数据已变化"):
                    cache.inspect_cache(job)

    def test_false_completed_status_is_rejected_before_http(self):
        self.populated(max_batches=1)
        self.edit_header(lambda header: header.update(status="encoded_complete"))
        with self.assertRaisesRegex(ValueError, "宣称完成但仍缺少向量"):
            cache.run_job(
                self.job,
                session_factory=lambda: self.fail("corrupt cache requested HTTP"),
            )

    def test_unknown_scope_is_rejected(self):
        self.populated(max_batches=1)
        self.edit_header(lambda header: header["scope"].update(mode="section-prefix"))
        with self.assertRaisesRegex(ValueError, "不兼容"):
            cache.inspect_cache(self.job)

    def test_unsupported_schema_version_is_rejected(self):
        self.populated(max_batches=1)
        self.execute("PRAGMA user_version=999")
        with self.assertRaisesRegex(ValueError, "schema版本不支持"):
            cache.inspect_cache(self.job)

    def test_unknown_schema_object_is_rejected(self):
        self.populated(max_batches=1)
        self.execute("CREATE TABLE unexpected (value TEXT)")
        with self.assertRaisesRegex(ValueError, "表结构或附加对象"):
            cache.inspect_cache(self.job)

    def test_unrelated_sqlite_is_rejected_without_modifying_bytes_or_journal_mode(self):
        self.job["cache_dir"].mkdir()
        with closing(sqlite3.connect(self.job["path"])) as connection:
            self.assertEqual(
                connection.execute("PRAGMA journal_mode=PERSIST").fetchone()[0],
                "persist",
            )
            connection.execute("CREATE TABLE unrelated (data TEXT)")
            connection.execute("INSERT INTO unrelated VALUES ('preserve me')")
            connection.commit()
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            before = {
                path.name: path.read_bytes() for path in self.job["cache_dir"].iterdir()
            }
            with self.assertRaisesRegex(ValueError, "不是本工具缓存|不是本工具的缓存"):
                cache.run_job(
                    self.job,
                    session_factory=lambda: self.fail("unknown cache created session"),
                )
            after = {
                path.name: path.read_bytes()
                for path in self.job["cache_dir"].iterdir()
                if path.name != cache.LOCK_NAME
            }
            self.assertEqual(after, before)
            self.assertEqual(
                connection.execute("PRAGMA journal_mode").fetchone()[0], journal_mode
            )

    def test_competing_writer_flock_rejects_before_session(self):
        with cache.writer_lock(self.job):
            with self.assertRaisesRegex(ValueError, "已有写入进程"):
                cache.run_job(
                    self.job,
                    session_factory=lambda: self.fail("locked writer created session"),
                )
        self.assertFalse(self.job["path"].exists())
        result, _ = self.populated(max_batches=1)
        self.assertEqual(result["done"], 2)

    def test_insert_failure_rolls_back_whole_batch_including_ledger(self):
        class FailSecondVectorInsert:
            def __init__(self, connection):
                self.connection = connection
                self.insertions = 0

            def execute(self, statement, *args):
                if statement.startswith("INSERT INTO vectors"):
                    self.insertions += 1
                    if self.insertions == 2:
                        raise sqlite3.IntegrityError(
                            "synthetic second vector insert failure"
                        )
                return self.connection.execute(statement, *args)

            def __getattr__(self, name):
                return getattr(self.connection, name)

        with cache.writer_lock(self.job):
            cache.create_cache(self.job, DIGEST)
            with cache.connect_cache(self.job, writable=True) as connection:
                failing = FailSecondVectorInsert(connection)
                with self.assertRaisesRegex(sqlite3.IntegrityError, "second vector"):
                    cache.commit_batch(
                        failing,
                        self.job,
                        self.job["rows"][:2],
                        [[1, 2], [3, 4]],
                        DIGEST,
                        17,
                        0.01,
                    )
                self.assertEqual(failing.insertions, 2)
                self.assertFalse(connection.in_transaction)
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 0
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0], 0
                )
        snapshot = cache.inspect_cache(self.job)
        self.assertEqual((snapshot["done"], snapshot["batches"]), (0, 0))
        self.assertEqual(self.populated(max_batches=1)[0]["done"], 2)

    def test_interrupted_initialization_never_publishes_empty_database(self):
        with patch.object(cache, "initialize_cache", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                cache.run_job(self.job, session_factory=SessionFactory())
        self.assertFalse(self.job["path"].exists())
        self.assertEqual(cache.inspect_cache(self.job)["status"], "not_created")
        self.assertEqual(self.populated(max_batches=1)[0]["done"], 2)

    def test_hot_rollback_journal_recovers_committed_vectors_without_http(self):
        job = synthetic_job(self.root / "hot-journal", count=8, dimension=4096)
        cache.run_job(job, session_factory=SessionFactory(dimension=4096))
        with closing(sqlite3.connect(job["path"])) as connection:
            committed = connection.execute(
                "SELECT chunk_id,vector,vector_sha FROM vectors ORDER BY ordinal"
            ).fetchall()

        # More than one page per vector forces SQLite to spill modified pages
        # while the child transaction is open. os._exit deliberately bypasses
        # connection cleanup, leaving a real crash-recovery journal behind.
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import os, sqlite3, sys\n"
                "connection = sqlite3.connect(sys.argv[1])\n"
                "connection.execute('PRAGMA cache_size=1')\n"
                "connection.execute('BEGIN IMMEDIATE')\n"
                "connection.execute('UPDATE vectors SET vector=zeroblob(length(vector))')\n"
                "os._exit(0)\n",
                str(job["path"]),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(child.returncode, 0)
        journal = Path(str(job["path"]) + "-journal")
        self.assertTrue(journal.is_file())
        self.assertGreater(journal.stat().st_size, 512)
        self.assertNotEqual(journal.read_bytes()[:8], b"\0" * 8)

        result = cache.run_job(
            job,
            session_factory=lambda: self.fail(
                "journal recovery created an HTTP session"
            ),
        )
        self.assertEqual(
            result,
            {
                "status": "encoded_complete",
                "done": 8,
                "newly_encoded": 0,
                "embedding_requests": 0,
                "http_requests": 0,
            },
        )
        with closing(sqlite3.connect(job["path"])) as connection:
            recovered = connection.execute(
                "SELECT chunk_id,vector,vector_sha FROM vectors ORDER BY ordinal"
            ).fetchall()
        self.assertEqual(recovered, committed)
        self.assertFalse(journal.exists())
        self.assertEqual(cache.inspect_cache(job)["status"], "encoded_complete")

    def test_readonly_inspect_preserves_all_file_contents(self):
        self.populated(max_batches=1)
        before = {
            path.name: path.read_bytes() for path in self.job["cache_dir"].iterdir()
        }
        self.assertEqual(cache.inspect_cache(self.job, required=True)["done"], 2)
        after = {
            path.name: path.read_bytes() for path in self.job["cache_dir"].iterdir()
        }
        self.assertEqual(after, before)
        with cache.connect_cache(self.job) as connection:
            with self.assertRaisesRegex(sqlite3.OperationalError, "readonly"):
                connection.execute("DELETE FROM vectors")

    def test_batch_size_change_resumes_with_same_scope(self):
        self.populated(max_batches=1)
        changed = {**self.job, "batch_size": 1}
        factory = SessionFactory()
        result = cache.run_job(changed, max_batches=2, session_factory=factory)
        self.assertEqual((result["done"], result["newly_encoded"]), (4, 2))
        self.assertEqual(
            (result["embedding_requests"], result["http_requests"]), (2, 6)
        )
        self.assertEqual(
            factory.post_inputs,
            [[self.job["rows"][2]["text"]], [self.job["rows"][3]["text"]]],
        )
        self.assertEqual(cache.inspect_cache(changed)["batches"], 3)
        final, _ = self.populated(job=changed)
        self.assertEqual((final["done"], final["newly_encoded"]), (5, 1))
        self.assertEqual(final["status"], "encoded_complete")

    def test_invalid_service_token_statistics_reject_without_committing(self):
        for index, tokens in enumerate((-1, True, 1.5, "17")):
            with self.subTest(tokens=tokens):
                job = synthetic_job(self.root / f"tokens-{index}")
                factory = SessionFactory(tokens=tokens)
                with self.assertRaisesRegex(ValueError, "Token统计不合法"):
                    cache.run_job(job, session_factory=factory)
                snapshot = cache.inspect_cache(job)
                self.assertEqual((snapshot["done"], snapshot["batches"]), (0, 0))
                self.assertEqual([call[0] for call in factory.calls], ["GET", "POST"])

    def test_tampered_persisted_token_statistics_are_rejected(self):
        self.populated(max_batches=1)
        self.execute("UPDATE batches SET processed_tokens=-1")
        with self.assertRaisesRegex(ValueError, "Token统计不合法"):
            cache.inspect_cache(self.job)

    def test_commit_rejects_invalid_statistics_without_opening_transaction(self):
        with cache.writer_lock(self.job):
            cache.create_cache(self.job, DIGEST)
            with cache.connect_cache(self.job, writable=True) as connection:
                for tokens, seconds in (
                    (-1, 1),
                    (True, 1),
                    (1, -1),
                    (1, float("inf")),
                    (1, True),
                ):
                    with self.subTest(tokens=tokens, seconds=seconds):
                        with self.assertRaisesRegex(
                            ValueError, "统计不合法|耗时不合法"
                        ):
                            cache.commit_batch(
                                connection,
                                self.job,
                                self.job["rows"][:1],
                                [[1, 2]],
                                DIGEST,
                                tokens,
                                seconds,
                            )
                        self.assertFalse(connection.in_transaction)
                self.assertEqual(cache.audit_cache(connection, self.job)["done"], 0)

    def test_digest_contract_requires_exact_sha256(self):
        self.assertEqual(model_digest({"digest": "sha256:" + "A" * 64}), DIGEST)
        for value in (
            None,
            "a" * 63,
            "g" * 64,
            "a" * 65,
            "https://example.invalid/model",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    model_digest({"digest": value})

    def test_local_endpoint_helper_accepts_only_literal_private_or_loopback_ips(self):
        for host in (
            "127.0.0.1",
            "10.1.2.3",
            "172.16.1.2",
            "192.168.1.2",
            "[::1]",
            "[fd00::1]",
        ):
            with self.subTest(host=host):
                endpoint = f"http://{host}:11434/api/embed"
                self.assertEqual(
                    local_endpoint_base({"endpoint": endpoint}), f"http://{host}:11434"
                )
        for host in (
            "8.8.8.8",
            "172.32.0.1",
            "169.254.1.1",
            "100.64.0.1",
            "example.invalid",
            "localhost",
            "[2606:4700:4700::1111]",
        ):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    local_endpoint_base({"endpoint": f"http://{host}:11434/api/embed"})


if __name__ == "__main__":
    unittest.main()
