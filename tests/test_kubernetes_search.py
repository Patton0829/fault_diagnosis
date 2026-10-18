"""Question-search contracts; model responses and all database inputs are synthetic.

The single SDK integration uses only a temporary three-row, four-dimensional
Milvus database. No test contacts a model or reads the project's real data/.env.
"""

import builtins
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

module = importlib.import_module("qa_platform.storage.kubernetes_search")
build = module.build
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
MODEL = "synthetic-query-embedding"
BASE_URL = "http://127.0.0.1:11434"


def fixture(directory):
    directory.mkdir()
    (directory / build.DATABASE).mkdir()
    vectors = np.asarray([[1, 0, 0, 0], [0.8, 0.6, 0, 0], [0, 1, 0, 0]], dtype="<f4")
    records = []
    for index in range(3):
        metadata = {
            "title": f"合成DNS文档{index}",
            "source_url": f"https://example.invalid/{index}",
            "sections": [["调试", "DNS"]],
            "synthetic": True,
        }
        records.append(
            {
                "chunk_id": f"chunk-{index}",
                "text": f"仅用于离线测试的正文{index}",
                "kb_id": "kubernetes_zh",
                "parent_id": "parent-a" if index < 2 else "parent-b",
                "snapshot_commit": "c" * 40,
                "review_status": "upstream_unreviewed",
                "metadata": metadata,
            }
        )
    plan = {
        "document_count": len(records),
        "documents": records,
        "document_snapshot_sha256": "d" * 64,
        "vector_snapshot_sha256": "e" * 64,
        "vector_sha256": {
            row["chunk_id"]: sha256(vector.tobytes()).hexdigest()
            for row, vector in zip(records, vectors)
        },
        "embedding": {
            "model": MODEL,
            "endpoint": BASE_URL + "/api/embed",
            "documented_dimension": 4,
            "actual_dimension": 4,
            "server_model_digest": DIGEST,
            "document_mode": "body-only",
            "query_format": "Instruct: {instruction}\nQuery: {query}",
            "query_instruction": "检索可回答该问题的文档",
            "truncate": False,
            "keep_alive": "5m",
        },
    }
    written = {
        key: plan[key]
        for key in (
            "document_count",
            "document_snapshot_sha256",
            "vector_snapshot_sha256",
        )
    }
    written.update(status="written_not_verified", writer_pid=11)
    verified = {
        **written,
        "status": "verified",
        "reader_pid": 12,
        "max_cosine_difference": 1e-7,
    }
    observed = {"observed_digest": DIGEST, "http_requests": 1}
    manifest = {key: value for key, value in plan.items() if key != "documents"}
    manifest.update(status="ready", verification=verified, model_observation=observed)
    files = {
        "build_plan.json": plan,
        "build_result.json": written,
        "verification.json": verified,
        "model_observation.json": observed,
        "manifest.json": manifest,
    }
    for name, value in files.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    return plan, {"records": records, "vectors": vectors}


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def json(self):
        return self.payload

    def close(self):
        self.closed = True


class QuerySession:
    def __init__(self, *, before=DIGEST, after=DIGEST, vector=None, tokens=23):
        self.digests = iter([before, after])
        self.vector = [1.0, 0.0, 0.0, 0.0] if vector is None else vector
        self.tokens = tokens
        self.trust_env = True
        self.calls = []
        self.responses = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def request(self, method, url, *, json, timeout, allow_redirects):
        assert self.trust_env is False
        assert allow_redirects is False
        self.calls.append((method, url, deepcopy(json), timeout))
        if method == "GET":
            assert url == BASE_URL + "/api/tags"
            assert json is None and timeout == (5, 15)
            payload = {"models": [{"name": MODEL, "digest": next(self.digests)}]}
        else:
            assert method == "POST" and url == BASE_URL + "/api/embed"
            payload = {
                "model": MODEL,
                "embeddings": [self.vector],
                "prompt_eval_count": self.tokens,
            }
        response = Response(payload)
        self.responses.append(response)
        return response


class KubernetesSearchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="k8s-search-tests-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "index"
        self.plan, self.bundle = fixture(self.directory)
        patcher = patch.object(
            build, "load_build", return_value=(self.plan, self.bundle)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def prepare(self, **kwargs):
        return module.prepare_search(self.directory, **kwargs)

    def rewrite(self, name, key, value):
        path = self.directory / name
        payload = json.loads(path.read_text())
        payload[key] = value
        path.write_text(json.dumps(payload))

    def test_default_main_is_readonly_without_sdk_or_http_imports(self):
        before = {p.name: p.read_bytes() for p in self.directory.glob("*.json")}
        original = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"requests", "pymilvus", "dotenv"}:
                raise AssertionError(f"offline plan imported {name}")
            return original(name, *args, **kwargs)

        with (
            patch.object(sys, "argv", ["query", "--index-dir", str(self.directory)]),
            patch.object(builtins, "__import__", side_effect=guarded_import),
            patch.object(
                module, "run_search", side_effect=AssertionError("must not run")
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(module.main(), 0)
        self.assertIn("实际HTTP=0", output.getvalue())
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.directory.glob("*.json")}
        )
        self.assertEqual(list((self.directory / build.DATABASE).iterdir()), [])

    def test_manifest_and_receipt_mismatch_are_rejected(self):
        mutations = [
            ("manifest.json", "status", "building"),
            ("verification.json", "reader_pid", 11),
            ("verification.json", "document_count", 2),
            ("verification.json", "max_cosine_difference", float("nan")),
            ("model_observation.json", "observed_digest", OTHER_DIGEST),
            ("build_result.json", "writer_pid", 99),
        ]
        for name, key, value in mutations:
            with self.subTest(file=name, field=key):
                path = self.directory / name
                original = path.read_bytes()
                self.rewrite(name, key, value)
                try:
                    with self.assertRaises(ValueError):
                        self.prepare()
                finally:
                    path.write_bytes(original)

    def test_filter_applies_same_rows_vectors_and_safe_milvus_expression(self):
        check = self.prepare(parent_id="parent-a", top_k=10)
        self.assertEqual(
            [row["chunk_id"] for row in check["records"]], ["chunk-0", "chunk-1"]
        )
        np.testing.assert_array_equal(check["vectors"], self.bundle["vectors"][:2])
        self.assertEqual(check["top_k"], 2)
        self.assertIn('parent_id == "parent-a"', check["filter"])
        self.assertIn('kb_id == "kubernetes_zh"', check["filter"])
        with self.assertRaisesRegex(ValueError, "parent-id未命中"):
            self.prepare(parent_id='parent-a" or kb_id != "')

    def test_invalid_query_topk_and_input_contract_fail_offline(self):
        for kwargs in (
            {"query": " "},
            {"query": "x" * 2001},
            {"top_k": True},
            {"top_k": 51},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prepare(**kwargs)
        self.plan["embedding"]["truncate"] = True
        with (
            patch.object(module, "load_ready", return_value=(self.plan, self.bundle)),
            self.assertRaisesRegex(ValueError, "输入合同"),
        ):
            self.prepare()

    def test_query_encoding_has_one_post_two_gets_and_exact_input_contract(self):
        check = self.prepare(query="如何查DNS？")
        session = QuerySession()
        with redirect_stdout(io.StringIO()):
            vector, stats = module.encode_query(check, 42, lambda: session)
        np.testing.assert_array_equal(vector, [1, 0, 0, 0])
        self.assertEqual(vector.dtype.str, "<f4")
        self.assertEqual([call[0] for call in session.calls], ["GET", "POST", "GET"])
        self.assertEqual(
            session.calls[1][2],
            {
                "model": MODEL,
                "input": ["Instruct: 检索可回答该问题的文档\nQuery: 如何查DNS？"],
                "truncate": False,
                "keep_alive": "5m",
            },
        )
        self.assertEqual(session.calls[1][3], (5, 42))
        self.assertEqual(stats["embedding_requests"], 1)
        self.assertEqual(stats["http_requests"], 3)
        self.assertEqual(stats["processed_tokens"], 23)
        self.assertTrue(session.closed)
        self.assertTrue(all(response.closed for response in session.responses))

    def test_digest_before_rejects_without_post_and_after_rejects_without_retry(self):
        check = self.prepare()
        for before, after, expected in (
            (OTHER_DIGEST, DIGEST, ["GET"]),
            (DIGEST, OTHER_DIGEST, ["GET", "POST", "GET"]),
        ):
            with self.subTest(before=before, after=after):
                session = QuerySession(before=before, after=after)
                with (
                    redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(ValueError, "digest"),
                ):
                    module.encode_query(check, session_factory=lambda: session)
                self.assertEqual([call[0] for call in session.calls], expected)
                self.assertTrue(session.closed)

    def test_invalid_query_vectors_and_token_statistics_are_rejected(self):
        check = self.prepare()
        invalid = [
            {"vector": [1, 0]},
            {"vector": [float("nan"), 0, 0, 0]},
            {"vector": [0, 0, 0, 0]},
            {"tokens": True},
            {"tokens": -1},
            {"tokens": 1.5},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                session = QuerySession(**kwargs)
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    module.encode_query(check, session_factory=lambda: session)
                self.assertEqual([call[0] for call in session.calls], ["GET", "POST"])
                self.assertTrue(session.closed)

    def test_numpy_ranking_is_cosine_not_raw_dot_product(self):
        records = self.plan["documents"]
        ranked = module.rank_numpy(
            records, [[2, 0, 0, 0], [80, 60, 0, 0], [0, 1, 0, 0]], [1, 0, 0, 0], 3
        )
        self.assertEqual(
            [hit["chunk_id"] for hit in ranked], ["chunk-0", "chunk-1", "chunk-2"]
        )
        np.testing.assert_allclose([hit["score"] for hit in ranked], [1, 0.8, 0])

    def test_topk_allows_boundary_tie_but_not_missing_strictly_better_result(self):
        records = self.plan["documents"]
        vectors = [[1, 0], [0, 1], [0, -1]]
        query = [1, 0]
        hits = [
            {"chunk_id": "chunk-0", "score": 1.0},
            {"chunk_id": "chunk-2", "score": 0.0},
        ]
        self.assertEqual(
            module.validate_top_k(hits, records, vectors, query, 2)["cutoff"], 0
        )
        with self.assertRaisesRegex(ValueError, "遗漏"):
            module.validate_top_k(
                [
                    {"chunk_id": "chunk-1", "score": 0.0},
                    {"chunk_id": "chunk-2", "score": 0.0},
                ],
                records,
                vectors,
                query,
                2,
            )
        hits[0]["score"] = 0.9
        with self.assertRaisesRegex(ValueError, "分数差"):
            module.validate_top_k(hits, records, vectors, query, 2)

    def test_milvus_rejects_unknown_duplicate_metadata_and_bad_scores(self):
        records = self.plan["documents"]
        valid = [
            {"chunk_id": row["chunk_id"], "entity": deepcopy(row), "distance": score}
            for row, score in zip(records[:2], [1.0, 0.8])
        ]
        mutations = [
            lambda hits: hits[0].update(chunk_id="unknown"),
            lambda hits: hits[1].update(chunk_id="chunk-0"),
            lambda hits: hits[0]["entity"].update(text="changed"),
            lambda hits: hits[0].update(distance=float("nan")),
            lambda hits: hits[0].update(distance=2.0),
            lambda hits: hits.reverse(),
        ]
        for mutate in mutations:
            hits = deepcopy(valid)
            mutate(hits)
            client = unittest.mock.Mock()
            client.search.return_value = [hits]
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                module.search_milvus(
                    client, records, np.asarray([1, 0, 0, 0]), 2, "safe-filter"
                )
        client = unittest.mock.Mock()
        client.search.return_value = [valid]
        module.search_milvus(
            client, records, np.asarray([1, 0, 0, 0]), 2, "safe-filter"
        )
        self.assertEqual(client.search.call_args.kwargs["filter"], "safe-filter")

    def test_database_failure_closes_client_before_any_model_http(self):
        check = self.prepare()
        client = unittest.mock.Mock()
        with (
            patch.object(module, "read_index", side_effect=ValueError("bad database")),
            self.assertRaisesRegex(ValueError, "bad database"),
        ):
            module.run_search(
                check,
                session_factory=lambda: self.fail("HTTP must not start"),
                client_factory=lambda **_: client,
            )
        client.close.assert_called_once_with()

    def test_vector_mismatch_closes_client_before_any_model_http(self):
        check = self.prepare()
        client = unittest.mock.Mock()
        with (
            patch.object(module, "read_index", return_value=self.bundle["vectors"] * 2),
            self.assertRaisesRegex(ValueError, "保存文档向量不一致"),
        ):
            module.run_search(
                check,
                session_factory=lambda: self.fail("HTTP must not start"),
                client_factory=lambda **_: client,
            )
        client.close.assert_called_once_with()

    def test_encoder_failure_also_closes_database_client(self):
        check = self.prepare()
        client = unittest.mock.Mock()
        session = QuerySession(after=OTHER_DIGEST)
        with (
            patch.object(module, "read_index", return_value=self.bundle["vectors"]),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "digest"),
        ):
            module.run_search(
                check,
                session_factory=lambda: session,
                client_factory=lambda **_: client,
            )
        client.close.assert_called_once_with()

    @unittest.skipUnless(
        importlib.util.find_spec("pymilvus"), "optional Milvus SDK unavailable"
    )
    def test_temporary_real_milvus_four_dimensional_query_and_parent_filter(self):
        from pymilvus import DataType, MilvusClient

        database = self.directory / build.DATABASE
        database.rmdir()
        client = MilvusClient(uri=str(database), timeout=30)
        try:
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(
                "chunk_id", DataType.VARCHAR, is_primary=True, max_length=64
            )
            schema.add_field("text", DataType.VARCHAR, max_length=build.TEXT_BYTES)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)
            for field, limit in build.SCALARS.items():
                schema.add_field(field, DataType.VARCHAR, max_length=limit)
            schema.add_field("metadata", DataType.JSON)
            indexes = client.prepare_index_params()
            indexes.add_index(
                field_name="vector", index_type="FLAT", metric_type="COSINE"
            )
            client.create_collection(
                collection_name=build.COLLECTION, schema=schema, index_params=indexes
            )
            client.insert(
                collection_name=build.COLLECTION,
                data=[
                    {**row, "vector": vector.tolist()}
                    for row, vector in zip(
                        self.plan["documents"], self.bundle["vectors"]
                    )
                ],
            )
            client.flush(collection_name=build.COLLECTION)
        finally:
            client.close()
        check = self.prepare(parent_id="parent-a", top_k=3)
        session = QuerySession()
        with redirect_stdout(io.StringIO()):
            result = module.run_search(
                check, repeat=1, preview_chars=100, session_factory=lambda: session
            )
        self.assertEqual(
            [hit["chunk_id"] for hit in result["memory"]], ["chunk-0", "chunk-1"]
        )
        self.assertEqual(
            [hit["chunk_id"] for hit in result["milvus"]], ["chunk-0", "chunk-1"]
        )
        self.assertEqual(result["http_requests"], 3)
        self.assertLess(result["max_score_difference"], module.SCORE_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
