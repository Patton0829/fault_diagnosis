"""Expanded teaching-index limits and paginated readback, using synthetic data only.

Header tests replace the frozen cache loader, so even upper-bound cases allocate
no document vectors. Readback tests use four-dimensional synthetic rows and a
fake Milvus SDK: no model, real database, project data, or secrets are accessed.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from qa_platform.storage import kubernetes_search as search


DIGEST = "a" * 64
FAKE_TYPES = SimpleNamespace(VARCHAR=21, FLOAT_VECTOR=101, JSON=23)


def ready_fixture(directory: Path, count: int = 3, dimension: int = 4):
    """Provide valid receipts while standing in for already-verified cache data."""
    directory.mkdir()
    (directory / search.build.DATABASE).mkdir()
    plan = {
        "document_count": count,
        "documents": [
            {"chunk_id": f"header-only-{ordinal}"} for ordinal in range(count)
        ],
        "document_snapshot_sha256": "b" * 64,
        "vector_snapshot_sha256": "c" * 64,
        "embedding": {
            "model": "synthetic-expanded-embedding",
            "actual_dimension": dimension,
            "server_model_digest": DIGEST,
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
    written.update(status="written_not_verified", writer_pid=101)
    verified = {
        **written,
        "status": "verified",
        "reader_pid": 102,
        "max_cosine_difference": 1e-7,
    }
    observed = {"observed_digest": DIGEST, "http_requests": 1}
    manifest = {key: value for key, value in plan.items() if key != "documents"}
    manifest.update(status="ready", verification=verified, model_observation=observed)
    files = {
        "manifest.json": manifest,
        "build_plan.json": plan,
        "build_result.json": written,
        "verification.json": verified,
        "model_observation.json": observed,
    }
    for name, payload in files.items():
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    return plan, {"records": plan["documents"], "vectors": []}


def replace_manifest(directory, mutation):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("count", "dimension"),
    [(1, 2), (3, 4), (1920, 4096), (5000, 4096), (4096, 8192)],
)
def test_supported_header_is_checked_before_cache_load(tmp_path, count, dimension):
    directory = tmp_path / "index"
    plan, bundle = ready_fixture(directory, count, dimension)
    with patch.object(search.build, "load_build", return_value=(plan, bundle)) as load:
        actual_plan, actual_bundle = search.load_ready(directory)
    load.assert_called_once_with(directory)
    assert actual_plan is plan
    assert actual_bundle is bundle


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_count", True),
        ("document_count", False),
        ("document_count", 0),
        ("document_count", -1),
        ("document_count", 5001),
        ("document_count", 1920.0),
        ("document_count", "1920"),
        ("actual_dimension", True),
        ("actual_dimension", False),
        ("actual_dimension", 0),
        ("actual_dimension", 1),
        ("actual_dimension", -1),
        ("actual_dimension", 8193),
        ("actual_dimension", 4096.0),
        ("actual_dimension", "4096"),
    ],
)
def test_illegal_header_rejected_before_cache_allocation(tmp_path, field, value):
    directory = tmp_path / "index"
    ready_fixture(directory)

    def mutate(manifest):
        target = manifest if field == "document_count" else manifest["embedding"]
        target[field] = value

    replace_manifest(directory, mutate)
    with (
        patch.object(search.build, "load_build") as load,
        pytest.raises(ValueError),
    ):
        search.load_ready(directory)
    load.assert_not_called()


def test_count_and_dimension_individually_valid_but_byte_budget_rejected(tmp_path):
    directory = tmp_path / "index"
    ready_fixture(directory, count=5000, dimension=8192)
    assert 5000 * 8192 * np.dtype("<f4").itemsize > 128 * 1024 * 1024
    with (
        patch.object(search.build, "load_build") as load,
        pytest.raises(ValueError),
    ):
        search.load_ready(directory)
    load.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        "null",
        "[]",
        "{}",
        '{"document_count": 3}',
        '{"document_count": 3, "embedding": null}',
        '{"document_count": 3, "embedding": []}',
        '{"document_count": 3, "embedding": {}}',
    ],
)
def test_missing_or_malformed_header_never_calls_cache_loader(tmp_path, payload):
    directory = tmp_path / "index"
    ready_fixture(directory)
    (directory / "manifest.json").write_text(payload, encoding="utf-8")
    with (
        patch.object(search.build, "load_build") as load,
        pytest.raises(ValueError),
    ):
        search.load_ready(directory)
    load.assert_not_called()


@pytest.mark.parametrize("field,value", [("document_count", 2), ("dimension", 8)])
def test_frozen_plan_cannot_differ_from_prechecked_header(tmp_path, field, value):
    directory = tmp_path / "index"
    plan, bundle = ready_fixture(directory)
    if field == "dimension":
        plan["embedding"]["actual_dimension"] = value
    else:
        plan[field] = value
    with (
        patch.object(search.build, "load_build", return_value=(plan, bundle)) as load,
        pytest.raises(ValueError),
    ):
        search.load_ready(directory)
    load.assert_called_once_with(directory)


@pytest.mark.parametrize("payload", ["", " " * (4 * 1024 * 1024 + 1)])
def test_empty_or_oversized_manifest_rejected_before_cache_load(tmp_path, payload):
    directory = tmp_path / "index"
    ready_fixture(directory)
    (directory / "manifest.json").write_text(payload, encoding="utf-8")
    with (
        patch.object(search.build, "load_build") as load,
        pytest.raises(ValueError),
    ):
        search.load_ready(directory)
    load.assert_not_called()


def readback_fixture(count):
    vectors = np.empty((count, 4), dtype="<f4")
    records = []
    for ordinal in range(count):
        vectors[ordinal] = [1, ordinal / max(count, 1), 0, 0]
        records.append(
            {
                "chunk_id": f"synthetic-{ordinal:04d}",
                "text": f"仅测试分页读取的合成正文 {ordinal}",
                "metadata": {"ordinal": ordinal, "synthetic": True},
                "kb_id": "kubernetes_zh",
                "parent_id": f"parent-{ordinal // 10}",
                "snapshot_commit": "d" * 40,
                "review_status": "test_data",
            }
        )
    plan = {
        "document_count": count,
        "documents": records,
        "embedding": {"actual_dimension": 4},
        "vector_sha256": {
            row["chunk_id"]: sha256(vector.tobytes()).hexdigest()
            for row, vector in zip(records, vectors)
        },
    }
    rows = {
        row["chunk_id"]: {**deepcopy(row), "vector": vector.tolist()}
        for row, vector in zip(records, vectors)
    }
    return plan, vectors, rows


def fake_client(plan, rows):
    client = Mock()
    fields = [
        {"name": "chunk_id", "type": FAKE_TYPES.VARCHAR, "is_primary": True},
        {"name": "text", "type": FAKE_TYPES.VARCHAR},
        {"name": "metadata", "type": FAKE_TYPES.JSON},
        {"name": "vector", "type": FAKE_TYPES.FLOAT_VECTOR, "params": {"dim": 4}},
        *[{"name": name, "type": FAKE_TYPES.VARCHAR} for name in search.build.SCALARS],
    ]
    client.has_collection.return_value = True
    client.describe_collection.return_value = {
        "fields": fields,
        "auto_id": False,
        "enable_dynamic_field": False,
    }
    client.list_indexes.return_value = ["vector_index"]
    client.describe_index.return_value = {
        "field_name": "vector",
        "index_type": "FLAT",
        "metric_type": "COSINE",
    }
    client.get_collection_stats.return_value = {"row_count": plan["document_count"]}

    def get(*, collection_name, ids, output_fields):
        assert collection_name == search.build.COLLECTION
        assert output_fields == search.build.OUTPUT_FIELDS
        assert len(ids) <= 128, "readback must be paginated before calling the SDK"
        # Milvus need not return rows in requested-primary-key order.
        return [deepcopy(rows[key]) for key in reversed(ids)]

    client.get.side_effect = get
    return client


@pytest.mark.parametrize("count", [1, 3, 127, 128, 129, 1920])
def test_read_index_pages_in_128_rows_without_changing_order_or_vector_bits(count):
    plan, vectors, rows = readback_fixture(count)
    client = fake_client(plan, rows)
    with patch.dict(sys.modules, {"pymilvus": SimpleNamespace(DataType=FAKE_TYPES)}):
        actual = search.read_index(client, plan)
    np.testing.assert_array_equal(actual, vectors)
    assert actual.dtype.str == "<f4"
    sizes = [len(call.kwargs["ids"]) for call in client.get.call_args_list]
    assert sizes == [min(128, count - start) for start in range(0, count, 128)]
    ids = [key for call in client.get.call_args_list for key in call.kwargs["ids"]]
    assert ids == [row["chunk_id"] for row in plan["documents"]]
    client.load_collection.assert_called_once_with(
        collection_name=search.build.COLLECTION
    )


@pytest.mark.parametrize(
    "corruption",
    ["missing", "duplicate", "wrong_page", "text", "metadata", "vector", "dimension"],
)
def test_later_page_corruption_still_rejected_before_query_encoding(corruption):
    plan, _, rows = readback_fixture(260)
    client = fake_client(plan, rows)
    original_get = client.get.side_effect

    def corrupt(**kwargs):
        result = original_get(**kwargs)
        if "synthetic-0259" in kwargs["ids"]:
            if corruption == "missing":
                result.pop()
            elif corruption == "duplicate":
                result[-1] = deepcopy(result[0])
            elif corruption == "wrong_page":
                result[0] = deepcopy(rows["synthetic-0000"])
            elif corruption == "text":
                result[0]["text"] = "篡改后的正文"
            elif corruption == "metadata":
                result[0]["metadata"]["ordinal"] = -1
            elif corruption == "vector":
                result[0]["vector"][0] = 0.5
            else:
                result[0]["vector"].append(0)
        return result

    client.get.side_effect = corrupt
    with (
        patch.dict(sys.modules, {"pymilvus": SimpleNamespace(DataType=FAKE_TYPES)}),
        pytest.raises(ValueError),
    ):
        search.read_index(client, plan)


@pytest.mark.parametrize("corruption", ["schema", "index", "count"])
def test_schema_index_and_count_checks_remain_before_paged_read(corruption):
    plan, _, rows = readback_fixture(1920)
    client = fake_client(plan, rows)
    if corruption == "schema":
        client.describe_collection.return_value["enable_dynamic_field"] = True
    elif corruption == "index":
        client.describe_index.return_value["index_type"] = "HNSW"
    else:
        client.get_collection_stats.return_value["row_count"] = 1919
    with (
        patch.dict(sys.modules, {"pymilvus": SimpleNamespace(DataType=FAKE_TYPES)}),
        pytest.raises(ValueError),
    ):
        search.read_index(client, plan)
    client.get.assert_not_called()
