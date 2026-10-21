"""Offline portable snapshot contracts; no model, database or service is contacted."""

from copy import deepcopy
import builtins
from hashlib import sha256
import json
import shutil
import socket
from unittest.mock import Mock

import numpy as np
import pytest

from qa_platform.storage import portable_knowledge as portable


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Portable snapshot tests must never contact a service")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def source(tmp_path):
    corpus = tmp_path / "source-corpus"
    corpus.mkdir()
    (corpus / "LICENSE").write_text(
        "Synthetic license fixture — CC BY 4.0 learning sample.\n", encoding="utf-8"
    )
    cache = tmp_path / "source-cache"
    cache.mkdir()
    (cache / "embeddings.sqlite3").write_bytes(b"fixture: never open this as SQLite")
    index = tmp_path / "source-index"
    index.mkdir()
    (index / "manifest.json").write_text("source fixture is untouched\n")
    rows = []
    for ordinal in range(2):
        chunk_id = f"portable-offline-{ordinal}"
        metadata = {
            "chunk_id": chunk_id,
            "kb_id": "kubernetes_zh",
            "parent_id": f"parent-{ordinal}",
            "snapshot_commit": "c" * 40,
            "review_status": "unreviewed_official_document",
            "title": f"合成 DNS 排查 {ordinal}",
            "section_path": ["故障排查", "DNS"],
            "source_url": f"https://github.com/kubernetes/website/blob/{'c' * 40}/content/zh-cn/docs/tasks/dns-{ordinal}.md",
            "license": "CC-BY-4.0",
            "is_test": False,
            "source": str(corpus / f"page-{ordinal}.md"),
            "local_path": str(corpus / f"page-{ordinal}.md"),
            "front_matter": {
                "internal_path": str(corpus),
                "min-kubernetes-server-version": "v1.28",
                "max-kubernetes-server-version": "v1.32",
                "title": "Do not export unrelated front matter",
            },
            "feature_states": [
                {
                    "state": "beta",
                    "for_k8s_version": "1.30",
                    "version_source": "explicit_shortcode_parameter",
                    "source_path": "content/zh-cn/docs/tasks/dns.md",
                    "original_shortcode": '{{< feature-state state="beta" for_k8s_version="1.30" >}}',
                }
            ],
            "feature_states_scope": "document_including_expanded_includes_not_per_chunk",
            "dependencies": [str(corpus / "included-file.md")],
        }
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": f"合成正文 {ordinal}：Pod 与 Service 的 DNS 查询。\n\n```sh\nnslookup example\n```",
                "kb_id": metadata["kb_id"],
                "parent_id": metadata["parent_id"],
                "snapshot_commit": metadata["snapshot_commit"],
                "review_status": metadata["review_status"],
                "metadata": metadata,
            }
        )
    matrix = np.asarray(
        [[1.0, 0.25, -0.5, 0.125], [0.5, 1.0, 0.25, -0.125]], dtype="<f4"
    )
    hashes = {
        row["chunk_id"]: sha256(matrix[i].tobytes()).hexdigest()
        for i, row in enumerate(rows)
    }
    embedding = {
        "provider": "ollama",
        "model": "offline-embedding:fixture",
        "endpoint": "http://127.0.0.1:11434/api/embed",
        "server_model_digest": "a" * 64,
        "actual_dimension": 4,
        "document_mode": "body-only",
        "document_query_prefix": False,
        "query_format": "Instruct: {instruction}\nQuery: {query}",
        "query_instruction": "Given a question, retrieve relevant documentation",
        "truncate": False,
        "dtype": "little_endian_float32",
        "metric_type": "COSINE",
        "keep_alive": "5m",
        "local_document": str(corpus / "model-guide.md"),
    }
    plan = {
        "document_count": len(rows),
        "documents": deepcopy(rows),
        "embedding": embedding,
        "corpus_path": str(corpus),
        "cache_path": str(cache / "embeddings.sqlite3"),
        "vector_sha256": hashes,
        "vector_snapshot_sha256": portable.fingerprint(portable.canonical(hashes)),
        "document_snapshot_sha256": portable.fingerprint(portable.canonical(rows)),
    }
    bundle = {
        "records": deepcopy(rows),
        "vectors": matrix.copy(),
        "hashes": hashes.copy(),
    }
    return {
        "index": index,
        "corpus": corpus,
        "cache": cache,
        "plan": plan,
        "bundle": bundle,
    }


def prepare(source):
    loader = Mock(return_value=(source["plan"], source["bundle"]))
    prepared = portable.prepare_source(source["index"], source_loader=loader)
    loader.assert_called_once_with(source["index"])
    return prepared


@pytest.fixture
def exported(source, tmp_path):
    prepared = prepare(source)
    target = tmp_path / "portable-output"
    manifest = portable.export_bundle(prepared, target)
    return prepared, target, manifest


def snapshot_files(directory):
    return {
        str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_prepare_source_is_read_only_and_retains_license_and_review(source):
    before = {key: snapshot_files(source[key]) for key in ("index", "corpus", "cache")}
    prepared = prepare(source)
    assert before == {key: snapshot_files(source[key]) for key in before}
    assert prepared["license"]["text"] == (source["corpus"] / "LICENSE").read_text()
    assert (
        prepared["license"]["sha256"]
        == sha256(prepared["license"]["text"].encode()).hexdigest()
    )
    assert prepared["records"][0]["review_status"] == "unreviewed_official_document"
    assert prepared["records"][0]["metadata"]["license"] == "CC-BY-4.0"
    assert prepared["records"][0]["metadata"]["section_path"] == ["故障排查", "DNS"]
    assert prepared["records"][0]["text"] == source["bundle"]["records"][0]["text"]


def test_prepare_source_drops_local_paths_and_deployment_only_contract_fields(source):
    prepared = prepare(source)
    serialized = json.dumps(
        {"records": prepared["records"], "embedding": prepared["embedding"]}
    )
    assert str(source["corpus"]) not in serialized
    assert "127.0.0.1:11434" not in serialized
    assert "endpoint" not in prepared["embedding"]
    assert "keep_alive" not in prepared["embedding"]
    metadata = prepared["records"][0]["metadata"]
    assert (
        not {"source", "local_path", "front_matter", "dependencies"} & metadata.keys()
    )
    assert metadata["source_url"] == (
        f"https://github.com/kubernetes/website/blob/{'c' * 40}/content/zh-cn/docs/tasks/dns-0.md"
    )
    assert prepared["embedding"]["dtype"] == "little_endian_float32"
    assert prepared["embedding"]["server_model_digest"] == "a" * 64
    assert (
        prepared["embedding"]["query_format"]
        == source["plan"]["embedding"]["query_format"]
    )


def test_prepare_retains_feature_state_and_explicit_version_conditions(source):
    prepared = prepare(source)
    original = source["bundle"]["records"][0]["metadata"]
    metadata = prepared["records"][0]["metadata"]
    assert metadata["feature_states"] == original["feature_states"]
    assert (
        metadata["feature_states_scope"]
        == "document_including_expanded_includes_not_per_chunk"
    )
    assert metadata["version_constraints"] == {
        "min-kubernetes-server-version": "v1.28",
        "max-kubernetes-server-version": "v1.32",
    }
    assert "front_matter" not in metadata
    assert "internal_path" not in portable.canonical(metadata)
    assert "Do not export unrelated front matter" not in portable.canonical(metadata)


def test_prepare_retains_numeric_version_without_coercing_to_string(source):
    rows = source["bundle"]["records"]
    rows[0]["metadata"]["front_matter"]["min-kubernetes-server-version"] = 1.2
    source["plan"]["documents"] = deepcopy(rows)
    source["plan"]["document_snapshot_sha256"] = portable.fingerprint(
        portable.canonical(rows)
    )
    metadata = prepare(source)["records"][0]["metadata"]
    assert metadata["version_constraints"]["min-kubernetes-server-version"] == 1.2
    assert (
        type(metadata["version_constraints"]["min-kubernetes-server-version"]) is float
    )


@pytest.mark.parametrize(
    "value", [True, False, float("nan"), float("inf"), -float("inf")]
)
def test_version_conditions_reject_boolean_and_non_finite_numbers(source, value):
    rows = source["bundle"]["records"]
    rows[0]["metadata"]["front_matter"]["min-kubernetes-server-version"] = value
    source["plan"]["documents"] = deepcopy(rows)
    # Non-finite values already violate the canonical source contract. Neither
    # direct source preparation nor a coherently rehashed boolean can publish.
    if isinstance(value, bool):
        source["plan"]["document_snapshot_sha256"] = portable.fingerprint(
            portable.canonical(rows)
        )
    with pytest.raises((ValueError, KeyError, TypeError)):
        prepare(source)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "http://127.0.0.1:11434/private-document",
        "file:///private/source.md",
        "https://example.invalid/source.md",
        f"http://github.com/kubernetes/website/blob/{'c' * 40}/content/zh-cn/docs/tasks/dns.md",
        f"https://github.com/other/website/blob/{'c' * 40}/content/zh-cn/docs/tasks/dns.md",
        f"https://github.com/kubernetes/website/blob/{'b' * 40}/content/zh-cn/docs/tasks/dns.md",
        f"https://github.com/kubernetes/website/blob/{'c' * 40}/content/en/docs/tasks/dns.md",
        f"https://github.com/kubernetes/website/blob/{'c' * 40}/content/zh-cn/docs/tasks/dns.md?token=private",
        f"https://github.com/kubernetes/website/blob/{'c' * 40}/content/zh-cn/docs/tasks/dns.md#private-fragment",
    ],
)
def test_source_requires_snapshot_pinned_public_upstream_url(source, value):
    rows = source["bundle"]["records"]
    if value is None:
        del rows[0]["metadata"]["source_url"]
    else:
        rows[0]["metadata"]["source_url"] = value
    source["plan"]["documents"] = deepcopy(rows)
    source["plan"]["document_snapshot_sha256"] = portable.fingerprint(
        portable.canonical(rows)
    )
    with pytest.raises((ValueError, KeyError, TypeError)):
        prepare(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("title", ""),
        ("title", None),
        ("license", "proprietary"),
        ("review_status", "approved"),
    ],
)
def test_source_requires_title_public_license_and_unreviewed_boundary(
    source, field, value
):
    rows = source["bundle"]["records"]
    rows[0]["metadata"][field] = value
    if field == "review_status":
        rows[0][field] = value
    source["plan"]["documents"] = deepcopy(rows)
    source["plan"]["document_snapshot_sha256"] = portable.fingerprint(
        portable.canonical(rows)
    )
    with pytest.raises((ValueError, KeyError, TypeError)):
        prepare(source)


def test_export_contains_exact_three_files_and_bit_identical_vectors(exported, source):
    _, target, manifest = exported
    assert {path.name for path in target.iterdir()} == {
        "manifest.json",
        "records.jsonl",
        "vectors.npy",
    }
    assert portable.SCHEMA == "portable-kb/v1"
    loaded = portable.load_bundle(target)
    assert loaded["manifest"] == manifest
    assert loaded["vectors"].dtype == np.dtype("<f4")
    assert loaded["vectors"].tobytes() == source["bundle"]["vectors"].tobytes()
    assert [row["chunk_id"] for row in loaded["records"]] == [
        row["chunk_id"] for row in source["bundle"]["records"]
    ]
    assert [row["text"] for row in loaded["records"]] == [
        row["text"] for row in source["bundle"]["records"]
    ]
    serialized = (target / "manifest.json").read_text() + (
        target / "records.jsonl"
    ).read_text()
    assert str(source["corpus"]) not in serialized
    assert str(source["cache"]) not in serialized
    assert "127.0.0.1:11434" not in serialized
    assert "Synthetic license fixture" in serialized


def test_relocated_bundle_verifies_without_source_files_or_loader(
    exported, source, tmp_path, monkeypatch
):
    _, target, _ = exported
    relocated = tmp_path / "another-machine" / "bundle"
    relocated.parent.mkdir()
    shutil.copytree(target, relocated)
    for key in ("index", "corpus", "cache"):
        shutil.rmtree(source[key])
    monkeypatch.setattr(
        portable,
        "prepare_source",
        Mock(side_effect=AssertionError("must not consult source")),
    )
    before = snapshot_files(relocated)
    result = portable.verify_bundle(relocated)
    assert result["document_count"] == 2
    assert result["dimension"] == 4
    assert result["model"] == "offline-embedding:fixture"
    assert result["self_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert result["network_requests"] == result["new_embeddings"] == 0
    assert snapshot_files(relocated) == before


@pytest.mark.parametrize("existing", ["directory", "file", "symlink", "broken_symlink"])
def test_existing_output_is_never_overwritten(source, tmp_path, existing):
    prepared = prepare(source)
    target = tmp_path / "occupied"
    if existing == "directory":
        target.mkdir()
        (target / "failure-scene").write_text("retain me")
    elif existing == "file":
        target.write_text("retain me")
    else:
        destination = tmp_path / "other"
        if existing == "symlink":
            destination.mkdir()
        target.symlink_to(destination, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        portable.export_bundle(prepared, target)
    assert target.is_symlink() if "symlink" in existing else target.exists()


@pytest.mark.parametrize("root", ["index", "corpus", "cache"])
def test_output_inside_any_source_root_is_rejected_before_write(source, root):
    prepared = prepare(source)
    before = snapshot_files(source[root])
    target = source[root] / "must-not-create"
    with pytest.raises(ValueError):
        portable.export_bundle(prepared, target)
    assert not target.exists()
    assert snapshot_files(source[root]) == before


@pytest.mark.parametrize("name", ["manifest.json", "records.jsonl", "vectors.npy"])
def test_missing_required_file_is_rejected(exported, name):
    _, target, _ = exported
    (target / name).unlink()
    with pytest.raises((ValueError, OSError)):
        portable.load_bundle(target)


@pytest.mark.parametrize("name", ["manifest.json", "records.jsonl", "vectors.npy"])
def test_package_file_symlink_is_rejected(exported, tmp_path, name):
    _, target, _ = exported
    original = tmp_path / f"real-{name}"
    shutil.move(target / name, original)
    (target / name).symlink_to(original)
    with pytest.raises(ValueError):
        portable.load_bundle(target)


def test_package_directory_symlink_is_rejected(exported, tmp_path):
    _, target, _ = exported
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        portable.load_bundle(alias)


def test_extra_file_is_rejected(exported):
    _, target, _ = exported
    (target / "hidden-source-pointer.json").write_text("{}")
    with pytest.raises(ValueError):
        portable.load_bundle(target)


@pytest.mark.parametrize("name", ["records.jsonl", "vectors.npy"])
def test_file_bytes_tamper_is_rejected(exported, name):
    _, target, _ = exported
    path = target / name
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError):
        portable.load_bundle(target)


def rewrite_manifest(target, manifest):
    (target / "manifest.json").write_text(
        portable.canonical(manifest) + "\n", encoding="utf-8"
    )


def rehash_payload(target, manifest, name):
    raw = (target / name).read_bytes()
    manifest["files"][name] = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    rewrite_manifest(target, manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_ids",
        "id_metadata_disagree",
        "empty_text",
        "local_path",
        "extra_top_field",
        "records_order",
    ],
)
def test_rehashed_records_still_obey_identity_and_schema(exported, mutation):
    _, target, manifest = exported
    rows = portable.load_bundle(target)["records"]
    if mutation == "duplicate_ids":
        rows[1]["chunk_id"] = rows[0]["chunk_id"]
        rows[1]["metadata"]["chunk_id"] = rows[0]["chunk_id"]
        manifest["chunk_ids"][1] = manifest["chunk_ids"][0]
    elif mutation == "id_metadata_disagree":
        rows[0]["chunk_id"] = "rehashed-new-id"
        manifest["chunk_ids"][0] = "rehashed-new-id"
    elif mutation == "empty_text":
        rows[0]["text"] = "  \n"
    elif mutation == "local_path":
        rows[0]["metadata"]["source"] = "/not/a/portable/path"
    elif mutation == "extra_top_field":
        rows[0]["database_path"] = "/private/runtime.db"
    elif mutation == "records_order":
        rows.reverse()
    (target / "records.jsonl").write_text(
        "".join(portable.canonical(row) + "\n" for row in rows), encoding="utf-8"
    )
    rehash_payload(target, manifest, "records.jsonl")
    with pytest.raises((ValueError, KeyError, TypeError)):
        portable.load_bundle(target)


@pytest.mark.parametrize(
    "mutation",
    ["zero", "nan", "inf", "float64", "wrong_shape", "swapped_rows", "object_pickle"],
)
def test_rehashed_npy_still_obeys_numeric_and_alignment_contract(exported, mutation):
    _, target, manifest = exported
    matrix = np.asarray(portable.load_bundle(target)["vectors"]).copy()
    if mutation == "zero":
        matrix[0] = 0
    elif mutation == "nan":
        matrix[0, 0] = np.nan
    elif mutation == "inf":
        matrix[0, 0] = np.inf
    elif mutation == "float64":
        matrix = matrix.astype("<f8")
    elif mutation == "wrong_shape":
        matrix = matrix.reshape(1, 8)
    elif mutation == "swapped_rows":
        matrix = matrix[::-1].copy()
    elif mutation == "object_pickle":
        matrix = np.asarray([[{"arbitrary": "object"}]], dtype=object)
    with (target / "vectors.npy").open("wb") as stream:
        np.save(stream, matrix, allow_pickle=True)
    rehash_payload(target, manifest, "vectors.npy")
    with pytest.raises((ValueError, TypeError)):
        portable.load_bundle(target)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "status",
        "count_bool",
        "dimension_bool",
        "dimension_too_large",
        "missing_file_hash",
        "vector_map_key",
        "license_text",
        "license_hash",
        "endpoint",
        "snapshot_sha",
        "extra_field",
    ],
)
def test_manifest_contract_tamper_is_rejected(exported, mutation):
    _, target, manifest = exported
    if mutation == "schema":
        manifest["schema_version"] = "portable-kb/v999"
    elif mutation == "status":
        manifest["status"] = "building"
    elif mutation == "count_bool":
        manifest["document_count"] = True
    elif mutation == "dimension_bool":
        manifest["embedding"]["actual_dimension"] = True
    elif mutation == "dimension_too_large":
        manifest["embedding"]["actual_dimension"] = 8193
    elif mutation == "missing_file_hash":
        del manifest["files"]["vectors.npy"]
    elif mutation == "vector_map_key":
        manifest["vector_sha256"]["unmapped-chunk"] = "a" * 64
    elif mutation == "license_text":
        manifest["license"]["text"] += "tampered license"
    elif mutation == "license_hash":
        manifest["license"]["sha256"] = "b" * 64
    elif mutation == "endpoint":
        manifest["embedding"]["endpoint"] = "http://127.0.0.1:11434/api/embed"
    elif mutation == "snapshot_sha":
        manifest["original_vector_snapshot_sha256"] = "f" * 64
    elif mutation == "extra_field":
        manifest["source_cache_path"] = "/private/cache.sqlite3"
    rewrite_manifest(target, manifest)
    with pytest.raises((ValueError, KeyError, TypeError)):
        portable.load_bundle(target)


@pytest.mark.parametrize(
    "constant", ["MAX_RECORD_BYTES", "MAX_MANIFEST_BYTES", "MAX_VECTOR_BYTES"]
)
def test_file_size_bound_is_enforced_before_loading_payload(
    exported, monkeypatch, constant
):
    _, target, _ = exported
    monkeypatch.setattr(portable, constant, 1)
    with pytest.raises(ValueError):
        portable.load_bundle(target)


def test_untrusted_manifest_name_cannot_read_outside_bundle(exported, tmp_path):
    _, target, _ = exported
    outside = tmp_path / "secret.json"
    outside.write_text("do not read")
    with pytest.raises(ValueError):
        portable.load_bundle(target, manifest_name="../secret.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_ids",
        "empty_text",
        "records_disagree",
        "wrong_vector_hash",
        "zero",
        "nan",
        "wrong_dimension",
        "float64",
    ],
)
def test_invalid_source_cannot_be_prepared(source, mutation):
    rows, matrix = source["bundle"]["records"], source["bundle"]["vectors"]
    if mutation == "duplicate_ids":
        rows[1]["chunk_id"] = rows[0]["chunk_id"]
        source["plan"]["documents"] = deepcopy(rows)
    elif mutation == "empty_text":
        rows[0]["text"] = "  \n"
        source["plan"]["documents"] = deepcopy(rows)
    elif mutation == "records_disagree":
        source["plan"]["documents"][0]["text"] = "different frozen plan"
    elif mutation == "wrong_vector_hash":
        source["bundle"]["hashes"][rows[0]["chunk_id"]] = "f" * 64
    else:
        if mutation == "zero":
            matrix[0] = 0
        elif mutation == "nan":
            matrix[0, 0] = np.nan
        elif mutation == "wrong_dimension":
            matrix = matrix[:, :3]
        elif mutation == "float64":
            matrix = matrix.astype("<f8")
        source["bundle"]["vectors"] = matrix
    with pytest.raises(ValueError):
        prepare(source)


@pytest.mark.parametrize("mutation", ["missing", "empty", "too_large", "invalid_utf8"])
def test_source_requires_preservable_license(source, mutation):
    license_file = source["corpus"] / "LICENSE"
    if mutation == "missing":
        license_file.unlink()
    elif mutation == "empty":
        license_file.write_bytes(b"")
    elif mutation == "too_large":
        license_file.write_bytes(b"x" * 65537)
    else:
        license_file.write_bytes(b"\xff\xfe\xfa")
    with pytest.raises((ValueError, OSError)):
        prepare(source)


@pytest.mark.parametrize("count", [0, True, 1001])
def test_source_count_is_bounded_non_boolean_integer(source, count):
    source["plan"]["document_count"] = count
    with pytest.raises(ValueError):
        prepare(source)


@pytest.mark.parametrize("dimension", [1, True, 8193])
def test_source_dimension_is_bounded_non_boolean_integer(source, dimension):
    source["plan"]["embedding"]["actual_dimension"] = dimension
    with pytest.raises(ValueError):
        prepare(source)


@pytest.mark.parametrize("mutation", ["id", "vector", "text", "embedding_model"])
def test_prepared_contract_mutation_cannot_publish_ready(source, tmp_path, mutation):
    prepared = prepare(source)
    if mutation == "id":
        prepared["records"][0]["chunk_id"] = "changed-after-preparation"
    elif mutation == "vector":
        prepared["vectors"][0, 0] += np.float32(0.25)
    elif mutation == "text":
        prepared["records"][0]["text"] = "different valid text after preparation"
    else:
        prepared["embedding"]["model"] = "different-model:fixture"
    target = tmp_path / "mutated-prepared"
    with pytest.raises((ValueError, KeyError, TypeError)):
        portable.export_bundle(prepared, target)
    assert not (target / "manifest.json").exists()


def test_prepare_makes_independent_nested_metadata_copy(source):
    prepared = prepare(source)
    prepared["records"][0]["metadata"]["section_path"].append("only output")
    assert source["bundle"]["records"][0]["metadata"]["section_path"] == [
        "故障排查",
        "DNS",
    ]


def test_export_failure_never_publishes_ready_manifest(source, tmp_path, monkeypatch):
    prepared = prepare(source)
    target = tmp_path / "interrupted"
    monkeypatch.setattr(np, "save", Mock(side_effect=OSError("synthetic save failure")))
    with pytest.raises((ValueError, OSError)):
        portable.export_bundle(prepared, target)
    assert not (target / "manifest.json").exists()
    if target.exists():
        with pytest.raises((ValueError, OSError)):
            portable.export_bundle(prepared, target)


def test_default_cli_prints_offline_plan_without_export(
    source, tmp_path, monkeypatch, capsys
):
    prepared = prepare(source)
    target = tmp_path / "plan-target"
    loader = Mock(return_value=prepared)
    monkeypatch.setattr(portable, "prepare_source", loader)
    monkeypatch.setattr(
        portable,
        "export_bundle",
        Mock(side_effect=AssertionError("plan must not write")),
    )
    assert (
        portable.main(
            ["--index-dir", str(source["index"]), "--output-dir", str(target)]
        )
        == 0
    )
    assert not target.exists()
    assert "0" in capsys.readouterr().out


def test_cli_plan_does_not_import_service_or_secret_sdks(source, tmp_path, monkeypatch):
    prepared = prepare(source)
    monkeypatch.setattr(portable, "prepare_source", Mock(return_value=prepared))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "pymilvus",
            "requests",
            "redis",
            "sqlalchemy",
            "openai",
            "dotenv",
        }
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert portable.main(["--output-dir", str(tmp_path / "no-write")]) == 0


def test_verify_cli_never_prepares_source_or_exports(exported, monkeypatch, capsys):
    _, target, _ = exported
    monkeypatch.setattr(
        portable,
        "prepare_source",
        Mock(side_effect=AssertionError("verify must be portable")),
    )
    monkeypatch.setattr(
        portable,
        "export_bundle",
        Mock(side_effect=AssertionError("verify must not export")),
    )
    before = snapshot_files(target)
    assert portable.main(["--verify", str(target)]) == 0
    assert snapshot_files(target) == before
    assert "0" in capsys.readouterr().out


def test_run_cli_exports_once(source, tmp_path, monkeypatch):
    prepared = prepare(source)
    target = tmp_path / "cli-output"
    monkeypatch.setattr(portable, "prepare_source", Mock(return_value=prepared))
    assert (
        portable.main(
            ["--index-dir", str(source["index"]), "--output-dir", str(target), "--run"]
        )
        == 0
    )
    assert portable.verify_bundle(target)["document_count"] == 2
