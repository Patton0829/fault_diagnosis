"""Rebuild only the preflight-approved source scope, without services or models."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import socket

import pytest
import requests

from qa_platform.ingestion import kubernetes_inventory as inventory
from qa_platform.ingestion.kubernetes_corpus import verify_directory, write_bundle


build = importlib.import_module("qa_platform.ingestion.kubernetes_full_corpus")
COMMIT = "a" * 40
BODY = "Pod 的 DNS 解析失败时，先核对命名空间与 DNS 配置。这是合成测试资料。\n"


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Full corpus preparation must not access network or models")

    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def page(root, relative, body=BODY, *, title="合成正文", front=""):
    path = root / "upstream/content/zh-cn/docs" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {title}\n{front}---\n{body}", encoding="utf-8")
    return path


def bytes_below(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def scanned(tmp_path):
    root = tmp_path / "source"
    upstream = root / "upstream"
    upstream.mkdir(parents=True)
    source = {
        "source": {"commit_sha": COMMIT},
        "license": {"spdx": "CC-BY-4.0", "attribution": "Synthetic test fixtures"},
        "counts": {"zh_docs_markdown_files": 5},
    }
    (root / "source_manifest.json").write_text(json.dumps(source), encoding="utf-8")
    (upstream / "hugo.toml").write_text('[params]\nversion = "v1.37"\n', encoding="utf-8")
    (upstream / "LICENSE").write_text("Synthetic CC BY 4.0 license fixture\n", encoding="utf-8")
    include = upstream / "content/zh-cn/includes/shared.md"
    include.parent.mkdir(parents=True)
    include.write_text(BODY, encoding="utf-8")
    page(root, "tasks/good.md", '{{< include "shared.md" >}}\n')
    page(
        root,
        "reference/glossary/dns.md",
        title="DNS 合成词条",
        front='min-kubernetes-server-version: "v1.25"\n',
    )
    page(root, "tasks/unknown.md", BODY + "{{< unknown-template >}}\n")
    page(root, "tasks/_index.md", "这是目录导航的合成正文。\n")
    page(root, "tasks/draft.md", front="draft: true\n")
    report = inventory.scan_snapshot(root)
    assert report["summary"]["eligible_documents"] == 2
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return root, report_path, report


def save_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def prepare(scanned):
    root, report_path, _ = scanned
    return build.prepare_from_report(report_path, corpus_root=root)


def test_rebuilds_only_the_approved_pages_and_keeps_every_report_record(scanned):
    root, _, report = scanned
    before = bytes_below(root)

    batch = prepare(scanned)

    assert len(batch["parents"]) == report["summary"]["eligible_documents"] == 2
    assert len(batch["chunks"]) == report["summary"]["candidate_chunks"]
    assert len(batch["audits"]) == 2
    rows = {row["path"]: row for row in batch["report"]["records"]}
    assert set(rows) == {row["relative_path"] for row in report["records"]}
    assert rows["tasks/good.md"]["status"] == "prepared"
    assert rows["reference/glossary/dns.md"]["status"] == "prepared"
    assert rows["tasks/_index.md"]["status"] == "excluded_by_scope"
    assert rows["tasks/draft.md"]["status"] == "excluded_by_scope"
    assert rows["tasks/unknown.md"]["status"] == "needs_review_cleaning"
    assert batch["report"]["candidate_chunks"] == len(batch["chunks"])
    assert batch["source"]["source"]["commit_sha"] == batch["commit"] == COMMIT
    assert batch["source_sha256"] == sha256((root / "source_manifest.json").read_bytes()).hexdigest()
    assert batch["license_bytes"] == (root / "upstream/LICENSE").read_bytes()
    assert batch["config_sha256"] == sha256((root / "upstream/hugo.toml").read_bytes()).hexdigest()
    assert bytes_below(root) == before


def test_source_roles_frontmatter_review_and_dependencies_survive(scanned):
    batch = prepare(scanned)
    parents = {row["metadata"]["corpus_relative_path"]: row for row in batch["parents"]}
    regular = parents["tasks/good.md"]["metadata"]
    glossary = parents["reference/glossary/dns.md"]["metadata"]

    assert regular["source_role_hint"] == "regular_page"
    assert glossary["source_role_hint"] == "glossary_data"
    assert glossary["front_matter"]["min-kubernetes-server-version"] == "v1.25"
    assert any(row["path"].endswith("shared.md") for row in regular["dependencies"])
    for row in batch["parents"] + batch["chunks"]:
        metadata = row["metadata"]
        assert metadata["review_status"] == "unreviewed_official_document"
        assert metadata["ready_for_embedding"] is False
        assert metadata["token_limit_checked"] is False
        assert metadata["source_role_hint"] in {"regular_page", "glossary_data"}
        assert metadata["snapshot_commit"] == COMMIT


def test_bundle_uses_existing_frozen_verifier_and_verifies_without_upstream(scanned, tmp_path):
    root, _, _ = scanned
    batch = prepare(scanned)
    output = tmp_path / "new-corpus"

    manifest = write_bundle(batch, output, 800)

    assert manifest["status"] == "prepared_partial"
    assert manifest["prepared_documents"] == 2
    assert manifest["candidate_chunks"] == len(batch["chunks"])
    assert manifest["ready_for_embedding"] is False
    root.rename(tmp_path / "moved-away-source")
    assert verify_directory(output) == manifest


def test_export_does_not_overwrite_existing_bundle(scanned, tmp_path):
    output = tmp_path / "existing"
    batch = prepare(scanned)
    write_bundle(batch, output, 800)
    before = bytes_below(output)
    with pytest.raises(FileExistsError):
        write_bundle(batch, output, 800)
    assert bytes_below(output) == before


def test_preparation_never_opens_environment_files_or_executes_document_commands(scanned, monkeypatch, tmp_path):
    root, report_path, _ = scanned
    sentinel = tmp_path / "must-not-exist"
    command = f"\n```shell\ntouch {sentinel}\n```\n"
    page(root, "tasks/good.md", BODY + command)
    report = inventory.scan_snapshot(root)
    save_report(report_path, report)
    original_read = Path.read_bytes

    def guarded_read(path):
        assert path.name != ".env" and path.suffix != ".env"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    batch = prepare((root, report_path, report))
    assert any(f"touch {sentinel}" in row["page_content"] for row in batch["chunks"])
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "target",
    [
        "upstream/content/zh-cn/docs/tasks/good.md",
        "upstream/content/zh-cn/docs/tasks/unknown.md",
        "upstream/content/zh-cn/docs/tasks/_index.md",
        "upstream/content/zh-cn/docs/tasks/draft.md",
        "upstream/content/zh-cn/includes/shared.md",
        "upstream/hugo.toml",
    ],
)
def test_any_fingerprinted_input_change_is_rejected(scanned, target):
    root, _, _ = scanned
    path = root / target
    path.write_bytes(path.read_bytes() + b"\nchanged after preflight\n")
    with pytest.raises((ValueError, OSError)):
        prepare(scanned)


def test_snapshot_commit_change_is_rejected(scanned):
    root, _, _ = scanned
    path = root / "source_manifest.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["source"]["commit_sha"] = "b" * 40
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare(scanned)


@pytest.mark.parametrize("kind", ["add", "remove", "symlink"])
def test_changed_markdown_scope_is_rejected(scanned, kind):
    root, _, _ = scanned
    if kind == "add":
        page(root, "tasks/added.md")
    elif kind == "remove":
        (root / "upstream/content/zh-cn/docs/tasks/good.md").unlink()
    else:
        path = root / "upstream/content/zh-cn/docs/tasks/good.md"
        content = root / "upstream/content/zh-cn/includes/shared.md"
        path.unlink()
        path.symlink_to(content)
    with pytest.raises((ValueError, OSError)):
        prepare(scanned)


def change(report, key, value):
    keys = key.split(".")
    target = report
    for part in keys[:-1]:
        target = target[part]
    target[keys[-1]] = value


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", "k8s-full-preflight/v0"),
        ("status", "source_changed"),
        ("source_commit", "b" * 40),
        ("summary.source_unchanged", False),
        ("summary.source_unchanged", 1),
        ("summary.eligible_documents", 3),
        ("summary.candidate_chunks", 999),
        ("summary.eligible_documents", True),
        ("summary.candidate_chunks", 2.0),
        ("summary.markdown_files", 4),
        ("summary.over_budget_chunks", 1),
        ("summary.max_chunk_chars", 1),
        ("parameters.rule_sha256", {}),
        ("parameters.chunk_size", True),
        ("changed_sources", ["content/zh-cn/docs/tasks/good.md"]),
    ],
)
def test_report_contract_and_counts_are_strict(scanned, key, value):
    _, report_path, original = scanned
    report = deepcopy(original)
    change(report, key, value)
    save_report(report_path, report)
    with pytest.raises((ValueError, TypeError, KeyError)):
        prepare(scanned)


@pytest.mark.parametrize(
    "key,value",
    [
        ("chunk_count", 999),
        ("max_chunk_chars", 1),
        ("over_budget_chunks", 1),
        ("raw_sha256", "b" * 64),
        ("raw_bytes", 1),
        ("source_role", "feature_gate_data"),
        ("relative_path", "../escape.md"),
    ],
)
def test_per_page_statistics_paths_and_role_cannot_be_forged(scanned, key, value):
    _, report_path, original = scanned
    report = deepcopy(original)
    row = next(row for row in report["records"] if row["relative_path"] == "tasks/good.md")
    row[key] = value
    save_report(report_path, report)
    with pytest.raises((ValueError, OSError, KeyError)):
        prepare(scanned)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "out-of-order"])
def test_inventory_records_are_a_complete_unique_ordered_scope(scanned, kind):
    _, report_path, original = scanned
    report = deepcopy(original)
    if kind == "missing":
        report["records"].pop()
    elif kind == "duplicate":
        report["records"].append(deepcopy(report["records"][0]))
    else:
        report["records"].reverse()
    save_report(report_path, report)
    with pytest.raises(ValueError):
        prepare(scanned)


@pytest.mark.parametrize("kind", ["changed", "missing", "escape"])
def test_dependency_report_is_checked_against_current_inputs(scanned, kind):
    _, report_path, original = scanned
    report = deepcopy(original)
    key = "content/zh-cn/includes/shared.md"
    assert key in report["dependency_sha256"]
    if kind == "changed":
        report["dependency_sha256"][key] = "b" * 64
    elif kind == "missing":
        del report["dependency_sha256"][key]
    else:
        report["dependency_sha256"]["../outside"] = "b" * 64
    save_report(report_path, report)
    with pytest.raises((ValueError, OSError, KeyError)):
        prepare(scanned)


def test_deferred_unknown_template_cannot_be_promoted_by_editing_report(scanned):
    _, report_path, original = scanned
    report = deepcopy(original)
    row = next(row for row in report["records"] if row["relative_path"] == "tasks/unknown.md")
    row.update(status="prepared_for_planning", messages=[], chunk_count=1)
    report["summary"]["eligible_documents"] += 1
    report["summary"]["candidate_chunks"] += 1
    report["summary"]["status_counts"]["prepared_for_planning"] += 1
    report["summary"]["status_counts"]["needs_review_cleaning"] -= 1
    save_report(report_path, report)
    with pytest.raises(ValueError):
        prepare(scanned)


@pytest.mark.parametrize("relative", ["tasks/_index.md", "tasks/draft.md"])
def test_navigation_or_hidden_page_cannot_be_promoted(scanned, relative):
    _, report_path, original = scanned
    report = deepcopy(original)
    row = next(row for row in report["records"] if row["relative_path"] == relative)
    row.update(status="prepared_for_planning", chunk_count=1)
    report["summary"]["eligible_documents"] += 1
    report["summary"]["candidate_chunks"] += 1
    save_report(report_path, report)
    with pytest.raises(ValueError):
        prepare(scanned)


def test_rule_code_changes_after_inventory_are_rejected(scanned, monkeypatch):
    original = build.corpus.rule_hashes
    monkeypatch.setattr(build.corpus, "rule_hashes", lambda: {**original(), "changed.py": "c" * 64})
    with pytest.raises(ValueError):
        prepare(scanned)
