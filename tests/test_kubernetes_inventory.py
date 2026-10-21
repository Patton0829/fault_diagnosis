"""Whole-snapshot planning tests: temporary sources, real cleaner, zero services."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import socket

import pytest


inventory = importlib.import_module("qa_platform.ingestion.kubernetes_inventory")
COMMIT = "a" * 40
BODY = "这是用于离线预检的合成说明。Pod 的 DNS 解析失败时，先核对名称空间与配置。\n"


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "source"
    docs = root / "upstream/content/zh-cn/docs"
    docs.mkdir(parents=True)
    (root / "source_manifest.json").write_text(
        json.dumps({"source": {"commit_sha": COMMIT}}), encoding="utf-8"
    )
    (root / "upstream/hugo.toml").write_text(
        '[params]\nversion = "v1.37"\n', encoding="utf-8"
    )
    return root


def page(corpus, relative, body=BODY, *, title="测试说明", front=""):
    path = corpus / "upstream/content/zh-cn/docs" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {title}\n{front}---\n{body}", encoding="utf-8")
    return path


def source_bytes(corpus):
    return {
        path.relative_to(corpus).as_posix(): path.read_bytes()
        for path in corpus.rglob("*")
        if path.is_file()
    }


def by_path(report):
    return {row["relative_path"]: row for row in report["records"]}


def test_real_cleaning_and_chunking_classify_every_markdown(corpus):
    page(corpus, "tasks/good.md", "<!-- removable English -->\n" + BODY)
    page(corpus, "tasks/_index.md", "目录导航可以有正文，但本轮按范围排除。\n")
    page(corpus, "concepts/draft.md", front="draft: true\n")
    page(corpus, "concepts/headless.md", front="headless: true\n")
    page(corpus, "tasks/template.md", BODY + "{{< unrecognized-widget >}}\n")
    page(corpus, "tasks/comment.md", BODY + "<!-- not closed\n")
    page(corpus, "tasks/orphan.md", "## 孤立标题\n")
    broken = page(corpus, "tasks/invalid-utf8.md")
    broken.write_bytes(b"\xff\xfe\x00")
    before = source_bytes(corpus)

    report = inventory.scan_snapshot(corpus)
    rows = by_path(report)

    assert rows["tasks/good.md"]["status"] == "prepared_for_planning"
    assert rows["tasks/_index.md"]["status"] == "excluded_navigation"
    assert rows["concepts/draft.md"]["status"] == "excluded_visibility"
    assert rows["concepts/headless.md"]["status"] == "excluded_visibility"
    assert rows["tasks/template.md"]["status"] == "needs_review_cleaning"
    assert rows["tasks/comment.md"]["status"] == "error_cleaning"
    assert rows["tasks/orphan.md"]["status"] == "needs_review_chunking"
    assert rows["tasks/invalid-utf8.md"]["status"] == "error_cleaning"
    assert report["summary"]["markdown_files"] == 8
    assert report["summary"]["eligible_documents"] == 1
    assert sum(report["summary"]["status_counts"].values()) == 8
    assert all(
        row["chunk_count"] == 0
        for row in rows.values()
        if row["status"] != "prepared_for_planning"
    )
    assert rows["tasks/good.md"]["cleaned_chars"] < rows["tasks/good.md"]["raw_bytes"]
    assert report["summary"]["source_unchanged"] is True
    assert source_bytes(corpus) == before


def test_estimates_use_candidate_chunks_not_document_or_all_file_counts(corpus):
    for index in range(9):
        page(corpus, f"tasks/page-{index}.md", title=f"离线说明 {index}")
    page(corpus, "tasks/_index.md")
    page(corpus, "tasks/unknown.md", BODY + "{{< unsupported >}}\n")

    report = inventory.scan_snapshot(corpus, chunk_size=800, batch_size=8)

    assert report["summary"]["markdown_files"] == 11
    assert report["summary"]["eligible_documents"] == 9
    assert report["summary"]["candidate_chunks"] == 9
    assert report["summary"]["embedding_batches"] == 2
    assert report["summary"]["raw_vector_bytes"] == 9 * 4096 * 4
    assert report["summary"]["all_documents_chunk_count"] is None
    assert report["parameters"]["model_requests"] == 0
    assert report["parameters"]["token_limit_checked"] is False
    assert report["sections"]["tasks"]["candidate_chunks"] == 9


def test_overlong_code_is_preserved_and_counted_not_silently_truncated(corpus):
    code = "\n".join(f"printf 'synthetic line {index}'" for index in range(50))
    page(corpus, "tasks/code.md", BODY + "\n```shell\n" + code + "\n```\n")

    report = inventory.scan_snapshot(corpus, chunk_size=100)
    row = report["records"][0]

    assert row["status"] == "prepared_for_planning"
    assert row["over_budget_chunks"] >= 1
    assert row["max_chunk_chars"] >= len(code)
    assert report["summary"]["over_budget_chunks"] == row["over_budget_chunks"]
    assert report["summary"]["max_chunk_chars"] == row["max_chunk_chars"]


def test_only_lowercase_md_under_chinese_docs_is_scanned(corpus):
    page(corpus, "tasks/deep/nested.md")
    page(corpus, "ROOT.md")
    docs = corpus / "upstream/content/zh-cn/docs"
    (docs / "skip.MD").write_text("not in lowercase glob", encoding="utf-8")
    (docs / "skip.yaml").write_text("kind: Pod", encoding="utf-8")
    foreign = corpus / "upstream/content/en/docs"
    foreign.mkdir(parents=True)
    (foreign / "not-chinese.md").write_text("not this scope", encoding="utf-8")

    report = inventory.scan_snapshot(corpus)

    assert sorted(by_path(report)) == ["ROOT.md", "tasks/deep/nested.md"]
    assert by_path(report)["ROOT.md"]["section"] == "root"
    assert by_path(report)["tasks/deep/nested.md"]["section"] == "tasks"


def test_report_does_not_export_body_text_or_vectors(corpus):
    secret_body = "BODY_ONLY_SENTINEL_NEVER_COPY_TO_REPORT"
    path = page(corpus, "tasks/check.md", BODY + secret_body)

    report = inventory.scan_snapshot(corpus)
    encoded = json.dumps(report, ensure_ascii=False)
    row = report["records"][0]

    assert row["raw_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert row["raw_bytes"] == len(path.read_bytes())
    assert secret_body not in encoded
    assert not {"page_content", "text", "vectors", "embedding"}.intersection(row)
    assert "not a token" not in encoded  # No fabricated tokenizer observation.


@pytest.mark.parametrize(
    "name,value",
    [
        ("chunk_size", 99),
        ("chunk_size", 10001),
        ("chunk_size", True),
        ("chunk_size", 800.0),
        ("chunk_size", "800"),
        ("batch_size", 0),
        ("batch_size", 257),
        ("batch_size", False),
        ("batch_size", 8.0),
        ("batch_size", "8"),
    ],
)
def test_invalid_parameters_fail_before_processing(corpus, name, value):
    page(corpus, "tasks/good.md")
    with pytest.raises(ValueError):
        inventory.scan_snapshot(corpus, **{name: value})


@pytest.mark.parametrize("commit", ["a" * 39, "A" * 40, "../bad", 123, None])
def test_invalid_snapshot_identity_is_rejected(corpus, commit):
    page(corpus, "tasks/good.md")
    (corpus / "source_manifest.json").write_text(
        json.dumps({"source": {"commit_sha": commit}}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        inventory.scan_snapshot(corpus)


def test_empty_snapshot_is_not_reported_as_success(corpus):
    with pytest.raises(ValueError):
        inventory.scan_snapshot(corpus)


def test_zero_network_and_environment_file_reads(corpus, monkeypatch):
    page(corpus, "tasks/good.md")
    before = source_bytes(corpus)
    original_read = Path.read_bytes

    def guarded_read(path):
        assert path.suffix != ".env" and path.name != ".env"
        return original_read(path)

    def no_network(*args, **kwargs):
        raise AssertionError("An offline inventory must not contact any service")

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)

    report = inventory.scan_snapshot(corpus)

    assert report["summary"]["eligible_documents"] == 1
    assert source_bytes(corpus) == before


def test_dependency_expansion_is_read_only_and_audited(corpus):
    page(corpus, "tasks/include.md", '{{< include "shared.md" >}}\n')
    include = corpus / "upstream/content/zh-cn/includes/shared.md"
    include.parent.mkdir(parents=True)
    include.write_text(BODY, encoding="utf-8")
    before = source_bytes(corpus)

    report = inventory.scan_snapshot(corpus)

    assert report["summary"]["eligible_documents"] == 1
    assert (
        report["dependency_sha256"]["content/zh-cn/includes/shared.md"]
        == sha256(include.read_bytes()).hexdigest()
    )
    assert report["summary"]["source_unchanged"] is True
    assert source_bytes(corpus) == before


def test_source_change_invalidates_estimations(corpus):
    source = page(corpus, "tasks/good.md")

    def change_after_processing(done, total):
        assert done == total == 1
        source.write_text(
            source.read_text(encoding="utf-8") + "changed\n", encoding="utf-8"
        )

    report = inventory.scan_snapshot(corpus, progress=change_after_processing)

    assert report["summary"]["source_unchanged"] is False
    assert report["status"] == "source_changed"
    assert report["summary"]["embedding_batches"] is None
    assert report["summary"]["raw_vector_bytes"] is None
    assert "content/zh-cn/docs/tasks/good.md" in report["changed_sources"]


def test_source_file_set_change_is_detected(corpus):
    page(corpus, "tasks/good.md")
    report = inventory.scan_snapshot(
        corpus, progress=lambda *_: page(corpus, "tasks/added.md")
    )
    assert report["summary"]["source_unchanged"] is False
    assert "Markdown文件集合" in report["changed_sources"]


@pytest.mark.parametrize("relative", ["source_manifest.json", "upstream/hugo.toml"])
def test_snapshot_manifest_and_config_changes_are_detected(corpus, relative):
    page(corpus, "tasks/good.md")
    target = corpus / relative

    def change_after_processing(*_):
        target.write_bytes(target.read_bytes() + b"\n")

    report = inventory.scan_snapshot(corpus, progress=change_after_processing)

    assert report["summary"]["source_unchanged"] is False
    assert report["summary"]["embedding_batches"] is None
    assert relative.removeprefix("upstream/") in report["changed_sources"]


@pytest.mark.parametrize("relative", ["source_manifest.json", "upstream/hugo.toml"])
def test_deleted_manifest_and_config_keep_source_changed_report(corpus, relative):
    page(corpus, "tasks/good.md")

    report = inventory.scan_snapshot(
        corpus, progress=lambda *_: (corpus / relative).unlink()
    )

    assert report["status"] == "source_changed"
    assert report["summary"]["source_unchanged"] is False
    assert report["summary"]["embedding_batches"] is None
    assert report["summary"]["raw_vector_bytes"] is None
    assert relative.removeprefix("upstream/") in report["changed_sources"]


@pytest.mark.parametrize(
    "relative,role",
    [
        (
            "reference/command-line-tools-reference/feature-gates/example.md",
            "feature_gate_data",
        ),
        ("reference/glossary/example.md", "glossary_data"),
        (
            "reference/setup-tools/kubeadm/generated/example.md",
            "generated_reference_fragment",
        ),
        ("tasks/example.md", "regular_page"),
    ],
)
def test_source_roles_are_descriptive_and_preserve_planning_status(
    corpus, relative, role
):
    page(corpus, relative)

    report = inventory.scan_snapshot(corpus)
    row = report["records"][0]

    assert row["source_role"] == role
    assert row["status"] == "prepared_for_planning"
    assert report["eligible_roles"][role] == {"documents": 1, "chunks": 1}
    assert report["summary"]["eligible_documents"] == 1
    assert "ready_for_embedding" not in row


def test_eligible_role_counts_exclude_deferred_and_navigation_pages(corpus):
    page(corpus, "reference/glossary/valid.md")
    page(
        corpus,
        "reference/glossary/deferred.md",
        BODY + "{{< unsupported >}}\n",
    )
    page(corpus, "reference/glossary/_index.md")
    page(corpus, "tasks/valid.md")

    report = inventory.scan_snapshot(corpus)

    assert report["eligible_roles"]["glossary_data"] == {
        "documents": 1,
        "chunks": 1,
    }
    assert report["eligible_roles"]["regular_page"] == {
        "documents": 1,
        "chunks": 1,
    }
    assert sum(item["documents"] for item in report["eligible_roles"].values()) == 2
    assert sum(item["chunks"] for item in report["eligible_roles"].values()) == 2
    assert report["summary"]["markdown_files"] == 4


def test_utf8_bom_is_reported_without_silently_changing_cleaning_rules(corpus):
    path = page(corpus, "tasks/bom.md")
    original = b"\xef\xbb\xbf" + path.read_bytes()
    path.write_bytes(original)
    page(corpus, "tasks/plain.md")

    report = inventory.scan_snapshot(corpus)
    rows = by_path(report)

    assert rows["tasks/bom.md"]["has_utf8_bom"] is True
    assert rows["tasks/bom.md"]["status"] == "error_cleaning"
    assert rows["tasks/bom.md"]["chunk_count"] == 0
    assert rows["tasks/bom.md"]["raw_sha256"] == sha256(original).hexdigest()
    assert rows["tasks/plain.md"]["has_utf8_bom"] is False
    assert rows["tasks/plain.md"]["status"] == "prepared_for_planning"
    assert path.read_bytes() == original
    assert report["summary"]["source_unchanged"] is True


def test_expanded_dependency_change_is_detected(corpus):
    page(corpus, "tasks/include.md", '{{< include "shared.md" >}}\n')
    include = corpus / "upstream/content/zh-cn/includes/shared.md"
    include.parent.mkdir(parents=True)
    include.write_text(BODY, encoding="utf-8")

    def change_after_processing(*_):
        include.write_text(BODY + "依赖已改变。\n", encoding="utf-8")

    report = inventory.scan_snapshot(corpus, progress=change_after_processing)

    assert report["summary"]["source_unchanged"] is False
    assert "content/zh-cn/includes/shared.md" in report["changed_sources"]
    assert report["summary"]["raw_vector_bytes"] is None


def test_processing_rule_change_invalidates_snapshot(corpus, monkeypatch):
    page(corpus, "tasks/good.md")
    real_hashes = inventory.rule_hashes

    def change_rules_after_processing(*_):
        monkeypatch.setattr(
            inventory, "rule_hashes", lambda: {**real_hashes(), "changed.py": "b" * 64}
        )

    report = inventory.scan_snapshot(corpus, progress=change_rules_after_processing)

    assert report["summary"]["source_unchanged"] is False
    assert "清洗/切块规则源码" in report["changed_sources"]
    assert report["summary"]["embedding_batches"] is None


def test_symbolic_link_documents_are_not_followed(corpus, tmp_path):
    page(corpus, "tasks/good.md")
    target = tmp_path / "outside.md"
    target.write_text("---\ntitle: outside\n---\n" + BODY, encoding="utf-8")
    link = corpus / "upstream/content/zh-cn/docs/tasks/linked.md"
    link.symlink_to(target)

    report = inventory.scan_snapshot(corpus)
    row = by_path(report)["tasks/linked.md"]

    assert row["status"] == "error_cleaning"
    assert row["raw_sha256"] is None
    assert row["chunk_count"] == 0
    assert report["summary"]["source_unchanged"] is False
    assert target.read_text(encoding="utf-8").startswith("---\ntitle: outside")


def test_unsupported_dependency_defers_whole_document_not_partial_text(corpus):
    page(corpus, "tasks/missing-include.md", BODY + '{{< include "missing.md" >}}\n')

    report = inventory.scan_snapshot(corpus)

    assert report["records"][0]["status"] == "needs_review_cleaning"
    assert report["records"][0]["chunk_count"] == 0
    assert report["summary"]["eligible_documents"] == 0
    assert report["summary"]["candidate_chunks"] == 0
    assert report["summary"]["embedding_batches"] == 0
    assert report["summary"]["raw_vector_bytes"] == 0


def test_index_visibility_and_cleaning_priority_is_explicit(corpus):
    # An invalid directory landing page remains scope-excluded, not silently empty.
    index = corpus / "upstream/content/zh-cn/docs/_index.md"
    index.write_bytes(b"\xff")
    page(corpus, "tasks/hidden.md", "{{< unsupported >}}", front="draft: true\n")

    report = inventory.scan_snapshot(corpus)

    assert by_path(report)["_index.md"]["status"] == "excluded_navigation"
    assert by_path(report)["tasks/hidden.md"]["status"] == "excluded_visibility"
    assert report["summary"]["eligible_documents"] == 0


def test_cleaning_error_is_per_document_not_fatal_to_remaining_files(corpus):
    missing_title = corpus / "upstream/content/zh-cn/docs/a-invalid.md"
    missing_title.write_text("---\nweight: 3\n---\n" + BODY, encoding="utf-8")
    page(corpus, "z-valid.md")

    report = inventory.scan_snapshot(corpus)

    assert by_path(report)["a-invalid.md"]["status"] == "error_cleaning"
    assert by_path(report)["z-valid.md"]["status"] == "prepared_for_planning"
    assert report["summary"]["markdown_files"] == 2
    assert report["summary"]["eligible_documents"] == 1


def test_existing_output_is_never_overwritten(corpus, tmp_path):
    page(corpus, "tasks/good.md")
    report = inventory.scan_snapshot(corpus)
    output = tmp_path / "report-output"
    inventory.write_report(report, output)
    before = source_bytes(output)

    assert set(before) == {"report.json", "report.html"}
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
    with pytest.raises(ValueError):
        inventory.write_report(report, output)
    assert source_bytes(output) == before


@pytest.mark.parametrize(
    "kind", ["file", "empty_directory", "symlink", "dangling_symlink"]
)
def test_any_existing_target_is_rejected(corpus, tmp_path, kind):
    page(corpus, "tasks/good.md")
    report = inventory.scan_snapshot(corpus)
    output = tmp_path / "occupied"
    if kind == "file":
        output.write_text("keep", encoding="utf-8")
    elif kind == "empty_directory":
        output.mkdir()
    else:
        target = tmp_path / "target"
        if kind == "symlink":
            target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        inventory.write_report(report, output)


def test_html_escapes_untrusted_labels_and_messages(corpus):
    page(corpus, "tasks/good.md")
    report = inventory.scan_snapshot(corpus)
    attack = '<img src=x onerror="alert(1)">'
    row = report["records"][0]
    row["relative_path"] = "tasks/" + attack + ".md"
    row["messages"] = [attack, "' onclick='alert(2)"]
    report["source_commit"] = attack
    report["warning_documents"] = {attack: 1}
    report["template_warning_documents"] = {attack: 1}
    report["sections"] = {
        attack: {"markdown_files": 1, "eligible_documents": 1, "candidate_chunks": 1}
    }
    report["summary"]["full_count_note"] = attack
    rendered = inventory.render_html(report)

    assert attack not in rendered
    assert "&lt;img" in rendered
    assert "<script" not in rendered.lower()
    assert "Content-Security-Policy" in rendered
    assert "default-src 'none'" in rendered
    assert "<iframe" not in rendered.lower()


def test_html_escapes_summary_and_parameter_fields(corpus):
    page(corpus, "tasks/good.md")
    report = inventory.scan_snapshot(corpus)
    attack = "<script>alert('numbers-are-still-input')</script>"
    report["summary"]["eligible_documents"] = attack
    report["parameters"]["chunk_size"] = attack
    assert attack not in inventory.render_html(report)


def test_plan_mode_does_not_scan_or_write(corpus, tmp_path, monkeypatch, capsys):
    page(corpus, "tasks/good.md")
    before = source_bytes(corpus)
    output = tmp_path / "never-created"

    def forbidden(*args, **kwargs):
        raise AssertionError("Plan mode must not clean, chunk, or write")

    monkeypatch.setattr(inventory, "scan_snapshot", forbidden)
    monkeypatch.setattr(inventory, "write_report", forbidden)
    assert (
        inventory.main(["--corpus-root", str(corpus), "--output-dir", str(output)]) == 0
    )
    assert "未逐篇处理" in capsys.readouterr().out
    assert not output.exists()
    assert source_bytes(corpus) == before


def test_run_mode_writes_report_without_changing_snapshot(corpus, tmp_path, capsys):
    page(corpus, "tasks/good.md")
    before = source_bytes(corpus)
    output = tmp_path / "results"

    assert (
        inventory.main(
            ["--run", "--corpus-root", str(corpus), "--output-dir", str(output)]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["candidate_chunks"] == 1
    assert "0模型请求" in capsys.readouterr().out
    assert source_bytes(corpus) == before


def test_main_rejects_output_inside_source(corpus):
    page(corpus, "tasks/good.md")
    output = corpus / "must-not-write"
    before = source_bytes(corpus)
    assert (
        inventory.main(
            ["--run", "--corpus-root", str(corpus), "--output-dir", str(output)]
        )
        == 1
    )
    assert not output.exists()
    assert source_bytes(corpus) == before


def test_main_source_changed_report_returns_failure(corpus, tmp_path, monkeypatch):
    page(corpus, "tasks/good.md")
    report = inventory.scan_snapshot(corpus)
    changed = deepcopy(report)
    changed["summary"]["source_unchanged"] = False
    changed["summary"]["embedding_batches"] = None
    changed["summary"]["raw_vector_bytes"] = None
    monkeypatch.setattr(inventory, "scan_snapshot", lambda *args, **kwargs: changed)
    output = tmp_path / "changed-report"
    assert (
        inventory.main(
            ["--run", "--corpus-root", str(corpus), "--output-dir", str(output)]
        )
        == 1
    )
    assert (output / "report.html").is_file()


def test_main_invalid_parameters_do_not_create_output(corpus, tmp_path):
    page(corpus, "tasks/good.md")
    output = tmp_path / "invalid-output"
    assert (
        inventory.main(
            [
                "--run",
                "--corpus-root",
                str(corpus),
                "--chunk-size",
                "0",
                "--output-dir",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
