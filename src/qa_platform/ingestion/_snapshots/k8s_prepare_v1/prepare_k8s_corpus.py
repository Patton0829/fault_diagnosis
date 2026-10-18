"""批量准备 Kubernetes 运维语料：默认只预检，--run 保存新目录，--verify 离线读回校验。

复用已有清洗/切块/块清理规则；只导出通过当前规则的子集，逐篇保留待处理原因。
准备文件不是向量索引，不调用模型，不读取 .env，不覆盖旧结果，不准备开发题。
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import yaml

from check_k8s_documents import (
    CLEANER_VERSION,
    CORPUS_ROOT,
    DocumentCleaner,
    safe_path,
    tomllib,
)
from check_k8s_chunking import SPLITTER_VERSION, scan_structure, structural_spans
from check_k8s_chunk_cleanup import CLEANUP_VERSION, candidate_documents, cleanup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER_VERSION = "k8s-prepare-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/k8s_corpora/troubleshooting_v2"
PAYLOAD_FILES = (
    "documents.jsonl",
    "chunks.jsonl",
    "audit.jsonl",
    "processing_report.json",
    "source_snapshot.json",
    "LICENSE",
)
RULE_FILES = (
    "check_k8s_documents.py",
    "check_k8s_chunking.py",
    "check_k8s_chunk_cleanup.py",
    "prepare_k8s_corpus.py",
)


class NeedsReview(ValueError):
    pass


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"不能序列化类型：{type(value).__name__}")


def json_text(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=json_default
    )


def normalized(value):
    return json.loads(json_text(value))


def rule_hashes() -> dict:
    return {
        name: digest((Path(__file__).parent / name).read_bytes()) for name in RULE_FILES
    }


def chunk_stats(chunks: list[dict]) -> dict:
    return {
        "over_budget_chunks": sum(chunk["metadata"]["over_budget"] for chunk in chunks),
        "max_chunk_chars": max(
            (len(chunk["page_content"]) for chunk in chunks), default=0
        ),
    }


def pack_page(loaded: dict, size: int) -> tuple[dict, list[dict], dict]:
    """把一篇通过清洗的父文档变成三个可序列化对象，不执行任何写入。"""
    text = loaded["cleaned"]
    metadata = {
        **loaded["metadata"],
        "builder_version": BUILDER_VERSION,
        "review_status": "unreviewed_official_document",
        "ready_for_embedding": False,
        "token_limit_checked": False,
    }
    sections, codes = scan_structure(text)
    before, _ = structural_spans(text, size, sections, codes)
    result = cleanup(text, before, sections, size)
    if result.deferred:
        raise NeedsReview(
            f"{len(result.deferred)}个孤立标题缺少同章节正文，整篇暂缓导出"
        )
    documents = candidate_documents(text, before, result, sections, metadata, size)
    if not documents:
        raise NeedsReview("没有可导出的正文候选")
    parent_id = documents[0].metadata["parent_id"]
    parent = {"id": parent_id, "page_content": text, "metadata": metadata}
    chunks = [
        {
            "id": document.id,
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]
    audit = {
        "parent_id": parent_id,
        "offset_scope": "cleaned_document_python_characters_half_open",
        "cleaned_sha256": metadata["cleaned_sha256"],
        "before": [asdict(span) for span in before],
        "kept": [asdict(span) for span in result.kept],
        "excluded": [asdict(item) for item in result.excluded],
        "merged": [[asdict(a), asdict(b)] for a, b in result.merged],
        "deferred": [],
    }
    return normalized(parent), normalized(chunks), normalized(audit)


def prepare_scope(size: int) -> dict:
    """确定范围并逐篇预检；有问题的整篇留在报告中，不混进成功子集。"""
    source_bytes = (CORPUS_ROOT / "source_manifest.json").read_bytes()
    source = json.loads(source_bytes)
    commit = source["source"]["commit_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("来源清单的提交编号不合法")
    root = CORPUS_ROOT / "upstream"
    docs_root = root / "content/zh-cn/docs"
    debug_root = safe_path(docs_root, "tasks/debug")
    if not debug_root.is_dir():
        raise FileNotFoundError("缺少本地 tasks/debug 文档目录，请先恢复上游快照")
    paths = sorted(
        {path.relative_to(docs_root).as_posix() for path in debug_root.rglob("*.md")}
        | {"tasks/administer-cluster/dns-debugging-resolution.md"}
    )
    config_path = safe_path(root, "hugo.toml")
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    params = {
        **config.get("params", {}),
        **config.get("languages", {}).get("zh-cn", {}).get("params", {}),
    }
    parents, chunks, audits, records = [], [], [], []
    for relative in paths:
        record = {"path": relative, "status": "pending", "messages": []}
        try:
            path = safe_path(docs_root, relative)
            record["raw_sha256"] = digest(path.read_bytes())
            if path.name == "_index.md":
                record.update(
                    status="excluded_by_scope",
                    messages=["本轮范围策略排除_index.md，不代表该页无内容或无价值"],
                )
            else:
                cleaner = DocumentCleaner(root, params)
                cleaner.record(config_path, config_bytes, "site_config")
                loaded = cleaner.load(relative, commit)
                if loaded["metadata"]["raw_sha256"] != record["raw_sha256"]:
                    raise ValueError("文档在处理期间发生变化，请重新运行")
                if cleaner.warnings:
                    record.update(
                        status="needs_review_cleaning", messages=cleaner.warnings
                    )
                elif any(
                    loaded["metadata"]["front_matter"].get(key) is True
                    for key in ("draft", "headless")
                ):
                    record.update(
                        status="needs_review_visibility",
                        messages=["draft/headless页面暂不作为独立正文导出"],
                    )
                else:
                    loaded["metadata"]["corpus_relative_path"] = relative
                    parent, page_chunks, audit = pack_page(loaded, size)
                    parents.append(parent)
                    chunks.extend(page_chunks)
                    audits.append(audit)
                    record.update(
                        status="prepared",
                        parent_id=parent["id"],
                        chunks=len(page_chunks),
                        **chunk_stats(page_chunks),
                    )
        except NeedsReview as error:
            record.update(status="needs_review_heading", messages=[str(error)])
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            StopIteration,
            yaml.YAMLError,
        ) as error:
            record.update(status="error", messages=[f"{type(error).__name__}: {error}"])
        records.append(record)
    summary = dict(Counter(record["status"] for record in records))
    report = {
        "scope": "tasks/debug/**/*.md plus tasks/administer-cluster/dns-debugging-resolution.md",
        "scope_policy": "exclude_index_pages_for_this_run_not_empty_page_detection",
        "paths": paths,
        "records": records,
        "summary": summary,
        "selected_files": len(paths),
        "prepared_documents": len(parents),
        "candidate_chunks": len(chunks),
        **chunk_stats(chunks),
        "source_commit": commit,
        "target_chars": size,
    }
    return {
        "parents": parents,
        "chunks": chunks,
        "audits": audits,
        "report": normalized(report),
        "source": source,
        "source_sha256": digest(source_bytes),
        "license_bytes": safe_path(root, "LICENSE").read_bytes(),
        "config_sha256": digest(config_bytes),
        "commit": commit,
    }


def validate_records(
    parents: list[dict], chunks: list[dict], audits: list[dict], size: int, commit: str
) -> None:
    """不读上游，利用已保存父正文重新计算块及账本，检查序列化是否一致。"""
    parent_ids = [parent["id"] for parent in parents]
    if not parent_ids or len(set(parent_ids)) != len(parent_ids):
        raise ValueError("父文档为空或身份重复")
    if len(audits) != len(parents) or len(
        {audit["parent_id"] for audit in audits}
    ) != len(audits):
        raise ValueError("父文档与账本数量或身份不一致")
    if len({chunk["id"] for chunk in chunks}) != len(chunks):
        raise ValueError("候选块身份重复")
    if any(chunk["metadata"]["parent_id"] not in parent_ids for chunk in chunks):
        raise ValueError("存在没有父文档的候选块")
    if {audit["parent_id"] for audit in audits} != set(parent_ids):
        raise ValueError("存在没有对应父文档的账本")
    for parent in parents:
        metadata = parent["metadata"]
        if metadata["snapshot_commit"] != commit or metadata[
            "cleaned_sha256"
        ] != digest(parent["page_content"].encode("utf-8")):
            raise ValueError("父正文指纹或提交编号不一致")
        expected_parent, expected_chunks, expected_audit = pack_page(
            {"cleaned": parent["page_content"], "metadata": metadata}, size
        )
        actual_chunks = [
            chunk for chunk in chunks if chunk["metadata"]["parent_id"] == parent["id"]
        ]
        actual_audit = next(
            audit for audit in audits if audit["parent_id"] == parent["id"]
        )
        if (
            parent != expected_parent
            or actual_chunks != expected_chunks
            or actual_audit != expected_audit
        ):
            raise ValueError("读回的正文、候选元数据或处置账本与规则计算不一致")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_directory(directory: Path, manifest_name: str = "manifest.json") -> dict:
    """只读取导出目录及本地规则代码，不需要上游文档或任何模型客户端。"""
    manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
    if (
        manifest["builder_version"] != BUILDER_VERSION
        or manifest["rule_sha256"] != rule_hashes()
    ):
        raise ValueError(
            "当前规则代码与构建时不一致，不能用本校验器直接验证旧结果；这不等于旧文件损坏，请保留旧包并另建新版本目录"
        )
    if set(manifest["files"]) != set(PAYLOAD_FILES):
        raise ValueError("清单文件集合不一致")
    for name in PAYLOAD_FILES:
        data = (directory / name).read_bytes()
        if manifest["files"][name] != {"sha256": digest(data), "bytes": len(data)}:
            raise ValueError(f"文件长度或SHA256不一致：{name}")
    parents = read_jsonl(directory / "documents.jsonl")
    chunks = read_jsonl(directory / "chunks.jsonl")
    audits = read_jsonl(directory / "audit.jsonl")
    validate_records(
        parents, chunks, audits, manifest["target_chars"], manifest["source_commit"]
    )
    report = json.loads(
        (directory / "processing_report.json").read_text(encoding="utf-8")
    )
    record_paths = [record["path"] for record in report["records"]]
    if record_paths != report["paths"] or record_paths != sorted(set(record_paths)):
        raise ValueError("范围清单与逐篇报告不一致或包含重复路径")
    actual_summary = dict(Counter(record["status"] for record in report["records"]))
    prepared_paths = {
        record["path"] for record in report["records"] if record["status"] == "prepared"
    }
    if actual_summary != report["summary"] or prepared_paths != {
        parent["metadata"]["corpus_relative_path"] for parent in parents
    }:
        raise ValueError("处理报告与父文档不一致")
    expected_counts = {
        "selected_files": len(record_paths),
        "prepared_documents": len(parents),
        "candidate_chunks": len(chunks),
        **chunk_stats(chunks),
    }
    if any(
        report[key] != count or manifest[key] != count
        for key, count in expected_counts.items()
    ):
        raise ValueError("清单或报告中的数量统计不一致")
    if (
        report["source_commit"] != manifest["source_commit"]
        or report["target_chars"] != manifest["target_chars"]
    ):
        raise ValueError("清单与处理报告的来源或切块参数不一致")
    parent_by_path = {
        parent["metadata"]["corpus_relative_path"]: parent for parent in parents
    }
    if len(parent_by_path) != len(parents):
        raise ValueError("多个父文档对应同一来源路径")
    for record in report["records"]:
        if record["status"] == "prepared":
            parent = parent_by_path[record["path"]]
            page_chunks = [
                chunk
                for chunk in chunks
                if chunk["metadata"]["parent_id"] == parent["id"]
            ]
            if (
                record["parent_id"] != parent["id"]
                or record["raw_sha256"] != parent["metadata"]["raw_sha256"]
                or record["chunks"] != len(page_chunks)
                or any(
                    record[key] != count
                    for key, count in chunk_stats(page_chunks).items()
                )
            ):
                raise ValueError("逐篇报告与实际父文档或候选统计不一致")
    source_snapshot = json.loads(
        (directory / "source_snapshot.json").read_text(encoding="utf-8")
    )
    source = source_snapshot["original_source_manifest"]
    if (
        source["source"]["commit_sha"] != manifest["source_commit"]
        or source_snapshot["original_file_sha256"] != manifest["source_manifest_sha256"]
        or source["license"] != manifest["license"]
    ):
        raise ValueError("来源快照与构建清单的提交、来源指纹或许可记录不一致")
    expected_partial = any(
        record["status"] not in ("prepared", "excluded_by_scope")
        for record in report["records"]
    )
    if manifest["status"] != (
        "prepared_partial" if expected_partial else "prepared_scope_complete"
    ):
        raise ValueError("处理状态不能反映待处理文档")
    if (
        manifest["ready_for_embedding"] is not False
        or manifest["token_limit_checked"] is not False
    ):
        raise ValueError("本阶段不能宣称Embedding条件已通过")
    return manifest


def write_bundle(batch: dict, output: Path, size: int) -> dict:
    """仅创建新目录；读回核对载荷后最后写manifest，失败目录不覆盖也不删除。"""
    if not batch["parents"]:
        raise ValueError("没有通过规则的正文，未创建输出目录")
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "documents.jsonl": "".join(
            json_text(item) + "\n" for item in batch["parents"]
        ).encode("utf-8"),
        "chunks.jsonl": "".join(
            json_text(item) + "\n" for item in batch["chunks"]
        ).encode("utf-8"),
        "audit.jsonl": "".join(
            json_text(item) + "\n" for item in batch["audits"]
        ).encode("utf-8"),
        "processing_report.json": (json_text(batch["report"]) + "\n").encode("utf-8"),
        "source_snapshot.json": (
            json_text(
                {
                    "original_source_manifest": batch["source"],
                    "original_file_sha256": batch["source_sha256"],
                }
            )
            + "\n"
        ).encode("utf-8"),
        "LICENSE": batch["license_bytes"],
    }
    for name, data in payload.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    for name, expected in payload.items():
        if (output / name).read_bytes() != expected:
            raise ValueError(f"写入后字节读回不一致：{name}")
    validate_records(
        read_jsonl(output / "documents.jsonl"),
        read_jsonl(output / "chunks.jsonl"),
        read_jsonl(output / "audit.jsonl"),
        size,
        batch["commit"],
    )
    partial = any(
        record["status"] not in ("prepared", "excluded_by_scope")
        for record in batch["report"]["records"]
    )
    manifest = {
        "builder_version": BUILDER_VERSION,
        "status": "prepared_partial" if partial else "prepared_scope_complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": batch["commit"],
        "source_manifest_sha256": batch["source_sha256"],
        "site_config_sha256": batch["config_sha256"],
        "cleaner_version": CLEANER_VERSION,
        "splitter_version": SPLITTER_VERSION,
        "cleanup_version": CLEANUP_VERSION,
        "rule_sha256": rule_hashes(),
        "target_chars": size,
        "overlap_chars": 0,
        "selected_files": batch["report"]["selected_files"],
        "prepared_documents": len(batch["parents"]),
        "candidate_chunks": len(batch["chunks"]),
        **chunk_stats(batch["chunks"]),
        "ready_for_embedding": False,
        "token_limit_checked": False,
        "quality_evaluated": False,
        "model_api_calls": 0,
        "license": batch["source"]["license"],
        "files": {
            name: {"sha256": digest(data), "bytes": len(data)}
            for name, data in payload.items()
        },
    }
    pending = output / ".manifest.pending.json"
    with pending.open("x", encoding="utf-8") as stream:
        stream.write(json_text(manifest) + "\n")
    verified = verify_directory(output, pending.name)
    # output由本进程独占新建；正式名称只在校验全部通过后出现。
    if (output / "manifest.json").exists():
        raise FileExistsError("正式清单已存在，拒绝发布")
    pending.rename(output / "manifest.json")
    return verified


def self_test() -> None:
    """内存合成文本仅测试导出结构，不创建问题集。"""
    text = "# 测试\n\n说明\n\n```yaml\nx: 1\n```\n"
    metadata = {
        "source_url": "https://example.invalid/test",
        "snapshot_commit": "a" * 40,
        "cleaned_sha256": digest(text.encode()),
        "cleaner_version": CLEANER_VERSION,
        "corpus_relative_path": "test.md",
    }
    parent, chunks, audit = pack_page({"cleaned": text, "metadata": metadata}, 20)
    validate_records([parent], chunks, [audit], 20, "a" * 40)
    checks = 1
    for field in ("page_content", "id"):
        bad = normalized(chunks)
        bad[0][field] += "tamper"
        try:
            validate_records([parent], bad, [audit], 20, "a" * 40)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("篡改数据应被拒绝")
    bad_audit = normalized(audit)
    bad_audit["kept"][0]["start"] += 1
    try:
        validate_records([parent], chunks, [bad_audit], 20, "a" * 40)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("账本区间篡改应被拒绝")
    assert (
        not chunks[0]["metadata"]["ready_for_embedding"]
        and "builder_version" not in metadata
    )
    checks += 1
    print(
        f"[自测通过] {checks}组：序列化重建、正文/ID/账本篡改检测及未就绪标记；无文件写入或模型请求。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run", action="store_true", help="将通过预检的子集保存到全新目录"
    )
    mode.add_argument("--verify", type=Path, help="仅校验已导出目录，不读取上游文档")
    mode.add_argument("--self-test", action="store_true", help="仅内存自测")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="字符预算目标，默认800，不是已评测最优参数",
    )
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 100000:
        parser.error("chunk-size须在1～100000之间")
    if args.self_test:
        self_test()
        return 0
    if args.verify:
        manifest = verify_directory(args.verify.resolve())
        print(
            f"[读回校验通过] {manifest['status']}；{manifest['prepared_documents']}篇；{manifest['candidate_chunks']}候选块。"
        )
        print(
            "[边界] 仅验证文件内部一致性与规则重建；未读上游、未调用模型，不代表检索质量或Token上限通过。"
        )
        return 0
    output = args.output.resolve()
    if args.run and (output.exists() or args.output.is_symlink()):
        raise FileExistsError(
            f"目标已存在，不覆盖、不追加：{output}；可用--verify校验或另选新目录"
        )
    print(f"[配置] Python：{sys.executable}；{BUILDER_VERSION}；目标：{output}")
    print(
        "[准备] 每次先做本地预检；默认只展示，加--run才保存。Embedding/Reranker/聊天请求均0，不读.env。"
    )
    batch = prepare_scope(args.chunk_size)
    report = batch["report"]
    for record in report["records"]:
        print(
            f"[{record['status']}] {record['path']}"
            + (f"；{record['chunks']}候选块" if "chunks" in record else "")
        )
        for message in record["messages"]:
            print(f"  {message}")
    print(
        f"[汇总] 范围{report['selected_files']}文件；状态={report['summary']}；通过{len(batch['parents'])}篇，候选{len(batch['chunks'])}块。"
    )
    print(
        f"[预算] 完整代码超字符目标的候选{report['over_budget_chunks']}块；最长{report['max_chunk_chars']}字符；字符数不是Token数。"
    )
    print(
        "[前后区别] 以前每次从Markdown清洗切块；保存后可直接读取JSONL复用，本步尚未接入检索。"
    )
    print(
        "[边界] 仅部分范围，不是全部1665文档；有问题的整篇暂缓，不用空结果冒充成功；超长代码仍待Token核查。"
    )
    if not args.run:
        print("[计划完成] 未创建输出目录或文件；加--run保存，已有目录永不覆盖。")
        return 0
    manifest = write_bundle(batch, output, args.chunk_size)
    print(f"[保存并读回通过] {output / 'manifest.json'}；status={manifest['status']}。")
    print(
        "[状态含义] 部分准备完成不等于全范围ready；没有构建Milvus索引，没有生成回答或准备开发题。"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        StopIteration,
        yaml.YAMLError,
    ) as error:
        print(f"[错误] {error}；已有输出不会自动删除或覆盖。", file=sys.stderr)
        raise SystemExit(2)
