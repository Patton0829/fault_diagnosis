"""把全量离线预检的通过页保存为兼容原编码/入库流程的新语料包。"""

import argparse
from collections import Counter
import json
from pathlib import Path

from qa_platform.ingestion import kubernetes_corpus as corpus
from qa_platform.ingestion import kubernetes_inventory as inventory
from qa_platform.ingestion.kubernetes_cleaning import (
    CORPUS_ROOT,
    DocumentCleaner,
    safe_path,
)
from qa_platform.settings import PROJECT_ROOT


DEFAULT_REPORT = PROJECT_ROOT / "reports/k8s-full-preflight-2026-10-21-03/report.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/k8s_corpora/full_preflight_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_sources(report, corpus_root):
    source, source_bytes, root, docs, config_bytes, params, paths = inventory._inputs(
        corpus_root
    )
    require(
        report["schema_version"] == inventory.SCHEMA_VERSION
        and report["summary"]["source_unchanged"] is True
        and report["status"] == "completed_with_deferred"
        and report["changed_sources"] == [],
        "需要未漂移的全量预检报告",
    )
    require(
        all(
            type(report["summary"][key]) is int and report["summary"][key] >= 0
            for key in (
                "markdown_files",
                "eligible_documents",
                "candidate_chunks",
                "over_budget_chunks",
                "max_chunk_chars",
            )
        ),
        "报告数量统计必须是非负整数",
    )
    require(
        report["source_commit"] == source["source"]["commit_sha"],
        "报告上游commit不一致",
    )
    require(
        report["parameters"]["rule_sha256"] == corpus.rule_hashes(),
        "预检与当前清洗/切块规则不一致",
    )
    require(report["parameters"]["target_overlap_chars"] == 0, "只支持原结构切块规则")
    require(
        [row["relative_path"] for row in report["records"]] == paths,
        "报告与源文件范围或顺序不一致",
    )
    require(report["summary"]["markdown_files"] == len(paths), "报告总文件数不一致")
    statuses = dict(Counter(row["status"] for row in report["records"]))
    require(
        statuses == report["summary"]["status_counts"]
        and set(statuses).issubset(inventory.STATUSES),
        "报告分类计数不一致",
    )
    for row in report["records"]:
        require(
            all(
                type(row[key]) is int and row[key] >= 0
                for key in (
                    "raw_bytes",
                    "cleaned_chars",
                    "chunk_count",
                    "over_budget_chunks",
                    "max_chunk_chars",
                )
            ),
            "逐篇数量统计必须是非负整数",
        )
        data = safe_path(docs, row["relative_path"]).read_bytes()
        require(
            corpus.digest(data) == row["raw_sha256"] and len(data) == row["raw_bytes"],
            "预检后源文档发生变化，请重新盘点",
        )
        require(
            row["source_role"] == inventory._source_role(row["relative_path"]),
            "报告资料角色提示不一致",
        )
    for relative, expected in report["dependency_sha256"].items():
        require(
            corpus.digest(safe_path(root, relative).read_bytes()) == expected,
            "预检后引用资源或站点配置发生变化",
        )
    return source, source_bytes, root, docs, config_bytes, params


def prepare_from_report(report_path, corpus_root=CORPUS_ROOT):
    """重建且核对每个通过页；不悄悄纳入待处理页，也不修改旧语料。"""
    report_path, corpus_root = Path(report_path), Path(corpus_root)
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    source, source_bytes, root, docs, config_bytes, params = _verify_sources(
        report, corpus_root
    )
    size = report["parameters"]["chunk_size"]
    inventory._parameters(size, report["parameters"]["batch_size"])
    commit = report["source_commit"]
    parents, chunks, audits, records = [], [], [], []
    for row in report["records"]:
        relative = row["relative_path"]
        record = {
            "path": relative,
            "status": row["status"],
            "messages": row["messages"],
            "raw_sha256": row["raw_sha256"],
            "source_role_hint": row["source_role"],
        }
        if row["status"] in {"excluded_navigation", "excluded_visibility"}:
            record["status"] = "excluded_by_scope"
        if row["status"] == "prepared_for_planning":
            require(Path(relative).name != "_index.md", "不能把导航页冒充预检通过页")
            cleaner = DocumentCleaner(root, params)
            cleaner.record(root / "hugo.toml", config_bytes, "site_config")
            loaded = cleaner.load(relative, commit)
            require(
                all(
                    report["dependency_sha256"].get(name) == item["sha256"]
                    for name, item in cleaner.dependencies.items()
                ),
                "报告缺少实际使用的引用依赖或指纹不符",
            )
            metadata = loaded["metadata"]
            require(
                not cleaner.warnings and metadata["raw_sha256"] == row["raw_sha256"],
                "通过页现在有清洗告警或源文件变化",
            )
            require(
                not any(
                    metadata["front_matter"].get(key) is True
                    for key in ("draft", "headless")
                ),
                "通过页包含draft/headless",
            )
            require(
                len(loaded["cleaned"]) == row["cleaned_chars"],
                "清洗正文长度与预检不一致",
            )
            metadata.update(
                corpus_relative_path=relative, source_role_hint=row["source_role"]
            )
            parent, page_chunks, audit = corpus.pack_page(loaded, size)
            inventory._chunk_contract(page_chunks)
            stats = corpus.chunk_stats(page_chunks)
            require(
                len(page_chunks) == row["chunk_count"]
                and all(row[key] == value for key, value in stats.items()),
                "重建块数或预算统计与预检不一致",
            )
            record.update(
                status="prepared",
                parent_id=parent["id"],
                chunks=len(page_chunks),
                **stats,
            )
            parents.append(parent)
            chunks.extend(page_chunks)
            audits.append(audit)
        records.append(record)
    require(
        bool(chunks)
        and len(parents) == report["summary"]["eligible_documents"]
        and len(chunks) == report["summary"]["candidate_chunks"],
        "通过页或候选总数不匹配",
    )
    require(
        corpus.chunk_stats(chunks)
        == {
            key: report["summary"][key]
            for key in ("over_budget_chunks", "max_chunk_chars")
        },
        "总预算统计不一致",
    )
    require(report_path.read_bytes() == report_bytes, "重建期间预检报告变化")
    checked = _verify_sources(report, corpus_root)
    require(
        checked[1] == source_bytes and checked[4] == config_bytes,
        "重建期间站点或来源清单变化",
    )
    processing = {
        "scope": "all Chinese docs: only offline-preflight passed pages",
        "scope_policy": "keep role hints; defer warnings/errors; exclude navigation/draft/headless",
        "preflight_report_sha256": corpus.digest(report_bytes),
        "paths": [row["path"] for row in records],
        "records": records,
        "summary": dict(Counter(row["status"] for row in records)),
        "selected_files": len(records),
        "prepared_documents": len(parents),
        "candidate_chunks": len(chunks),
        "source_commit": commit,
        "target_chars": size,
        **corpus.chunk_stats(chunks),
    }
    return {
        "parents": parents,
        "chunks": chunks,
        "audits": audits,
        "report": processing,
        "source": source,
        "source_sha256": corpus.digest(source_bytes),
        "license_bytes": (root / "LICENSE").read_bytes(),
        "config_sha256": corpus.digest(config_bytes),
        "commit": commit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="保存全量预检通过的语料；不编码、不连接Milvus"
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = args.output.absolute()
        require(
            not output.exists() and not output.is_symlink(),
            "输出已存在，保留原包；请另选新目录",
        )
        require(
            not output.resolve().is_relative_to(args.corpus_root.resolve()),
            "不能写入原始语料目录",
        )
        batch = prepare_from_report(args.report, args.corpus_root)
        print(
            f"[范围] {len(batch['parents'])}篇通过页，{len(batch['chunks'])}块；保留资料角色、原始frontmatter和未审核标记。"
        )
        print(
            "[边界] 不纳入待处理页，不重切、不截断；未请求模型，不读.env、不写数据库。"
        )
        if not args.run:
            print("[计划完成] 未写文件；--run才保存新语料包。")
            return 0
        manifest = corpus.write_bundle(batch, output, batch["report"]["target_chars"])
        print(
            f"[已保存并核验] {output}；{manifest['prepared_documents']}篇/{manifest['candidate_chunks']}块。"
        )
        print(
            "[下一步] build_k8s_embeddings.py --corpus <本目录> --run，然后build_k8s_milvus.py创建独立索引。"
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        print(
            "[未完成] 预检报告、原始资料、规则或目标目录校验失败；不覆盖、不自动重试。"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
