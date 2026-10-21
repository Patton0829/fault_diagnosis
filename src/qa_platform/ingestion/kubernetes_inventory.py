"""全量中文文档只读盘点：复用156块库规则，不编码、不创建语料或索引。"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import time
from urllib.parse import quote
from uuid import uuid4

import yaml

from qa_platform.ingestion.kubernetes_cleaning import (
    CLEANER_VERSION,
    CORPUS_ROOT,
    DocumentCleaner,
    front_matter,
    safe_path,
    tomllib,
)
from qa_platform.ingestion.kubernetes_corpus import (
    BUILDER_VERSION,
    json_text,
    pack_page,
    rule_hashes,
)
from qa_platform.settings import PROJECT_ROOT


SCHEMA_VERSION = "k8s-full-preflight/v1"
STATUSES = {
    "excluded_navigation": "目录导航（范围策略排除）",
    "excluded_visibility": "draft/headless（不作为独立正文）",
    "prepared_for_planning": "离线预检通过（不是已入库）",
    "needs_review_cleaning": "清洗告警，整篇待处理",
    "error_cleaning": "读取或清洗未完成",
    "needs_review_chunking": "切块或检索字段待处理",
}
SOURCE_ROLES = {
    "feature_gate_data": "特性门数据页（路径提示）",
    "glossary_data": "术语数据页（路径提示）",
    "generated_reference_fragment": "生成式参考片段（路径提示）",
    "regular_page": "其他页面（仍需筛选）",
}


def _source_role(relative):
    for prefix, role in (
        ("reference/command-line-tools-reference/feature-gates/", "feature_gate_data"),
        ("reference/glossary/", "glossary_data"),
        ("reference/setup-tools/kubeadm/generated/", "generated_reference_fragment"),
    ):
        if relative.startswith(prefix):
            return role
    return "regular_page"


def _hash(data):
    return sha256(data).hexdigest()


def _parameters(chunk_size, batch_size):
    if type(chunk_size) is not int or not 100 <= chunk_size <= 10000:
        raise ValueError("chunk_size须为100～10000的整数")
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch_size须为1～256的整数；只用于估算")


def _inputs(corpus_root):
    corpus_root = Path(corpus_root).resolve()
    source_bytes = (corpus_root / "source_manifest.json").read_bytes()
    source = json.loads(source_bytes)
    commit = source["source"]["commit_sha"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("来源commit须为40位十六进制")
    root = corpus_root / "upstream"
    config_bytes = (root / "hugo.toml").read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    params = {
        **config.get("params", {}),
        **config.get("languages", {}).get("zh-cn", {}).get("params", {}),
    }
    docs = root / "content/zh-cn/docs"
    if not docs.is_dir():
        raise ValueError("本地中文docs快照不存在")
    paths = sorted(path.relative_to(docs).as_posix() for path in docs.rglob("*.md"))
    if not paths:
        raise ValueError("没有中文Markdown文件")
    return source, source_bytes, root, docs, config_bytes, params, paths


def _chunk_contract(chunks):
    # 与现有检索分词和Milvus字段一致；不构建BM25索引，也不调用模型Tokenizer。
    from qa_platform.retrieval.bm25 import tokenize

    for chunk in chunks:
        text, metadata = chunk["page_content"], chunk["metadata"]
        if not tokenize(text):
            raise ValueError("存在BM25分词后为空的候选块")
        if len(text.encode("utf-8")) > 65535:
            raise ValueError("候选正文超过Milvus字段字节限制")
        if (
            max(
                len(json_text(metadata).encode()),
                len(json.dumps(metadata, ensure_ascii=True).encode()),
            )
            > 60000
        ):
            raise ValueError("候选完整元数据超过Milvus字段字节限制")


def scan_snapshot(
    corpus_root=CORPUS_ROOT, *, chunk_size=800, batch_size=8, progress=None
):
    """逐文件计数，失败不终止其余文件；不把带告警的部分正文计入可用块。"""
    _parameters(chunk_size, batch_size)
    started = time.perf_counter()
    source, source_bytes, root, docs, config_bytes, params, paths = _inputs(corpus_root)
    commit = source["source"]["commit_sha"]
    rules = rule_hashes()
    baselines, dependencies, changed, records = {}, {}, set(), []
    warning_documents, template_documents = Counter(), Counter()
    lengths = []
    for relative in paths:
        try:
            path = safe_path(docs, relative)
            if path.is_symlink() or (docs / relative).is_symlink():
                raise ValueError("不扫描符号链接文档")
            data = path.read_bytes()
            baselines[relative] = (_hash(data), len(data))
        except (OSError, ValueError):
            baselines[relative] = (None, 0)

    for ordinal, relative in enumerate(paths, 1):
        raw_hash, raw_bytes = baselines[relative]
        cleaner = None
        record = {
            "relative_path": relative,
            "section": relative.split("/")[0] if "/" in relative else "root",
            "raw_sha256": raw_hash,
            "raw_bytes": raw_bytes,
            "source_role": _source_role(relative),
            "has_utf8_bom": False,
            "cleaned_chars": 0,
            "chunk_count": 0,
            "over_budget_chunks": 0,
            "max_chunk_chars": 0,
            "messages": [],
            "status": "error_cleaning",
        }
        try:
            if raw_hash is None:
                raise ValueError("原文件无法安全读取")
            path = safe_path(docs, relative)
            if path.name == "_index.md":
                record["status"] = "excluded_navigation"
            else:
                data = path.read_bytes()
                record["has_utf8_bom"] = data.startswith(b"\xef\xbb\xbf")
                if _hash(data) != raw_hash:
                    changed.add("content/zh-cn/docs/" + relative)
                    raise ValueError("源文件在预检期间变化")
                fields, _ = front_matter(data.decode("utf-8"))
                if any(fields.get(key) is True for key in ("draft", "headless")):
                    record["status"] = "excluded_visibility"
                else:
                    cleaner = DocumentCleaner(root, params)
                    cleaner.record(root / "hugo.toml", config_bytes, "site_config")
                    loaded = cleaner.load(relative, commit)
                    if loaded["metadata"]["raw_sha256"] != raw_hash:
                        changed.add("content/zh-cn/docs/" + relative)
                        raise ValueError("源文件在清洗期间变化")
                    record.update(
                        title=loaded["metadata"]["title"],
                        cleaned_chars=len(loaded["cleaned"]),
                    )
                    if cleaner.warnings:
                        record.update(
                            status="needs_review_cleaning", messages=cleaner.warnings
                        )
                        # 每类提示按文档去重计数，同一篇可能命中多类，不与文档数相加。
                        warning_documents.update(
                            set(_warning_kind(message) for message in cleaner.warnings)
                        )
                        template_documents.update(
                            set(
                                name
                                for message in cleaner.warnings
                                for name in re.findall(r"\{\{[<%]\s*([/\w-]+)", message)
                            )
                        )
                    else:
                        record["status"] = "needs_review_chunking"
                        try:
                            _, chunks, _ = pack_page(loaded, chunk_size)
                            _chunk_contract(chunks)
                            page_lengths = [
                                len(chunk["page_content"]) for chunk in chunks
                            ]
                            record.update(
                                status="prepared_for_planning",
                                chunk_count=len(chunks),
                                over_budget_chunks=sum(
                                    length > chunk_size for length in page_lengths
                                ),
                                max_chunk_chars=max(page_lengths, default=0),
                            )
                            lengths.extend(page_lengths)
                        except (
                            OSError,
                            ValueError,
                            KeyError,
                            TypeError,
                            StopIteration,
                        ) as error:
                            record["messages"] = [f"{type(error).__name__}: {error}"]
        except (
            OSError,
            UnicodeError,
            ValueError,
            KeyError,
            TypeError,
            StopIteration,
            yaml.YAMLError,
        ) as error:
            record.update(
                status="error_cleaning", messages=[f"{type(error).__name__}: {error}"]
            )
        if cleaner is not None:
            for name, item in cleaner.dependencies.items():
                if name in dependencies and dependencies[name] != item["sha256"]:
                    changed.add(name)
                dependencies.setdefault(name, item["sha256"])
        records.append(record)
        if progress is not None and (ordinal % 100 == 0 or ordinal == len(paths)):
            progress(ordinal, len(paths))

    # 在统计完成后重读源文档和已展开依赖；不把源文件的前后变化冒充一致快照。
    for relative, (expected, _) in baselines.items():
        if expected is not None:
            try:
                if _hash(safe_path(docs, relative).read_bytes()) != expected:
                    changed.add("content/zh-cn/docs/" + relative)
            except (OSError, ValueError):
                changed.add("content/zh-cn/docs/" + relative)
    for relative, expected in dependencies.items():
        try:
            if _hash(safe_path(root, relative).read_bytes()) != expected:
                changed.add(relative)
        except (OSError, ValueError):
            changed.add(relative)
    for label, path, original in (
        (
            "source_manifest.json",
            Path(corpus_root) / "source_manifest.json",
            source_bytes,
        ),
        ("hugo.toml", root / "hugo.toml", config_bytes),
    ):
        try:
            if path.read_bytes() != original:
                changed.add(label)
        except OSError:
            changed.add(label)
    final_paths = sorted(
        path.relative_to(docs).as_posix() for path in docs.rglob("*.md")
    )
    if final_paths != paths:
        changed.add("Markdown文件集合")
    try:
        rules_changed = rule_hashes() != rules
    except OSError:
        rules_changed = True
    if rules_changed:
        changed.add("清洗/切块规则源码")
    statuses = dict(sorted(Counter(record["status"] for record in records).items()))
    eligible = statuses.get("prepared_for_planning", 0)
    unchanged = not changed and all(
        value[0] is not None for value in baselines.values()
    )
    sections = {}
    for section in sorted({record["section"] for record in records}):
        selected = [record for record in records if record["section"] == section]
        sections[section] = {
            "markdown_files": len(selected),
            "eligible_documents": sum(
                row["status"] == "prepared_for_planning" for row in selected
            ),
            "candidate_chunks": sum(row["chunk_count"] for row in selected),
        }
    roles = {}
    for role in SOURCE_ROLES:
        selected = [
            row
            for row in records
            if row["source_role"] == role and row["status"] == "prepared_for_planning"
        ]
        roles[role] = {
            "documents": len(selected),
            "chunks": sum(row["chunk_count"] for row in selected),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "status": "completed_with_deferred" if unchanged else "source_changed",
        "parameters": {
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "dimension": 4096,
            "dtype": "float32",
            "cleaner_version": CLEANER_VERSION,
            "builder_version": BUILDER_VERSION,
            "rule_sha256": rules,
            "target_overlap_chars": 0,
            "model_requests": 0,
            "token_limit_checked": False,
            "bm25_tokenizer_checked": True,
            "bm25_check_scope": "仅计入候选的通过页；未建立BM25索引",
        },
        "summary": {
            "markdown_files": len(paths),
            "declared_markdown_files": source.get("counts", {}).get(
                "zh_docs_markdown_files"
            ),
            "raw_bytes": sum(value[1] for value in baselines.values()),
            "status_counts": statuses,
            "eligible_documents": eligible,
            "candidate_chunks": len(lengths),
            "deferred_documents": sum(
                value
                for key, value in statuses.items()
                if key
                not in {
                    "prepared_for_planning",
                    "excluded_navigation",
                    "excluded_visibility",
                }
            ),
            "embedding_batches": (len(lengths) + batch_size - 1) // batch_size
            if unchanged
            else None,
            "raw_vector_bytes": len(lengths) * 4096 * 4 if unchanged else None,
            "over_budget_chunks": sum(length > chunk_size for length in lengths),
            "max_chunk_chars": max(lengths, default=0),
            "source_unchanged": unchanged,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "all_documents_chunk_count": None,
            "full_count_note": "仅统计通过当前规则的候选；待处理文档的块数未知，不把部分范围冒充全量。",
        },
        "sections": sections,
        "eligible_roles": roles,
        "warning_documents": dict(warning_documents.most_common()),
        "template_warning_documents": dict(template_documents.most_common()),
        "dependency_sha256": dict(sorted(dependencies.items())),
        "changed_sources": sorted(changed),
        "records": records,
    }


def _warning_kind(message):
    if "四空格/Tab" in message:
        return "缩进Markdown需核对"
    if "feature-state" in message:
        return "功能状态模板参数未支持"
    if "缺少引用资源" in message:
        return "引用资源缺失"
    if "未闭合" in message:
        return "标记未闭合或跨边界"
    if "{{" in message:
        return "模板语义未支持"
    return "其他清洗提示"


def render_html(report):
    """静态一页报告：所有输入转义，无脚本或远程资源；导航页不是无价值文档。"""

    def text(value):
        return escape(str(value), quote=True)

    summary = report["summary"]
    vector_bytes = summary["raw_vector_bytes"]
    vector_size = (
        f"{vector_bytes / 1024**2:.2f} MiB"
        if type(vector_bytes) is int and vector_bytes >= 0
        else "未计算"
    )
    batches = (
        summary["embedding_batches"]
        if summary["source_unchanged"]
        else "源变更，需重跑后计算"
    )
    stats = "".join(
        f"<tr><td>{text(STATUSES.get(key, key))}</td><td>{text(value)}</td></tr>"
        for key, value in summary["status_counts"].items()
    )
    sections = "".join(
        f"<tr><td>{text(key)}</td><td>{text(value['markdown_files'])}</td>"
        f"<td>{text(value['eligible_documents'])}</td><td>{text(value['candidate_chunks'])}</td></tr>"
        for key, value in report["sections"].items()
    )
    roles = "".join(
        f"<tr><td>{text(SOURCE_ROLES.get(key, key))}</td><td>{text(value['documents'])}</td>"
        f"<td>{text(value['chunks'])}</td></tr>"
        for key, value in report["eligible_roles"].items()
    )
    warnings = "".join(
        f"<li>{text(key)}：{text(value)}篇</li>"
        for key, value in report["warning_documents"].items()
    )
    templates = "".join(
        f"<li><code>{text(key)}</code>：{text(value)}篇</li>"
        for key, value in report["template_warning_documents"].items()
    )
    groups = []
    for status, label in STATUSES.items():
        rows = []
        for record in report["records"]:
            if record["status"] != status:
                continue
            url = (
                f"https://github.com/kubernetes/website/blob/{report['source_commit']}/content/zh-cn/docs/"
                + quote(record["relative_path"], safe="/")
            )
            messages = "<br>".join(text(message) for message in record["messages"])
            hints = [SOURCE_ROLES.get(record["source_role"], record["source_role"])]
            if record["has_utf8_bom"]:
                hints.append(
                    "有UTF-8 BOM；旧frontmatter解析兼容性待核对，不代表实际缺少标题"
                )
            messages += "<br>" + text("；".join(hints))
            rows.append(
                f"<tr><td><a href='{text(url)}' rel='noreferrer'>{text(record['relative_path'])}</a></td>"
                f"<td>{text(record['chunk_count'])}</td><td>{messages or '—'}</td></tr>"
            )
        groups.append(
            f"<details><summary>{text(label)} · {len(rows)}篇</summary><div class='scroll'><table>"
            "<tr><th>文件 / 固定快照原文</th><th>候选块</th><th>原因</th></tr>"
            + "".join(rows)
            + "</table></div></details>"
        )
    return (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>Kubernetes 全量离线盘点</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f3f6f8;"
        "color:#243746;max-width:1100px;margin:32px auto;padding:0 20px}section,details{background:white;padding:24px;"
        "border:1px solid #d7e1e6;border-radius:12px;margin:16px 0}h1{margin-bottom:8px}h2{font-size:21px}"
        ".numbers{display:flex;gap:32px;flex-wrap:wrap}.numbers strong{font-size:32px;color:#087c72;display:block}"
        "table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #e4eaee;padding:10px;"
        "vertical-align:top;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}.scroll{overflow:auto}"
        "a{color:#087c72}code{overflow-wrap:anywhere}.note{color:#647785}</style>"
        "<h1>Kubernetes 全量离线盘点</h1><p>0模型请求。不是全量入库，也不是答案质量评测。</p><section class='numbers'>"
        f"<div><strong>{text(summary['markdown_files'])}</strong>中文Markdown文件</div>"
        f"<div><strong>{text(summary['eligible_documents'])}</strong>通过本次规则的文档</div>"
        f"<div><strong>{text(summary['candidate_chunks'])}</strong>已统计候选块</div>"
        f"<div><strong>{text(summary['deferred_documents'])}</strong>待处理文档</div></section>"
        f"<section><h2>范围去向</h2><table>{stats}</table><p>{text(summary['full_count_note'])}</p>"
        "<p>导航页按本轮范围策略排除，不等于没有价值；有清洗告警的整篇暂缓，不删除未知模板后强行计数。</p></section>"
        "<section><h2>编码规划，不是已经发出的请求</h2>"
        f"<p>仅上述候选、从零编码、每批{text(report['parameters']['batch_size'])}块：约{text(batches)}次Embedding。"
        f"4096维float32原始向量约{text(vector_size)}，不含索引和元数据。</p>"
        f"<p>目标{text(report['parameters']['chunk_size'])}字符；超预算块{text(summary['over_budget_chunks'])}个；"
        f"最长{text(summary['max_chunk_chars'])}字符。字符不是Token，尚未做模型Token/可编码性检查，也不估算编码总耗时。</p>"
        "<p>沿用默认156块知识库的结构切块及代码保护规则；不是215块扩展索引的章节递归规则。"
        "当前多格式接入及便携包工具另有1000块上限；这是应用限制，不是Milvus容量上限，不能直接把大范围交给旧入口。</p></section>"
        "<section><h2>通过规则，不等于独立运维文章</h2><table><tr><th>资料角色提示</th><th>通过文件</th><th>候选块</th></tr>"
        + roles
        + "</table><p>特性门和术语数据不等于独立排障教程。很多无frontmatter文件是被其他页引用的生成片段，"
        "不是坏文档；_build、stages、removed等元数据含义仍需核对。本次不改旧规则，也不自动决定是否独立入库。</p></section>"
        "<section><h2>按栏目分布</h2><table><tr><th>栏目</th><th>文件</th><th>通过文档</th><th>候选块</th></tr>"
        + sections
        + "</table></section><section><h2>待处理原因</h2><ul>"
        + warnings
        + "</ul><details><summary>查看告警涉及的模板清单</summary><ul>"
        + templates
        + "</ul></details><p class='note'>按文档去重；同一篇可涉及多类原因，这些数字不可相加当作文档总数。</p></section>"
        + "".join(groups)
        + f"<p class='note'>来源快照：{text(report['source_commit'])}<br>"
        f"源文档/引用/规则前后观测一致：{text(summary['source_unchanged'])}；盘点耗时{text(summary['elapsed_seconds'])}秒。"
        "不代表过程中每个时刻均未变化；未读取.env、未连接模型、数据库或缓存服务，未改原始文档、索引或当前问答服务。</p></html>"
    )


def write_report(report, output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("目标已存在，不覆盖；请换新目录")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in (
        ("report.json", json_text(report) + "\n"),
        ("report.html", render_html(report)),
    ):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description="全量中文Markdown零模型清洗切块预检")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        _parameters(args.chunk_size, args.batch_size)
        *_, paths = _inputs(args.corpus_root)
        print(
            f"[范围] 本地中文docs全部{len(paths)}个Markdown文件；原156块库规则；目标{args.chunk_size}字符。"
        )
        print(
            "[边界] 只清洗、切块与本地分词/字段预检；0模型请求，不读.env、不编码、不入库、不切换服务。"
        )
        if not args.run:
            print("[计划完成] 未逐篇处理、未写报告；加--run执行全量离线盘点。")
            return 0
        output = (
            args.output_dir
            or PROJECT_ROOT / "reports" / f"k8s-full-preflight-{uuid4().hex[:12]}"
        )
        if (
            output.exists()
            or output.is_symlink()
            or output.resolve().is_relative_to(args.corpus_root.resolve())
        ):
            raise ValueError("目标已存在或位于源语料内，拒绝写入")
        report = scan_snapshot(
            args.corpus_root,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            progress=lambda done, total: print(
                f"[离线进度] {done}/{total}", flush=True
            ),
        )
        write_report(report, output)
        print("[汇总]", json_text(report["summary"]))
        print(f"[报告] {output / 'report.html'}；逐篇明细：report.json。")
        print("[未完成] 待处理文档的块数未知；通过预检也不是已编码、已审核或已入库。")
        return 0 if report["summary"]["source_unchanged"] else 1
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, yaml.YAMLError):
        print(
            "[未完成] 本地输入、参数或报告目录校验失败；不覆盖、不重试，已有文件保留。"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
