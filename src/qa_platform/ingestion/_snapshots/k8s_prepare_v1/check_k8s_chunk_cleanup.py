"""IT 运维知识库第三步：对照结构块清理前后，零模型请求、不入库。

纯空白块退出检索候选但保留账本；完整孤立标题优先在同章节合并，
超预算时转为同章节正文的元数据上下文。没有对应正文则保留并提示。
不按长度随意删短块，不删除原文；只产生内存中的候选和去向记录。
"""

import argparse
from dataclasses import dataclass
import json
import re
import sys

from check_k8s_documents import CORPUS_ROOT, DEFAULT_FILES, DocumentCleaner, tomllib
from check_k8s_chunking import (
    HEADING,
    Section,
    Span,
    excerpt,
    make_documents,
    metrics,
    scan_structure,
    structural_spans,
    verify_partition,
)


CLEANUP_VERSION = "k8s-chunk-cleanup-v1"


@dataclass(frozen=True)
class Excluded:
    span: Span
    reason: str
    section_start: int | None = None


@dataclass
class CleanupResult:
    kept: list[Span]
    excluded: list[Excluded]
    merged: list[tuple[Span, Span]]
    deferred: list[Span]


def section_for(span: Span, sections: list[Section]) -> Section:
    return next(
        section for section in sections if section.start <= span.start < section.end
    )


def heading_only(text: str, span: Span, sections: list[Section]) -> bool:
    """只认全局章节起点处的完整标题，不把代码里的 # 或标题碎片误删。"""
    section = section_for(span, sections)
    if span.start != section.start:
        return False
    newline = text.find("\n", span.start)
    line_end = newline + 1 if newline >= 0 else len(text)
    if line_end > span.end:
        return False
    return (
        bool(HEADING.fullmatch(text[span.start : line_end].rstrip("\r\n")))
        and not text[line_end : span.end].strip()
    )


def cleanup(
    text: str, before: list[Span], sections: list[Section], size: int
) -> CleanupResult:
    """保守后处理：不跨标题拼接，不切开新正文，不超预算合并。"""
    verify_partition(text, before, size, True)
    result = CleanupResult([], [], [], [])
    index = 0
    while index < len(before):
        span = before[index]
        if not text[span.start : span.end].strip():
            result.excluded.append(Excluded(span, "whitespace"))
            index += 1
            continue
        if not heading_only(text, span, sections):
            result.kept.append(span)
            index += 1
            continue
        own_section = section_for(span, sections)
        following = index + 1
        while (
            following < len(before)
            and not text[before[following].start : before[following].end].strip()
        ):
            following += 1
        target = before[following] if following < len(before) else None
        same_section_body = (
            target is not None
            and section_for(target, sections).start == own_section.start
            and not heading_only(text, target, sections)
        )
        if same_section_body and target.end - span.start <= size:
            combined = Span(span.start, target.end, "packed")
            result.kept.append(combined)
            result.merged.append((span, combined))
            index = following + 1
            continue
        if same_section_body:
            # 完整正文已占满预算时，不为塞入标题而再切正文或偷偷扩大预算。
            result.excluded.append(
                Excluded(span, "heading_to_context", own_section.start)
            )
        else:
            result.kept.append(span)
            result.deferred.append(span)
        index += 1
    validate_cleanup(text, result, sections, size)
    return result


def validate_cleanup(
    text: str, result: CleanupResult, sections: list[Section], size: int
) -> None:
    """候选+账本必须完整覆盖输入；不能借‘过滤’隐藏代码或有内容的正文。"""
    accounted = sorted(
        result.kept + [item.span for item in result.excluded],
        key=lambda span: span.start,
    )
    verify_partition(text, accounted, size, True)
    _, codes = scan_structure(text)
    for span in result.kept:
        if not text[span.start : span.end].strip():
            raise ValueError("清理后候选仍含纯空白块。")
    for item in result.excluded:
        span = item.span
        if any(span.start < code.end and code.start < span.end for code in codes):
            raise ValueError("不能把代码范围移到排除账本。")
        if item.reason == "whitespace":
            if text[span.start : span.end].strip():
                raise ValueError("有内容的正文被误标为纯空白。")
        elif item.reason == "heading_to_context":
            if (
                not heading_only(text, span, sections)
                or section_for(span, sections).start != item.section_start
            ):
                raise ValueError("标题上下文的源区间不合法。")
            if not any(
                section_for(candidate, sections).start == item.section_start
                and not heading_only(text, candidate, sections)
                for candidate in result.kept
            ):
                raise ValueError("标题没有对应的同章节正文，不能仅转为元数据。")
        else:
            raise ValueError("未知排除原因。")
    if any(
        not any(
            span.start <= code.start and code.end <= span.end for span in result.kept
        )
        for code in codes
    ):
        raise ValueError("清理后代码未完整保留在检索候选中。")


def candidate_documents(
    text: str,
    before: list[Span],
    result: CleanupResult,
    sections: list[Section],
    metadata: dict,
    size: int,
):
    """沿用 Document 来源字段；新身份含清理版本，记录原块及标题上下文区间。"""
    originals = make_documents(text, before, sections, metadata, "structure_code", size)
    documents = make_documents(
        text,
        result.kept,
        sections,
        metadata,
        f"structure_cleaned/{CLEANUP_VERSION}",
        size,
    )
    for document, span in zip(documents, result.kept):
        own_section = section_for(span, sections)
        document.metadata.update(
            {
                "cleanup_version": CLEANUP_VERSION,
                "original_chunk_ids": [
                    original.id
                    for original, old in zip(originals, before)
                    if span.start < old.end and old.start < span.end
                ],
                "heading_context_ranges": [
                    {
                        "start": item.span.start,
                        "end": item.span.end,
                        "scope": "cleaned_document",
                    }
                    for item in result.excluded
                    if item.reason == "heading_to_context"
                    and item.section_start == own_section.start
                ],
                "needs_heading_review": span in result.deferred,
                "token_limit_checked": False,
                "ready_for_embedding": False,
            }
        )
    return documents


def show_comparison(
    text: str, metadata: dict, size: int, show_all: bool, preview_chars: int
) -> tuple[int, int]:
    sections, codes = scan_structure(text)
    before, _ = structural_spans(text, size, sections, codes)
    result = cleanup(text, before, sections, size)
    print(f"\n[文档] {metadata['title']}；清洗正文={len(text)}字符")
    print(
        "阶段 | 候选数 | 候选字符 | 最短 | 最长 | 空白块 | 完整孤立标题 | 超预算代码块"
    )
    for label, spans in (("清理前B", before), ("清理后C", result.kept)):
        values = metrics(spans, sections, codes, size)
        blank = sum(not text[span.start : span.end].strip() for span in spans)
        orphan = sum(heading_only(text, span, sections) for span in spans)
        print(
            f"{label} | {len(spans):6} | {values['chars']:8} | {values['min']:4} | {values['max']:4} | {blank:6} | {orphan:12} | {values['over']:12}"
        )
    for heading, combined in result.merged:
        print(
            f"[标题合并] [{heading.start},{heading.end}) → 同章节块[{combined.start},{combined.end})；合并后{combined.end - combined.start}字符。"
        )
    for item in result.excluded:
        action = (
            "退出检索候选，仍保留在原文与账本"
            if item.reason == "whitespace"
            else "不再独立检索，转为同章节正文元数据"
        )
        print(f"[去向] [{item.span.start},{item.span.end})；{item.reason}；{action}。")
        print(
            f"[原内容] {excerpt(text[item.span.start : item.span.end], preview_chars)!r}"
        )
    if not result.excluded and not result.merged:
        print("[变化] 本次没有可安全处理的空白块或孤立标题，不为制造差异而修改块。")
    for span in result.deferred:
        print(
            f"[保留待核对] [{span.start},{span.end})只有标题，但找不到紧随其后的同章节正文；未跨到其他章节合并。"
        )
    kept_chars = sum(span.end - span.start for span in result.kept)
    context_chars = sum(
        item.span.end - item.span.start
        for item in result.excluded
        if item.reason == "heading_to_context"
    )
    blank_chars = sum(
        item.span.end - item.span.start
        for item in result.excluded
        if item.reason == "whitespace"
    )
    print(
        f"[覆盖账本] 候选{kept_chars} + 标题上下文{context_chars} + 排除空白{blank_chars} = 输入{len(text)}字符。"
    )
    print(
        "[校验] 候选与账本的原区间排序后可完整还原输入，无重叠；代码仍完整保留。不是把候选正文减少伪装成原文没有变化。"
    )
    documents = candidate_documents(text, before, result, sections, metadata, size)
    selected = documents if show_all else documents[:2]
    for document in selected:
        info = document.metadata
        print(
            f"[候选{info['chunk_index'] + 1}] id={document.id}；[{info['cleaned_char_start']},{info['cleaned_char_end']})；{info['body_chars']}字符；超预算={info['over_budget']}"
        )
        print(
            f"[章节元数据] {info['sections']}；标题上下文区间={info['heading_context_ranges']}"
        )
        print(f"[正文预览] {excerpt(document.page_content, preview_chars)}")
    if not show_all and len(documents) > len(selected):
        print(
            f"[展示范围] 仅前{len(selected)}个候选，全部共{len(documents)}个；--show-all可查看全部摘要。"
        )
    for document in documents:
        if document.metadata["over_budget"]:
            print(
                f"[仍待处理] 候选{document.metadata['chunk_index'] + 1}含完整超长代码：{len(document.page_content)}字符；本步未核查模型Token上限，不可直接宣称可送入模型。"
            )
    print(f"[来源] {metadata['source_url']}")
    print("[来源范围] 区间属于清洗正文；依赖仍是整篇范围，不是原始文件精确行号。")
    return len(before), len(result.kept)


def self_test() -> None:
    """纯内存样本；不创建开发题，不评测检索质量。"""
    checks = 0

    def run(text, size, before=None):
        sections, codes = scan_structure(text)
        spans = (
            before
            if before is not None
            else structural_spans(text, size, sections, codes)[0]
        )
        return sections, spans, cleanup(text, spans, sections, size)

    text = "# 标题\n\n正文\n"
    end = text.index("正文")
    _, _, result = run(text, 100, [Span(0, end), Span(end, len(text))])
    assert len(result.kept) == 1 and result.merged and not result.excluded
    checks += 1
    text = "# 标题\n\n" + "正文" * 15
    end = text.index("正文")
    _, _, result = run(text, 30, [Span(0, end), Span(end, len(text))])
    assert (
        len(result.excluded) == 1 and result.excluded[0].reason == "heading_to_context"
    )
    checks += 1
    text = "# A\n## B\n短事实\n"
    _, _, result = run(text, 100)
    assert result.deferred and result.kept[0].start == 0
    assert not result.merged  # 不跨过另一标题合并。
    checks += 1
    for text in (
        "",
        "\n\n",
        "# 只有标题\n",
        "短事实",
        "# A\r\n\r\n正文😀\r\n",
        "# " + "长标题" * 20 + "\n",
        "# A\n\n```yaml\n# not heading\n" + "  x: 1\n" * 12 + "```\n\n",
    ):
        for size in (1, 20, 80, 800):
            _, _, result = run(text, size)
            assert all(text[span.start : span.end].strip() for span in result.kept)
            checks += 1
    text = "# A\n\n" + "x" * 20 + "\n\n"
    end = text.index("x")
    before = [Span(0, end), Span(end, end + 20), Span(end + 20, len(text))]
    sections, before, result = run(text, 20, before)
    assert sorted(item.reason for item in result.excluded) == [
        "heading_to_context",
        "whitespace",
    ]
    metadata = {
        "source_url": "https://example.invalid/doc",
        "cleaned_sha256": "x" * 64,
        "cleaner_version": "test",
    }
    documents = candidate_documents(text, before, result, sections, metadata, 20)
    second = candidate_documents(text, before, result, sections, metadata, 20)
    assert (
        documents[0].id == second[0].id
        and documents[0].metadata["heading_context_ranges"]
    )
    assert (
        documents[0].page_content == "x" * 20
        and not documents[0].metadata["ready_for_embedding"]
    )
    assert "cleanup_version" not in metadata
    checks += 1
    bad = CleanupResult([Span(0, 1)], [Excluded(Span(1, 3), "whitespace")], [], [])
    try:
        validate_cleanup("abc", bad, scan_structure("abc")[0], 10)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("不能把有内容的正文谎称为空白")
    print(
        f"[自测通过] {checks}组：合并/转上下文、无正文标题保留、短事实/代码/Unicode保护、覆盖账本、身份与错误排除。"
    )
    print("[边界] 只用内存合成文本，不准备开发题，不联网，不写文件，不调用模型。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", action="append", help="相对中文docs的.md路径，可重复，默认三篇样本"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="字符目标，默认200以观察碎块；不是最终参数建议",
    )
    parser.add_argument("--preview-chars", type=int, default=120)
    parser.add_argument("--show-all", action="store_true", help="显示全部候选摘要")
    parser.add_argument("--self-test", action="store_true", help="仅内存自测")
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 100000 or not 1 <= args.preview_chars <= 2000:
        parser.error("chunk-size须在1～100000，preview-chars须在1～2000之间。")
    if args.self_test:
        self_test()
        return 0
    manifest = json.loads(
        (CORPUS_ROOT / "source_manifest.json").read_text(encoding="utf-8")
    )
    commit = manifest["source"]["commit_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("来源清单提交编号不合法。")
    root = CORPUS_ROOT / "upstream"
    config_path = root / "hugo.toml"
    config_data = config_path.read_bytes()
    config = tomllib.loads(config_data.decode("utf-8"))
    params = {
        **config.get("params", {}),
        **config.get("languages", {}).get("zh-cn", {}).get("params", {}),
    }
    loaded = []
    for file in args.file or DEFAULT_FILES:
        cleaner = DocumentCleaner(root, params)
        cleaner.record(config_path, config_data, "site_config")
        result = cleaner.load(file, commit)
        if cleaner.warnings or result["metadata"]["is_navigation"]:
            raise ValueError(
                f"{file}：清洗有待处理提示或属于导航页，请先核对清洗输出。"
            )
        loaded.append(result)
    print(
        f"[配置] Python：{sys.executable}；{CLEANUP_VERSION}；{len(loaded)}篇；字符目标={args.chunk_size}"
    )
    print(
        "[实验] 原结构块B → 保守清理后候选C；共享正文、预算与代码保护规则；不是检索效果对照。"
    )
    print(
        "[请求计划] Embedding=0；Reranker=0；聊天模型=0；不读.env、不联网、不入库、不写报告或原文。"
    )
    print(
        "[规则] 不按长度随意删短块；标题只与紧随其后的同章节正文合并，合并超预算则转为元数据；无对应正文则保留待核对。"
    )
    totals = [0, 0]
    for result in loaded:
        before, after = show_comparison(
            result["cleaned"],
            result["metadata"],
            args.chunk_size,
            args.show_all,
            args.preview_chars,
        )
        totals[0] += before
        totals[1] += after
    print(
        f"\n[完成] 原B={totals[0]}块，清理候选C={totals[1]}块；只在内存中保留Document与去向账本。"
    )
    print(
        "[边界] 标题留在元数据不等于已加入Embedding文本；本步不保证标题语义可检索或质量提高。"
    )
    print(
        "[仍未完成] Token硬上限核查、片段级原文映射、全库兼容性与检索质量评测；未准备开发题，未选定最终策略。"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"[错误] {error}", file=sys.stderr)
        raise SystemExit(2)
