"""IT 运维知识库第二步：对照两种切块规则，只读、零模型请求、不入库。

A：固定字符窗口，严格满足预算。B：按标题/段落分组并保护代码围栏；
超长代码单独保留，明确允许超预算。两者都不重叠、不改写、不丢弃正文。
这是结构对照，不是检索准确率评测，也不是完整 Markdown 解析器。
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import re
import sys

from langchain_core.documents import Document

from check_k8s_documents import CORPUS_ROOT, DEFAULT_FILES, DocumentCleaner, tomllib


SPLITTER_VERSION = "k8s-chunk-demo-v1"
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)")
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")


@dataclass(frozen=True)
class Span:
    """半开字符区间 [start, end)，坐标始终属于同一份清洗正文。"""

    start: int
    end: int
    kind: str = "prose"


@dataclass(frozen=True)
class Section:
    start: int
    end: int
    headings: tuple[str, ...]


def short_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:16]


def scan_structure(text: str) -> tuple[list[Section], list[Span]]:
    """扫描 ATX 标题和代码围栏；代码内的 # 不能被当成章节标题。"""
    headings, codes = [], []
    active: list[tuple[int, str]] = []
    opened = None
    offset = 0
    for line in text.splitlines(keepends=True):
        if opened is not None:
            start, marker = opened
            close = re.fullmatch(
                r" {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*",
                line.rstrip("\r\n"),
            )
            if close:
                codes.append(Span(start, offset + len(line), "code"))
                opened = None
        elif match := FENCE.match(line):
            opened = (offset, match.group(1))
        elif match := HEADING.match(line.rstrip("\r\n")):
            level = len(match.group(1))
            title = re.sub(r"\s+\{#[^}]+\}\s*$", "", match.group(2))
            title = re.sub(r"\s+#+\s*$", "", title)
            while active and active[-1][0] >= level:
                active.pop()
            active.append((level, title))
            headings.append((offset, tuple(title for _, title in active)))
        offset += len(line)
    if opened is not None:
        raise ValueError("清洗正文含未闭合代码围栏；停止切块，避免错误判断代码边界。")
    if not text:
        return [], []
    if not headings or headings[0][0] != 0:
        headings.insert(0, (0, ()))
    sections = [
        Section(start, headings[i + 1][0] if i + 1 < len(headings) else len(text), path)
        for i, (start, path) in enumerate(headings)
    ]
    return sections, codes


def fixed_spans(text: str, size: int) -> list[Span]:
    """A：不分析内容，每 size 个 Python 字符切一次，重叠为零。"""
    if size <= 0:
        raise ValueError("切块预算必须大于 0。")
    return [
        Span(start, min(start + size, len(text)), "fixed")
        for start in range(0, len(text), size)
    ]


def paragraph_spans(text: str, start: int, end: int):
    """按空行找段落边界，分隔符也保留，不能通过 strip 悄悄丢字。"""
    cursor = start
    for match in re.finditer(r"\n[ \t\r]*\n", text[start:end]):
        stop = start + match.end()
        yield Span(cursor, stop)
        cursor = stop
    if cursor < end:
        yield Span(cursor, end)


def structural_spans(
    text: str, size: int, sections: list[Section], codes: list[Span]
) -> tuple[list[Span], int]:
    """B：不跨章节，优先保持段落；超长普通段落硬切，超长代码单独保留。"""
    if size <= 0:
        raise ValueError("切块预算必须大于 0。")
    result, long_paragraphs = [], 0
    for section in sections:
        units, cursor = [], section.start
        for code in codes:
            if section.start <= code.start < section.end:
                units.extend(paragraph_spans(text, cursor, code.start))
                units.append(code)
                cursor = code.end
        units.extend(paragraph_spans(text, cursor, section.end))
        pieces = []
        for unit in units:
            if unit.kind != "code" and unit.end - unit.start > size:
                long_paragraphs += 1
                pieces.extend(
                    Span(start, min(start + size, unit.end))
                    for start in range(unit.start, unit.end, size)
                )
            else:
                pieces.append(unit)
        pending = None
        for piece in pieces:
            if piece.end - piece.start > size:
                if pending is not None:
                    result.append(pending)
                    pending = None
                result.append(piece)  # 只允许完整代码作为超预算例外。
            elif pending is None:
                pending = Span(piece.start, piece.end, "packed")
            elif piece.end - pending.start <= size:
                pending = Span(pending.start, piece.end, "packed")
            else:
                result.append(pending)
                pending = Span(piece.start, piece.end, "packed")
        if pending is not None:
            result.append(pending)
    return result, long_paragraphs


def touched_sections(span: Span, sections: list[Section]) -> list[Section]:
    return [
        section
        for section in sections
        if span.start < section.end and section.start < span.end
    ]


def verify_partition(
    text: str, spans: list[Span], size: int, allow_long_code: bool
) -> None:
    """检查坐标连续、无重叠、无丢字；B 的超预算必须确实为完整代码。"""
    cursor = 0
    sections, codes = scan_structure(text)
    for span in spans:
        if span.start != cursor or not span.start < span.end <= len(text):
            raise ValueError("切块坐标存在空隙、重叠或越界。")
        if span.end - span.start > size:
            if not allow_long_code or span.kind != "code" or span not in codes:
                raise ValueError("发现非完整代码的超预算块。")
        cursor = span.end
    if (
        cursor != len(text)
        or "".join(text[span.start : span.end] for span in spans) != text
    ):
        raise ValueError("切块后的正文不能完整还原原输入。")
    if allow_long_code:
        if any(
            not any(span.start <= code.start and code.end <= span.end for span in spans)
            for code in codes
        ):
            raise ValueError("结构路线未把代码围栏完整保留在单个块中。")
        if any(len(touched_sections(span, sections)) > 1 for span in spans):
            raise ValueError("结构路线出现跨章节块。")


def make_documents(
    text: str,
    spans: list[Span],
    sections: list[Section],
    metadata: dict,
    strategy: str,
    size: int,
) -> list[Document]:
    """用 LangChain Document 包装正文与元数据，两条路线使用相同的来源规则。"""
    parent_id = short_hash(
        json.dumps(
            [
                metadata["source_url"],
                metadata["cleaned_sha256"],
                metadata["cleaner_version"],
            ],
            ensure_ascii=False,
        )
    )
    documents = []
    for index, span in enumerate(spans):
        touched = touched_sections(span, sections)
        chunk_id = short_hash(
            json.dumps(
                [parent_id, SPLITTER_VERSION, strategy, size, 0, span.start, span.end]
            )
        )
        documents.append(
            Document(
                id=chunk_id,
                page_content=text[span.start : span.end],
                metadata={
                    **metadata,
                    "parent_id": parent_id,
                    "chunk_id": chunk_id,
                    "strategy": strategy,
                    "splitter_version": SPLITTER_VERSION,
                    "chunk_index": index,
                    "chunk_count": len(spans),
                    "target_chars": size,
                    "overlap_chars": 0,
                    "cleaned_char_start": span.start,
                    "cleaned_char_end": span.end,
                    "offset_scope": "cleaned_document_python_characters_half_open",
                    "sections": [list(section.headings) for section in touched],
                    "dependencies_scope": "parent_document_not_individual_chunk",
                    "body_chars": span.end - span.start,
                    "over_budget": span.end - span.start > size,
                    "over_budget_reason": "intact_code_block"
                    if span.end - span.start > size
                    else None,
                },
            )
        )
    return documents


def metrics(
    spans: list[Span], sections: list[Section], codes: list[Span], size: int
) -> dict:
    lengths = [span.end - span.start for span in spans]
    cut = [
        code
        for code in codes
        if not any(span.start <= code.start and code.end <= span.end for span in spans)
    ]
    return {
        "chunks": len(spans),
        "chars": sum(lengths),
        "min": min(lengths, default=0),
        "max": max(lengths, default=0),
        "mean": sum(lengths) / len(lengths) if lengths else 0,
        "over": sum(length > size for length in lengths),
        "cross_section": sum(
            len(touched_sections(span, sections)) > 1 for span in spans
        ),
        "cut_code": len(cut),
        "code_total": len(codes),
    }


def excerpt(text: str, limit: int) -> str:
    return text[:limit].replace("\n", "⏎").replace("\r", "") + (
        "…" if len(text) > limit else ""
    )


def show_comparison(
    text: str, metadata: dict, size: int, show_all: bool, preview_chars: int
) -> tuple[int, int]:
    sections, codes = scan_structure(text)
    baseline = fixed_spans(text, size)
    structured, fallbacks = structural_spans(text, size, sections, codes)
    verify_partition(text, baseline, size, False)
    verify_partition(text, structured, size, True)
    routes = (("A", "fixed", baseline), ("B", "structure_code", structured))
    print(
        f"\n[文档] {metadata['title']}；清洗正文={len(text)}字符；章节段={len(sections)}；代码围栏={len(codes)}"
    )
    print(
        "路线 | 块数 | 字符总和 | 最短 | 最长 | 平均长度 | 超预算块 | 跨章节块 | 被分开的代码块"
    )
    document_routes = {}
    for label, strategy, spans in routes:
        values = metrics(spans, sections, codes, size)
        blank_count = sum(not text[span.start : span.end].strip() for span in spans)
        print(
            f"{label:4} | {values['chunks']:4} | {values['chars']:8} | {values['min']:4} | {values['max']:4} | {values['mean']:8.1f} | {values['over']:8} | {values['cross_section']:8} | {values['cut_code']}/{values['code_total']}"
        )
        document_routes[label] = make_documents(
            text, spans, sections, metadata, strategy, size
        )
        if blank_count:
            print(
                f"[小块提醒] {label}有{blank_count}个纯空白块；本实验为检查原文覆盖而保留，不能直接作为检索内容入库。"
            )
    print(
        f"[校验] 两路均可逐块拼接完整还原正文；零重叠；B触发字符硬切的长段落数={fallbacks}。"
    )
    print(
        "[指标口径] 被分开=同一原始围栏跨度未完整落在任何单块中；含围栏与换行，不等于程序语法错误率。"
    )
    print(
        "[章节口径] 按任一级ATX标题划分；章节名存在元数据中，不额外复制进块正文，避免隐藏重复字符。"
    )
    for label, _, _ in routes:
        documents = document_routes[label]
        selected = documents if show_all else documents[:2]
        for document in selected:
            info = document.metadata
            path = " | ".join(
                " / ".join(headings) or "无标题" for headings in info["sections"]
            )
            print(
                f"[{label}块{info['chunk_index'] + 1:02}] id={document.id}；清洗字符[{info['cleaned_char_start']},{info['cleaned_char_end']})；{len(document.page_content)}字符；超预算={info['over_budget']}"
            )
            print(f"[章节元数据] {path}")
            print(
                f"[正文片段，非完整块] {excerpt(document.page_content, preview_chars)}"
            )
        if not show_all and len(documents) > 2:
            print(
                f"[{label}展示范围] 只预览前2块，共{len(documents)}块；--show-all可查看所有块的摘要。"
            )
    boundary = next(
        (
            (span.end, code)
            for span in baseline[:-1]
            for code in codes
            if code.start < span.end < code.end
        ),
        None,
    )
    if boundary:
        position, code = boundary
        containing = next(
            span
            for span in structured
            if span.start <= code.start and code.end <= span.end
        )
        print(
            f"[同段代码对照] 清洗正文代码区间[{code.start},{code.end})，共{code.end - code.start}字符。"
        )
        print(
            f"[A切点] 字符{position}；左侧：{excerpt(text[max(code.start, position - 65) : position], 65)}"
        )
        print(
            f"[A切点] 右侧：{excerpt(text[position : min(code.end, position + 65)], 65)}"
        )
        print(
            f"[B处理] 同一代码完整位于[{containing.start},{containing.end})；该块{containing.end - containing.start}字符；超预算={containing.end - containing.start > size}。"
        )
    else:
        print(
            "[同段代码对照] 本次A恰未分开代码围栏，不能据此宣称固定切块总能保护代码。"
        )
    for document in document_routes["B"]:
        if document.metadata["over_budget"]:
            print(
                f"[超预算例外] B块{document.metadata['chunk_index'] + 1}：{len(document.page_content)}字符；完整代码暂不拆分，后续仍需检查模型Token上限。"
            )
    print(f"[来源] {metadata['source_url']}")
    print(
        f"[版本] cleaner={metadata['cleaner_version']}；splitter={SPLITTER_VERSION}；清洗SHA256={metadata['cleaned_sha256'][:16]}…"
    )
    print(
        f"[元数据范围] 标题、快照、许可、前提字段和{len(metadata['dependencies'])}个文档级依赖随块携带；不提供原始文件精确行号。"
    )
    return len(baseline), len(structured)


def self_test() -> None:
    """纯内存测试；覆盖不丢字、预算例外、标题层级、代码边界和身份。"""
    samples = (
        "",
        "中文😀测试\n",
        "# 标题\n\n正文\n",
        "# A\n\n" + "长段落" * 50,
        "# A\n\n```yaml\n# not a heading\n" + "  x: 1\n" * 15 + "```\n\n结尾\n",
        "# A\n## B\n### C\n文字\n## D\n后文\n",
        "# 重复\n正文\n# 重复\n正文\n",
        "# A\r\n\r\n~~~yaml\r\nx: 1\r\n~~~\r\n结尾\r\n",
        "# A\n\n````text\n```\n## not heading\n````\n",
        "## A ###\n\n行\n\n## B {#anchor}\n\n末行",
    )
    checks = 0
    for text in samples:
        sections, codes = scan_structure(text)
        for size in (1, 20, 80, 1000):
            a = fixed_spans(text, size)
            b, _ = structural_spans(text, size, sections, codes)
            verify_partition(text, a, size, False)
            verify_partition(text, b, size, True)
            assert metrics(b, sections, codes, size)["cut_code"] == 0
            assert metrics(b, sections, codes, size)["cross_section"] == 0
            checks += 1
    text = "# Root\n## A\n### Detail\ntext\n## B\n```text\n## fake\n```\n"
    sections, codes = scan_structure(text)
    assert sections[-1].headings == ("Root", "B") and len(codes) == 1
    assert len(sections) == 4
    checks += 1
    for bad in ("```yaml\nunclosed\n", "~~~\n````\n"):
        try:
            scan_structure(bad)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("应拒绝未闭合围栏")
    text = "# 重复\n正文\n# 重复\n正文\n"
    sections, codes = scan_structure(text)
    spans, _ = structural_spans(text, 80, sections, codes)
    metadata = {
        "source_url": "https://example.invalid/doc",
        "cleaned_sha256": short_hash(text),
        "cleaner_version": "test",
    }
    first = make_documents(text, spans, sections, metadata, "structure_code", 80)
    second = make_documents(text, spans, sections, metadata, "structure_code", 80)
    assert [doc.id for doc in first] == [doc.id for doc in second]
    assert len({doc.id for doc in first}) == len(first)
    assert all(
        doc.metadata["offset_scope"].startswith("cleaned_document") for doc in first
    )
    assert "chunk_id" not in metadata
    checks += 1
    for function in (
        lambda: fixed_spans("abc", 0),
        lambda: structural_spans("abc", 0, [], []),
        lambda: verify_partition("abc", [Span(0, 1), Span(2, 3)], 5, False),
    ):
        try:
            function()
        except ValueError:
            checks += 1
        else:
            raise AssertionError("本应拒绝非法预算或有缺口的分块")
    print(
        f"[自测通过] {checks}组内存检查；覆盖还原、零重叠、代码保护、超长例外、标题、Unicode/CRLF、确定性ID及非法输入。"
    )
    print("[边界] 不读语料、不联网、不写文件；不是全库兼容性或检索质量认证。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", action="append", help="相对中文docs的路径；可重复指定，默认三篇样本"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="正文字符预算目标，默认800，不是Token数",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=160,
        help="每块摘要最多预览多少原文字符，默认160",
    )
    parser.add_argument(
        "--show-all", action="store_true", help="显示所有块的摘要，默认每路前两块"
    )
    parser.add_argument("--self-test", action="store_true", help="仅运行内存自测")
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 100000 or not 1 <= args.preview_chars <= 2000:
        parser.error("chunk-size须在1～100000之间，preview-chars须在1～2000之间。")
    if args.self_test:
        self_test()
        return 0
    manifest = json.loads(
        (CORPUS_ROOT / "source_manifest.json").read_text(encoding="utf-8")
    )
    commit = manifest["source"]["commit_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("来源清单中的提交编号格式不正确。")
    root = CORPUS_ROOT / "upstream"
    config_path = root / "hugo.toml"
    data = config_path.read_bytes()
    config = tomllib.loads(data.decode("utf-8"))
    params = {
        **config.get("params", {}),
        **config.get("languages", {}).get("zh-cn", {}).get("params", {}),
    }
    loaded = []
    for file in args.file or DEFAULT_FILES:
        cleaner = DocumentCleaner(root, params)
        cleaner.record(config_path, data, "site_config")
        result = cleaner.load(file, commit)
        if cleaner.warnings or result["metadata"]["is_navigation"]:
            raise ValueError(
                f"{file}：清洗有待处理提示或属于导航页，停止切块；请先用check_k8s_documents.py核对。"
            )
        loaded.append(result)
    print(
        f"[配置] Python：{sys.executable}；{len(loaded)}篇；快照={commit[:12]}；{SPLITTER_VERSION}"
    )
    print(
        f"[控制条件] 每篇只清洗一次，两路共享完全相同的正文；预算目标={args.chunk_size}字符；重叠=0。"
    )
    print("[路线A] 固定字符窗口；严格不超过预算。")
    print(
        "[路线B] 标题/段落分组+代码保护；超长段落硬切，超长代码独立保留并允许超预算。"
    )
    print(
        "[实验口径] 两路不是相同硬上限；B同时改变几项边界规则，本轮不分别归因、不评测检索准确率。"
    )
    print(
        "[请求计划] Embedding=0；Reranker=0；聊天模型=0；不读.env、不联网、不入库、不写报告。"
    )
    print("[先预测] 哪路会产生更多小块？代码完整性与长度上限冲突时该如何取舍？")
    totals = [0, 0]
    for result in loaded:
        counts = show_comparison(
            result["cleaned"],
            result["metadata"],
            args.chunk_size,
            args.show_all,
            args.preview_chars,
        )
        totals = [total + count for total, count in zip(totals, counts)]
    print(
        f"\n[完成] {len(loaded)}篇；A共{totals[0]}块，B共{totals[1]}块；仅在内存中构造LangChain Document。"
    )
    print(
        "[实验边界] 块更多/跨章节更少/代码不分开均不等于回答更准；结构规则也可能让解释与代码分处两块。"
    )
    print(
        "[保护范围] 本演示只识别ATX标题、空行段落和围栏代码，不保证列表、表格、Setext标题或HTML结构完整。"
    )
    print(
        "[来源边界] 字符区间属于清洗正文，不是原始Markdown行号；依赖信息是整篇范围，未实现片段级来源映射。"
    )
    print(
        "[下一步前] 先核对块边界与超预算代价；本步未选择最终策略、未替换现有RAG或构建新索引。"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"[错误] {error}", file=sys.stderr)
        raise SystemExit(2)
