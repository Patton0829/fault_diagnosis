"""第四步：解析真实知识库，按完整条目分块并保留来源元数据。"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import re
import sys

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qa_platform.settings import PROJECT_ROOT


KNOWLEDGE_SOURCES = {
    "experience": PROJECT_ROOT / "knowledge" / "列车故障诊断经验知识库.md",
    "test": PROJECT_ROOT / "knowledge" / "故障诊断问答测试知识库.txt",
}
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
TEST_HEADING_PATTERN = re.compile(r"^##\s+(TEST-[A-Z0-9]+-\d+)\s+(.+)$")
ENTRY_ID_PATTERN = re.compile(r"(?:TEST-)?[A-Z][A-Z0-9]*-\d+")
PREVIEW_FIELDS = [
    "chunk_id",
    "parent_id",
    "kb_id",
    "entry_id",
    "title",
    "section",
    "source",
    "line_start",
    "line_end",
    "table_header_line",
    "is_test",
    "document_version",
    "part_index",
    "part_count",
    "source_line_scope",
    "parent_char_start",
    "parent_char_end",
]


def digest(text: str) -> str:
    """生成确定性的短哈希，相同文本每次都会得到相同结果。"""
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def split_table_row(line: str) -> list[str]:
    """按未转义的竖线分列，保留公式中的转义竖线。"""
    row = line.strip()
    if not row.startswith("|") or not row.endswith("|"):
        raise ValueError("当前表格解析器要求每行有首尾竖线。")
    cells, buffer = [], []
    escaped = False
    for character in row[1:-1]:
        if escaped:
            # 只还原转义的竖线；其他反斜杠内容保持原样。
            buffer.extend([character] if character == "|" else ["\\", character])
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(character)
    if escaped:
        buffer.append("\\")
    cells.append("".join(buffer).strip())
    return cells


def is_table_separator(line: str) -> bool:
    """识别 Markdown 表头下面的 |---|---| 分隔行。"""
    if not line.strip().startswith("|") or not line.strip().endswith("|"):
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in split_table_row(line))


def make_record(content: str, source_metadata: dict, record_metadata: dict) -> Document:
    """用 Document 包装一个完整父条目：正文与来源信息分开存放。"""
    metadata = {**source_metadata, **record_metadata}
    # 有编号时使用编号；无编号时用章节、表头和首字段确定条目身份。
    identity = metadata["entry_id"] or json.dumps(
        [metadata["section"], metadata.get("table_headers"), metadata["title"]],
        ensure_ascii=False,
    )
    parent_id = digest(f"{metadata['document_id']}\n{identity}")
    metadata.update(parent_id=parent_id, record_hash=digest(content))
    return Document(id=parent_id, page_content=content, metadata=metadata)


def parse_markdown_tables(text: str, source_metadata: dict) -> list[Document]:
    """维护章节路径，将每条表格数据转成带字段名的完整记录。"""
    lines = text.splitlines()
    records, headings = [], []
    headers = None
    header_line = None
    for position, line in enumerate(lines):
        heading = HEADING_PATTERN.match(line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
            headers = None
            continue
        if not line.strip().startswith("|"):
            headers = None
            continue

        cells = split_table_row(line)
        if position + 1 < len(lines) and is_table_separator(lines[position + 1]):
            if not all(cells) or len(set(cells)) != len(cells):
                raise ValueError(f"第 {position + 1} 行的表头为空或重复。")
            headers, header_line = cells, position + 1
            continue
        if is_table_separator(line):
            if headers is None or len(cells) != len(headers):
                raise ValueError(f"第 {position + 1} 行的表格分隔符与表头不一致。")
            continue
        if headers is None or len(cells) != len(headers):
            raise ValueError(f"第 {position + 1} 行缺少表头或列数不匹配。")

        fields = dict(zip(headers, cells))
        entry_id = fields.get("条目ID")
        if entry_id is not None and not ENTRY_ID_PATTERN.fullmatch(entry_id):
            raise ValueError(f"第 {position + 1} 行的条目 ID 格式不正确。")
        title = next(
            (value for key, value in fields.items() if key != "条目ID"), entry_id
        )
        content = "\n".join(f"{key}：{value}" for key, value in fields.items())
        records.append(
            make_record(
                content,
                source_metadata,
                {
                    "entry_id": entry_id,
                    "title": title,
                    "section": " / ".join(title for _, title in headings),
                    "record_type": "table_row",
                    "line_start": position + 1,
                    "line_end": position + 1,
                    "table_header_line": header_line,
                    "table_headers": headers.copy(),
                    "raw_text": line,
                },
            )
        )
    return records


def parse_test_entries(text: str, source_metadata: dict) -> list[Document]:
    """以 TEST 标题和等号分隔线确定边界，保留完整现象、原因和步骤。"""
    lines = text.splitlines()
    records = []
    position = 0
    while position < len(lines):
        heading = TEST_HEADING_PATTERN.match(lines[position])
        if not heading:
            position += 1
            continue
        start, stop = position, position + 1
        while stop < len(lines):
            if TEST_HEADING_PATTERN.match(lines[stop]) or re.fullmatch(
                r"={3,}", lines[stop].strip()
            ):
                break
            stop += 1
        end = stop
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        content = "\n".join(lines[start + 1 : end]).strip()
        if not content:
            raise ValueError(f"第 {start + 1} 行的 TEST 条目没有正文。")
        entry_id, title = heading.groups()
        records.append(
            make_record(
                content,
                source_metadata,
                {
                    "entry_id": entry_id,
                    "title": title,
                    "section": f"{lines[0].strip()} / {entry_id} {title}",
                    "record_type": "test_entry",
                    "line_start": start + 1,
                    "line_end": end,
                    "table_header_line": None,
                    "raw_text": "\n".join(lines[start:end]),
                },
            )
        )
        position = stop
    return records


def load_records(kb: str = "all") -> list[Document]:
    """读取两个指定知识库，生成父记录并保存文档级说明和内容版本。"""
    if kb != "all" and kb not in KNOWLEDGE_SOURCES:
        raise ValueError(f"未知知识库：{kb}")
    records = []
    for kb_id, path in KNOWLEDGE_SOURCES.items():
        if kb != "all" and kb != kb_id:
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        source_metadata = {
            "kb_id": kb_id,
            "is_test": kb_id == "test",
            "source": str(path.resolve()),
            "source_name": path.name,
            "document_id": digest(str(path.relative_to(PROJECT_ROOT))),
            "document_version": sha256(raw).hexdigest(),
            "review_status": "test_data" if kb_id == "test" else "unreviewed",
        }
        parser = parse_markdown_tables if path.suffix == ".md" else parse_test_entries
        parsed = parser(text, source_metadata)
        if not parsed:
            raise ValueError(f"{path.name} 没有解析出有效条目。")

        covered_lines = set()
        for record in parsed:
            covered_lines.update(
                range(record.metadata["line_start"], record.metadata["line_end"] + 1)
            )
        notes = [
            f"第 {number} 行：{line}"
            for number, line in enumerate(text.splitlines(), start=1)
            if (
                line.startswith(">")
                if kb_id == "experience"
                else number not in covered_lines
                and line.strip()
                and not re.fullmatch(r"={3,}", line.strip())
            )
        ]
        # 文件说明与测试建议只作为资料保存，不执行其中的文字指令。
        for record in parsed:
            record.metadata["document_notes"] = notes.copy()
        records.extend(parsed)

    seen_ids = set()
    for record in records:
        if record.id in seen_ids:
            raise ValueError(
                f"父记录身份重复：{record.metadata['source_name']} / {record.metadata['title']}"
            )
        seen_ids.add(record.id)
    return records


def split_records(
    records: list[Document], chunk_size: int = 800, chunk_overlap: int = 100
) -> list[Document]:
    """短条目保持完整；超长条目递归切分，每个子块补回章节与条目标题。"""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("必须满足 chunk_size > 0 且 0 <= chunk_overlap < chunk_size")
    chunks = []
    for record in records:
        entry_label = (
            f"{record.metadata['entry_id'] or ''} {record.metadata['title']}".strip()
        )
        prefix = f"章节：{record.metadata['section']}\n条目：{entry_label}\n\n"
        body_budget = chunk_size - len(prefix)
        if body_budget <= 0:
            raise ValueError("chunk_size 太小，无法容纳章节和条目标题。")

        if len(record.page_content) <= body_budget:
            pieces = [
                Document(
                    page_content=record.page_content,
                    metadata={**record.metadata, "start_index": 0},
                )
            ]
        else:
            if chunk_overlap >= body_budget:
                raise ValueError(
                    "扣除标题后，正文空间不足以容纳 chunk_overlap，请调大块大小或减小重叠。"
                )
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=body_budget,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
                keep_separator="end",
                add_start_index=True,
                strip_whitespace=False,
            )
            pieces = splitter.split_documents([record])

        for part_index, piece in enumerate(pieces):
            start = piece.metadata.pop("start_index")
            if (
                start < 0
                or record.page_content[start : start + len(piece.page_content)]
                != piece.page_content
            ):
                raise ValueError("无法在父条目正文中定位子块。")
            piece.metadata.update(
                part_index=part_index,
                part_count=len(pieces),
                parent_char_start=start,
                parent_char_end=start + len(piece.page_content),
                source_line_scope="parent_record",
            )
            # 字符偏移指向解析后的父记录正文，不是原文件的字符偏移。
            piece.page_content = prefix + piece.page_content
            piece.metadata["content_hash"] = digest(piece.page_content)
            piece.id = digest(f"{record.id}\n{start}\n{piece.metadata['content_hash']}")
            piece.metadata["chunk_id"] = piece.id
            chunks.append(piece)
    return chunks


def main() -> int:
    """本地解析、统计和预览；不读取密钥、不调用模型、不写入数据库。"""
    parser = argparse.ArgumentParser(description="验证知识库解析与章节感知分块")
    parser.add_argument(
        "--kb", choices=["all", *KNOWLEDGE_SOURCES], default="all", help="选择知识库"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="每块最大字符数，包含标题；不是 Token 数",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="同一父条目内子块的目标重叠字符数",
    )
    parser.add_argument(
        "--entry-id", help="只预览指定条目，例如 TEST-SW-006 或 BRG-001"
    )
    parser.add_argument("--show", type=int, default=2, help="最多预览多少个子块")
    args = parser.parse_args()
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("必须满足 chunk_size > 0 且 0 <= chunk_overlap < chunk_size")
    if args.show <= 0:
        parser.error("--show 必须大于 0")

    try:
        records = load_records(args.kb)
        chunks = split_records(records, args.chunk_size, args.chunk_overlap)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[错误] 知识库解析或分块失败：{exc}", file=sys.stderr)
        return 1

    print(f"[配置] Python：{sys.executable}")
    print(
        f"[配置] 最大块大小：{args.chunk_size} 字符；目标重叠：{args.chunk_overlap} 字符"
    )
    parent_counts = Counter(record.metadata["kb_id"] for record in records)
    chunk_counts = Counter(chunk.metadata["kb_id"] for chunk in chunks)
    for kb_id, count in parent_counts.items():
        print(f"[统计] {kb_id}：父记录 {count} 条，检索子块 {chunk_counts[kb_id]} 个")
    print(
        f"[统计] 共 {len(records)} 条父记录、{len(chunks)} 个子块；最长子块 {max(len(c.page_content) for c in chunks)} 字符"
    )

    previews = chunks
    if args.entry_id:
        previews = [
            chunk for chunk in chunks if chunk.metadata["entry_id"] == args.entry_id
        ]
        if not previews:
            print(f"[错误] 当前知识库没有条目：{args.entry_id}", file=sys.stderr)
            return 1
    for number, chunk in enumerate(previews[: args.show], start=1):
        print(f"\n[预览 {number}] {len(chunk.page_content)} 字符")
        print(chunk.page_content)
        print("[元数据]")
        metadata = {key: chunk.metadata.get(key) for key in PREVIEW_FIELDS}
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\n[成功] 已在内存中生成 LangChain Document，可供后续检索使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
