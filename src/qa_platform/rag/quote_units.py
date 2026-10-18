"""可定位的引用单位：保留原文软换行，不改写、不补全、不跨块取文。"""

import re


LINE_STRATEGY = "line-v1"
PARAGRAPH_STRATEGY = "paragraph-v1"
MAX_QUOTE_CHARS = 240
MIN_QUOTE_CHARS = 6

_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}\s")
_LIST = re.compile(r"^[ \t]*(?:[-+*]|[0-9]+[.)])[ \t]+")
_NOTE = re.compile(r"^\[(?:说明|说明结束|警告|警告结束|注意|注意结束)\]$")
_SENTENCE_END = re.compile(r"[。！？!?][\"”’）)\]]*|[.](?=\s|$)")


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split(text: str, start: int, end: int, kind: str) -> list[dict]:
    """长单位有上限：优先句末、其次换行，否则硬切，并显式标注。"""
    start, end = _trim(text, start, end)
    split = end - start > MAX_QUOTE_CHARS
    units = []
    cursor = start
    while cursor < end:
        cut = min(cursor + MAX_QUOTE_CHARS, end)
        if cut < end:
            candidates = [
                cursor + match.end()
                for match in _SENTENCE_END.finditer(text[cursor:cut])
                if match.end() >= MIN_QUOTE_CHARS
            ]
            if candidates:
                cut = candidates[-1]
            else:
                newline = text.rfind("\n", cursor, cut)
                if newline + 1 - cursor >= MIN_QUOTE_CHARS:
                    cut = newline + 1
            # 不因硬上限而把最后不足6字符的尾部悄悄丢弃。
            if end - cut < MIN_QUOTE_CHARS:
                cut = min(cursor + MAX_QUOTE_CHARS, end - MIN_QUOTE_CHARS)
        left, right = _trim(text, cursor, cut)
        if right - left >= MIN_QUOTE_CHARS:
            units.append({
                "text": text[left:right], "start": left, "end": right,
                "kind": kind, "split": split,
            })
        cursor = cut
    return units


def paragraph_units(text: str) -> list[dict]:
    """轻量结构规则，不声称是完整Markdown解析器或语义完整性判定器。

    空行、标题、列表新项和代码围栏形成边界；普通段落内软换行保留。
    start/end是Python Unicode码点位置，[start,end)始终定位原文连续子串。
    """
    if not isinstance(text, str):
        raise ValueError("引用正文须为字符串")
    units, pending = [], None
    position = 0
    fence = None
    code_start = None

    def flush():
        nonlocal pending
        if pending is not None:
            start, end, kind = pending
            units.extend(_split(text, start, end, kind))
            pending = None

    for raw_line in text.splitlines(keepends=True):
        plain = raw_line.rstrip("\r\n")
        end = position + len(raw_line)
        if fence is not None:
            closing = re.fullmatch(
                rf"[ \t]{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*", plain,
            )
            if closing:
                units.extend(_split(text, code_start, position, "code"))
                fence, code_start = None, None
            position = end
            continue
        opening = _FENCE.match(plain)
        if opening:
            flush()
            fence, code_start = opening.group(1), end
        elif not plain.strip() or _HEADING.match(plain) or _NOTE.fullmatch(plain.strip()):
            flush()
        elif _LIST.match(plain):
            flush()
            pending = (position, end, "list")
        elif pending is None:
            pending = (position, end, "paragraph")
        else:
            pending = (pending[0], end, pending[2])
        position = end
    if fence is not None:
        # 块边界可能截断代码围栏；仅保留现有正文，不臆造结束标记或邻块。
        units.extend(_split(text, code_start, len(text), "code"))
    flush()
    return units


def paragraph_quotes(text: str) -> dict[str, str]:
    quotes = {f"Q{index}": unit["text"] for index, unit in enumerate(paragraph_units(text), 1)}
    if not quotes:
        raise ValueError("证据块中没有可供引用的原文单位")
    return quotes
