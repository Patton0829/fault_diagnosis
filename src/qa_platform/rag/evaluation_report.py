"""纯离线评测报告：把技术结果、引用原文和待人工核验项放在同一页。"""

from html import escape
import json
from math import isfinite
from urllib.parse import urlsplit


ROUTES = (("bm25", "BM25"), ("dense", "Dense"), ("rrf", "RRF"),
          ("reranker", "Reranker"))


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _items(value):
    return value if isinstance(value, (list, tuple)) else []


def _text(value):
    return escape("未提供" if value is None else str(value), quote=True)


def _metric(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
        return f"{value:.4f}"
    return "不适用"


def _source_url(value):
    """只给固定上游仓库的 HTTPS 原文生成链接；其他 URL 仅显示文字。"""
    if not isinstance(value, str) or any(ord(char) <= 32 for char in value):
        return _text(value)
    try:
        parsed = urlsplit(value)
        allowed = (parsed.scheme == "https" and parsed.netloc == "github.com"
                   and parsed.path.startswith("/kubernetes/website/blob/")
                   and bool(parsed.path.removeprefix("/kubernetes/website/blob/")))
    except ValueError:
        allowed = False
    if allowed:
        return f'<a href="{_text(value)}" rel="noopener noreferrer">{_text(value)}</a>'
    return _text(value)


def _list(values):
    rows = "".join(f"<li>{_text(value)}</li>" for value in _items(values))
    return f"<ul>{rows}</ul>" if rows else "<p>未提供</p>"


def _quote(reference, sources):
    reference = _mapping(reference)
    source_id, quote_id = reference.get("source_id"), reference.get("quote_id")
    # 异常引用也只作为文字展示，不把模型输出拼成元素 ID 或选择器。
    source = _mapping(sources.get(source_id)) if isinstance(source_id, str) else {}
    quotes = _mapping(source.get("quotes"))
    quote = quotes.get(quote_id) if isinstance(quote_id, str) else None
    label = f"[{_text(source_id)}/{_text(quote_id)}]"
    if not isinstance(quote, str):
        return f'<div class="quote missing">{label} 引用未找到，需人工核对。</div>'
    metadata = _mapping(source.get("metadata"))
    sections = metadata.get("section_path")
    section_text = " / ".join(str(item) for item in sections) if isinstance(sections, list) else sections
    return (f'<div class="quote"><strong>{label} 原文</strong><pre>{_text(quote)}</pre>'
            f'<p>块：{_text(source.get("chunk_id"))} · {_text(metadata.get("title"))}'
            f' / {_text(section_text)}</p>'
            f'<p>来源：{_source_url(metadata.get("source_url"))}<br>'
            f'审核：{_text(metadata.get("review_status"))}</p></div>')


def _answer(row):
    answer = row.get("answer")
    if not isinstance(answer, dict):
        return "<p>没有可展示的已校验回答。技术运行未完成不代表答案事实错误。</p>"
    sources = _mapping(row.get("sources"))
    claims = []
    for claim in _items(answer.get("claims")):
        claim = _mapping(claim)
        quotes = "".join(_quote(reference, sources) for reference in _items(claim.get("evidence")))
        if not quotes:
            quotes = '<p class="missing">未提供引用，需人工核对。</p>'
        claims.append(f'<li><p>{_text(claim.get("text"))}</p>{quotes}</li>')
    claim_list = f'<ol>{"".join(claims)}</ol>' if claims else "<p>本题未列回答要点。</p>"
    return (f'<p>回答状态：{_text(answer.get("status"))}</p>{claim_list}'
            f'<h4>缺少的信息</h4>{_list(answer.get("missing_info"))}')


def _case(index, row, k):
    row = _mapping(row)
    case = _mapping(row.get("case"))
    retrieval = _mapping(row.get("retrieval"))
    route_rows = []
    for key, title in ROUTES:
        route = _mapping(retrieval.get(key))
        top_ids = ", ".join(str(value) for value in _items(route.get("top_chunk_ids")))
        completed = "完成" if route.get("completed") is True else "未完成"
        hit_value = route.get("hit_at_k")
        hit = "命中" if hit_value is True else "未命中" if hit_value is False else "不适用"
        route_rows.append(f"<tr><td>{title}</td><td>{completed}</td><td>{hit}</td>"
                          f'<td>{_text(route.get("first_relevant_rank"))}</td>'
                          f'<td>{_metric(route.get("reciprocal_rank_at_k"))}</td>'
                          f"<td>{_text(top_ids or None)}</td></tr>")
    technical_status = "技术完成" if row.get("technical_status") == "completed" else "技术失败 / 未完成"
    error = _mapping(row.get("error"))
    error_html = (f'<p class="missing">阶段：{_text(error.get("stage"))}；'
                  f'类别：{_text(error.get("kind"))}；{_text(error.get("message"))}。'
                  '这是运行状态，不是事实错误判定。</p>') if error else ""
    checks = row.get("checks")
    checks_html = (f'<pre>{_text(json.dumps(checks, ensure_ascii=False, indent=2, default=str))}</pre>'
                   if checks is not None else "<p>自动检查未完成。</p>")
    manual = _mapping(row.get("manual_review"))
    points = manual.get("points") or case.get("review_points")
    counts = _mapping(row.get("completed_request_counts"))
    counts_html = ("<p>已完成阶段累计请求：" + "；".join(
        f"{label} {_text(counts.get(key))}" for key, label in (
            ("embedding", "Embedding"), ("reranker", "重排"), ("chat", "回答生成"), ("http", "HTTP")))
        + "。</p>") if counts else ""
    if row.get("failed_request_counts_unknown") is True:
        counts_html += '<p class="missing">失败阶段可能已发送请求并产生费用；上面只统计此前已完成阶段，不是精确请求总数。</p>'
    checks_pass = "通过" if row.get("automatic_checks_pass") is True else "未通过 / 未完成"
    return (f'<section><h2>{index}. {_text(case.get("case_id"))}</h2>'
            f'<p class="question">{_text(case.get("question"))}</p>'
            f'<p>{technical_status} · 自动技术检查：{checks_pass} · 耗时：{_text(row.get("elapsed_seconds"))} 秒</p>'
            f'{error_html}{counts_html}'
            f'<p>预置相关块标签（待用户确认）：{_text(", ".join(str(value) for value in _items(case.get("relevant_chunk_ids"))) or "无；本题不计检索指标")}</p>'
            '<div class="scroll"><table><thead><tr><th>路线</th><th>检索状态</th>'
            f'<th>Hit@{_text(k)}</th><th>首个相关块名次</th><th>倒数排名</th><th>候选块 ID</th>'
            f'</tr></thead><tbody>{"".join(route_rows)}</tbody></table></div>'
            '<h3>回答与逐条引用</h3>' + _answer(row)
            + '<details><summary>自动技术检查（不是正确性裁判）</summary>' + checks_html + '</details>'
            '<h3>人工审核：待进行</h3><p>自动检查不能代替语义支持性、适用版本和现场风险核验。</p>'
            + _list(points) + '</section>')


def render_html(report: dict) -> str:
    """返回独立 HTML；不读取文件、密钥、服务，不运行脚本或发出模型请求。"""
    report = _mapping(report)
    summary = _mapping(report.get("summary"))
    k = report.get("k", 5)
    retrieval = _mapping(summary.get("retrieval"))
    rows = []
    for key, title in ROUTES:
        route = _mapping(retrieval.get(key))
        rows.append(f'<tr><td>{title}</td><td>{_text(route.get("hits"))}'
                    f' / {_text(route.get("denominator"))}</td>'
                    f'<td>{_metric(route.get("hit_at_k"))}</td>'
                    f'<td>{_metric(route.get("mrr_at_k"))}</td></tr>')
    metadata = "".join(f"<dt>{label}</dt><dd>{_text(report.get(key))}</dd>" for key, label in (
        ("schema_version", "报告结构版本"), ("created_at", "生成时间"), ("model", "回答模型"),
        ("snapshot_commit", "文档上游快照"), ("document_snapshot_sha256", "文档快照指纹"),
        ("vector_snapshot_sha256", "向量快照指纹")))
    cases = "".join(_case(index, row, k) for index, row in enumerate(_items(report.get("cases")), 1))
    return ('''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Kubernetes 开发冒烟评测</title>
<style>
body{margin:0;background:#f3f6f8;color:#243749;font:16px/1.7 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:24px}section,header{background:white;border:1px solid #d9e2e8;border-radius:12px;padding:24px;margin-bottom:20px}
h1{font-size:28px}h2{font-size:23px}h3{font-size:19px;color:#126d65}h4{font-size:16px}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border:1px solid #d9e2e8;vertical-align:top;overflow-wrap:anywhere}th{background:#edf3f5}.scroll{overflow:auto}
.notice{background:#edf5f4;padding:15px;border-left:4px solid #1b8278}.question{font-size:18px;font-weight:600}.quote{border-left:3px solid #aac7c2;margin:10px 0;padding:8px 15px;background:#f6f9fa}.quote p{font-size:14px}.missing{color:#9b442a}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,monospace}dd{margin-left:0;overflow-wrap:anywhere}dt{font-weight:600}a{color:#126d65;overflow-wrap:anywhere}li{margin:8px 0}summary{cursor:pointer}
</style></head><body><main><header><h1>Kubernetes 开发冒烟评测</h1>
<p class="notice">这是少量固定开发题的冒烟检查，不是准确率报告，也不是独立基准测试。相关块标签由助手对照原文预置，尚待用户确认，不是独立人审金标；未标注不等于不相关。技术完成、自动检查通过与答案正确是不同的判断。</p>'''
            f'<p>技术完成 {_text(summary.get("completed"))} / {_text(summary.get("total"))}；'
            f'运行失败 {_text(summary.get("failed"))}；自动技术检查通过 {_text(summary.get("automatic_checks_passed"))}；'
            f'人工审核待进行 {_text(summary.get("manual_review_pending"))}。</p>'
            f'<h2>检索指标对照（前 {_text(k)} 块）</h2><div class="scroll"><table><thead><tr>'
            f'<th>路线</th><th>命中数 / 可回答题分母</th><th>Hit@{_text(k)}</th><th>MRR@{_text(k)}</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            '<p>分母只包括带相关块标签的可回答题。检索失败仍在分母中计 0；生成失败不抹去已完成的检索结果。分数不是故障概率；各路线原始分数不横向比较。</p>'
            '<details><summary>实验身份与快照</summary><dl>' + metadata + '</dl></details></header>'
            + (cases or '<section><p>本报告尚无题目结果。</p></section>')
            + '<footer><p>本页仅展示保存的结果；打开、阅读或打印不会重新检索、请求模型或执行文档命令。JSON报告保留候选排名和实际引用原文，不含未引用候选全文。</p></footer>'
            '</main></body></html>')
