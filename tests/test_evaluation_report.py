"""离线报告可阅读且不把任何模型/文档字符串当作 HTML 执行。"""

from copy import deepcopy
from html import escape
from html.parser import HTMLParser
import socket

import pytest

from qa_platform.rag.evaluation_report import render_html


class Elements(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.tags = []
        self.links = []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "a":
            self.links.append(dict(attrs))


@pytest.fixture
def report():
    routes = {route: {"denominator": 1, "hits": 1, "hit_at_k": 1.0, "mrr_at_k": 0.5}
              for route in ("bm25", "dense", "rrf", "reranker")}
    case_routes = {route: {"top_chunk_ids": ["other", "dns-namespace"],
                          "first_relevant_rank": 2, "hit_at_k": True,
                          "reciprocal_rank_at_k": 0.5, "completed": True}
                   for route in routes}
    return {
        "schema_version": 1, "created_at": "2026-10-21", "model": "synthetic-model",
        "snapshot_commit": "a" * 40, "document_snapshot_sha256": "b" * 64,
        "vector_snapshot_sha256": "c" * 64, "k": 5,
        "summary": {"total": 1, "completed": 1, "failed": 0,
                    "automatic_checks_passed": 1, "manual_review_pending": 1,
                    "retrieval": routes},
        "cases": [{
            "case": {"case_id": "namespace", "question": "不同命名空间如何查询？",
                     "relevant_chunk_ids": ["dns-namespace"], "review_points": ["核对条件"]},
            "technical_status": "completed", "retrieval": case_routes,
            "automatic_checks_pass": True,
            "checks": {"valid_citations": True}, "elapsed_seconds": 1.25,
            "completed_request_counts": {"embedding": 1, "reranker": 1, "chat": 1, "http": 5},
            "failed_request_counts_unknown": False,
            "answer": {"status": "partial", "claims": [{"text": "应带上命名空间。",
                        "evidence": [{"source_id": "S1", "quote_id": "Q1"}]}],
                       "missing_info": ["缺少集群版本"]},
            "sources": {"S1": {"chunk_id": "dns-namespace", "text": "FULL_BODY_NOT_NEEDED",
                               "quotes": {"Q1": "原文：Pod 与 Service 不同命名空间。", "Q2": "UNRELATED_QUOTE"},
                               "metadata": {"title": "调试 DNS", "section_path": ["DNS", "名字空间"],
                                            "review_status": "unreviewed_official_document",
                                            "source_url": "https://github.com/kubernetes/website/blob/abc/content/zh-cn/docs/example.md"}}},
            "error": None, "manual_review": {"status": "pending", "points": ["每点是否受原文支持"]},
        }],
    }


def test_report_places_claims_next_to_only_the_cited_quotes(report):
    before = deepcopy(report)
    document = render_html(report)
    assert "应带上命名空间。" in document
    assert "[S1/Q1] 原文" in document
    assert "原文：Pod 与 Service 不同命名空间。" in document
    assert "FULL_BODY_NOT_NEEDED" not in document and "UNRELATED_QUOTE" not in document
    assert "DNS / 名字空间" in document
    assert "人工审核：待进行" in document and "每点是否受原文支持" in document
    assert "自动技术检查：通过" in document
    assert "Embedding 1；重排 1；回答生成 1；HTTP 5" in document
    assert "不是准确率报告" in document and "未标注不等于不相关" in document
    assert "Hit@5" in document and "MRR@5" in document and "0.5000" in document
    assert report == before


def test_all_dynamic_strings_are_escaped_in_text_and_attributes(report):
    attack = '\"><script>alert("x")</script><img src=x onerror=alert(1)>'
    report["model"] = attack
    report["created_at"] = attack
    report["cases"][0]["case"]["question"] = attack
    report["cases"][0]["case"]["relevant_chunk_ids"] = [attack]
    report["cases"][0]["answer"]["claims"][0]["text"] = attack
    report["cases"][0]["answer"]["missing_info"] = [attack]
    source = report["cases"][0]["sources"]["S1"]
    source["quotes"]["Q1"] = attack
    source["metadata"]["title"] = attack
    source["metadata"]["review_status"] = attack
    source["metadata"]["source_url"] += '?q=\"><script>alert(1)</script>'
    report["cases"][0]["checks"] = {attack: attack}
    report["cases"][0]["manual_review"]["points"] = [attack]
    document = render_html(report)
    elements = Elements(document)
    assert "script" not in elements.tags and "img" not in elements.tags
    assert escape(attack, quote=True) in document
    assert len(elements.links) == 1
    assert set(elements.links[0]) == {"href", "rel"}
    assert elements.links[0]["href"] == source["metadata"]["source_url"]


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "http://github.com/kubernetes/website/blob/abc/doc.md",
    "https://github.com.evil.invalid/kubernetes/website/blob/abc/doc.md",
    "https://github.com@evil.invalid/kubernetes/website/blob/abc/doc.md",
    "https://github.com:443/kubernetes/website/blob/abc/doc.md",
    "https://github.com/other/website/blob/abc/doc.md",
    "https://github.com/kubernetes/website/blob/", "//github.com/kubernetes/website/blob/abc/doc.md",
    "https://github.com/kubernetes/website/blob/a\nb/doc.md", "https://[broken",
])
def test_non_whitelisted_source_urls_are_plain_text(report, url):
    report["cases"][0]["sources"]["S1"]["metadata"]["source_url"] = url
    document = render_html(report)
    assert not Elements(document).links
    assert escape(url, quote=True) in document


def test_missing_answer_and_failed_generation_do_not_become_factual_failure(report):
    row = report["cases"][0]
    row["technical_status"] = "failed"
    row["answer"] = None
    row["checks"] = None
    row["error"] = {"stage": "answer", "kind": "validation_error", "message": "输出结构不符合约定"}
    document = render_html(report)
    assert "没有可展示的已校验回答" in document
    assert "技术运行未完成不代表答案事实错误" in document
    assert "输出结构不符合约定" in document
    assert "人工审核：待进行" in document
    assert "检索失败仍在分母中计 0" in document


def test_failed_request_counters_explicitly_exclude_unknown_failed_stage(report):
    row = report["cases"][0]
    row["failed_request_counts_unknown"] = True
    row["completed_request_counts"] = {"embedding": 1, "reranker": 1, "chat": 0, "http": 4}
    document = render_html(report)
    assert "HTTP 4" in document
    assert "失败阶段可能已发送请求并产生费用" in document
    assert "不是精确请求总数" in document


def test_none_metrics_and_empty_report_are_supported(report):
    for route in report["summary"]["retrieval"].values():
        route.update(denominator=0, hits=0, hit_at_k=None, mrr_at_k=None)
    assert render_html(report).count("不适用") >= 8
    assert "本报告尚无题目结果" in render_html({})


@pytest.mark.parametrize("reference", [{"source_id": "missing", "quote_id": "Q1"},
                                       {"source_id": "S1", "quote_id": "missing"},
                                       {"source_id": "<script>", "quote_id": ["bad"]}, None])
def test_invalid_citation_displays_missing_without_executing(report, reference):
    report["cases"][0]["answer"]["claims"][0]["evidence"] = [reference]
    document = render_html(report)
    assert "引用未找到" in document
    assert not Elements(document).links
    assert "script" not in Elements(document).tags


def test_renderer_is_offline_and_has_no_executable_or_network_assets(report, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("renderer must not perform I/O")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    document = render_html(report)
    tags = Elements(document).tags
    assert not set(tags).intersection({"script", "iframe", "img", "link", "form", "audio", "video"})
    assert "default-src 'none'" in document
    assert "不会重新检索、请求模型或执行文档命令" in document
