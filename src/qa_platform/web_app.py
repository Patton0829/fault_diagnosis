"""本地只读文档处理对照台：复用已有规则，零模型请求，不写语料或数据库。

运行：指定环境 python -B serve_data_lab.py；浏览器访问 http://127.0.0.1:8765。
--self-test 在进程内检查 API 和四篇样例，不启动监听端口。
"""

import argparse
from difflib import SequenceMatcher
from itertools import zip_longest
import json
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
import yaml

from qa_platform.ingestion.kubernetes_cleaning import (
    CORPUS_ROOT,
    DocumentCleaner,
    safe_path,
    tomllib,
)
from qa_platform.ingestion.kubernetes_chunking import (
    fixed_spans,
    make_documents,
    metrics,
    scan_structure,
    structural_spans,
    verify_partition,
)
from qa_platform.ingestion.kubernetes_cleanup import candidate_documents, cleanup
from qa_platform.settings import PROJECT_ROOT as PROJECT_ROOT, WEB_PAGE


PAGE = WEB_PAGE
# 用户只能选择这四篇，不把请求参数当文件路径使用。
SAMPLES = {
    "dns": {
        "title": "DNS 故障排查",
        "description": "观察代码被切开、长代码保护和块清理",
        "path": "tasks/administer-cluster/dns-debugging-resolution.md",
    },
    "pods": {
        "title": "调试 Pod",
        "description": "观察英文注释、标题与示例的清洗",
        "path": "tasks/debug/debug-application/debug-pods.md",
    },
    "crictl": {
        "title": "使用 crictl 调试",
        "description": "观察 feature-state 转换及完整代码",
        "path": "tasks/debug/debug-cluster/crictl.md",
    },
    "metrics": {
        "title": "资源指标管道（有告警）",
        "description": "观察未知 Mermaid 模板为何仍需核对",
        "path": "tasks/debug/debug-cluster/resource-metrics-pipeline.md",
    },
}
BUDGETS = (200, 800)


def diff_rows(raw: str, cleaned: str) -> list[dict]:
    """行级文字差异；replace的并排配对不代表原文行号的精确来源映射。"""
    before, after = raw.splitlines(), cleaned.splitlines()
    rows = []
    matcher = SequenceMatcher(None, before, after, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        for left, right in zip_longest(range(a0, a1), range(b0, b1)):
            rows.append(
                {
                    "tag": tag,
                    "raw_line": None if left is None else left + 1,
                    "cleaned_line": None if right is None else right + 1,
                    "raw_text": None if left is None else before[left],
                    "cleaned_text": None if right is None else after[right],
                }
            )
    return rows


def route_payload(label, documents, spans, text, sections, codes, size) -> dict:
    """只包装既有算法结果，不在界面层重新实现切块。"""
    chunks = []
    for document, span in zip(documents, spans):
        metadata = document.metadata
        chunks.append(
            {
                "id": document.id,
                "index": metadata["chunk_index"],
                "start": span.start,
                "end": span.end,
                "text": document.page_content,
                "chars": len(document.page_content),
                "sections": metadata["sections"],
                "over_budget": metadata["over_budget"],
                "cut_code": any(
                    span.start < code.end
                    and code.start < span.end
                    and not (span.start <= code.start and code.end <= span.end)
                    for code in codes
                ),
                "needs_heading_review": metadata.get("needs_heading_review", False),
                "heading_context_ranges": [
                    {**item, "text": text[item["start"] : item["end"]]}
                    for item in metadata.get("heading_context_ranges", [])
                ],
            }
        )
    return {
        "label": label,
        "stats": metrics(spans, sections, codes, size),
        "chunks": chunks,
        "omitted": [],
    }


def build_chunking(text: str, metadata: dict, size: int) -> dict:
    sections, codes = scan_structure(text)
    a = fixed_spans(text, size)
    b, _ = structural_spans(text, size, sections, codes)
    verify_partition(text, a, size, False)
    verify_partition(text, b, size, True)
    c = cleanup(text, b, sections, size)  # 内部验证候选＋排除账本完整覆盖。
    routes = {
        "A": route_payload(
            "固定字符切块",
            make_documents(text, a, sections, metadata, "fixed", size),
            a,
            text,
            sections,
            codes,
            size,
        ),
        "B": route_payload(
            "结构与代码保护",
            make_documents(text, b, sections, metadata, "structure_code", size),
            b,
            text,
            sections,
            codes,
            size,
        ),
        "C": route_payload(
            "结构块清理后",
            candidate_documents(text, b, c, sections, metadata, size),
            c.kept,
            text,
            sections,
            codes,
            size,
        ),
    }
    routes["C"].update(
        omitted=[
            {
                "start": item.span.start,
                "end": item.span.end,
                "text": text[item.span.start : item.span.end],
                "reason": item.reason,
                "section_start": item.section_start,
            }
            for item in c.excluded
        ],
        merges=[
            {
                "from_start": old.start,
                "from_end": old.end,
                "to_start": new.start,
                "to_end": new.end,
            }
            for old, new in c.merged
        ],
        deferred=[{"start": span.start, "end": span.end} for span in c.deferred],
    )
    return {
        "available": True,
        "error": None,
        "codes": [
            {"start": code.start, "end": code.end, "kind": code.kind} for code in codes
        ],
        "routes": routes,
        "offset_scope": "cleaned_document_python_characters_half_open",
    }


def build_preview(document_id: str, size: int) -> dict:
    if document_id not in SAMPLES or size not in BUDGETS:
        raise ValueError("仅允许四篇样例及200/800字符预算")
    root = CORPUS_ROOT / "upstream"
    source = json.loads(
        (CORPUS_ROOT / "source_manifest.json").read_text(encoding="utf-8")
    )
    commit = source["source"]["commit_sha"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("来源清单提交编号不合法")
    config_path = safe_path(root, "hugo.toml")
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    params = {
        **config.get("params", {}),
        **config.get("languages", {}).get("zh-cn", {}).get("params", {}),
    }
    cleaner = DocumentCleaner(root, params)
    cleaner.record(config_path, config_bytes, "site_config")
    loaded = cleaner.load(SAMPLES[document_id]["path"], commit)
    raw, cleaned = loaded["raw"], loaded["cleaned"]
    try:
        chunking = build_chunking(cleaned, loaded["metadata"], size)
    except ValueError as error:
        # 未知语法/切块失败仍可观察清洗结果，但不伪造可用候选。
        chunking = {"available": False, "error": str(error), "codes": [], "routes": {}}
    return {
        "document": {
            "id": document_id,
            **SAMPLES[document_id],
            "metadata": loaded["metadata"],
        },
        "budget": {
            "target_chars": size,
            "overlap_chars": 0,
            "unit": "python_characters",
            "model_api_calls": 0,
            "ready_for_embedding": False,
            "token_limit_checked": False,
            "preview_only": True,
        },
        "cleaning": {
            "raw": raw,
            "cleaned": cleaned,
            "raw_chars": len(raw),
            "cleaned_chars": len(cleaned),
            "raw_lines": len(raw.splitlines()),
            "cleaned_lines": len(cleaned.splitlines()),
            "counts": dict(cleaner.counts),
            "warnings": list(cleaner.warnings),
            "actions": [
                {"name": name, "before": before, "after": after}
                for name, (before, after) in cleaner.actions.items()
            ],
            "diff_rows": diff_rows(raw, cleaned),
        },
        "chunking": chunking,
    }


app = FastAPI(title="文档处理对照台", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def readonly_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; img-src 'none'; font-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'none'"
    )
    return response


@app.get("/")
def index():
    return FileResponse(PAGE, media_type="text/html")


@app.get("/api/documents")
def document_list():
    return {
        "documents": [{"id": key, **sample} for key, sample in SAMPLES.items()],
        "default_id": "dns",
    }


@app.get("/api/preview")
def preview(
    document: str = Query("dns", max_length=24),
    chunk_size: int = Query(200, ge=100, le=2000),
):
    if document not in SAMPLES or chunk_size not in BUDGETS:
        raise HTTPException(422, "请选择内置样例和200/800字符预算")
    try:
        return build_preview(document, chunk_size)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        raise HTTPException(422, f"本地资料预览失败：{error}") from error


def self_test():
    """合成区间＋真实样例＋进程内HTTP检查；不监听端口，不写文件。"""
    from fastapi.testclient import TestClient

    checks = 0
    rows = diff_rows("甲\n<script>not executed</script>\n", "甲\n乙😀\n")
    assert [row["raw_text"] for row in rows if row["raw_line"]] == [
        "甲",
        "<script>not executed</script>",
    ]
    assert [row["cleaned_text"] for row in rows if row["cleaned_line"]] == [
        "甲",
        "乙😀",
    ]
    checks += 1
    for document_id in SAMPLES:
        for size in BUDGETS:
            result = build_preview(document_id, size)
            assert (
                result["budget"]["model_api_calls"] == 0
                and not result["budget"]["ready_for_embedding"]
            )
            assert result["chunking"]["available"]
            text = result["cleaning"]["cleaned"]
            for route in result["chunking"]["routes"].values():
                ledger = sorted(
                    route["chunks"] + route["omitted"], key=lambda item: item["start"]
                )
                assert "".join(item["text"] for item in ledger) == text
                assert all(
                    item["text"] == text[item["start"] : item["end"]] for item in ledger
                )
            if document_id == "dns":
                routes = result["chunking"]["routes"]
                assert [routes[key]["stats"]["chunks"] for key in "ABC"] == (
                    [44, 55, 52] if size == 200 else [11, 19, 19]
                )
                assert routes["A"]["stats"]["cut_code"] > 0
                assert (
                    routes["B"]["stats"]["cut_code"]
                    == routes["C"]["stats"]["cut_code"]
                    == 0
                )
            if document_id == "metrics":
                assert any(
                    "mermaid" in warning for warning in result["cleaning"]["warnings"]
                )
            checks += 1
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert len(client.get("/api/documents").json()["documents"]) == 4
        assert client.get("/api/preview?document=dns&chunk_size=200").status_code == 200
        checks += 1
        for query in (
            "document=../models/.env",
            "document=unknown",
            "chunk_size=0",
            "chunk_size=300",
            "chunk_size=abc",
        ):
            assert client.get("/api/preview?" + query).status_code == 422
            checks += 1
        assert client.post("/api/preview").status_code == 405
        assert (
            client.get(
                "/api/documents", headers={"host": "outside.example"}
            ).status_code
            == 400
        )
        assert client.get("/api/documents").headers["cache-control"] == "no-store"
        checks += 1
    print(
        f"[自测通过] {checks}组：四篇×两档、DNS已知结果、区间账本、告警、参数/路径/Host/只读API；零模型请求。"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 1024 <= args.port <= 65535:
        parser.error("port须在1024～65535之间")
    if not PAGE.is_file():
        parser.error(f"缺少页面：{PAGE}")
    import uvicorn

    print(f"[本地预览] http://127.0.0.1:{args.port}", flush=True)
    print(
        "[边界] 只读四篇样例；不读.env、不联网调用模型、不改上游/JSONL/Milvus；Ctrl+C停止。",
        flush=True,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
