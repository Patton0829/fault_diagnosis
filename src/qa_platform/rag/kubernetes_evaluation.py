"""三道Kubernetes开发回归题；真实图、部分相关标签、自动检查与人工审核分开。"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from qa_platform.settings import PACKAGE_ROOT, PROJECT_ROOT


DEFAULT_INDEX = PROJECT_ROOT / "data" / "milvus_indexes" / "kubernetes_zh_v1"
SNAPSHOT_COMMIT = "2f85f2802e59eb2128543dbfbf114ad641db4be0"
DOCUMENT_SNAPSHOT = "a33e0289f9eefa50a5bf9bf3e0f4810f85cea4c5dbda50d40a7588d8bb6129c8"
ROUTES = ("bm25", "dense", "rrf", "reranker")
K = 5
DEFAULT_MODEL = "qwen3.8-flash"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    expected_statuses: tuple[str, ...]
    forbidden_literals: tuple[str, ...]
    review_points: tuple[str, ...]


CASES = (
    EvalCase(
        "namespace", "Pod 与 Service 不在同一命名空间时，DNS 查询应该指定什么？请简短回答。",
        ("8c3b126c9dd63e67", "4fe4ff99d7c2e8c4"), ("answered", "partial"), (),
        ("是否明确指定Service所在命名空间，而非Pod所在命名空间", "是否把示例default误当成固定名称"),
    ),
    EvalCase(
        "dns_config", "完全限定名称可以解析，但相对名称不能解析时，应检查 Pod 中的哪个配置文件？",
        ("a78c05ca1bacb446",), ("answered", "partial"), (),
        ("是否回答Pod内的/etc/resolv.conf", "是否误答CoreDNS Corefile或节点的其他resolv.conf路径"),
    ),
    EvalCase(
        "out_of_scope", "英国的首都是什么？", (), ("insufficient_evidence",), ("伦敦", "London"),
        ("是否按当前Kubernetes知识库不足拒答", "missing_info是否绕过引用规则偷偷给出答案"),
    ),
)
LABEL_EXCERPTS = {
    "8c3b126c9dd63e67": "如果 Pod 和 Service 的名字空间不相同，则 DNS 查询必须指定 Service 所在的名字空间。",
    "4fe4ff99d7c2e8c4": "请尝试使用限定命名空间的名称（同样在 Pod 内运行）：",
    "a78c05ca1bacb446": "如果你能够使用完全限定的名称查找，但不能使用相对名称，则需要检查你 Pod 中的\n`/etc/resolv.conf` 文件是否正确。",
}


def ranking_metrics(chunk_ids: list[str], relevant_chunk_ids: tuple[str, ...], k: int = K) -> dict:
    """MRR@K截断在K；没有正例标签的拒答题不进入检索指标分母。"""
    if (type(k) is not int or k < 1 or not isinstance(chunk_ids, list)
            or any(not isinstance(value, str) or not value for value in chunk_ids)
            or len(chunk_ids) != len(set(chunk_ids))):
        raise ValueError("排名必须是唯一块ID列表，K须为正整数")
    rank = next((rank for rank, value in enumerate(chunk_ids, 1) if value in relevant_chunk_ids), None)
    hit = bool(rank and rank <= k)
    return {"top_chunk_ids": chunk_ids[:k], "first_relevant_rank": rank, "completed": True,
            "hit_at_k": hit if relevant_chunk_ids else None,
            "reciprocal_rank_at_k": (1 / rank if hit else 0.0) if relevant_chunk_ids else None}


def prepare_suite(index_dir: Path = DEFAULT_INDEX, case_ids: list[str] | None = None) -> dict:
    """只离线核验既有ready合同与标注原文，不打开Milvus或读取认证。"""
    from qa_platform.retrieval import kubernetes_hybrid as hybrid

    selected = {case.case_id for case in CASES} if case_ids is None else set(case_ids)
    if not selected or not selected.issubset({case.case_id for case in CASES}):
        raise ValueError("评测题目编号无效")
    cases = [case for case in CASES if case.case_id in selected]
    check = hybrid.prepare(index_dir, cases[0].question)
    plan = check["plan"]
    if (plan["document_snapshot_sha256"] != DOCUMENT_SNAPSHOT
            or {row["snapshot_commit"] for row in plan["documents"]} != {SNAPSHOT_COMMIT}):
        raise ValueError("相关标签只适用于已核对的文档快照；变更语料须重新审核标签")
    texts = {row["chunk_id"]: row["text"] for row in check["records"]}
    for case in cases:
        for chunk_id in case.relevant_chunk_ids:
            if LABEL_EXCERPTS[chunk_id] not in texts.get(chunk_id, ""):
                raise ValueError("相关标签原文不匹配；未发送模型请求")
    return {"cases": cases, "check": check}


def answer_checks(case: EvalCase, public: dict) -> dict:
    """检查显式规则，不用词语命中宣称语义正确，也不增加裁判模型。"""
    answer = public["answer"]
    cited = {public["sources"][ref["source_id"]]["chunk_id"]
             for claim in answer["claims"] for ref in claim["evidence"]}
    free_text = "\n".join([claim["text"] for claim in answer["claims"]] + answer["missing_info"])
    forbidden = [word for word in case.forbidden_literals if word.casefold() in free_text.casefold()]
    refusal = (answer["status"] == "insufficient_evidence" and not answer["claims"]
               and bool(answer["missing_info"]) and not forbidden)
    return {"expected_status_match": answer["status"] in case.expected_statuses,
            "citation_locators_valid": True if answer["claims"] else None,
            "cites_known_relevant_chunk": bool(cited.intersection(case.relevant_chunk_ids))
            if case.relevant_chunk_ids else None,
            "forbidden_literals_found": forbidden,
            "refusal_rule_pass": refusal if not case.relevant_chunk_ids else None}


def evaluate_case(case: EvalCase, index_dir: Path, model: str, *, graph_factory=None) -> dict:
    """驱动原LangGraph一次；旁观已有节点update，不复制RAG或访问会话/Redis。"""
    from langsmith import tracing_context
    from qa_platform.workflows import live_rag as live

    row = {
        "case": asdict(case), "technical_status": "failed",
        "retrieval": {route: {**ranking_metrics([], case.relevant_chunk_ids), "completed": False}
                      for route in ROUTES},
        "checks": None, "answer": None, "sources": {}, "error": None,
        "automatic_checks_pass": False,
        "manual_review": {"status": "pending", "points": [*case.review_points,
            "逐条判断引用原句是否充分支持回答，而不只是引用编号存在",
            "是否保留条件、示例和版本边界；资料未审核不代表生产规程"]},
        "completed_request_counts": {"embedding": 0, "reranker": 0, "chat": 0, "http": 0},
        "failed_request_counts_unknown": False,
        "elapsed_seconds": 0.0,
    }
    started, completed, stage, result = perf_counter(), [], "retrieve", None
    try:
        with tracing_context(enabled=False):
            graph = (graph_factory or live.build_graph)(index_dir)
            for update in graph.stream(
                {"query": case.question, "model": model},
                config={"callbacks": [], "recursion_limit": 10}, stream_mode="updates",
            ):
                if len(completed) >= len(live.PIPELINE):
                    raise ValueError("图返回多余节点")
                stage = live.PIPELINE[len(completed)]
                if set(update) != {stage} or set(update[stage]) != {"result"}:
                    raise ValueError("节点顺序或字段不符合原图")
                result = update[stage]["result"]
                if result["query"] != case.question:
                    raise ValueError("节点问题与当前评测题不一致")
                counts = result["request_counts"]
                if any(type(counts[key]) is not int or counts[key] < 0 for key in row["completed_request_counts"]):
                    raise ValueError("请求计数无效")
                expected_counts = {
                    "embedding": 1, "http": {"retrieve": 3, "rerank": 4, "answer": 5}[stage],
                    "reranker": int(stage != "retrieve"), "chat": int(stage == "answer"),
                }
                if counts != expected_counts or "cache" in result:
                    raise ValueError("评测要求无缓存的原三节点请求合同")
                if stage == "retrieve":
                    for route in ROUTES[:3]:
                        row["retrieval"][route] = ranking_metrics(
                            [hit["chunk_id"] for hit in result["routes"][route]], case.relevant_chunk_ids,
                        )
                elif stage == "rerank":
                    row["retrieval"]["reranker"] = ranking_metrics(
                        [hit["chunk_id"] for hit in result["routes"]["reranker"]], case.relevant_chunk_ids,
                    )
                row["completed_request_counts"] = {key: counts[key] for key in row["completed_request_counts"]}
                completed.append(stage)
                stage = live.PIPELINE[min(len(completed), len(live.PIPELINE) - 1)]
            if completed != list(live.PIPELINE):
                raise ValueError("图未完整结束")
            public = live._public_result(result, model)  # 沿用网页相同的JSON/引用与公开字段边界。
        checks = answer_checks(case, public)
        required = [checks["expected_status_match"]]
        if case.relevant_chunk_ids:
            required.extend((row["retrieval"]["reranker"]["hit_at_k"], checks["cites_known_relevant_chunk"]))
        else:
            required.append(checks["refusal_rule_pass"])
        row.update(technical_status="completed", checks=checks, answer=public["answer"],
                   sources=public["sources"], automatic_checks_pass=all(required))
    except Exception as exc:
        row["failed_request_counts_unknown"] = True
        row["error"] = {"stage": stage,
                        "kind": "validation_error" if isinstance(exc, (ValueError, KeyError, TypeError)) else "execution_error",
                        "message": "本阶段未完成；保留先前检索指标，不显示未经校验的草稿或原始异常，不自动重试。"}
    row["elapsed_seconds"] = round(perf_counter() - started, 3)
    return row


def summarize(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["case"]["relevant_chunk_ids"]]
    retrieval = {}
    for route in ROUTES:
        hits = sum(row["retrieval"][route]["hit_at_k"] is True for row in positive)
        rr = sum(row["retrieval"][route]["reciprocal_rank_at_k"] or 0 for row in positive)
        denominator = len(positive)  # 检索失败仍在分母；回答失败不抹去已完成检索。
        retrieval[route] = {"denominator": denominator, "hits": hits,
                            "hit_at_k": hits / denominator if denominator else None,
                            "mrr_at_k": rr / denominator if denominator else None}
    completed = sum(row["technical_status"] == "completed" for row in rows)
    return {"total": len(rows), "completed": completed, "failed": len(rows) - completed,
            "automatic_checks_passed": sum(row["automatic_checks_pass"] for row in rows),
            "manual_review_pending": sum(row["manual_review"]["status"] == "pending" for row in rows),
            "retrieval": retrieval}


def run_suite(prepared: dict, model: str) -> dict:
    plan = prepared["check"]["plan"]
    rows = []
    for case in prepared["cases"]:
        print(f"[评测] {case.case_id}：{case.question}", flush=True)
        row = evaluate_case(case, prepared["check"]["index_dir"], model)
        rows.append(row)
        print(f"[结果] 技术状态={row['technical_status']}；规则检查={row['automatic_checks_pass']}；人工审核=pending", flush=True)
    source_files = ("rag/kubernetes_evaluation.py", "workflows/live_rag.py", "rag/kubernetes_answer.py",
                    "retrieval/kubernetes_hybrid.py", "retrieval/kubernetes_rerank.py")
    return {
        "schema_version": "k8s-eval/v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model, "k": K, "snapshot_commit": SNAPSHOT_COMMIT,
        "document_snapshot_sha256": plan["document_snapshot_sha256"],
        "vector_snapshot_sha256": plan["vector_snapshot_sha256"],
        "embedding_model": plan["embedding"]["model"],
        "source_versions": {name: sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest() for name in source_files},
        "cache": "disabled", "conversation_storage": "not_used",
        "label_scope": "已知问题的开发回归题；助手对照原文预置的部分正例标签，待用户确认，非独立人审金标或准确率。",
        "cases": rows, "summary": summarize(rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Kubernetes三题最小开发评测，默认纯离线计划")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--case", action="append", choices=[case.case_id for case in CASES])
    parser.add_argument("--model", choices=["qwen3.8-flash", "qwen3.8-max", "glm"], default=DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        prepared = prepare_suite(args.index_dir, args.case)
        print(f"[范围] {len(prepared['cases'])}道开发回归题；固定已核对快照；不使用Redis或会话历史。")
        for case in prepared["cases"]:
            print(f"  {case.case_id} | {case.question}")
        print(f"[计划] 正常每题Embedding1/重排1/回答1/HTTP5；本次共约{5 * len(prepared['cases'])}次HTTP。无重试或裁判模型。")
        print("[边界] 已知相关块为部分标签；Hit@5/MRR@5不是答案正确率，引用定位不是语义支持。人工审核始终待进行。")
        if not args.run:
            print("[计划完成] 未读密钥、未建HTTP/Milvus客户端、未写报告；加--run才执行。")
            return 0
        # CLI与网页共用本机Lite索引：先停应用，禁止两个进程同时打开它。
        from qa_platform.deployment import require_exclusive_app
        from qa_platform.rag.evaluation_report import render_html

        require_exclusive_app()
        output = args.output_dir or PROJECT_ROOT / "reports" / f"k8s-eval-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        output.mkdir(mode=0o700, exist_ok=False)  # 已有目录不覆盖；失败现场保留。
        if args.model == "glm":
            print("[提醒] GLM配置使用明文HTTP；本次仅固定公开文档与测试问题，不应替换为敏感资料。")
        report = run_suite(prepared, args.model)
        for name, content in (("report.json", json.dumps(report, ensure_ascii=False, indent=2)),
                              ("report.html", render_html(report))):
            with (output / name).open("x", encoding="utf-8") as stream:
                stream.write(content)
        print(f"[汇总] 技术完成{report['summary']['completed']}/{len(report['cases'])}；人工审核仍待进行。")
        print(f"[报告] {output / 'report.html'}")
        return 0 if report["summary"]["failed"] == 0 else 1
    except Exception:
        print("[未完成] 请检查索引/标签合同、应用是否已停止、报告目录是否为新目录；不显示原始异常，不覆盖、不回退或自动重试。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
