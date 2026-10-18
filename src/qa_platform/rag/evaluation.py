"""第十步：四道固定开发回归题；分开记录检索、引用、拒答与人工审核。"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
)
from dotenv import load_dotenv

from qa_platform.ingestion.railway import load_records, split_records
from qa_platform.providers.chat import DEFAULT_MODEL, MODEL_CONFIGS
from qa_platform.rag.answering import (
    SYSTEM_PROMPT,
    AnswerValidationError,
    GroundedAnswer,
    TruncatedAnswerError,
    build_quotes,
    build_sources,
    generate_answer,
    parse_answer,
    print_answer,
    print_answer_diagnostics,
)
from qa_platform.retrieval.dense import DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE
from qa_platform.retrieval.pipeline import (
    EMBEDDING_API_URL,
    EMBEDDING_MODEL,
    RERANK_API_URL,
    RERANK_MODEL,
    RetrievalError,
    run_retrieval,
)
from qa_platform.settings import ENV_PATH, PACKAGE_ROOT, PROJECT_ROOT


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    target_entry_ids: tuple[str, ...]
    expected_statuses: tuple[str, ...]
    forbidden_literals: tuple[str, ...]
    review_points: tuple[str, ...]


CASES = [
    EvalCase(
        "motor",
        "电机还在转，但丝杆一点也不动，可能是什么原因？",
        ("TEST-SW-006",),
        ("answered", "partial"),
        (),
        (
            "是否只列可能原因而非确诊",
            "是否区分实际停转与反馈异常",
            "是否擅自补充原文没有的阈值或操作",
        ),
    ),
    EvalCase(
        "distance",
        "那个测距离的数一直停在老数上，设备动了好几次也不更新，可电流和转速都在变，怎么查？",
        ("TEST-SW-005",),
        ("answered", "partial"),
        (),
        (
            "是否理解测距读数冻结而其他数据变化",
            "回答是否受所引原文支持，而非泛谈电机故障",
        ),
    ),
    EvalCase(
        "out_of_scope",
        "英国的首都是什么？",
        (),
        ("insufficient_evidence",),
        ("伦敦", "London"),
        ("是否依据知识库不足拒答，而不是靠常识作答", "missing_info是否偷偷泄露答案"),
    ),
    EvalCase(
        "test_code",
        "测试知识库中，QZ-7319-BLUE-WHALE对应什么设备？这是真实设备资料吗？",
        ("TEST-SW-007",),
        ("answered", "partial"),
        (),
        (
            "是否准确识别为蓝鲸测试转辙机",
            "是否明确说明设备资料是虚构测试",
            "不得把虚构复位步骤推广为真实维护规程",
            "出现测试或虚构字样本身不等于理解边界",
        ),
    ),
]


def retrieval_metrics(case: EvalCase, hits: list) -> dict:
    """仅对有目标条目的题计算；排名单位是最终重排子块，命中按entry_id认定。"""
    entries = [hit.candidate.document.metadata.get("entry_id") for hit in hits]
    target_rank = next(
        (
            rank
            for rank, entry in enumerate(entries, 1)
            if entry in case.target_entry_ids
        ),
        None,
    )
    return {
        "candidate_entry_ids": entries,
        "target_rank": target_rank,
        "hit_at_k": target_rank is not None if case.target_entry_ids else None,
        "reciprocal_rank": (1.0 / target_rank if target_rank else 0.0)
        if case.target_entry_ids
        else None,
    }


def answer_checks(case: EvalCase, answer: GroundedAnswer, sources: dict) -> dict:
    """机器检查只检查明确规则，不能代替人工判断答案是否正确。"""
    answer = parse_answer(answer.model_dump_json(), sources)
    cited_entries = list(
        dict.fromkeys(
            sources[reference.source_id].candidate.document.metadata.get("entry_id")
            for claim in answer.claims
            for reference in claim.evidence
        )
    )
    # 仅检查模型生成的自由文本，不把程序固定警告或引用原文误算成模型能力。
    free_text = "\n".join([claim.text for claim in answer.claims] + answer.missing_info)
    forbidden = [
        term
        for term in case.forbidden_literals
        if term.casefold() in free_text.casefold()
    ]
    refusal = None
    if not case.target_entry_ids:
        refusal = (
            answer.status == "insufficient_evidence"
            and not answer.claims
            and bool(answer.missing_info)
            and not forbidden
        )
    return {
        "expected_status_match": answer.status in case.expected_statuses,
        "citation_check": "passed" if answer.claims else "not_applicable_no_claims",
        "cited_entry_ids": cited_entries,
        "cites_target": bool(set(cited_entries) & set(case.target_entry_ids))
        if case.target_entry_ids
        else None,
        "forbidden_literals_found": forbidden,
        "closed_book_refusal_check": refusal,
        "fiction_word_in_model_text": "虚构" in free_text
        if case.case_id == "test_code"
        else None,
    }


def evidence_snapshot(sources: dict) -> dict:
    """保存本次实际候选和原文编号，便于后来审核；不保存密钥或服务原始响应。"""
    return {
        source_id: {
            "metadata": hit.candidate.document.metadata,
            "content": hit.candidate.document.page_content,
            "quotes": build_quotes(hit),
            "rerank_score": hit.score,
            "rrf_rank": hit.input_index + 1,
        }
        for source_id, hit in sources.items()
    }


def safe_error(exc: Exception, stage: str) -> dict:
    """接口错误不输出响应体；模型原始草稿也不写进失败报告。"""
    if isinstance(exc, RetrievalError):
        kind, message = "retrieval_error", str(exc)
    elif isinstance(exc, TruncatedAnswerError):
        kind, message = "truncated", str(exc)
    elif isinstance(exc, AnswerValidationError):
        kind, message = "validation_error", str(exc)
    elif isinstance(exc, (APITimeoutError, httpx.TimeoutException)):
        kind, message = "timeout", "模型请求超时。"
    elif isinstance(exc, APIStatusError):
        kind, message = "http_error", f"聊天服务返回HTTP {exc.status_code}。"
    elif isinstance(exc, (APIConnectionError, httpx.RequestError)):
        kind, message = "connection_error", "模型服务连接失败。"
    elif isinstance(exc, OpenAIError):
        kind, message = "sdk_error", type(exc).__name__
    else:
        kind, message = "validation_error", str(exc)
    result: dict = {"stage": stage, "kind": kind, "message": message}
    # 保留既有stage/kind/message字段，只增补脱敏结构；不保存原始异常对象。
    if (
        isinstance(exc, (AnswerValidationError, TruncatedAnswerError))
        and exc.diagnostics
    ):
        result["diagnostics"] = exc.diagnostics
    return result


def evaluate_case(
    case: EvalCase,
    chunks: list,
    retrieval_client: httpx.Client,
    chat_client: OpenAI,
    args: argparse.Namespace,
) -> dict:
    """每题独立记账；运行失败与状态不合预期是两种不同结果。"""
    row = {
        "case": asdict(case),
        "technical_status": "failed",
        "retrieval_ok": False,
        "retrieval": retrieval_metrics(case, []),
        "checks": None,
        "answer": None,
        "sources": {},
        "error": None,
        "automatic_checks_pass": False,
        "manual_review": {
            "status": "pending",
            "points": list(case.review_points)
            + [
                "每个要点是否受所引原文直接支持，且实际回答了问题",
                "虚构测试资料不能经简单确认就作为真实规程",
            ],
        },
        "timings": {
            "retrieval_seconds": None,
            "generation_seconds": None,
            "total_seconds": None,
        },
    }
    started = perf_counter()
    stage = "retrieval"
    stage_started = started
    try:
        hits = run_retrieval(
            retrieval_client,
            case.question,
            chunks,
            args.recall_k,
            args.fusion_k,
            args.top_k,
            args.rrf_constant,
            args.batch_size,
            show_candidates=False,
        )
        row["timings"]["retrieval_seconds"] = perf_counter() - stage_started
        row["retrieval_ok"] = True
        row["retrieval"] = retrieval_metrics(case, hits)
        sources = build_sources(hits)
        row["sources"] = evidence_snapshot(sources)
        stage = "generation"
        stage_started = perf_counter()
        answer = generate_answer(
            chat_client, case.question, sources, args.model, args.max_tokens
        )
        row["timings"]["generation_seconds"] = perf_counter() - stage_started
        stage = "answer_checks"
        row["checks"] = answer_checks(case, answer, sources)
        row["answer"] = answer.model_dump()
        row["technical_status"] = "completed"
        required = [row["checks"]["expected_status_match"]]
        if case.target_entry_ids:
            required += [row["retrieval"]["hit_at_k"], row["checks"]["cites_target"]]
        else:
            required += [row["checks"]["closed_book_refusal_check"]]
        row["automatic_checks_pass"] = all(required)
        print_answer(answer, sources)
    except (
        RetrievalError,
        TruncatedAnswerError,
        OpenAIError,
        httpx.RequestError,
        ValueError,
        TypeError,
        OverflowError,
    ) as exc:
        if stage in {"retrieval", "generation"}:
            row["timings"][f"{stage}_seconds"] = perf_counter() - stage_started
        row["error"] = safe_error(exc, stage)
        print(f"[运行失败] {row['error']['kind']}：{row['error']['message']}")
        print_answer_diagnostics(row["error"].get("diagnostics", {}))
    row["timings"]["total_seconds"] = perf_counter() - started
    print(
        f"[自动检查] {'通过' if row['automatic_checks_pass'] else '未通过'}；人工审核仍待进行"
    )
    print(f"[人工审核点] {'；'.join(row['manual_review']['points'])}")
    return row


def summarize(results: list[dict], top_k: int) -> dict:
    """失败题保留在分母里；不可回答题不参加目标条目命中率与MRR。"""
    answerable = [row for row in results if row["case"]["target_entry_ids"]]
    refusal_rows = [row for row in results if not row["case"]["target_entry_ids"]]
    hits = sum(bool(row["retrieval"]["hit_at_k"]) for row in answerable)
    return {
        "planned_cases": len(results),
        "completed_cases": sum(
            row["technical_status"] == "completed" for row in results
        ),
        "failed_cases": sum(row["technical_status"] != "completed" for row in results),
        "automatic_checks_passed": sum(row["automatic_checks_pass"] for row in results),
        "answerable_cases": len(answerable),
        "retrieval_completed_answerable": sum(
            row["retrieval_ok"] for row in answerable
        ),
        "hit_count": hits,
        "top_k": top_k,
        "hit_at_k": hits / len(answerable) if answerable else None,
        "mrr": sum(row["retrieval"]["reciprocal_rank"] for row in answerable)
        / len(answerable)
        if answerable
        else None,
        "refusal_cases": len(refusal_rows),
        "refusal_checks_passed": sum(
            bool(row["checks"] and row["checks"]["closed_book_refusal_check"])
            for row in refusal_rows
        ),
        "manual_reviews_pending": len(results),
    }


def make_metadata(
    cases: list[EvalCase], chunks: list, args: argparse.Namespace
) -> dict:
    """记录开发回归题、知识版本、提示词与代码版本；固定版本也不保证模型完全确定。"""
    source_files = [
        PACKAGE_ROOT / name
        for name in (
            "rag/evaluation.py",
            "rag/answering.py",
            "retrieval/pipeline.py",
            "ingestion/railway.py",
            "providers/chat.py",
            "providers/cloud_embedding.py",
            "retrieval/bm25.py",
            "retrieval/dense.py",
            "retrieval/hybrid.py",
            "providers/cloud_reranker.py",
            "settings.py",
        )
    ]
    params = {
        name: getattr(args, name)
        for name in (
            "model",
            "max_tokens",
            "recall_k",
            "fusion_k",
            "top_k",
            "rrf_constant",
            "chunk_size",
            "chunk_overlap",
            "batch_size",
        )
    }
    return {
        "scope": "development_regression_not_independent_benchmark",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python": sys.executable,
        "packages": {
            name: version(name)
            for name in (
                "openai",
                "httpx",
                "numpy",
                "pydantic",
                "langchain-core",
                "langchain-text-splitters",
                "jieba",
                "rank-bm25",
            )
        },
        "suite_sha256": sha256(
            json.dumps(
                [asdict(case) for case in CASES], ensure_ascii=False, sort_keys=True
            ).encode()
        ).hexdigest(),
        "selected_case_ids": [case.case_id for case in cases],
        "parameters": params,
        "models": {
            "embedding": EMBEDDING_MODEL,
            "embedding_api_url": EMBEDDING_API_URL,
            "reranker": RERANK_MODEL,
            "reranker_api_url": RERANK_API_URL,
            "chat": args.model,
            "chat_base_url": MODEL_CONFIGS[args.model]["base_url"],
        },
        "knowledge_versions": {
            chunk.metadata["source"]: chunk.metadata["document_version"]
            for chunk in chunks
        },
        "prompt_sha256": sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "code_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_files
        },
        "limitations": [
            "只有4道开发回归题，不代表整体准确率",
            "引用机械校验不等于语义支持验证",
            "虚构标记由程序展示，不作为模型理解虚构能力的分数",
            "自动检查通过仍需人工审核",
        ],
    }


def save_report(path: Path, report: dict) -> None:
    """仅创建新报告，不覆盖已有结果。"""
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(serialized + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="固定测试库的4题RAG开发回归评测；默认零请求"
    )
    parser.add_argument(
        "--case",
        choices=["all", *(case.case_id for case in CASES)],
        default="all",
        help="只选一道或运行全部",
    )
    parser.add_argument(
        "--run", action="store_true", help="显式开始在线评测；不加则只展示计划"
    )
    parser.add_argument(
        "--output", type=Path, help="可选JSON报告路径；禁止覆盖已有文件"
    )
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), default=DEFAULT_MODEL)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--fusion-k", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if (
        min(
            args.recall_k, args.fusion_k, args.top_k, args.rrf_constant, args.max_tokens
        )
        <= 0
    ):
        parser.error("数量、RRF常数和max-tokens必须是正整数")
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size 必须在1到{MAX_BATCH_SIZE}之间")
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("必须满足 chunk_size > 0 且 0 <= chunk_overlap < chunk_size")
    output = args.output.expanduser() if args.output else None
    if output and (output.exists() or output.is_symlink()):
        parser.error("报告路径已存在，请更换文件名，避免覆盖旧结果")
    output = output.resolve() if output else None
    cases = [case for case in CASES if args.case in {"all", case.case_id}]
    try:
        chunks = split_records(
            load_records("test"), args.chunk_size, args.chunk_overlap
        )
        available_ids = {chunk.metadata.get("entry_id") for chunk in chunks}
        if any(set(case.target_entry_ids) - available_ids for case in cases):
            raise ValueError("题集目标条目不在当前测试库中，请先核对知识版本。")
        metadata = make_metadata(cases, chunks, args)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[错误] 本地准备失败：{exc}", file=sys.stderr)
        return 1
    per_case = (len(chunks) + args.batch_size - 1) // args.batch_size + 3
    print(f"[配置] Python：{sys.executable}；回答模型：{args.model}")
    print(f"[范围] 仅测试库，{len(chunks)}个子块；选中{len(cases)}道开发回归题")
    for case in cases:
        print(f"[{case.case_id}] {case.question}")
        print(
            f"  目标条目：{', '.join(case.target_entry_ids) or '无，预期依据知识库不足拒答'}"
        )
    print(
        f"[计划] 每题正常完成需{per_case}次模型请求，全量选中题最多约{per_case * len(cases)}次；失败或空证据可能更少"
    )
    print(
        "[边界] 不增加裁判模型、不自动重试、不缓存向量；自动检查不是答案正确率，全部题仍需人审。"
    )
    if not args.run:
        print(
            "[计划完成] 未读取密钥、未创建HTTP客户端、未发送请求、未写报告；加--run才执行。"
        )
        return 0
    if not ENV_PATH.is_file():
        print(f"[错误] 配置文件不存在：{ENV_PATH}", file=sys.stderr)
        return 1
    load_dotenv(ENV_PATH, override=False)
    retrieval_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    config = MODEL_CONFIGS[args.model]
    chat_key = (
        os.getenv(config["api_key_env"], "").strip()
        if config["api_key_env"]
        else "not-needed"
    )
    if not retrieval_key or not chat_key:
        print("[错误] 检索或聊天认证配置缺失。", file=sys.stderr)
        return 1
    results = []
    with httpx.Client(
        headers={"Authorization": f"Bearer {retrieval_key}"}, timeout=60.0
    ) as retrieval_client:
        with OpenAI(
            api_key=chat_key, base_url=config["base_url"], timeout=60.0, max_retries=0
        ) as chat_client:
            for case in cases:
                print(f"\n[开始评测] {case.case_id}：{case.question}", flush=True)
                results.append(
                    evaluate_case(case, chunks, retrieval_client, chat_client, args)
                )
    summary = summarize(results, args.top_k)
    print(
        f"\n[汇总] 技术完成 {summary['completed_cases']}/{len(cases)}；运行失败 {summary['failed_cases']}；自动检查通过 {summary['automatic_checks_passed']}/{len(cases)}"
    )
    if summary["answerable_cases"]:
        print(
            f"[检索] Hit@{args.top_k}={summary['hit_count']}/{summary['answerable_cases']}；MRR={summary['mrr']:.4f}；检索完成题={summary['retrieval_completed_answerable']}"
        )
        print(
            "[分母] 仅包含可回答题；检索失败保留在分母并记0，生成失败不抹去已完成的检索结果。"
        )
    else:
        print("[检索] Hit@K和MRR不适用：本次未选择可回答题。")
    print(
        f"[拒答] 状态/空要点/禁答词检查 {summary['refusal_checks_passed']}/{summary['refusal_cases']}；人工审核待进行 {len(cases)}题"
    )
    if output:
        try:
            save_report(
                output,
                {
                    "schema_version": 1,
                    "metadata": metadata,
                    "summary": summary,
                    "results": results,
                },
            )
        except (OSError, ValueError) as exc:
            print(f"[错误] 保存报告失败：{exc}", file=sys.stderr)
            return 1
        print(f"[报告] {output}")
    if summary["failed_cases"]:
        return 1
    return 0 if summary["automatic_checks_passed"] == len(cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())
