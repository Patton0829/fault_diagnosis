"""第十四步：将Milvus Dense接回RAG；可对照内存检索，共享等价的下游输入。"""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import httpx
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
)

from qa_platform.providers.chat import DEFAULT_MODEL, MODEL_CONFIGS
from qa_platform.providers.cloud_embedding import embed_texts
from qa_platform.rag.answering import (
    AnswerValidationError,
    TruncatedAnswerError,
    build_sources,
    generate_answer,
    print_answer,
    print_answer_diagnostics,
)
from qa_platform.retrieval.bm25 import BM25Index, tokenize
from qa_platform.retrieval.dense import DEFAULT_QUERY, rank_documents, validate_vectors
from qa_platform.retrieval.hybrid import print_recall_results, reciprocal_rank_fusion
from qa_platform.retrieval.pipeline import (
    EMBEDDING_API_URL,
    EMBEDDING_MODEL,
    RERANK_API_URL,
    RERANK_MODEL,
    RetrievalError,
    print_fusion_pool,
    rerank_candidates,
)
from qa_platform.settings import ENV_PATH
from qa_platform.storage.milvus_build import DEFAULT_INDEX_DIR
from qa_platform.storage.milvus_search import (
    load_snapshot,
    print_comparison,
    read_vectors,
    search_milvus,
    validate_search,
)
from qa_platform.storage.milvus_smoke import require


class ComparisonDifference(RuntimeError):
    """对照池不同不是数据库故障；停止新增模型请求，先观察原因。"""


def fuse(
    documents: list[Document],
    bm25_hits: list,
    dense_hits: list,
    args: argparse.Namespace,
) -> list:
    """两路使用同一套融合函数和参数；不混加BM25与余弦原始分数。"""
    return reciprocal_rank_fusion(
        documents,
        {
            "BM25": [doc for doc, _, _ in bm25_hits],
            "Dense": [doc for doc, _ in dense_hits],
        },
        args.rrf_constant,
        args.fusion_k,
    )


def same_ordered_pool(left: list, right: list) -> bool:
    """重排输入是有序正文列表，不能只比较ID集合或后面临时分配的S1。"""

    def identity(pool):
        return [
            (hit.document.id, hit.document.page_content, hit.document.metadata)
            for hit in pool
        ]

    return identity(left) == identity(right)


def retrieve(
    client: httpx.Client,
    database,
    documents: list[Document],
    vectors: np.ndarray,
    args: argparse.Namespace,
    timings: dict,
) -> list:
    """只替换Dense来源，复用BM25、RRF和Reranker；不调用旧的文档重编码流程。"""
    stage = "BM25"
    try:
        started = perf_counter()
        tokens = tokenize(args.query)
        bm25_hits = BM25Index(documents).search(tokens, args.recall_k) if tokens else []
        timings["BM25构建与搜索（两路共用）"] = perf_counter() - started
        print(f"[分词] {' / '.join(tokens) if tokens else '无有效词，BM25为空路'}")
        stage = "问题Embedding"
        print("[请求] 编码一次问题：text_type=query；文档Embedding为0次。", flush=True)
        started = perf_counter()
        with np.errstate(over="ignore", invalid="ignore"):
            query_matrix = embed_texts(client, [args.query], "query").astype(np.float32)
        validate_vectors(query_matrix, 1, vectors.shape[1])
        query_vector = query_matrix[0]
        timings["问题编码（两路共用）"] = perf_counter() - started

        stage = "Milvus Dense"
        started = perf_counter()
        dense_hits = search_milvus(
            database, documents, query_vector, min(args.recall_k, len(documents))
        )
        timings["Milvus Dense客户端搜索及结果校验"] = perf_counter() - started
        stage = "Milvus RRF"
        started = perf_counter()
        pool = fuse(documents, bm25_hits, dense_hits, args)
        timings["Milvus路线RRF"] = perf_counter() - started
        print_recall_results(bm25_hits, dense_hits)

        if args.compare:
            stage = "NumPy对照"
            started = perf_counter()
            memory_hits = rank_documents(
                documents, vectors, query_vector, args.recall_k
            )
            timings["NumPy Dense内存计算及向量校验"] = perf_counter() - started
            validate_search(
                dense_hits, documents, vectors, query_vector, len(dense_hits)
            )
            started = perf_counter()
            memory_pool = fuse(documents, bm25_hits, memory_hits, args)
            timings["NumPy路线RRF"] = perf_counter() - started
            print("\n[Dense对照：同一问题向量、同一已保存文档向量]")
            print_comparison(
                memory_hits, dense_hits, "NumPy COSINE", "Milvus FLAT/COSINE"
            )
            print("\n[RRF对照：同一BM25路、同一融合参数；以下分数是RRF而非余弦]")
            print_comparison(
                [(hit.document, hit.score) for hit in memory_pool],
                [(hit.document, hit.score) for hit in pool],
                "NumPy路线RRF",
                "Milvus路线RRF",
            )
            if not same_ordered_pool(memory_pool, pool):
                raise ComparisonDifference(
                    "两路有序RRF候选不同，停止重排和生成；可能涉及同分边界或排序差异，不直接判定Milvus错误。"
                )
            print(
                "[共享条件] 有序候选ID、正文和元数据一致；Reranker输入等价，采用Milvus池只重排一次、回答一次。"
            )
            print(
                "[对照边界] 这是检索对照与共享下游，不是两个独立端到端回答，更不是答案准确率评测。"
            )

        print_fusion_pool(pool)
        stage = "Reranker"
        started = perf_counter()
        hits = rerank_candidates(client, args.query, pool, args.top_k)
        timings["Reranker请求及校验"] = perf_counter() - started
        return hits
    except httpx.TimeoutException:
        raise RetrievalError(f"{stage}请求超时。") from None
    except httpx.HTTPStatusError as exc:
        raise RetrievalError(
            f"{stage}返回HTTP {exc.response.status_code}；不显示响应体。"
        ) from None
    except httpx.RequestError:
        raise RetrievalError(f"{stage}连接失败；不显示请求信息。") from None
    except (ValueError, TypeError, OverflowError):
        raise RetrievalError(f"{stage}解析或校验失败，未进入后续生成。") from None


def run(
    args: argparse.Namespace,
    manifest: dict,
    written: dict,
    documents: list[Document],
    counts: dict,
    timings: dict,
) -> None:
    """数据库先核验，检索和聊天分别认证；已有提示词和引用校验保持不变。"""
    started = perf_counter()
    from pymilvus import MilvusClient

    database = MilvusClient(uri=str(args.index_dir / "knowledge.db"), timeout=20)
    try:
        timings["SDK导入与打开数据库"] = perf_counter() - started
        started = perf_counter()
        vectors = read_vectors(database, manifest, written, documents)
        timings["集合与全部记录/向量读回校验"] = perf_counter() - started
        print(
            f"[校验] 已核对{len(documents)}块的原文、元数据与float32向量指纹，尚未调用模型。",
            flush=True,
        )
        require(ENV_PATH.is_file(), "models/.env不存在。")
        load_dotenv(ENV_PATH, override=False, encoding="utf-8")
        retrieval_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        require(bool(retrieval_key), "缺少检索服务密钥；不显示密钥。")
        config = MODEL_CONFIGS[args.model]
        chat_key = (
            os.getenv(config["api_key_env"], "").strip()
            if config["api_key_env"]
            else "not-needed"
        )
        require(bool(chat_key), "缺少聊天服务认证配置；未发送模型请求。")

        # 请求钩子只统计尝试，不记录请求头、密钥、问题、证据或响应正文。
        def count_retrieval(request: httpx.Request) -> None:
            if request.url == httpx.URL(EMBEDDING_API_URL):
                require(
                    json.loads(request.content)["parameters"]["text_type"] == "query",
                    "本入口禁止重新编码文档。",
                )
                counts["问题Embedding"] += 1
            elif request.url == httpx.URL(RERANK_API_URL):
                counts["Reranker"] += 1
            else:
                raise ValueError("出现未约定的检索请求地址。")

        with httpx.Client(
            headers={"Authorization": f"Bearer {retrieval_key}"},
            timeout=60.0,
            event_hooks={"request": [count_retrieval]},
        ) as client:
            hits = retrieve(client, database, documents, vectors, args, timings)
    finally:
        database.close()

    sources = build_sources(hits)
    print(
        f"\n[回答证据] {len(sources)}块；只展示Milvus主路的实际重排结果，候选不等于证据充分。"
    )
    for source_id, hit in sources.items():
        metadata = hit.candidate.document.metadata
        print(
            f"[{source_id}] {metadata['entry_id']} | 重排={hit.score:.4f} | chunk_id={metadata['chunk_id']} | is_test={metadata['is_test']}"
        )

    def count_chat(request: httpx.Request) -> None:
        counts["聊天"] += 1

    started = perf_counter()
    # 不能把检索客户端的千问Authorization交给聊天客户端，尤其是GLM地址。
    with httpx.Client(timeout=60.0, event_hooks={"request": [count_chat]}) as chat_http:
        with OpenAI(
            api_key=chat_key,
            base_url=config["base_url"],
            timeout=60.0,
            max_retries=0,
            http_client=chat_http,
        ) as chat_client:
            if sources:
                print(
                    "[请求] 生成JSON回答；完整收集后复用原有结构与引用校验，不展示未经校验的正文。",
                    flush=True,
                )
            answer = generate_answer(
                chat_client, args.query, sources, args.model, args.max_tokens
            )
    timings["回答生成与结构/引用校验"] = perf_counter() - started
    print_answer(answer, sources)


def main() -> int:
    parser = argparse.ArgumentParser(description="Milvus接入单轮RAG；默认只展示计划")
    parser.add_argument(
        "--run", action="store_true", help="明确发起模型请求；不重建或更新索引"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="增加同向量NumPy/RRF对照；输入一致时共享重排与回答",
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), default=DEFAULT_MODEL)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--fusion-k", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if (
        not args.query.strip()
        or min(
            args.recall_k, args.fusion_k, args.top_k, args.rrf_constant, args.max_tokens
        )
        <= 0
    ):
        parser.error("问题不能为空；候选数量、RRF常数与max-tokens须大于0")
    args.index_dir = args.index_dir.expanduser().absolute()
    counts = {"文档Embedding": 0, "问题Embedding": 0, "Reranker": 0, "聊天": 0}
    timings = {}
    attempted = False
    started = perf_counter()
    try:
        manifest, written, documents = load_snapshot(args.index_dir)
        config = MODEL_CONFIGS[args.model]
        old_document_calls = (len(documents) + manifest["batch_size"] - 1) // manifest[
            "batch_size"
        ]
        print(f"[配置] Python：{sys.executable}")
        print(
            f"[索引] {args.index_dir}；仅测试库；{len(documents)}块；{written['dimension']}维；FLAT/COSINE"
        )
        print(
            f"[模型] Embedding={EMBEDDING_MODEL}；Reranker={RERANK_MODEL}；回答={args.model}；max_tokens={args.max_tokens}"
        )
        print(f"[聊天地址] {config['base_url']}")
        if not config["api_key_env"]:
            print(
                "[认证] GLM仅使用占位认证，不接收千问检索密钥；问题和证据将发送到上述地址。"
            )
        print(
            f"[参数] 每路召回={args.recall_k}；融合池={args.fusion_k}；回答证据最多={args.top_k}；RRF常数={args.rrf_constant}"
        )
        print(f"[问题] {args.query}")
        print("[链路] BM25 + Milvus Dense → RRF → Reranker → 原有结构化回答/引用校验")
        print(
            f"[请求计划对照] 相同分块、批量配置下，旧流程不缓存：文档{old_document_calls} + 问题1 + 重排1 + 回答1 = {old_document_calls + 3}次"
        )
        print(
            "[本入口计划] 文档0 + 问题1 + 重排1 + 回答1 = 正常完成3次模型请求；不自动重试。"
        )
        print(
            "[历史成本] 文档已在建库时编码；知识或模型变更后仍需更新索引。请求数减少不等于费用、总耗时同比减少。"
        )
        print(
            f"[内存对照] {'开启；同一次问题编码、同一份已存文档向量；共享BM25。' if args.compare else '未开启；加--compare可同时展示NumPy Dense和RRF。'}"
        )
        if args.compare:
            print(
                "[分歧保护] 有序RRF池不同则只完成问题编码便停止；一致才共享一次重排和回答。"
            )
        print(
            "[边界] 全量读回是小库教学校验；不是大规模检索优化。索引模型服务版本仍未固定。"
        )
        print(
            "[边界] 测试条目是虚构资料；保留原有拒答、来源标记及人工核验要求；不设置通用可信分数阈值。"
        )
        print(
            "[保留] 原check_rag.py与check_eval.py仍走原内存流程，本入口不会自动改变它们。"
        )
        if not args.run:
            print(
                "[计划完成] 未读取.env、未导入Milvus SDK、未创建客户端、未发请求或写文件；加--run才执行。"
            )
            return 0
        attempted = True
        run(args, manifest, written, documents, counts, timings)
    except ComparisonDifference as exc:
        print(f"[对照分歧] {exc}", file=sys.stderr)
        return 3
    except RetrievalError as exc:
        print(f"[检索失败] {exc}", file=sys.stderr)
        return 1
    except TruncatedAnswerError as exc:
        print(f"[截断] {exc} 未展示不完整回答。", file=sys.stderr)
        print_answer_diagnostics(exc.diagnostics, file=sys.stderr)
        return 2
    except AnswerValidationError as exc:
        print(f"[回答校验失败] {exc}", file=sys.stderr)
        print_answer_diagnostics(exc.diagnostics, file=sys.stderr)
        return 1
    except APITimeoutError:
        print("[回答失败] 聊天请求超时。", file=sys.stderr)
        return 1
    except APIConnectionError:
        print("[回答失败] 聊天连接失败。", file=sys.stderr)
        return 1
    except APIStatusError as exc:
        print(
            f"[回答失败] 聊天服务返回HTTP {exc.status_code}；不显示响应体。",
            file=sys.stderr,
        )
        return 1
    except OpenAIError as exc:
        print(
            f"[回答失败] SDK错误类型：{type(exc).__name__}；不显示请求或响应。",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"[校验失败] {exc}", file=sys.stderr)
        return 1
    except (OSError, TypeError, KeyError, OverflowError):
        print(
            "[失败] 本地文件、数据格式或字段校验失败；不展示未经校验的回答。",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"[失败] {type(exc).__name__}：操作未完成；不显示请求或服务响应。",
            file=sys.stderr,
        )
        return 1
    finally:
        if attempted:
            print(
                "\n[本轮HTTP请求尝试] "
                + "；".join(f"{name}={count}" for name, count in counts.items())
                + f"；合计={sum(counts.values())}"
            )
            for name, seconds in timings.items():
                print(f"[分段耗时] {name}：{seconds:.4f} s")
            print(
                f"[本轮总耗时] {perf_counter() - started:.3f} s；含准备、校验、终端输出和可选对照，非纯数据库基准。"
            )
    print("\n[成功] Milvus主路RAG完成；未更新索引或写报告，结果仍需人工核验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
