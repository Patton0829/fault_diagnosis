"""真实单轮RAG：复用已有函数，用三个LangGraph节点自动完成问答。"""

from pathlib import Path
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

from langsmith import tracing_context

from qa_platform.rag import kubernetes_answer as answering
from qa_platform.rag.answering import GroundedAnswer
from qa_platform.retrieval import kubernetes_hybrid as hybrid
from qa_platform.retrieval import kubernetes_rerank as reranking
from qa_platform.workflows.progress import RunCancelled, report


PIPELINE = ("retrieve", "rerank", "answer")
NOTICE = (
    "一次提问自动完成混合检索、全候选池重排、回答与引用校验。"
    "使用已有Milvus文档向量，只编码问题；正常完成需Embedding、Reranker、聊天各1次，"
    "另有2次本地模型身份GET，共5次模型服务HTTP。"
    "问题发送到自部署Embedding；问题与候选正文发送到云端重排，"
    "问题与前5块证据发送到所选回答服务；不发送本机路径或向量到云端。"
    "资料未审核、适用版本需核实；引用校验不是事实正确性证明，不执行文档命令。"
    "不自动重试、不补相邻上下文、不外发LangSmith追踪。默认单次请求；可选短期会话理解追问，历史不是证据。"
    "每次主动提问都是新请求，可能收费。"
)


class LiveRAGState(TypedDict, total=False):
    """初始只有query/model；节点写入result，再逐步用新结果覆盖它。"""

    query: str
    model: str
    result: dict


def build_graph(index_dir: Path):
    """注册已有业务函数，不在建图时连接数据库或调用模型。"""
    from langgraph.graph import END, START, StateGraph

    def retrieve(state: LiveRAGState) -> dict:
        report("retrieve", "started")
        check = hybrid.prepare(index_dir, state["query"])
        result = hybrid.compare(check, "hybrid")
        # 旧回答接口要求本轮身份；该值只在服务端传递，不由模型或浏览器提供。
        result["experiment_id"] = uuid4().hex
        report("retrieve", "completed")
        return {"result": result}

    def rerank(state: LiveRAGState) -> dict:
        report("rerank", "started")
        result = reranking.rerank(state["result"])
        report("rerank", "completed")
        return {"result": result}

    def answer(state: LiveRAGState) -> dict:
        report("answer", "started")
        # generate内部已完成段落引用准备、一次生成、JSON/引用校验。
        result = answering.generate(state["result"], state["model"])
        report("answer", "completed")
        return {"result": result}

    builder = StateGraph(LiveRAGState)
    for name, function in (("retrieve", retrieve), ("rerank", rerank), ("answer", answer)):
        builder.add_node(name, function, retry_policy=None)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=None, cache=None)


class RAGRunError(RuntimeError):
    """只携带固定阶段名；不向API泄露原始响应、密钥或未经校验的正文。"""

    def __init__(self, stage: str, completed_steps: list[str]):
        self.stage = stage
        self.completed_steps = list(completed_steps)
        labels = {"retrieve": "混合检索", "rerank": "云端重排", "answer": "回答生成或引用校验"}
        super().__init__(
            f"{labels[stage]}未完成；已完成{len(completed_steps)}个阶段。"
            "未展示未经校验的回答；失败前请求可能已发生并产生费用，不自动重试。"
            "再次点击提问会从检索开始新的一轮。"
        )


def describe() -> dict:
    """打开页面只读固定配置，不加载密钥、索引或模型客户端。"""
    from qa_platform.workflows.agent_rag import describe as describe_agent
    from qa_platform.workflows.conversation import describe as describe_conversation
    from qa_platform.web_stream import describe as describe_stream

    config = answering.describe()
    return {
        "schema_version": "rag-live-plan/v1",
        "default_model": config["default_model"], "models": config["models"],
        "reranker": {key: reranking.describe()[key] for key in ("model", "url")},
        "pipeline": list(PIPELINE), "notice": NOTICE, "agent": describe_agent(),
        "conversation": describe_conversation(),
        "stream": describe_stream(),
    }


def _public_result(result: dict, model: str) -> dict:
    """只展示校验后的回答及实际引用来源；不返回全池、向量或本机路径。"""
    generation = result["answer_generation"]
    if generation["model"] != model:
        raise ValueError("回答模型与本轮选择不一致")
    answer = GroundedAnswer.model_validate(generation["answer"]).model_dump()
    if answer["status"] == "insufficient_evidence":
        if answer["claims"] or not answer["missing_info"]:
            raise ValueError("拒答状态不符合约定")
    elif not answer["claims"] or (answer["status"] == "partial" and not answer["missing_info"]):
        raise ValueError("回答状态不符合约定")
    sources = {}
    for claim in answer["claims"]:
        for ref in claim["evidence"]:
            source_id, quote_id = ref["source_id"], ref["quote_id"]
            row = generation["sources"][source_id]
            if not row["quotes"][quote_id] or row["quotes"][quote_id] not in row["text"]:
                raise ValueError("引用不能定位到原文")
            if source_id not in sources:
                sources[source_id] = {
                    "chunk_id": row["chunk_id"], "text": row["text"], "quotes": row["quotes"],
                    "metadata": {key: row["metadata"][key] for key in (
                        "title", "section_path", "source_url", "review_status", "kubernetes_version",
                    ) if key in row["metadata"]},
                }
    return {"query": result["query"], "model": model, "answer": answer,
            "sources": sources, "request_counts": result["request_counts"]}


def run(query: str, model: str, index_dir: Path) -> dict:
    """一次stream驱动三个真实节点；任一步失败即停止，不循环或隐式重试。"""
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("问题须为1～2000字符")
    if not isinstance(model, str) or model not in answering.MODEL_CONFIGS:
        raise ValueError("回答模型不在允许列表")
    started, completed, result = perf_counter(), [], None
    try:
        # 即使环境开启追踪，也不把问题、文档和模型输出交给第三方追踪服务。
        with tracing_context(enabled=False):
            graph = build_graph(index_dir)
            for update in graph.stream(
                {"query": query.strip(), "model": model},
                config={"callbacks": [], "recursion_limit": 10}, stream_mode="updates",
            ):
                name = PIPELINE[len(completed)]
                if set(update) != {name} or set(update[name]) != {"result"}:
                    raise ValueError("图返回了非预期节点更新")
                result = update[name]["result"]
                completed.append(name)
            if completed != list(PIPELINE) or result["query"] != query.strip():
                raise ValueError("本轮执行不完整或问题不匹配")
            public = _public_result(result, model)
    except RunCancelled:
        raise
    except Exception:
        stage = PIPELINE[min(len(completed), len(PIPELINE) - 1)]
        raise RAGRunError(stage, completed) from None
    return {"schema_version": "rag-live/v1", **public, "steps": completed,
            "elapsed_seconds": round(perf_counter() - started, 3)}
