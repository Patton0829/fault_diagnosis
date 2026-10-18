"""同一批纯节点：显式函数调用 vs 真正的 LangGraph；不创建模型或数据库客户端。"""

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
import json
from math import isfinite
from typing import TypedDict

from langchain_core.documents import Document
from langsmith import tracing_context

from qa_platform.rag.answering import parse_answer
from qa_platform.rag.quote_units import PARAGRAPH_STRATEGY, paragraph_quotes
from qa_platform.retrieval.hybrid import RRFHit, reciprocal_rank_fusion
from qa_platform.retrieval.pipeline import RerankedHit


SCHEMA = "rag-workflow-replay/v1"
VALIDATED_VERSION = "1.2.16"
ZERO_REQUESTS = {"embedding": 0, "http": 0, "reranker": 0, "chat": 0}
NOTICE = (
    "这是教学合成快照的零模型回放：召回、重排分数、草稿来自快照，RRF、引用切分和校验在本地重算。"
    "两路使用相同节点与输入；界面逐步查看的是已经执行完的轨迹，不是实时模型执行。"
    "模型服务HTTP为0，本机页面API通信另计。不读.env、不连接Milvus、不外发LangSmith追踪，"
    "没有自动重试、持久化检查点、多轮记忆或真实RAG接管。"
    "结果一致只验证本样例编排等价，不证明检索质量、语义支持或性能更好；普通Python也能自行实现日志与分支。"
)
NODES = [
    {"id": "retrieve", "label": "召回回放", "reads": ["query"], "writes": ["bm25", "dense"], "operation": "读取合成快照"},
    {"id": "fuse", "label": "RRF 融合", "reads": ["bm25", "dense"], "writes": ["rrf_pool"], "operation": "本地计算"},
    {"id": "rerank", "label": "重排回放", "reads": ["rrf_pool"], "writes": ["reranked"], "operation": "读取合成快照"},
    {"id": "prepare_evidence", "label": "准备段落引用", "reads": ["reranked"], "writes": ["evidence"], "operation": "本地计算"},
    {"id": "generate", "label": "草稿回放", "reads": ["evidence"], "writes": ["draft"], "operation": "读取合成快照"},
    {"id": "validate", "label": "回答与引用校验", "reads": ["draft", "evidence"], "writes": ["answer"], "operation": "本地校验"},
]


class ReplayState(TypedDict, total=False):
    """共享状态各字段默认覆盖更新；本步不使用累加 reducer。"""

    query: str
    bm25: list[dict]
    dense: list[dict]
    rrf_pool: list[dict]
    reranked: list[dict]
    evidence: dict
    draft: dict
    answer: dict


def load_fixture() -> dict:
    """只读随源码打包的合成JSON；不接受文件路径或浏览器上传。"""
    return json.loads(files("qa_platform.workflows").joinpath("replay_fixture.json").read_text(encoding="utf-8"))


def _fixture_info(fixture: dict) -> dict:
    return {key: deepcopy(fixture[key]) for key in ("id", "title", "origin", "query", "notice")}


def describe() -> dict:
    try:
        installed = version("langgraph")
    except PackageNotFoundError:
        installed = None
    return {
        "schema_version": SCHEMA, "available": installed is not None,
        "langgraph_version": installed, "fixture": _fixture_info(load_fixture()),
        "nodes": deepcopy(NODES), "request_counts": dict(ZERO_REQUESTS), "notice": NOTICE,
        "dependency_message": (
            f"本步验证版本为LangGraph {VALIDATED_VERSION}；当前缺少依赖，请先安装项目workflow可选依赖，不会伪装成图执行。"
            if installed is None else f"本步验证版本为LangGraph {VALIDATED_VERSION}；读取计划本身不会安装或升级包。"
        ),
    }


def _documents(fixture: dict) -> list[Document]:
    if fixture.get("origin") != "synthetic" or not fixture.get("query"):
        raise ValueError("仅接受教学合成快照")
    documents = [Document(page_content=row["text"], metadata={
        "chunk_id": row["chunk_id"], "title": row["title"], "is_test": True,
        "review_status": "test_data", "document_mode": "body-only", "quote_strategy": PARAGRAPH_STRATEGY,
    }) for row in fixture["documents"]]
    ids = [doc.metadata["chunk_id"] for doc in documents]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("合成文档身份重复或为空")
    for route in ("bm25", "dense"):
        hits = fixture[route]
        hit_ids = [hit["chunk_id"] for hit in hits]
        if len(hit_ids) != len(set(hit_ids)) or not set(hit_ids) <= set(ids):
            raise ValueError("合成召回身份非法")
        if any(type(hit["score"]) not in (int, float) or not isfinite(hit["score"]) for hit in hits):
            raise ValueError("合成分数非法")
    return documents


def make_nodes(fixture: dict) -> dict:
    """节点只返回增量，不原地改state；两条路线复用这些相同函数。"""
    frozen = deepcopy(fixture)
    documents = _documents(frozen)
    by_id = {doc.metadata["chunk_id"]: doc for doc in documents}

    def retrieve(state: ReplayState) -> dict:
        if state["query"] != frozen["query"]:
            raise ValueError("快照问题不一致，不能把旧结果当成新问题结果")
        return {"bm25": deepcopy(frozen["bm25"]), "dense": deepcopy(frozen["dense"])}

    def fuse(state: ReplayState) -> dict:
        hits = reciprocal_rank_fusion(documents, {
            key: [by_id[row["chunk_id"]] for row in state[key]] for key in ("bm25", "dense")
        }, rank_constant=60, top_k=len(documents))
        return {"rrf_pool": [{"chunk_id": hit.document.metadata["chunk_id"], "score": hit.score,
                              "ranks": hit.ranks, "contributions": hit.contributions} for hit in hits]}

    def rerank(state: ReplayState) -> dict:
        rows = []
        for index, hit in enumerate(state["rrf_pool"]):
            score = frozen["reranker_scores"][hit["chunk_id"]]
            if type(score) not in (int, float) or not isfinite(score):
                raise ValueError("合成重排分数非法")
            rows.append({"chunk_id": hit["chunk_id"], "score": score, "rrf_rank": index + 1})
        rows.sort(key=lambda row: (-row["score"], row["rrf_rank"]))
        return {"reranked": rows}

    def prepare_evidence(state: ReplayState) -> dict:
        return {"evidence": {f"S{index}": {
            "chunk_id": row["chunk_id"], "text": by_id[row["chunk_id"]].page_content,
            "quotes": paragraph_quotes(by_id[row["chunk_id"]].page_content),
        } for index, row in enumerate(state["reranked"][:2], 1)}}

    def generate(state: ReplayState) -> dict:
        if not state["evidence"]:
            raise ValueError("没有准备好的证据")
        return {"draft": deepcopy(frozen["answer"])}

    def validate(state: ReplayState) -> dict:
        sources = {}
        for source_id, evidence in state["evidence"].items():
            document = by_id[evidence["chunk_id"]]
            if evidence["text"] != document.page_content or evidence["quotes"] != paragraph_quotes(document.page_content):
                raise ValueError("引用准备结果与合成原文不一致")
            sources[source_id] = RerankedHit(RRFHit(document, 0.0, {}, {}), 0, 0.0)
        # 真正执行既有JSON/状态/引用定位校验；不是把草稿直接当成成功回答。
        answer = parse_answer(json.dumps(state["draft"], ensure_ascii=False), sources)
        return {"answer": answer.model_dump()}

    return {"retrieve": retrieve, "fuse": fuse, "rerank": rerank,
            "prepare_evidence": prepare_evidence, "generate": generate, "validate": validate}


def _event(step: int, node: str, before: dict, update: dict, after: dict) -> dict:
    return deepcopy({"step": step, "node": node, "before": before, "update": update, "after": after})


def _run_plain(fixture: dict, nodes: dict) -> dict:
    state: ReplayState = {"query": fixture["query"]}
    events = []
    # 顺序由Python调用方指定，状态由调用方合并；与图使用相同的节点函数。
    for node in ("retrieve", "fuse", "rerank", "prepare_evidence", "generate", "validate"):
        before = deepcopy(state)
        update = nodes[node](deepcopy(state))
        state.update(deepcopy(update))
        events.append(_event(len(events) + 1, node, before, update, state))
    return {"engine": "python", "events": events, "final_state": deepcopy(state)}


def run_plain(fixture: dict) -> dict:
    with tracing_context(enabled=False):
        return _run_plain(fixture, make_nodes(fixture))


def build_graph(fixture: dict, *, nodes: dict | None = None):
    """真正StateGraph；没有checkpointer、缓存、重试策略或云端绘图。"""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ReplayState)
    for node_id, function in (nodes or make_nodes(fixture)).items():
        builder.add_node(node_id, function, retry_policy=None)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "fuse")
    builder.add_edge("fuse", "rerank")
    builder.add_edge("rerank", "prepare_evidence")
    builder.add_edge("prepare_evidence", "generate")
    builder.add_edge("generate", "validate")
    builder.add_edge("validate", END)
    return builder.compile(checkpointer=None, cache=None)


def _run_langgraph(fixture: dict, graph) -> dict:
    events, previous, pending = [], None, None
    # values由LangGraph实际合并后产生；不在应用里伪造图的after状态。
    with tracing_context(enabled=False):
        for kind, data in graph.stream(
            {"query": fixture["query"]}, stream_mode=["updates", "values"],
            config={"callbacks": [], "recursion_limit": 20},
        ):
            if kind == "updates":
                if pending is not None or len(data) != 1:
                    raise ValueError("本步只支持单节点顺序轨迹")
                pending = next(iter(data.items()))
            elif kind == "values":
                if pending is not None:
                    node, update = pending
                    events.append(_event(len(events) + 1, node, previous, update, data))
                    pending = None
                previous = deepcopy(data)
    if pending is not None or len(events) != len(NODES):
        raise ValueError("图没有产生完整轨迹")
    return {"engine": "langgraph", "events": events, "final_state": deepcopy(previous)}


def run_langgraph(fixture: dict) -> dict:
    with tracing_context(enabled=False):
        return _run_langgraph(fixture, build_graph(fixture))


def compare_replay() -> dict:
    fixture = load_fixture()
    # 禁用自动LangSmith tracing，即使用户环境中已开启，也不外发合成或未来接入的数据。
    with tracing_context(enabled=False):
        nodes = make_nodes(fixture)
        plain = _run_plain(fixture, nodes)
        graph = build_graph(fixture, nodes=nodes)
        graphed = _run_langgraph(fixture, graph)
        structure = graph.get_graph()
    return {
        "schema_version": SCHEMA, "fixture": _fixture_info(fixture), "nodes": deepcopy(NODES),
        "langgraph_version": version("langgraph"),
        "graph": {"nodes": [{"id": key, "label": value.name} for key, value in structure.nodes.items()],
                  "edges": [{"source": edge.source, "target": edge.target} for edge in structure.edges]},
        "plain": plain, "langgraph": graphed,
        "comparison": {
            "same_steps": [event["node"] for event in plain["events"]] == [event["node"] for event in graphed["events"]],
            "same_updates": [event["update"] for event in plain["events"]] == [event["update"] for event in graphed["events"]],
            "same_final_state": plain["final_state"] == graphed["final_state"],
            "same_answer": plain["final_state"]["answer"] == graphed["final_state"]["answer"],
        },
        "request_counts": dict(ZERO_REQUESTS), "notice": NOTICE,
    }
