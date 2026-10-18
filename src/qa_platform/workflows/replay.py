"""同一批纯节点：显式函数调用 vs 真正的 LangGraph；不创建模型或数据库客户端。"""

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
import inspect
import json
from math import isfinite
from textwrap import dedent
from typing import TypedDict, get_origin

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
        "definitions": describe_definitions(),
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


def _execute_shared(fixture: dict):
    """shared_nodes是普通字典：名字映射到函数对象，两路使用同一批函数。"""
    shared_nodes = make_nodes(fixture)
    plain = _run_plain(fixture, shared_nodes)
    graph = build_graph(fixture, nodes=shared_nodes)
    graphed = _run_langgraph(fixture, graph)
    return plain, graph, graphed, shared_nodes


def _source_record(obj) -> dict:
    """只展示本模块已知定义；没有浏览器可指定的文件路径或代码执行入口。"""
    obj = inspect.unwrap(obj)
    if getattr(obj, "__module__", None) != __name__ or inspect.getsourcefile(obj) != inspect.getsourcefile(ReplayState):
        raise ValueError("只能展示本回放模块的源码定义")
    lines, start_line = inspect.getsourcelines(obj)
    return {
        "name": obj.__name__, "code": dedent("".join(lines)).rstrip("\n"),
        "start_line": start_line, "end_line": start_line + len(lines) - 1,
    }


def describe_definitions(fixture: dict | None = None, nodes: dict | None = None) -> dict:
    """提取实际类型和源码文本，不执行节点、建图、检索或生成回答。"""
    shared_nodes = nodes if nodes is not None else make_nodes(fixture if fixture is not None else load_fixture())
    if list(shared_nodes) != [row["id"] for row in NODES]:
        raise ValueError("节点注册表与教学顺序不一致")
    node_sources = {node_id: _source_record(function) for node_id, function in shared_nodes.items()}
    if any(source["name"] != node_id for node_id, source in node_sources.items()):
        raise ValueError("本教学注册名称与对应函数不一致")
    state = _source_record(ReplayState)
    state["fields"] = [{
        "name": name,
        "type": str(annotation) if get_origin(annotation) is not None else getattr(annotation, "__name__", str(annotation)),
        "required": name in ReplayState.__required_keys__,
        "read_by": [row["id"] for row in NODES if name in row["reads"]],
        "written_by": [row["id"] for row in NODES if name in row["writes"]],
    } for name, annotation in ReplayState.__annotations__.items()]
    return {
        "schema_version": "workflow-definitions/v1",
        "source_file": "src/qa_platform/workflows/replay.py",
        "state": state, "factory": _source_record(make_nodes),
        "setup": _source_record(_execute_shared),
        "nodes": node_sources,
        "graph_builder": _source_record(build_graph),
        "notice": (
            "这里展示本模块真实定义的只读源码，不是可执行编辑器，不接受任意文件路径。"
            "ReplayState是TypedDict类型约定，不自动创建字段，也不做运行时数据校验；"
            "total=False允许各字段分阶段出现，不代表节点可以缺少自己需要的字段。"
            "shared_nodes只是名字到函数对象的Python字典，由make_nodes创建，不是LangGraph内置变量。"
            "当前节点读写说明来自教学声明，实际值仍以所选步骤轨迹为准。"
        ),
    }


def compare_replay() -> dict:
    fixture = load_fixture()
    # 禁用自动LangSmith tracing，即使用户环境中已开启，也不外发合成或未来接入的数据。
    with tracing_context(enabled=False):
        plain, graph, graphed, shared_nodes = _execute_shared(fixture)
        structure = graph.get_graph()
        definitions = describe_definitions(fixture, shared_nodes)
    return {
        "schema_version": SCHEMA, "fixture": _fixture_info(fixture), "nodes": deepcopy(NODES),
        "langgraph_version": version("langgraph"),
        "definitions": definitions,
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
