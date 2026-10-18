"""条件边小实验：同一批节点，Python if/else 与 LangGraph 条件路由。"""

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import inspect
from textwrap import dedent
from typing import Literal

from langsmith import tracing_context

from qa_platform.workflows import replay
from qa_platform.workflows.replay import ReplayState


SCHEMA = "rag-workflow-branching/v2"
SCENARIOS = [
    {"id": "with_evidence", "label": "有候选资料", "description": "保留教学召回，证据非空，进入草稿回放；不代表资料充分或可靠。"},
    {"id": "empty_evidence", "label": "召回为空", "description": "同一问题和资料，仅将两路召回置空，走固定拒答；不会自动补检索或调用模型。"},
]
PREFIX = ("retrieve", "fuse", "rerank", "prepare_evidence")
NODES = deepcopy(replay.NODES)
NODES.insert(-1, {
    "id": "refuse", "label": "固定拒答", "reads": ["evidence"], "writes": ["draft"],
    "operation": "本地固定模板，不调用生成模型",
})
NOTICE = (
    "仅教学合成数据：非空证据只决定能否进入草稿步骤，不是证据质量、相关性或可信度判定。"
    "空召回场景只改两路召回列表，问题与资料不变。页面只运行和展示LangGraph，未选中的分支不执行。"
    "模型服务HTTP为0；不读.env、不连接Milvus、不外发LangSmith追踪；本机页面API通信另计。"
    "没有自动重试、补检索、持久化、多轮记忆或真实RAG接管；错误会停止，不伪装成正常拒答。"
)


def load_fixture(scenario: str = "with_evidence") -> dict:
    """只允许两个固定场景；空证据通过清空召回构造，不上传或改写资料。"""
    if type(scenario) is not str or scenario not in {row["id"] for row in SCENARIOS}:
        raise ValueError("未知教学场景")
    fixture = replay.load_fixture()
    if scenario == "empty_evidence":
        fixture["bm25"], fixture["dense"] = [], []
    return fixture


def route_after_evidence(state: ReplayState) -> Literal["generate", "refuse"]:
    """路由不是节点：读取状态，返回下一节点名，不返回状态更新。"""
    if not isinstance(state["evidence"], dict):
        raise ValueError("证据字段必须为字典；格式错误不能冒充正常空召回")
    return "generate" if state["evidence"] else "refuse"


def refuse(state: ReplayState) -> dict:
    """证据为空才允许固定拒答；它不是生成模型，也不是异常兜底。"""
    if not isinstance(state["evidence"], dict) or state["evidence"]:
        raise ValueError("只有合法空证据字典才能进入本拒答分支")
    return {"draft": {
        "status": "insufficient_evidence", "claims": [],
        "missing_info": ["本教学场景两路召回均为空，未得到可引用资料；需要补充可检索资料，不能据此给出排查结论。"],
    }}


def make_nodes(fixture: dict) -> dict:
    """复用上一课六个函数，另加固定拒答函数，返回七个名称到函数的绑定。"""
    nodes = replay.make_nodes(fixture)
    nodes["refuse"] = refuse
    return {row["id"]: nodes[row["id"]] for row in NODES}


def _observe_router(decisions: list):
    """仅记录路由真实调用的结果；既不写state，也不额外再调用一次路由。"""
    def observed(state: ReplayState) -> Literal["generate", "refuse"]:
        target = route_after_evidence(state)
        decisions.append({
            "after_node": "prepare_evidence", "target": target,
            "evidence_count": len(state["evidence"]),
            "reason": "证据字典非空，进入草稿回放；尚未判定质量。" if target == "generate" else "证据字典为空，跳过草稿回放，使用固定拒答。",
        })
        return target
    return observed


def _run_plain(fixture: dict, nodes: dict) -> dict:
    state: ReplayState = {"query": fixture["query"]}
    events, decisions = [], []

    def call(node_id: str):
        before = deepcopy(state)
        update = nodes[node_id](deepcopy(state))
        state.update(deepcopy(update))
        events.append(replay._event(len(events) + 1, node_id, before, update, state))

    for node_id in PREFIX:
        call(node_id)
    # 普通Python由调用者直接选择分支：不是同时调用两个节点。
    target = _observe_router(decisions)(deepcopy(state))
    if target == "generate":
        call("generate")
    else:
        call("refuse")
    call("validate")
    return {"engine": "python", "events": events, "decisions": decisions, "final_state": deepcopy(state)}


def run_plain(fixture: dict) -> dict:
    with tracing_context(enabled=False):
        return _run_plain(fixture, make_nodes(fixture))


def build_graph(fixture: dict, *, nodes: dict | None = None, router=None):
    """条件边替代prepare_evidence的固定后继；两个分支只执行其一。"""
    from langgraph.graph import END, START, StateGraph

    # 1. 创建图的构建器：先指定共享状态格式；这时没有运行任何节点。
    builder = StateGraph(ReplayState)
    # 2. 注册函数对象：节点名称绑定到函数，不是在这里调用函数。
    for node_id, function in (nodes if nodes is not None else make_nodes(fixture)).items():
        builder.add_node(node_id, function, retry_policy=None)
    # 3. 连固定边：START进入召回，再按PREFIX连接前四个节点。
    builder.add_edge(START, "retrieve")
    for source, target in zip(PREFIX, PREFIX[1:]):
        builder.add_edge(source, target)
    # 4. 连条件边：证据准备结束后，路由只选择generate或refuse之一。
    builder.add_conditional_edges(
        "prepare_evidence", router if router is not None else route_after_evidence,
        {"generate": "generate", "refuse": "refuse"},
    )
    # 5. 连接汇合与终点：两个备选分支都通往validate，再到END。
    builder.add_edge("generate", "validate")
    builder.add_edge("refuse", "validate")
    builder.add_edge("validate", END)
    # 6. 编译成可运行的图；compile本身仍不执行节点，之后stream才执行。
    return builder.compile(checkpointer=None, cache=None)


def _run_langgraph(fixture: dict, graph, decisions: list) -> dict:
    events, previous, pending = [], None, None
    # after来自LangGraph实际values事件，不用Python自己合并来假扮框架输出。
    for kind, data in graph.stream(
        {"query": fixture["query"]}, stream_mode=["updates", "values"],
        config={"callbacks": [], "recursion_limit": 20},
    ):
        if kind == "updates":
            if pending is not None or len(data) != 1:
                raise ValueError("本实验只允许单节点顺序轨迹")
            pending = next(iter(data.items()))
        elif kind == "values":
            if pending is not None:
                node_id, update = pending
                events.append(replay._event(len(events) + 1, node_id, previous, update, data))
                pending = None
            previous = deepcopy(data)
    if pending is not None or len(decisions) != 1:
        raise ValueError("条件图没有产生唯一完整路由")
    expected = [*PREFIX, decisions[0]["target"], "validate"]
    if [event["node"] for event in events] != expected:
        raise ValueError("实际执行轨迹与路由结果不一致")
    return {"engine": "langgraph", "events": events, "decisions": deepcopy(decisions), "final_state": deepcopy(previous)}


def run_langgraph(fixture: dict) -> dict:
    with tracing_context(enabled=False):
        decisions = []
        graph = build_graph(fixture, router=_observe_router(decisions))
        return _run_langgraph(fixture, graph, decisions)


def _source_record(obj) -> dict:
    obj = inspect.unwrap(obj)
    allowed = {__name__: __file__, replay.__name__: replay.__file__}
    if getattr(obj, "__module__", None) not in allowed or inspect.getsourcefile(obj) != allowed[obj.__module__]:
        raise ValueError("只能读取两个已知教学模块的函数定义")
    lines, start = inspect.getsourcelines(obj)
    module_file = "branching.py" if obj.__module__ == __name__ else "replay.py"
    return {"file": f"src/qa_platform/workflows/{module_file}", "name": obj.__name__,
            "code": dedent("".join(lines)).rstrip("\n"), "start_line": start, "end_line": start + len(lines) - 1}


def describe_definitions(fixture: dict | None = None, nodes: dict | None = None) -> dict:
    shared = nodes if nodes is not None else make_nodes(fixture if fixture is not None else load_fixture())
    if list(shared) != [row["id"] for row in NODES]:
        raise ValueError("节点注册表不完整或顺序不一致")
    sources = {node_id: _source_record(fn) for node_id, fn in shared.items()}
    if any(record["name"] != node_id for node_id, record in sources.items()):
        raise ValueError("函数绑定与节点名称不一致")
    return {
        "state": _source_record(ReplayState), "router": _source_record(route_after_evidence),
        "base_factory": _source_record(replay.make_nodes), "factory": _source_record(make_nodes),
        "graph_builder": _source_record(build_graph), "runner": _source_record(_run_langgraph),
        "entry": _source_record(run_replay), "nodes": sources,
    }


def _common(fixture: dict, nodes: dict | None = None) -> dict:
    try:
        installed = version("langgraph")
    except PackageNotFoundError:
        installed = None
    return {
        "schema_version": SCHEMA, "fixture": replay._fixture_info(fixture),
        "nodes": deepcopy(NODES), "scenarios": deepcopy(SCENARIOS),
        "definitions": describe_definitions(fixture, nodes),
        "langgraph_version": installed, "request_counts": dict(replay.ZERO_REQUESTS), "notice": NOTICE,
    }


def describe() -> dict:
    data = _common(load_fixture())
    return {**data, "available": data["langgraph_version"] is not None,
            "dependency_message": f"沿用本项目验证的LangGraph {replay.VALIDATED_VERSION}；缺少依赖时不会伪装为图执行或自动安装。"}


def run_replay(scenario: str = "with_evidence") -> dict:
    """页面入口：准备固定输入，构图，执行，再返回真实图和状态轨迹。"""
    fixture = load_fixture(scenario)
    with tracing_context(enabled=False):
        shared_nodes = make_nodes(fixture)
        decisions = []
        # _observe_router仅记录路由结果，不改变route_after_evidence的分支规则。
        graph = build_graph(fixture, nodes=shared_nodes, router=_observe_router(decisions))
        graphed = _run_langgraph(fixture, graph, decisions)
        structure = graph.get_graph()
        common = _common(fixture, shared_nodes)
    return {
        **common, "scenario": deepcopy(next(row for row in SCENARIOS if row["id"] == scenario)),
        "graph": {"nodes": [{"id": key, "label": value.name} for key, value in structure.nodes.items()],
                  "edges": [{"source": edge.source, "target": edge.target, "conditional": edge.conditional} for edge in structure.edges]},
        "langgraph": graphed,
    }


def compare_replay(scenario: str = "with_evidence") -> dict:
    """仅供离线回归对照；当前学习页不会调用此入口。"""
    fixture = load_fixture(scenario)
    with tracing_context(enabled=False):
        shared_nodes = make_nodes(fixture)
        plain = _run_plain(fixture, shared_nodes)
        decisions = []
        graph = build_graph(fixture, nodes=shared_nodes, router=_observe_router(decisions))
        graphed = _run_langgraph(fixture, graph, decisions)
        structure = graph.get_graph()
        common = _common(fixture, shared_nodes)
    return {
        **common, "scenario": deepcopy(next(row for row in SCENARIOS if row["id"] == scenario)),
        "graph": {"nodes": [{"id": key, "label": value.name} for key, value in structure.nodes.items()],
                  "edges": [{"source": edge.source, "target": edge.target, "conditional": edge.conditional} for edge in structure.edges]},
        "plain": plain, "langgraph": graphed,
        "comparison": {
            "same_steps": [event["node"] for event in plain["events"]] == [event["node"] for event in graphed["events"]],
            "same_updates": [event["update"] for event in plain["events"]] == [event["update"] for event in graphed["events"]],
            "same_final_state": plain["final_state"] == graphed["final_state"],
            "same_answer": plain["final_state"]["answer"] == graphed["final_state"]["answer"],
            "same_decisions": plain["decisions"] == graphed["decisions"],
        },
    }
