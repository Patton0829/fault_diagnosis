"""生成前真实暂停与恢复：只用合成资料和进程内检查点，不调用模型。"""

from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
import inspect
from textwrap import dedent
from typing import Literal
from uuid import uuid4

from langsmith import tracing_context

from qa_platform.workflows import replay
from qa_platform.workflows.replay import ReplayState


SCHEMA = "rag-workflow-approval/v1"
PREFIX = ("retrieve", "fuse", "rerank", "prepare_evidence")
NODES = deepcopy(replay.NODES)
NODES.insert(4, {"id": "review", "label": "人工确认", "reads": ["evidence"],
                 "writes": ["approved"], "operation": "interrupt暂停；恢复后记录布尔决定"})
NODES.insert(-1, {"id": "cancel", "label": "用户取消", "reads": ["approved"],
                  "writes": ["cancelled"], "operation": "正常结束，不产生草稿或回答"})
NOTICE = (
    "合成教学数据；模型推理及模型服务HTTP均为0，本机页面API通信另计。"
    "暂停来自真实interrupt和InMemorySaver，不是浏览器隐藏结果。检查点仅在本服务进程内存："
    "刷新页面可查看当前会话，重启服务或重新开始会丢弃它；不是磁盘持久化、多轮记忆或崩溃恢复。"
    "恢复会从确认节点开头重新进入，interrupt之前的代码会再运行；前四个已完成节点不重跑。"
    "计数仅为教学观测；确认节点前不做业务写入或模型请求。人工确认只允许继续教学回放，不代表资料通过审核。"
    "不读.env、不连接Milvus、不外发LangSmith追踪、不接管真实RAG、不自动重试。"
)


class ApprovalState(ReplayState, total=False):
    """沿用前课状态；确认后才出现approved，取消路径才写入cancelled。"""

    approved: bool
    cancelled: bool


def review_evidence(state: ApprovalState) -> dict:
    """这个节点会在恢复时从头进入；interrupt前不能放不可重复的业务操作。"""
    from langgraph.types import interrupt

    if not isinstance(state["evidence"], dict) or not state["evidence"]:
        raise ValueError("本课只对非空合成证据演示人工确认")
    approved = interrupt({
        "kind": "review_evidence", "question": "是否继续读取合成草稿？这是教学确认，不是资料审核或云端授权。",
        "evidence_count": len(state["evidence"]), "source_ids": list(state["evidence"]),
    })
    if type(approved) is not bool:
        raise ValueError("确认值必须为布尔值")
    return {"approved": approved}


def route_after_review(state: ApprovalState) -> Literal["generate", "cancel"]:
    if type(state["approved"]) is not bool:
        raise ValueError("缺少合法的人工决定")
    return "generate" if state["approved"] else "cancel"


def cancel(state: ApprovalState) -> dict:
    """取消不是证据不足拒答，也不是错误：不创建draft或answer。"""
    if state["approved"] is not False:
        raise ValueError("没有取消决定，不能进入取消节点")
    return {"cancelled": True}


def make_nodes(fixture: dict) -> dict:
    functions = replay.make_nodes(fixture)
    functions.update(review=review_evidence, cancel=cancel)
    return {row["id"]: functions[row["id"]] for row in NODES}


def build_graph(nodes: dict, checkpointer):
    """编译时接入检查点；确认后只选择生成或取消其中一个节点。"""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ApprovalState)
    for name, function in nodes.items():
        builder.add_node(name, function, retry_policy=None)
    builder.add_edge(START, "retrieve")
    for source, target in zip(PREFIX, PREFIX[1:]):
        builder.add_edge(source, target)
    builder.add_edge("prepare_evidence", "review")
    builder.add_conditional_edges("review", route_after_review, {"generate": "generate", "cancel": "cancel"})
    builder.add_edge("generate", "validate")
    builder.add_edge("validate", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=checkpointer, cache=None)


class ApprovalConflict(ValueError):
    """旧会话、已结束会话或未暂停会话均不允许恢复。"""


@dataclass
class _Session:
    session_id: str
    graph: object
    config: dict
    run_counts: dict
    events: list = field(default_factory=list)
    phase: str = "starting"
    view: dict | None = None


def _observed_nodes(functions: dict, counts: dict) -> dict:
    """只观测实际进入/正常返回次数；中断的节点不记作已完成。"""
    def observe(name, function):
        @wraps(function)
        def observed(state):
            counts[name]["entered"] += 1
            update = function(state)
            counts[name]["completed"] += 1
            return update
        return observed
    return {name: observe(name, fn) for name, fn in functions.items()}


def _drive(session: _Session, graph_input):
    previous, pending = None, None
    for kind, data in session.graph.stream(
        graph_input, config=session.config, stream_mode=["updates", "values"],
    ):
        if kind == "updates":
            if "__interrupt__" in data:
                if pending is not None:
                    raise ValueError("中断之前存在未完成的状态事件")
                continue  # 这是运行器控制事件，不是review节点成功返回的update。
            if pending is not None or len(data) != 1:
                raise ValueError("本课只允许单节点顺序事件")
            pending = next(iter(data.items()))
        elif kind == "values":
            if pending is not None:
                name, update = pending
                session.events.append(replay._event(len(session.events) + 1, name, previous, update, data))
                pending = None
            previous = deepcopy(data)
    if pending is not None:
        raise ValueError("没有取得LangGraph合并后的状态")
    snapshot = session.graph.get_state(session.config)
    interrupts = snapshot.interrupts
    if interrupts:
        if len(interrupts) != 1 or snapshot.next != ("review",) or "approved" in snapshot.values:
            raise ValueError("不是预期的人工确认检查点")
        session.phase = "paused"
    elif snapshot.next:
        raise ValueError("图仍有未完成任务，但没有合法确认中断")
    else:
        session.phase = "cancelled" if snapshot.values.get("cancelled") is True else "completed"
    structure = session.graph.get_graph()
    session.view = {
        "phase": session.phase, "session_id": session.session_id,
        "graph": {"nodes": [{"id": key, "label": value.name} for key, value in structure.nodes.items()],
                  "edges": [{"source": edge.source, "target": edge.target, "conditional": edge.conditional} for edge in structure.edges]},
        "values": deepcopy(snapshot.values), "events": deepcopy(session.events),
        "run_counts": deepcopy(session.run_counts),
        "checkpoint": {"thread_id": session.session_id,
                       "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"], "next": list(snapshot.next)},
        "interrupt": {"id": interrupts[0].id, "value": deepcopy(interrupts[0].value)} if interrupts else None,
    }


class ApprovalLesson:
    """每个本机app只保留一轮；外层API使用与真实RAG相同的串行锁。"""

    def __init__(self):
        self._session = None

    def current(self) -> dict:
        data = describe()
        return {**data, **deepcopy(self._session.view)} if self._session is not None else data

    def start(self) -> dict:
        """从固定输入开始，真实运行到interrupt；重新开始会替换旧内存会话。"""
        from langgraph.checkpoint.memory import InMemorySaver

        self._session = None
        with tracing_context(enabled=False):
            fixture = replay.load_fixture()
            session_id = uuid4().hex
            counts = {row["id"]: {"entered": 0, "completed": 0} for row in NODES}
            functions = _observed_nodes(make_nodes(fixture), counts)
            graph = build_graph(functions, InMemorySaver())
            session = _Session(session_id, graph, {
                "configurable": {"thread_id": session_id}, "callbacks": [], "recursion_limit": 20,
            }, counts)
            _drive(session, {"query": fixture["query"]})
            if session.phase != "paused":
                raise ValueError("初次执行没有在人工确认处暂停")
            response = {**describe(), **deepcopy(session.view)}
            self._session = session
            return response

    def resume(self, session_id: str, approved: bool) -> dict:
        """使用同一个thread_id和Command恢复；不是再次传query从头启动。"""
        from langgraph.types import Command

        if type(approved) is not bool:
            raise ValueError("确认值必须为布尔值")
        session = self._session
        if session is None or session_id != session.session_id or session.phase != "paused":
            raise ApprovalConflict("当前会话不存在或不在等待确认；请重新开始，不能重复恢复。")
        try:
            with tracing_context(enabled=False):
                _drive(session, Command(resume=approved))
                if session.phase not in {"completed", "cancelled"}:
                    raise ValueError("恢复未正常结束")
                return self.current()
        except Exception:
            self._session = None  # 不允许用旧暂停画面再次提交可能已部分执行的恢复。
            raise


def _source_record(obj) -> dict:
    obj = inspect.unwrap(obj)
    if getattr(obj, "__module__", None) != __name__ or inspect.getsourcefile(obj) != __file__:
        raise ValueError("只能查看本课已知源码")
    lines, start = inspect.getsourcelines(obj)
    return {"file": "src/qa_platform/workflows/approval.py", "name": obj.__name__,
            "code": dedent("".join(lines)).rstrip("\n"), "start_line": start, "end_line": start + len(lines) - 1}


def describe() -> dict:
    try:
        installed = version("langgraph")
    except PackageNotFoundError:
        installed = None
    return {
        "schema_version": SCHEMA, "available": installed is not None, "langgraph_version": installed,
        "fixture": replay._fixture_info(replay.load_fixture()), "nodes": deepcopy(NODES),
        "definitions": {"state": _source_record(ApprovalState), "review": _source_record(review_evidence),
                        "router": _source_record(route_after_review), "graph_builder": _source_record(build_graph),
                        "start": _source_record(ApprovalLesson.start), "resume": _source_record(ApprovalLesson.resume)},
        "request_counts": dict(replay.ZERO_REQUESTS), "notice": NOTICE,
        "phase": "idle", "session_id": None, "graph": None, "values": {}, "events": [],
        "run_counts": {row["id"]: {"entered": 0, "completed": 0} for row in NODES},
        "checkpoint": None, "interrupt": None,
    }
