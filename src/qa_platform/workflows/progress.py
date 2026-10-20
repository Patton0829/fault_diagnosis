"""本次执行的固定阶段通知；没有监听者时不产生外部动作。"""

from contextlib import contextmanager
from contextvars import ContextVar


STAGES = frozenset({"contextualize", "plan", "search", "retrieve", "rerank", "answer"})
STATUSES = frozenset({"started", "completed"})
_scope = ContextVar("qa_progress", default=None)


class RunCancelled(Exception):
    """客户端已离线；只能在阶段边界合作停止，不能杀死在途SDK请求。"""


def check_cancelled() -> None:
    scope = _scope.get()
    if scope is not None and scope[1].is_set():
        raise RunCancelled()


@contextmanager
def progress_scope(sink, cancelled):
    token = _scope.set((sink, cancelled))
    try:
        check_cancelled()
        yield
    finally:
        _scope.reset(token)


def report(stage: str, status: str) -> None:
    if stage not in STAGES or status not in STATUSES:
        raise ValueError("非预期进度事件")
    check_cancelled()
    scope = _scope.get()
    if scope is not None:
        scope[0]({"schema_version": "qa-progress/v1", "stage": stage, "status": status})
        check_cancelled()
