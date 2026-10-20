"""将同步RAG工作线程的阶段事件桥接成单向SSE，不转发模型Token。"""

import asyncio
import json
from threading import Event, Thread

from starlette.responses import StreamingResponse

from qa_platform.workflows.progress import RunCancelled, check_cancelled, progress_scope


HEARTBEAT_SECONDS = 10
MAX_PENDING_EVENTS = 64


def describe() -> dict:
    return {"url": "/api/qa/stream", "protocol": "qa-progress/v1",
            "heartbeat_seconds": HEARTBEAT_SECONDS}


def _frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_response(work, release, error_detail) -> StreamingResponse:
    """锁已由调用方取得；只有工作线程真正退出后才释放，断流不提前放锁。"""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=MAX_PENDING_EVENTS)
    cancelled = Event()

    def publish(event, data):
        if cancelled.is_set():
            return

        def enqueue():
            if cancelled.is_set():
                return
            try:
                queue.put_nowait((event, data))
            except asyncio.QueueFull:
                # 真实有界图远少于64条；异常生产者不能积压无限正文或继续调用模型。
                cancelled.set()

        try:
            loop.call_soon_threadsafe(enqueue)
        except RuntimeError:  # 应用事件循环已经关闭。
            cancelled.set()

    def worker():
        try:
            with progress_scope(lambda data: publish("progress", data), cancelled):
                result = work()
                check_cancelled()
                publish("result", result)
        except RunCancelled:
            pass
        except Exception as exc:
            publish("error", {"detail": error_detail(exc)})
        finally:
            release()

    async def body():
        try:
            while not cancelled.is_set():
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # 心跳不是工作进度，也不请求模型。
                    continue
                yield _frame(event, data)
                if event in {"result", "error"}:
                    return
        finally:
            cancelled.set()

    class Response(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                cancelled.set()  # 即使生成器尚未迭代就断开，也通知工作线程。

    try:
        response = Response(body(), media_type="text/event-stream", headers={
            "Cache-Control": "no-store", "X-Accel-Buffering": "no",
        })
        Thread(target=worker, name="qa-stream", daemon=True).start()
    except Exception:
        release()
        raise
    return response
