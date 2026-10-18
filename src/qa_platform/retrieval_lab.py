"""独立的本地检索学习台；GET不调用模型，按钮明确选择零模型或一次问题编码。"""

import argparse
import json
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from qa_platform.retrieval import kubernetes_hybrid as hybrid
from qa_platform.settings import DATA_DIR, PROJECT_ROOT


DEFAULT_INDEX = DATA_DIR / "milvus_indexes" / "kubernetes_zh_v1"
PAGE = PROJECT_ROOT / "web" / "retrieval_lab.html"


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=2000)
    mode: str


def create_app(index_dir: Path = DEFAULT_INDEX) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    busy = Lock()

    @app.middleware("http")
    async def local_only(request, call_next):
        response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                    "img-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
                ),
            }
        )
        return response

    @app.get("/")
    def page():
        return FileResponse(PAGE)

    @app.get("/api/plan")
    def plan():
        try:
            return hybrid.describe(hybrid.prepare(index_dir))
        except (ValueError, OSError, KeyError, TypeError):
            raise HTTPException(503, "已有Kubernetes索引或本地语料未就绪；不会自动建库。") from None

    @app.post("/api/compare")
    async def compare(request: Request):
        # 自定义头要求浏览器跨站预检；不启用CORS，不接受外站表单/跨源调用。
        origin = request.headers.get("origin")
        expected_origin = str(request.base_url).rstrip("/")
        if request.headers.get("x-retrieval-lab") != "1" or (
            origin is not None and origin != expected_origin
        ):
            raise HTTPException(403, "只接受本机页面主动发起的检索")
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise HTTPException(415, "需要application/json")
        # 流式限长，不能在检查前一次性读入任意大的请求体。
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > 16384:
                raise HTTPException(413, "请求体过大")
        try:
            payload = ComparisonRequest.model_validate(json.loads(body))
            if not payload.query.strip() or payload.mode not in {"bm25", "hybrid"}:
                raise ValueError("无效问题或模式")
        except (ValueError, UnicodeError, ValidationError):
            raise HTTPException(422, "问题须为1～2000字符；mode为bm25或hybrid；不接受其他参数。") from None
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "已有一轮检索在执行，请等待；不会并发请求模型。")
        try:
            def execute():
                check = hybrid.prepare(index_dir, payload.query.strip())
                return hybrid.compare(check, payload.mode)

            return await run_in_threadpool(execute)
        except Exception as exc:
            # 不把模型服务响应/未校验正文或环境信息发送到页面。
            print(f"[检索未完成] {type(exc).__name__}；无自动重试。", flush=True)
            raise HTTPException(
                503,
                "检索未完成：请检查已有索引、模型连通性与模型版本。未展示部分结果，未自动重试；失败前的模型请求可能已经发生。",
            ) from None
        finally:
            busy.release()

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="本地Kubernetes BM25/Dense/RRF对照台")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port须为1024～65535")
    import uvicorn

    print(f"[打开] http://127.0.0.1:{args.port}/；仅本机，不发布。")
    print("[计划] 打开页面0模型；仅BM25=0；三路对照=1次本地问题Embedding+2次身份GET。")
    print("[边界] 不读.env、不编码文档、不请求云端、不写报告；数据库运行时可能更新锁/日志。")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
