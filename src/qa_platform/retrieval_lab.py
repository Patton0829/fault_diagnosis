"""本机检索与重排学习台；只有主动点击云端重排才读取密钥、发送正文。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from qa_platform.retrieval import kubernetes_hybrid as hybrid
from qa_platform.retrieval import kubernetes_rerank as reranking
from qa_platform.settings import DATA_DIR, PROJECT_ROOT


DEFAULT_INDEX = DATA_DIR / "milvus_indexes" / "kubernetes_zh_v1"
PAGE = PROJECT_ROOT / "web" / "retrieval_lab.html"
EXPORT_SCRIPT = PROJECT_ROOT / "web" / "retrieval_export.js"


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=2000)
    mode: str


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    experiment_id: str = Field(min_length=1, max_length=64)


def create_app(index_dir: Path = DEFAULT_INDEX) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    busy = Lock()
    # 仅保留最近一轮的服务端冻结结果；浏览器不能上传任意正文或路径。
    snapshot = None

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

    @app.get("/assets/retrieval_export.js")
    def export_script():
        # 固定静态资源，不接收路径参数，不触发检索或模型调用。
        return FileResponse(EXPORT_SCRIPT, media_type="text/javascript")

    @app.get("/api/plan")
    def plan():
        try:
            return {**hybrid.describe(hybrid.prepare(index_dir)), "reranker": reranking.describe()}
        except (ValueError, OSError, KeyError, TypeError):
            raise HTTPException(503, "已有Kubernetes索引或本地语料未就绪；不会自动建库。") from None

    async def read_payload(request: Request, model, error_message: str):
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
            return model.model_validate(json.loads(body))
        except (ValueError, UnicodeError, ValidationError):
            raise HTTPException(422, error_message) from None

    @app.post("/api/compare")
    async def compare(request: Request):
        nonlocal snapshot
        payload = await read_payload(
            request, ComparisonRequest,
            "问题须为1～2000字符；mode为bm25或hybrid；不接受其他参数。",
        )
        try:
            if not payload.query.strip() or payload.mode not in {"bm25", "hybrid"}:
                raise ValueError("无效问题或模式")
        except ValueError:
            raise HTTPException(422, "问题须为1～2000字符；mode为bm25或hybrid；不接受其他参数。") from None
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "已有一轮检索在执行，请等待；不会并发请求模型。")
        try:
            snapshot = None  # 即使新一轮失败，也不允许旧token冒充当前问题。

            def execute():
                check = hybrid.prepare(index_dir, payload.query.strip())
                return hybrid.compare(check, payload.mode)

            result = deepcopy(await run_in_threadpool(execute))
            if payload.mode == "hybrid":
                result["experiment_id"] = uuid4().hex
                snapshot = deepcopy(result)
            return result
        except Exception as exc:
            # 不把模型服务响应/未校验正文或环境信息发送到页面。
            print(f"[检索未完成] {type(exc).__name__}；无自动重试。", flush=True)
            raise HTTPException(
                503,
                "检索未完成：请检查已有索引、模型连通性与模型版本。未展示部分结果，未自动重试；失败前的模型请求可能已经发生。",
            ) from None
        finally:
            busy.release()

    @app.post("/api/rerank")
    async def rerank(request: Request):
        nonlocal snapshot
        payload = await read_payload(
            request, RerankRequest, "只接受当前一轮的experiment_id，不接受问题、正文或模型参数。",
        )
        if not payload.experiment_id.strip():
            raise HTTPException(422, "experiment_id不能为空")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "已有一轮检索或重排在执行，请等待；不会并发请求模型。")
        try:
            if snapshot is None or snapshot["experiment_id"] != payload.experiment_id:
                raise HTTPException(409, "该轮候选已失效，请重新运行三路对照；未请求云端模型。")
            if "reranker" in snapshot["routes"]:
                # 成功结果幂等复用：重复点击或重复发送token不再收费。
                return deepcopy(snapshot)
            result = await run_in_threadpool(reranking.rerank, deepcopy(snapshot))
            result["experiment_id"] = snapshot["experiment_id"]
            snapshot = deepcopy(result)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            print(f"[重排未完成] {type(exc).__name__}；保留原召回，无自动重试。", flush=True)
            raise HTTPException(
                503,
                "云端重排未完成：请检查重排密钥、服务连通性或响应格式。原三路结果仍保留；"
                "失败前请求可能已经发生，计数仅覆盖已完成阶段。再次主动点击会另发请求；不会自动重试。",
            ) from None
        finally:
            busy.release()

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="本地Kubernetes检索与云端重排对照台")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port须为1024～65535")
    import uvicorn

    print(f"[打开] http://127.0.0.1:{args.port}/；仅本机，不发布。")
    print("[计划] 打开页面0模型；仅BM25=0；三路对照=1次本地问题Embedding+2次身份GET。")
    print("[重排] 三路完成后可主动点击：固定同一并集，读密钥并向云端发送问题与正文，增加1次HTTP；可能收费。")
    print("[边界] 打开页面/普通检索不读.env、不请求云端；不编码文档、不聊天、不写报告，无自动重试。")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
