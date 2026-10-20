"""本机检索实验台与一次提问的真实RAG入口；只有主动提交才调用模型。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from qa_platform.rag import kubernetes_answer as answering
from qa_platform.retrieval import kubernetes_hybrid as hybrid
from qa_platform.retrieval import kubernetes_rerank as reranking
from qa_platform.settings import DATA_DIR, PROJECT_ROOT


DEFAULT_INDEX = DATA_DIR / "milvus_indexes" / "kubernetes_zh_v1"
PAGE = PROJECT_ROOT / "web" / "retrieval_lab.html"
EXPORT_SCRIPT = PROJECT_ROOT / "web" / "retrieval_export.js"
WORKFLOW_PAGE = PROJECT_ROOT / "web" / "workflow_lab.html"
BRANCHING_PAGE = PROJECT_ROOT / "web" / "branching_lab.html"
APPROVAL_PAGE = PROJECT_ROOT / "web" / "approval_lab.html"
QA_PAGE = PROJECT_ROOT / "web" / "qa.html"


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=2000)
    mode: str


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    experiment_id: str = Field(min_length=1, max_length=64)


class AnswerRequest(RerankRequest):
    model: str = Field(min_length=1, max_length=80)


class ReplayRequest(BaseModel):
    """最小回放只接受空对象；不允许把路径、正文或服务地址交给后端。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class BranchReplayRequest(BaseModel):
    """只选择固定教学场景，不接受任意快照或问题。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    scenario: Literal["with_evidence", "empty_evidence"]


class ApprovalResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    approved: bool


class QARequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=2000)
    model: str = Field(min_length=1, max_length=80)
    confirmed: bool
    mode: Literal["single", "agent"] = "single"


def create_app(index_dir: Path = DEFAULT_INDEX) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    busy = Lock()
    # 仅保留最近一轮的服务端冻结结果；浏览器不能上传任意正文或路径。
    snapshot = None
    approval_lesson = None  # 按需创建；只保留本app的一轮合成内存检查点。

    def get_approval_lesson():
        nonlocal approval_lesson
        from qa_platform.workflows.approval import ApprovalLesson

        if approval_lesson is None:
            approval_lesson = ApprovalLesson()
        return approval_lesson

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

    @app.get("/qa")
    def qa_page():
        return FileResponse(QA_PAGE)

    @app.get("/api/qa/plan")
    def qa_plan():
        from qa_platform.workflows.live_rag import describe

        return describe()  # 不读密钥/索引，不启动图或模型请求。

    @app.get("/assets/retrieval_export.js")
    def export_script():
        # 固定静态资源，不接收路径参数，不触发检索或模型调用。
        return FileResponse(EXPORT_SCRIPT, media_type="text/javascript")

    @app.get("/workflow")
    def workflow_page():
        return FileResponse(WORKFLOW_PAGE)

    @app.get("/api/workflow/plan")
    def workflow_plan():
        # 独立教学页不要求当前Milvus索引或真实问答快照已经就绪。
        from qa_platform.workflows.replay import describe

        return describe()

    @app.get("/workflow/branching")
    def branching_page():
        return FileResponse(BRANCHING_PAGE)

    @app.get("/api/workflow/branching/plan")
    def branching_plan():
        from qa_platform.workflows.branching import describe

        return describe()

    @app.get("/workflow/approval")
    def approval_page():
        return FileResponse(APPROVAL_PAGE)

    @app.get("/api/workflow/approval/state")
    def approval_state():
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "当前有操作正在执行；请稍后刷新，读取不会执行节点。")
        try:
            return get_approval_lesson().current()
        except Exception:
            raise HTTPException(503, "无法读取本地教学状态；未启动或恢复图，未调用模型。") from None
        finally:
            busy.release()

    @app.get("/api/plan")
    def plan():
        try:
            return {
                **hybrid.describe(hybrid.prepare(index_dir)),
                "reranker": reranking.describe(), "answer": answering.describe(),
            }
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

    @app.post("/api/qa")
    async def qa(request: Request):
        payload = await read_payload(
            request, QARequest, "只接受query、允许的model、布尔confirmed及可选single/agent模式；问题须为1～2000字符。",
        )
        if not payload.query.strip() or payload.model not in answering.MODEL_CONFIGS or payload.confirmed is not True:
            raise HTTPException(422, "请输入非空问题、选择允许的模型，并确认本次完整问答的数据发送。")
        if payload.mode == "agent":
            from qa_platform.workflows.agent_rag import ENABLED_MODELS

            if payload.model not in ENABLED_MODELS:
                raise HTTPException(422, "多步检索暂仅支持已验证原生工具协议的qwen3.8-flash，未请求模型。")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "已有操作正在执行；本次提问尚未开始，请等待后主动提交。")
        try:
            if payload.mode == "agent":
                from qa_platform.workflows.agent_rag import run
            else:
                from qa_platform.workflows.live_rag import run

            # 独立的一次执行；不冒充教学回放，也不读写原实验台snapshot。
            return await run_in_threadpool(run, payload.query, payload.model, index_dir)
        except Exception as exc:
            from qa_platform.workflows.agent_rag import AgentRunError
            from qa_platform.workflows.live_rag import RAGRunError

            if isinstance(exc, (RAGRunError, AgentRunError)):
                raise HTTPException(503, str(exc)) from None
            print(f"[自动问答未完成] {type(exc).__name__}；不自动重试。", flush=True)
            raise HTTPException(
                503, "自动问答未完成，请检查索引、LangGraph依赖及模型服务。未展示未经校验的回答；"
                "请求可能已经发生并产生费用，不自动重试。",
            ) from None
        finally:
            busy.release()

    @app.post("/api/workflow/replay")
    async def workflow_replay(request: Request):
        await read_payload(request, ReplayRequest, "回放只接受空JSON对象，不接受问题、路径、正文或模型参数。")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "当前有操作正在执行，请等待；未开始回放，也不会请求模型。")
        try:
            from qa_platform.workflows.replay import compare_replay

            # 不读取或修改真实检索snapshot；既有回答和实验编号原样保留。
            return await run_in_threadpool(compare_replay)
        except Exception as exc:
            print(f"[工作流回放未完成] {type(exc).__name__}；未调用模型。", flush=True)
            raise HTTPException(503, "本地回放未完成，请检查LangGraph依赖或合成快照；未调用模型、未自动重试。") from None
        finally:
            busy.release()

    @app.post("/api/workflow/branching/replay")
    async def branching_replay(request: Request):
        payload = await read_payload(
            request, BranchReplayRequest,
            "只接受固定scenario：with_evidence或empty_evidence；不接受问题、路径、正文或模型参数。",
        )
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "当前有操作正在执行；未开始条件回放，也不会请求模型。")
        try:
            from qa_platform.workflows.branching import run_replay

            # 与原线性课件一样，不接管或清除真实检索snapshot。
            return await run_in_threadpool(run_replay, payload.scenario)
        except Exception as exc:
            print(f"[条件回放未完成] {type(exc).__name__}；未调用模型。", flush=True)
            raise HTTPException(503, "本地条件回放未完成；未调用模型、未自动重试，也未把异常当作正常拒答。") from None
        finally:
            busy.release()

    @app.post("/api/workflow/approval/start")
    async def approval_start(request: Request):
        await read_payload(request, ReplayRequest, "开始只接受空JSON对象，不接受问题、正文或模型配置。")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "已有操作执行中，未开始新会话。")
        try:
            return await run_in_threadpool(get_approval_lesson().start)
        except Exception as exc:
            print(f"[确认课件未开始] {type(exc).__name__}；无自动重试。", flush=True)
            raise HTTPException(503, "本地教学图未成功暂停；未调用模型、未自动重试。") from None
        finally:
            busy.release()

    @app.post("/api/workflow/approval/resume")
    async def approval_resume(request: Request):
        payload = await read_payload(request, ApprovalResumeRequest, "只接受当前32位session_id和布尔approved，不接受其他参数。")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "已有操作执行中，未提交恢复。")
        try:
            from qa_platform.workflows.approval import ApprovalConflict

            try:
                return await run_in_threadpool(get_approval_lesson().resume, payload.session_id, payload.approved)
            except ApprovalConflict:
                raise HTTPException(409, "会话已失效或不在等待确认；请刷新查看或重新开始，不能重复恢复。") from None
        except HTTPException:
            raise
        except Exception as exc:
            print(f"[确认课件未恢复] {type(exc).__name__}；无自动重试。", flush=True)
            raise HTTPException(503, "本地恢复未完成，会话已失效；可能已执行部分节点，请重新开始。未调用模型、未自动重试。") from None
        finally:
            busy.release()

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

    @app.post("/api/answer")
    async def answer(request: Request):
        nonlocal snapshot
        payload = await read_payload(
            request, AnswerRequest, "只接受当前experiment_id和允许的model，不接受问题、正文、地址或密钥。",
        )
        if not payload.experiment_id.strip() or payload.model not in answering.MODEL_CONFIGS:
            raise HTTPException(422, "实验编号不能为空；回答模型必须在允许列表中。")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "已有一轮检索、重排或生成在执行，请等待；不会并发请求模型。")
        try:
            if snapshot is None or snapshot["experiment_id"] != payload.experiment_id:
                raise HTTPException(409, "本轮证据已失效，请重新检索并重排；未请求回答模型。")
            if "reranker" not in snapshot["routes"]:
                raise HTTPException(409, "请先完成本轮云端重排；未请求回答模型。")
            if "answer_generation" in snapshot:
                if snapshot["answer_generation"]["model"] != payload.model:
                    raise HTTPException(409, "本轮已有回答；更换模型需重新开始一轮，不会自动新增请求。")
                return deepcopy(snapshot)
            result = await run_in_threadpool(answering.generate, deepcopy(snapshot), payload.model)
            result["experiment_id"] = snapshot["experiment_id"]
            snapshot = deepcopy(result)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            print(f"[回答未完成] {type(exc).__name__}；保留检索与重排，无自动重试。", flush=True)
            raise HTTPException(
                503,
                "回答未完成：模型连接、完整结束、JSON结构或引用校验未通过。"
                "未展示未经校验的回答，已有检索、重排与证据仍保留；失败请求可能已发生，"
                "计数仅覆盖已完成阶段。再次主动点击会另发请求，不会自动重试。",
            ) from None
        finally:
            busy.release()

    @app.post("/api/quotes")
    async def quote_preview(request: Request):
        nonlocal snapshot
        payload = await read_payload(
            request, RerankRequest, "只接受当前experiment_id；引用预览不接受正文、地址或模型参数。",
        )
        if not payload.experiment_id.strip():
            raise HTTPException(422, "experiment_id不能为空")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "当前有操作正在执行，请等待；预览不会调用模型。")
        try:
            if snapshot is None or snapshot["experiment_id"] != payload.experiment_id:
                raise HTTPException(409, "实验已失效，请重新检索并重排；未调用模型。")
            if "reranker" not in snapshot["routes"]:
                raise HTTPException(409, "请先完成本轮重排；引用预览未调用模型。")
            if "quote_comparison" not in snapshot:
                comparison = await run_in_threadpool(
                    answering.prepare_quote_comparison, deepcopy(snapshot),
                )
                updated = deepcopy(snapshot)
                updated["quote_comparison"] = comparison
                snapshot = updated
            # 已有回答也只增加对照，不重新生成或改写旧引用编号。
            return deepcopy(snapshot)
        except HTTPException:
            raise
        except Exception as exc:
            print(f"[引用预览未完成] {type(exc).__name__}；零模型请求。", flush=True)
            raise HTTPException(503, "本地引用预览未完成，已有结果保留；未读取密钥、未调用模型。") from None
        finally:
            busy.release()

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="本地Kubernetes问答与检索学习台")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port须为1024～65535")
    import uvicorn

    print(f"[打开] http://127.0.0.1:{args.port}/；仅本机，不发布。")
    print(f"[问答] http://127.0.0.1:{args.port}/qa；一次提问由LangGraph自动完成真实检索、重排和回答。")
    print("[问答计划] 默认单次：问题Embedding、云端重排、聊天各1次，另有2次身份GET。")
    print("[多步可选] Flash原生工具循环：最多2次检索；Embedding2/重排2/聊天3，共最多11次HTTP；不自动重试。")
    print("[计划] 打开页面0模型；仅BM25=0；三路对照=1次本地问题Embedding+2次身份GET。")
    print("[重排] 三路完成后可主动点击：固定同一并集，读密钥并向云端发送问题与正文，增加1次HTTP；可能收费。")
    print("[回答] 重排完成后另行确认：选前5块生成带引用的回答，增加1次HTTP；可选既有千问或GLM，GLM地址为HTTP明文。")
    print("[引用] 可单独预览按行/按段对照；只在本机计算，新增模型请求为0，不改原文或向量。")
    print(f"[编排] http://127.0.0.1:{args.port}/workflow；独立教学合成快照，普通函数与LangGraph零模型回放。")
    print("[边界] 打开页面/普通检索不读.env、不请求云端；不重编码文档、不执行命令、不写报告，无自动重试。")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
