"""只读知识检索工具：复用混合检索与重排，跨轮冻结语料及模型合同。"""

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from qa_platform.ingestion.embedding_plan import canonical, fingerprint, require
from qa_platform.rag import kubernetes_answer as answering
from qa_platform.rag.answering import build_quotes
from qa_platform.rag.quote_units import PARAGRAPH_STRATEGY
from qa_platform.retrieval import kubernetes_hybrid as hybrid
from qa_platform.retrieval import kubernetes_rerank as reranking
from qa_platform.retrieval.pipeline import RerankedHit
from qa_platform.workflows.progress import report


TOOL_NAME = "search_knowledge"
MAX_SOURCES = 5


class SearchArgs(BaseModel):
    """模型只能提供检索文本，不接受本机路径、服务地址或检索参数。"""

    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=2000)


TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "只读检索Kubernetes中文知识库，返回相关正文和原文片段。资料未审核，命令不会执行。",
        "parameters": SearchArgs.model_json_schema(),
    },
}]


def _contract(plan: dict) -> str:
    """指纹覆盖正文、向量及完整Embedding配置；不把路径发送给模型。"""
    return fingerprint(canonical({
        "document_snapshot_sha256": plan["document_snapshot_sha256"],
        "vector_snapshot_sha256": plan["vector_snapshot_sha256"],
        "embedding": plan["embedding"],
    }))


def _validate_sources(round: dict) -> None:
    """局部适配对象须仍指向冻结的原文和来源，不接受另塞入的证据。"""
    sources = round["sources"]
    require(isinstance(sources, dict) and 1 <= len(sources) <= MAX_SOURCES, "检索证据数量无效")
    require(list(sources) == [f"S{i}" for i in range(1, len(sources) + 1)], "检索证据编号无效")
    seen = set()
    for hit in sources.values():
        document = hit.candidate.document
        chunk_id = document.metadata["chunk_id"]
        require(chunk_id not in seen and chunk_id in round["result"]["records"], "检索证据身份无效")
        seen.add(chunk_id)
        row = round["result"]["records"][chunk_id]
        metadata = deepcopy(row["metadata"])
        metadata["document_mode"] = "body-only"
        metadata["quote_strategy"] = PARAGRAPH_STRATEGY
        metadata.setdefault("is_test", False)
        require(
            document.page_content == row["text"] and document.metadata == metadata,
            "检索证据正文或元数据与冻结快照不同",
        )


def search_knowledge(
    index_dir: Path, query: str, *, expected_contract: str | None = None,
) -> dict:
    """一次问题编码、一次重排；模型或索引合同变化时在发请求前停止。"""
    report("retrieve", "started")
    args = SearchArgs.model_validate({"query": query})
    check = hybrid.prepare(index_dir, args.query)
    contract = _contract(check["plan"])
    require(expected_contract is None or contract == expected_contract, "多步检索的知识/模型合同已变化")
    result = hybrid.compare(check, "hybrid")
    result["experiment_id"] = uuid4().hex
    report("retrieve", "completed")
    report("rerank", "started")
    result = reranking.rerank(result)
    sources = answering._sources(result)
    report("rerank", "completed")
    return {"result": result, "contract": contract, "sources": sources}


def tool_reply(round: dict) -> dict:
    """工具消息仅含问题、正文、引用片段和必要审核标记。"""
    _validate_sources(round)
    evidence = []
    for hit in round["sources"].values():
        document = hit.candidate.document
        evidence.append({
            "chunk_id": document.metadata["chunk_id"],
            "text": document.page_content,
            "quotes": build_quotes(hit),
            "is_test": document.metadata.get("is_test", False),
            "review_status": document.metadata.get("review_status", "unreviewed"),
        })
    return {"query": round["result"]["query"], "evidence": evidence}


def merge_sources(rounds: list[dict]) -> dict[str, RerankedHit]:
    """轮流取各轮前列并去重；不同问题的重排原始分数不可直接比较。"""
    require(bool(rounds), "没有可合并的检索轮次")
    contract = rounds[0]["contract"]
    known = {}
    for round in rounds:
        require(round["contract"] == contract, "检索轮次来自不同知识/模型合同")
        _validate_sources(round)
        # 全候选正文也须一致，不能只检查最后选中的五块。
        for chunk_id, row in round["result"]["records"].items():
            if chunk_id in known:
                require(row == known[chunk_id], "同一文本块在检索轮次中的正文或元数据不同")
            known[chunk_id] = row
    ordered = [list(round["sources"].values()) for round in rounds]
    merged, seen = {}, set()
    for position in range(max(map(len, ordered))):
        for hits in ordered:
            if position >= len(hits):
                continue
            hit = hits[position]
            document = hit.candidate.document
            chunk_id = document.metadata["chunk_id"]
            require(document.page_content == known[chunk_id]["text"], "候选正文与检索快照不符")
            if chunk_id not in seen and len(merged) < MAX_SOURCES:
                merged[f"S{len(merged) + 1}"] = deepcopy(hit)
                seen.add(chunk_id)
    require(bool(merged), "没有可供回答的证据")
    return merged


def evidence_payload(sources: dict[str, RerankedHit]) -> dict:
    """为现有回答展示保留来源；完整元数据只留在本机响应，不进入工具消息。"""
    evidence = {}
    for source_id, hit in sources.items():
        document = hit.candidate.document
        metadata = deepcopy(document.metadata)
        metadata.pop("document_mode", None)
        metadata.pop("quote_strategy", None)
        evidence[source_id] = {
            "chunk_id": document.metadata["chunk_id"],
            "text": document.page_content,
            "quotes": build_quotes(hit),
            "metadata": metadata,
        }
    return evidence
