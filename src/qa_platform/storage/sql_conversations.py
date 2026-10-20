"""SQLAlchemy会话持久化：每次操作独立事务，服务端不隐式建表或回退内存。"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from math import isfinite
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from qa_platform.rag.answering import GroundedAnswer
from qa_platform.workflows.conversation import MAX_HISTORY_TURNS, _Query, _Turn, _summary


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ConversationRecord(Base):
    __tablename__ = "qa_conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    last_turn: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class TurnRecord(Base):
    __tablename__ = "qa_turns"

    conversation_id: Mapped[str] = mapped_column(ForeignKey("qa_conversations.id"), primary_key=True)
    turn_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    effective_query: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(80))
    summary: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class DatabaseError(RuntimeError):
    """数据库错误只向调用方提供固定说明，不暴露SQL、参数或连接凭据。"""

    def __init__(self):
        super().__init__("会话数据库操作未完成，请检查数据库服务、配置和表结构；未回退内存，也未自动重试。")


def check_schema(engine) -> None:
    """只读检查两张表；建表必须由用户显式运行初始化脚本。"""
    try:
        with engine.connect() as connection:
            # 让数据库解析全部所需列，但不读取任何已有问答正文。
            connection.execute(select(ConversationRecord).limit(0))
            connection.execute(select(TurnRecord).limit(0))
    except SQLAlchemyError:
        raise DatabaseError() from None


def _public_payload(result: dict) -> dict:
    """重新拣选公开字段；未知字段、私有来源、向量、候选池和草稿不入库。"""
    if result.get("schema_version") != "rag-live/v1":
        raise ValueError("只保存已完成的公开问答响应")
    model = result["model"]
    if not isinstance(model, str) or not model.strip() or len(model) > 80:
        raise ValueError("回答模型无效")
    public = {
        "schema_version": "rag-live/v1", "query": _Query.model_validate({"query": result["query"]}).query,
        "model": model, "answer": GroundedAnswer.model_validate(result["answer"]).model_dump(),
        "sources": {},
    }
    metadata_fields = ("title", "section_path", "source_url", "review_status", "kubernetes_version")
    # 只存回答真正引用的来源，不保留多余候选。
    for claim in public["answer"]["claims"]:
        for ref in claim["evidence"]:
            source_id = ref["source_id"]
            if source_id not in public["sources"]:
                row = result["sources"][source_id]
                public["sources"][source_id] = {
                    "chunk_id": row["chunk_id"], "text": row["text"], "quotes": deepcopy(row["quotes"]),
                    "metadata": {key: deepcopy(row["metadata"][key]) for key in metadata_fields
                                 if key in row.get("metadata", {})},
                }
    counts = result["request_counts"]
    if any(type(counts[key]) is not int or counts[key] < 0 for key in ("embedding", "reranker", "chat", "http")):
        raise ValueError("请求计数无效")
    public["request_counts"] = {key: counts[key] for key in ("embedding", "reranker", "chat", "http")}
    steps = result["steps"]
    if not isinstance(steps, list) or any(stage not in {"contextualize", "plan", "search", "retrieve", "rerank", "answer"}
                                         for stage in steps):
        raise ValueError("问答步骤无效")
    public["steps"] = list(steps)
    elapsed = result["elapsed_seconds"]
    if type(elapsed) not in (int, float) or not isfinite(elapsed) or elapsed < 0:
        raise ValueError("耗时无效")
    public["elapsed_seconds"] = elapsed
    if "cache" in result:
        from qa_platform.storage.retrieval_cache import validate_info

        if result.get("mode") == "agent" or "agent" in result:
            raise ValueError("多步检索本阶段不使用缓存")
        public["cache"] = validate_info(result["cache"])
        hit = public["cache"]["status"] == "hit"
        extra = 1 if steps and steps[0] == "contextualize" else 0
        expected = {"embedding": 0 if hit else 1, "reranker": 0 if hit else 1,
                    "chat": 1 + extra, "http": (1 if hit else 5) + extra}
        if public["request_counts"] != expected:
            raise ValueError("缓存状态与持久化请求计数不一致")
    if "mode" in result:
        if result["mode"] not in {"single", "agent"}:
            raise ValueError("问答模式无效")
        public["mode"] = result["mode"]
    if "agent" in result:
        agent = result["agent"]
        public["agent"] = {key: deepcopy(agent[key]) for key in ("max_tool_calls", "planner_calls", "stop_reason")}
        public["agent"]["tool_calls"] = [
            {key: deepcopy(call[key]) for key in ("id", "name", "query", "evidence_ids")}
            for call in agent["tool_calls"]
        ]
    # JSON往返既断开所有共享引用，也拒绝非JSON对象和NaN等非法值。
    return json.loads(json.dumps(public, ensure_ascii=False, allow_nan=False))


class SQLConversationStore:
    """Engine可跨线程复用；Session仅属于当前操作，成功问答才原子提交。"""

    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def _active(session: Session, session_id: str, *, lock: bool = False) -> ConversationRecord:
        statement = select(ConversationRecord).where(ConversationRecord.id == session_id)
        if lock:
            statement = statement.with_for_update()
        row = session.scalar(statement)
        if row is None or not row.active:
            raise KeyError(session_id)
        return row

    @staticmethod
    def _recent(session: Session, session_id: str) -> list[TurnRecord]:
        rows = list(session.scalars(select(TurnRecord).where(TurnRecord.conversation_id == session_id)
                                   .order_by(TurnRecord.turn_no.desc()).limit(MAX_HISTORY_TURNS)))
        return list(reversed(rows))

    def get(self, session_id: str) -> list[dict]:
        try:
            with Session(self.engine) as session, session.begin():
                self._active(session, session_id)
                return [_Turn.model_validate({"query": row.query, "effective_query": row.effective_query,
                                              "answer": row.summary}).model_dump()
                        for row in self._recent(session, session_id)]
        except SQLAlchemyError:
            raise DatabaseError() from None

    def commit(self, session_id: str | None, query: str, effective_query: str, result: dict) -> dict:
        query = _Query.model_validate({"query": query}).query
        effective_query = _Query.model_validate({"query": effective_query}).query
        if result.get("query") != query:
            raise ValueError("会话问题与已校验结果不一致")
        summary = _summary(result)
        public = _public_payload(result)
        try:
            with Session(self.engine) as session, session.begin():
                if session_id is None:
                    session_id = uuid4().hex
                    conversation = ConversationRecord(id=session_id, last_turn=0, active=True)
                    session.add(conversation)
                    session.flush()  # 明确先插入父记录，随后才插入带外键的问答。
                else:
                    conversation = self._active(session, session_id, lock=True)
                turn_no = conversation.last_turn + 1
                meta = {"id": session_id, "turn": turn_no,
                        "history_turns": min(conversation.last_turn, MAX_HISTORY_TURNS),
                        "retained_turns": min(turn_no, MAX_HISTORY_TURNS),
                        "effective_query": effective_query, "rewritten": effective_query != query}
                public["conversation"] = deepcopy(meta)
                session.add(TurnRecord(conversation_id=session_id, turn_no=turn_no, query=query,
                                       effective_query=effective_query, model=public["model"],
                                       summary=summary, result=public))
                conversation.last_turn = turn_no
                session.flush()
            return meta
        except SQLAlchemyError:
            raise DatabaseError() from None

    def clear(self, session_id: str) -> None:
        try:
            with Session(self.engine) as session, session.begin():
                self._active(session, session_id, lock=True).active = False
        except SQLAlchemyError:
            raise DatabaseError() from None

    def load(self, session_id: str) -> dict:
        try:
            with Session(self.engine) as session, session.begin():
                conversation = self._active(session, session_id)
                return {"schema_version": "qa-conversation/v1", "id": session_id,
                        "turn": conversation.last_turn,
                        "turns": [deepcopy(row.result) for row in self._recent(session, session_id)]}
        except SQLAlchemyError:
            raise DatabaseError() from None
