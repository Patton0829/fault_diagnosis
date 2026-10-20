"""SQL persistence tests use explicit temporary SQLite schemas, never MySQL."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import socket
from threading import Event
from unittest.mock import Mock

import dotenv
from fastapi.testclient import TestClient
import httpx
import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, event, func, inspect, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from qa_platform import retrieval_lab as lab
from qa_platform.storage import sql_conversations as sql
from qa_platform.workflows import conversation, live_rag


MODEL = "qwen3.8-flash"
QUERY = "Synthetic DNS question?"
PRIVATE = "PRIVATE_PATH_VECTOR_CANDIDATE_TOKEN_MUST_NOT_PERSIST"
HEADERS = {"X-Retrieval-Lab": "1", "Origin": "http://127.0.0.1"}


@pytest.fixture(autouse=True)
def no_network_or_env(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SQL tests must not read secrets or contact MySQL/model services")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)


def public_result(query=QUERY):
    text = "Synthetic conditional source paragraph."
    return {
        "schema_version": "rag-live/v1", "query": query, "model": MODEL,
        "answer": {"status": "partial", "claims": [{
            "text": "Synthetic conditional explanation.",
            "evidence": [{"source_id": "S1", "quote_id": "Q1"}],
        }], "missing_info": ["Actual cluster information is required."]},
        "sources": {"S1": {"chunk_id": "synthetic-source", "text": text,
                            "quotes": {"Q1": text}, "metadata": {
                                "title": "Synthetic source", "section_path": ["Synthetic"],
                                "source_url": "https://example.invalid/source", "review_status": "unreviewed",
                                "kubernetes_version": "unknown", "source_path": PRIVATE, "vector": [PRIVATE],
                            }}},
        "request_counts": {"embedding": 1, "http": 5, "reranker": 1, "chat": 1},
        "steps": ["retrieve", "rerank", "answer"], "elapsed_seconds": 0.1,
    }


@pytest.fixture
def engine(tmp_path):
    engine = create_engine("sqlite+pysqlite:///" + str(tmp_path / "conversations.sqlite3"))
    sql.Base.metadata.create_all(engine)  # Explicit test setup, not application behavior.
    yield engine
    engine.dispose()


@pytest.fixture
def store(engine):
    return sql.SQLConversationStore(engine)


def test_constructor_never_connects_or_creates_schema(tmp_path):
    engine = create_engine("sqlite+pysqlite:///" + str(tmp_path / "uninitialized.sqlite3"))
    connected = Mock()
    event.listen(engine, "connect", connected)
    try:
        sql.SQLConversationStore(engine)
        connected.assert_not_called()
        with pytest.raises(sql.DatabaseError):
            sql.check_schema(engine)
        assert inspect(engine).get_table_names() == []
    finally:
        engine.dispose()


def test_schema_check_is_read_only_and_valid_explicit_schema_passes(engine):
    statements = []

    def listener(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    event.listen(engine, "before_cursor_execute", listener)
    try:
        sql.check_schema(engine)
    finally:
        event.remove(engine, "before_cursor_execute", listener)
    assert statements and all(statement.lstrip().upper().startswith("SELECT") for statement in statements)


def test_schema_check_rejects_missing_payload_columns_without_reading_existing_rows(tmp_path):
    engine = create_engine("sqlite+pysqlite:///" + str(tmp_path / "old-schema.sqlite3"))
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE qa_conversations (id TEXT, last_turn INTEGER, active BOOLEAN)")
            connection.exec_driver_sql("CREATE TABLE qa_turns (conversation_id TEXT, turn_no INTEGER, summary JSON, result JSON)")
        with pytest.raises(sql.DatabaseError):
            sql.check_schema(engine)
    finally:
        engine.dispose()


def test_commit_load_and_reopen_engine_restore_validated_result(store, engine):
    meta = store.commit(None, QUERY, QUERY, public_result())
    assert meta["turn"] == 1 and meta["retained_turns"] == 1
    loaded = store.load(meta["id"])
    assert loaded["schema_version"] == "qa-conversation/v1" and loaded["id"] == meta["id"]
    assert loaded["turn"] == 1 and loaded["turns"][0]["conversation"] == meta
    url = engine.url
    engine.dispose()
    reopened = create_engine(url)
    try:
        restored = sql.SQLConversationStore(reopened)
        assert restored.load(meta["id"]) == loaded
        assert restored.get(meta["id"])[0]["query"] == QUERY
    finally:
        reopened.dispose()


def test_storage_whitelists_metadata_and_only_keeps_actually_cited_sources(store, engine):
    result = public_result()
    result.update(draft=PRIVATE, candidate_pool=[PRIVATE], api_key=PRIVATE)
    result["sources"]["S2"] = {**deepcopy(result["sources"]["S1"]), "text": PRIVATE}
    original = deepcopy(result)
    meta = store.commit(None, QUERY, QUERY, result)
    loaded = store.load(meta["id"])["turns"][0]
    assert set(loaded["sources"]) == {"S1"}
    assert set(loaded["sources"]["S1"]["metadata"]) == {
        "title", "section_path", "source_url", "review_status", "kubernetes_version",
    }
    assert PRIVATE not in json.dumps(loaded) and result == original
    with Session(engine) as session:
        record = session.scalar(select(sql.TurnRecord))
        assert PRIVATE not in json.dumps(record.result)
        assert PRIVATE not in json.dumps(record.summary)


def test_recent_context_and_load_limit_to_three_but_all_turns_remain_in_database(store, engine):
    identity = None
    for number in range(5):
        query = f"Synthetic question {number}"
        meta = store.commit(identity, query, query, public_result(query))
        identity = meta["id"]
    assert meta["turn"] == 5 and meta["history_turns"] == meta["retained_turns"] == 3
    assert [turn["query"] for turn in store.get(identity)] == [f"Synthetic question {n}" for n in (2, 3, 4)]
    assert [turn["conversation"]["turn"] for turn in store.load(identity)["turns"]] == [3, 4, 5]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(sql.TurnRecord)) == 5


def test_get_and_load_are_detached_deep_copies(store):
    identity = store.commit(None, QUERY, QUERY, public_result())["id"]
    history, loaded = store.get(identity), store.load(identity)
    history[0]["answer"]["claims"][0] = PRIVATE
    loaded["turns"][0]["sources"]["S1"]["text"] = PRIVATE
    assert PRIVATE not in json.dumps(store.get(identity))
    assert PRIVATE not in json.dumps(store.load(identity))


def test_sql_store_does_not_apply_sixteen_session_memory_eviction(store):
    identities = [store.commit(None, QUERY, QUERY, public_result())["id"] for _ in range(17)]
    assert store.get(identities[0]) and store.get(identities[-1])


def test_reset_soft_closes_conversation_and_preserves_rows(store, engine):
    identity = store.commit(None, QUERY, QUERY, public_result())["id"]
    store.clear(identity)
    for operation in (store.get, store.load):
        with pytest.raises(KeyError):
            operation(identity)
    with pytest.raises(KeyError):
        store.commit(identity, QUERY, QUERY, public_result())
    with Session(engine) as session:
        assert session.get(sql.ConversationRecord, identity).active is False
        assert session.scalar(select(func.count()).select_from(sql.TurnRecord)) == 1


@pytest.mark.parametrize("existing", [False, True])
def test_transaction_failure_rolls_back_parent_counter_and_turn_insert(store, engine, existing):
    identity = store.commit(None, QUERY, QUERY, public_result())["id"] if existing else None

    def fail_insert(mapper, connection, target):
        raise SQLAlchemyError(PRIVATE)

    event.listen(sql.TurnRecord, "before_insert", fail_insert)
    try:
        with pytest.raises(sql.DatabaseError) as caught:
            store.commit(identity, QUERY, QUERY, public_result())
        assert PRIVATE not in str(caught.value) and caught.value.__cause__ is None
    finally:
        event.remove(sql.TurnRecord, "before_insert", fail_insert)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(sql.ConversationRecord)) == int(existing)
        assert session.scalar(select(func.count()).select_from(sql.TurnRecord)) == int(existing)
    if existing:
        assert store.load(identity)["turn"] == 1


def test_composite_primary_key_prevents_duplicate_turn_overwriting_existing_answer(store, engine):
    identity = store.commit(None, QUERY, QUERY, public_result())["id"]
    before = store.load(identity)
    with Session(engine) as session:
        row = session.scalar(select(sql.TurnRecord))
        duplicate = sql.TurnRecord(conversation_id=identity, turn_no=1, query=PRIVATE,
                                  effective_query=PRIVATE, model=MODEL, summary=row.summary, result=row.result)
    with Session(engine) as session, pytest.raises(IntegrityError):
        with session.begin():
            session.add(duplicate)
    assert store.load(identity) == before


@pytest.mark.parametrize("mutation", ["query", "citation", "counts"])
def test_invalid_result_never_creates_database_rows(store, engine, mutation):
    result = public_result()
    if mutation == "query":
        result["query"] = PRIVATE
    elif mutation == "citation":
        result["sources"]["S1"]["quotes"]["Q1"] = PRIVATE
    else:
        result["request_counts"]["http"] = True
    with pytest.raises(ValueError):
        store.commit(None, QUERY, QUERY, result)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(sql.ConversationRecord)) == 0
        assert session.scalar(select(func.count()).select_from(sql.TurnRecord)) == 0


def test_missing_schema_errors_are_sanitized_without_memory_fallback(tmp_path):
    engine = create_engine("sqlite+pysqlite:///" + str(tmp_path / "missing.sqlite3"))
    try:
        store = sql.SQLConversationStore(engine)
        with pytest.raises(sql.DatabaseError) as caught:
            store.commit(None, QUERY, QUERY, public_result())
        assert "未回退内存" in str(caught.value) and caught.value.__cause__ is None
        assert inspect(engine).get_table_names() == []
    finally:
        engine.dispose()


@pytest.fixture
def app(store, tmp_path):
    return lab.create_app(tmp_path / "missing-index", conversation_store=store, storage_backend="mysql")


@pytest.fixture
def api(app):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = {"run": [], "context": []}

    def run(query, model, path):
        calls["run"].append(query)
        return public_result(query)

    def context(query, history, model):
        calls["context"].append(deepcopy(history))
        return "Synthetic complete follow-up?"

    monkeypatch.setattr(live_rag, "run", run)
    monkeypatch.setattr(conversation, "contextualize", context)
    return calls


def ask(api, query=QUERY, **updates):
    return api.post("/api/qa", headers=HEADERS,
                    json={"query": query, "model": MODEL, "confirmed": True, "conversation": True, **updates})


def test_api_plan_reports_sql_persistence_without_model_calls(api, fake_pipeline):
    plan = api.get("/api/qa/plan")
    assert plan.status_code == 200
    assert plan.json()["storage"]["backend"] == "mysql" and plan.json()["storage"]["persistent"] is True
    assert plan.json()["conversation"]["max_sessions"] is None
    assert fake_pipeline == {"run": [], "context": []}


def test_default_memory_plan_keeps_sixteen_session_limit(tmp_path):
    with TestClient(lab.create_app(tmp_path), base_url="http://127.0.0.1") as client:
        data = client.get("/api/qa/plan").json()
    assert data["storage"]["backend"] == "memory" and data["storage"]["persistent"] is False
    assert data["conversation"]["max_sessions"] == 16


def test_sql_app_recreation_loads_answers_without_rerunning_models(api, store, engine, fake_pipeline, tmp_path):
    first = ask(api).json()
    identity = first["conversation"]["id"]
    url = engine.url
    reopened = create_engine(url)
    try:
        app = lab.create_app(tmp_path, conversation_store=sql.SQLConversationStore(reopened), storage_backend="mysql")
        with TestClient(app, base_url="http://127.0.0.1") as client:
            loaded = client.post("/api/qa/conversation/load", headers=HEADERS, json={"conversation_id": identity})
            assert loaded.status_code == 200
            assert loaded.json()["turns"][0]["query"] == QUERY
            assert loaded.json()["turns"][0]["conversation"] == first["conversation"]
            assert PRIVATE not in loaded.text
    finally:
        reopened.dispose()
    assert fake_pipeline["run"] == [QUERY] and fake_pipeline["context"] == []


def test_sql_followup_uses_saved_summary_and_persists_display_query(api, store, fake_pipeline):
    first = ask(api).json()
    identity = first["conversation"]["id"]
    second = ask(api, "And that namespace?", conversation_id=identity)
    assert second.status_code == 200 and second.json()["conversation"]["turn"] == 2
    loaded = store.load(identity)["turns"][-1]
    assert loaded["query"] == "And that namespace?"
    assert loaded["conversation"]["effective_query"] == "Synthetic complete follow-up?"
    assert loaded["request_counts"] == {"embedding": 1, "http": 6, "reranker": 1, "chat": 2}
    assert len(fake_pipeline["context"][0]) == 1 and PRIVATE not in json.dumps(fake_pipeline["context"])


def test_failed_generation_does_not_append_persistent_turn(api, store, fake_pipeline, monkeypatch):
    identity = ask(api).json()["conversation"]["id"]
    failed = Mock(side_effect=RuntimeError(PRIVATE))
    monkeypatch.setattr(live_rag, "run", failed)
    response = ask(api, "Follow-up?", conversation_id=identity)
    assert response.status_code == 503 and PRIVATE not in response.text
    assert store.load(identity)["turn"] == 1 and len(store.get(identity)) == 1
    failed.assert_called_once()


def test_api_database_error_is_not_returned_as_success_or_memory_session(api, store, fake_pipeline, monkeypatch):
    monkeypatch.setattr(store, "commit", Mock(side_effect=sql.DatabaseError()))
    response = ask(api)
    assert response.status_code == 503 and "conversation" not in response.json()
    assert len(fake_pipeline["run"]) == 1


@pytest.mark.parametrize("headers,body,status", [
    ({}, {"conversation_id": "a" * 32}, 403),
    ({**HEADERS, "Origin": "https://evil.invalid"}, {"conversation_id": "a" * 32}, 403),
    (HEADERS, {"conversation_id": "bad-id"}, 422),
    (HEADERS, {"conversation_id": "a" * 32, "history": [PRIVATE]}, 422),
    (HEADERS, {"conversation_id": "a" * 32}, 410),
])
def test_load_guards_or_unknown_session_never_call_models(api, fake_pipeline, headers, body, status):
    response = api.post("/api/qa/conversation/load", headers=headers, json=body)
    assert response.status_code == status and PRIVATE not in response.text
    assert fake_pipeline == {"run": [], "context": []}


def test_reset_soft_closes_persisted_session_and_load_returns_gone(api, store, engine, fake_pipeline):
    identity = ask(api).json()["conversation"]["id"]
    body = {"conversation_id": identity}
    assert api.post("/api/qa/conversation/reset", headers=HEADERS, json=body).status_code == 200
    assert api.post("/api/qa/conversation/reset", headers=HEADERS, json=body).status_code == 200
    assert api.post("/api/qa/conversation/load", headers=HEADERS, json=body).status_code == 410
    with Session(engine) as session:
        assert session.get(sql.ConversationRecord, identity).active is False
        assert session.scalar(select(func.count()).select_from(sql.TurnRecord)) == 1
    assert fake_pipeline["run"] == [QUERY]


def test_load_holds_shared_busy_lock_until_database_read_finishes(api, store, fake_pipeline, monkeypatch):
    identity = ask(api).json()["conversation"]["id"]
    original = store.load
    entered, release = Event(), Event()

    def blocked(session_id):
        entered.set()
        assert release.wait(5)
        return original(session_id)

    monkeypatch.setattr(store, "load", blocked)
    body = {"conversation_id": identity}
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(api.post, "/api/qa/conversation/load", headers=HEADERS, json=body)
        try:
            assert entered.wait(5)
            assert api.post("/api/qa/conversation/reset", headers=HEADERS, json=body).status_code == 409
            assert ask(api).status_code == 409
            assert fake_pipeline["run"] == [QUERY]
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
    assert ask(api).status_code == 200
