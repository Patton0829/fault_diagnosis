"""本地文档编码缓存：SQLite只作事务检查点，不是向量搜索库。

默认只读计划；--run才调用本地Embedding并保存；--verify纯离线读回。
同一语料/输入合同可断点续跑；不做跨语料版本增量迁移，不调用Milvus、重排或聊天。
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from time import perf_counter

from qa_platform.ingestion.embedding_plan import (
    CORPUS,
    POLICY_VERSION,
    canonical,
    fingerprint,
    load_plan,
    request_body,
    require,
)
from qa_platform.providers.local_embedding import (
    http_json,
    local_endpoint_base,
    model_digest,
    model_entry,
    validate_embeddings,
)
from qa_platform.settings import DATA_DIR


SCHEMA_VERSION = 1
APPLICATION_ID = 0x51414543
DATABASE_NAME = "embeddings.sqlite3"
LOCK_NAME = ".writer.lock"


def prepare_job(
    corpus: Path, batch_size: int = 8, cache_dir: Path | None = None
) -> dict:
    """核验冻结语料，保留完整chunk；本函数只读，不创建任何目录。"""
    require(
        type(batch_size) is int and 1 <= batch_size <= 8,
        "本步batch-size只允许1～8；8是小样本验证起点，不是服务上限",
    )
    corpus = corpus.resolve()
    plan = load_plan(corpus, batch_size, "body-only")
    base_url = local_endpoint_base(plan["profile"])
    chunk_bytes = (corpus / "chunks.jsonl").read_bytes()
    require(
        sha256(chunk_bytes).hexdigest()
        == plan["manifest"]["files"]["chunks.jsonl"]["sha256"],
        "准备期间候选文件发生变化",
    )
    chunks = [
        json.loads(line)
        for line in chunk_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    rows = plan["inputs"]["body-only"]
    require(
        [chunk["id"] for chunk in chunks] == [row["chunk_id"] for row in rows],
        "候选顺序发生变化",
    )
    scope = {
        "corpus_sha256": plan["corpus_sha256"],
        "input_snapshot_sha256": plan["input_snapshot_sha256"],
        "input_contract_sha256": plan["contract_sha256"],
        "profile": plan["profile"],
        "policy": POLICY_VERSION,
        "mode": "body-only",
        "dtype": "<f4",
        "dimension": plan["profile"]["documented_dimension"],
        "expected_count": len(rows),
    }
    directory = (
        cache_dir
        or DATA_DIR
        / "embedding_cache"
        / f"kubernetes_zh_{fingerprint(canonical(scope))[:16]}"
    ).absolute()
    return {
        "plan": plan,
        "corpus_path": corpus,
        "scope": scope,
        "rows": rows,
        "chunks": {chunk["id"]: chunk for chunk in chunks},
        "by_id": {row["chunk_id"]: row for row in rows},
        "ordinal": {row["chunk_id"]: index for index, row in enumerate(rows)},
        "cache_dir": directory,
        "path": directory / DATABASE_NAME,
        "base_url": base_url,
        "batch_size": batch_size,
    }


def check_paths(job: dict) -> None:
    directory, path = job["cache_dir"], job["path"]
    require(
        not directory.is_symlink() and not path.is_symlink(),
        "缓存目录/文件不能是软链接",
    )
    if directory.exists():
        require(directory.is_dir(), "cache-dir不是目录")
        allowed = {DATABASE_NAME, DATABASE_NAME + "-journal", LOCK_NAME}
        require(
            all(
                (
                    item.name in allowed
                    or re.fullmatch(
                        r"\.initializing-[a-z0-9_-]+\.sqlite3(?:-journal)?", item.name
                    )
                )
                and item.is_file()
                and not item.is_symlink()
                for item in directory.iterdir()
            ),
            "缓存目录含非本工具文件；不覆盖、不清理，请另选目录",
        )
    require(not path.exists() or path.is_file(), "缓存目标不是普通文件")


@contextmanager
def writer_lock(job: dict):
    """锁覆盖校验、网络与全部提交，避免两个进程先后重复编码。"""
    check_paths(job)
    job["cache_dir"].mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        job["cache_dir"] / LOCK_NAME, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "同一缓存已有写入进程；本次不请求模型，请等待后重试"
            ) from error
        check_paths(job)
        yield
    finally:
        os.close(descriptor)


@contextmanager
def connect_cache(job: dict, *, writable: bool = False, create: bool = False):
    """mode=ro防止计划/校验意外建库；rw只在持有写锁时使用。"""
    if writable and not create:
        # 先纯文件读确认类型，之后才允许SQLite恢复本工具的hot journal。
        # 60/68偏移分别是SQLite文件头的user_version/application_id。
        with job["path"].open("rb") as stream:
            header = stream.read(100)
        require(
            len(header) == 100
            and header[:16] == b"SQLite format 3\0"
            and int.from_bytes(header[68:72], "big") == APPLICATION_ID
            and int.from_bytes(header[60:64], "big") == SCHEMA_VERSION,
            "文件未完整初始化、不是本工具缓存或schema不兼容；保留原文件，请另选目录",
        )
    mode = "rwc" if create else "rw" if writable else "ro"
    connection = sqlite3.connect(
        job["path"].as_uri() + f"?mode={mode}",
        uri=True,
        isolation_level=None,
        timeout=0,
    )
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA foreign_keys=ON")
        if writable:
            connection.execute("PRAGMA synchronous=FULL")
        else:
            connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def initialize_cache(connection, job: dict, digest: str) -> None:
    """一次事务初始化；已有缓存永不DROP、REPLACE或覆盖。"""
    require(
        not connection.execute("SELECT name FROM sqlite_master").fetchall(),
        "目标数据库不是全新空库，拒绝初始化",
    )
    require(model_digest({"digest": digest}) == digest, "初始化digest格式不合法")
    require(
        connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete",
        "无法启用DELETE事务日志",
    )
    header = {
        "schema_version": SCHEMA_VERSION,
        "scope": job["scope"],
        "server_model_digest": digest,
        "status": "building",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_path_at_creation": str(job["corpus_path"]),
        "corpus_manifest": job["plan"]["manifest"],
    }
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        connection.execute(
            "CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), header_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE batches (batch_id INTEGER PRIMARY KEY, ids_json TEXT NOT NULL, request_sha TEXT NOT NULL, processed_tokens INTEGER, elapsed_seconds REAL NOT NULL, digest_before TEXT NOT NULL, digest_after TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE vectors (chunk_id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL, input_id TEXT NOT NULL, input_sha TEXT NOT NULL, record_json TEXT NOT NULL, vector BLOB NOT NULL, vector_sha TEXT NOT NULL, batch_id INTEGER NOT NULL REFERENCES batches(batch_id))"
        )
        connection.execute("INSERT INTO metadata VALUES (1, ?)", (canonical(header),))
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def create_cache(job: dict, digest: str) -> None:
    """先在临时库完成初始化，再排他发布；中途崩溃不留下正式空库。"""
    descriptor, name = tempfile.mkstemp(
        prefix=".initializing-", suffix=".sqlite3", dir=job["cache_dir"]
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        staging = {**job, "path": temporary}
        with connect_cache(staging, writable=True, create=True) as connection:
            initialize_cache(connection, job, digest)
            audit_cache(connection, job)
        os.link(temporary, job["path"])  # 目标已存在时失败，绝不覆盖。
    finally:
        temporary.unlink(missing_ok=True)


def vector_identity(row: dict, digest: str) -> str:
    return fingerprint(canonical([row["embedding_input_id"], digest, "<f4"]))


def audit_cache(connection, job: dict) -> dict:
    """读回所有已提交行；身份、原文、元数据、向量和批次账本一起核验。"""
    import numpy as np

    require(
        connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID,
        "不是本工具的缓存，拒绝使用",
    )
    require(
        connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION,
        "缓存schema版本不支持",
    )
    require(
        connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete",
        "缓存日志模式不支持；不会自动切换",
    )
    objects = {
        (row[0], row[1])
        for row in connection.execute(
            "SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
    }
    require(
        objects == {("table", "metadata"), ("table", "batches"), ("table", "vectors")},
        "缓存表结构或附加对象不符合约定",
    )
    columns = {
        "metadata": ("id", "header_json"),
        "batches": (
            "batch_id",
            "ids_json",
            "request_sha",
            "processed_tokens",
            "elapsed_seconds",
            "digest_before",
            "digest_after",
        ),
        "vectors": (
            "chunk_id",
            "ordinal",
            "input_id",
            "input_sha",
            "record_json",
            "vector",
            "vector_sha",
            "batch_id",
        ),
    }
    for table, expected in columns.items():
        require(
            tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            == expected,
            "缓存字段结构不支持",
        )
    require(
        [row[0] for row in connection.execute("PRAGMA quick_check")] == ["ok"],
        "SQLite完整性检查未通过",
    )
    headers = connection.execute("SELECT id,header_json FROM metadata").fetchall()
    require(len(headers) == 1 and headers[0]["id"] == 1, "缓存缺少唯一元信息")
    header = json.loads(headers[0]["header_json"])
    require(
        header["schema_version"] == SCHEMA_VERSION and header["scope"] == job["scope"],
        "缓存语料/模型配置/输入规则不兼容；请另选新cache-dir",
    )
    require(header["corpus_manifest"] == job["plan"]["manifest"], "缓存来源清单不一致")
    digest = model_digest({"digest": header["server_model_digest"]})
    require(
        digest == header["server_model_digest"]
        and header["status"] in ("building", "encoded_complete"),
        "缓存模型身份或状态不合法",
    )
    batches = {
        row["batch_id"]: dict(row)
        for row in connection.execute("SELECT * FROM batches ORDER BY batch_id")
    }
    require(list(batches) == list(range(1, len(batches) + 1)), "缓存批次编号缺失或非法")
    per_batch = {batch_id: [] for batch_id in batches}
    done = set()
    for saved in connection.execute("SELECT * FROM vectors ORDER BY ordinal"):
        chunk_id = saved["chunk_id"]
        require(
            chunk_id in job["by_id"] and chunk_id not in done, "缓存出现未知或重复块"
        )
        row = job["by_id"][chunk_id]
        require(
            saved["ordinal"] == job["ordinal"][chunk_id]
            and saved["input_sha"] == row["input_sha256"]
            and saved["input_id"] == vector_identity(row, digest),
            "缓存输入身份或顺序不一致",
        )
        require(
            saved["record_json"] == canonical(job["chunks"][chunk_id]),
            "缓存原文或完整元数据已变化",
        )
        blob = saved["vector"]
        require(
            isinstance(blob, bytes) and len(blob) == job["scope"]["dimension"] * 4,
            "向量字节长度不匹配",
        )
        require(sha256(blob).hexdigest() == saved["vector_sha"], "缓存向量SHA256不一致")
        vector = np.frombuffer(blob, dtype="<f4")
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        require(
            np.isfinite(vector).all() and math.isfinite(norm) and norm > 0,
            "缓存向量非有限或为零",
        )
        require(saved["batch_id"] in batches, "向量缺少所属批次")
        per_batch[saved["batch_id"]].append(chunk_id)
        done.add(chunk_id)
    for batch_id, batch in batches.items():
        ids = json.loads(batch["ids_json"])
        require(
            isinstance(ids, list) and ids and ids == per_batch[batch_id],
            "批次清单与提交向量不一致",
        )
        require(
            batch["digest_before"] == batch["digest_after"] == digest,
            "批次模型digest与缓存不一致",
        )
        require(
            batch["request_sha"]
            == fingerprint(
                canonical(
                    request_body(
                        job["plan"]["profile"], [job["by_id"][key] for key in ids]
                    )
                )
            ),
            "批次请求内容指纹不一致",
        )
        tokens, seconds = batch["processed_tokens"], batch["elapsed_seconds"]
        require(
            tokens is None or (type(tokens) is int and tokens >= 0),
            "批次Token统计不合法",
        )
        require(
            type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
            "批次耗时不合法",
        )
    missing = [row for row in job["rows"] if row["chunk_id"] not in done]
    require(
        header["status"] != "encoded_complete" or not missing,
        "缓存宣称完成但仍缺少向量",
    )
    return {
        "header": header,
        "status": header["status"],
        "done": len(done),
        "missing": missing,
        "batches": len(batches),
    }


def inspect_cache(job: dict, *, required: bool = False) -> dict:
    check_paths(job)
    if not job["path"].exists():
        require(not required, "缓存尚不存在，请先--run编码")
        return {
            "header": None,
            "status": "not_created",
            "done": 0,
            "missing": job["rows"],
            "batches": 0,
        }
    with connect_cache(job) as connection:
        connection.execute("BEGIN")  # 同一读快照，避免恰逢写入提交造成跨表不一致。
        return audit_cache(connection, job)


def commit_batch(
    connection,
    job: dict,
    group: list[dict],
    matrix,
    digest: str,
    tokens,
    seconds: float,
) -> None:
    """向量与批次记录全有或全无；不用REPLACE掩盖重复输入。"""
    import numpy as np

    ids = [row["chunk_id"] for row in group]
    require(
        tokens is None or (type(tokens) is int and tokens >= 0), "待提交Token统计不合法"
    )
    require(
        type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
        "待提交耗时不合法",
    )
    header = json.loads(
        connection.execute("SELECT header_json FROM metadata WHERE id=1").fetchone()[0]
    )
    require(
        header["server_model_digest"] == digest and header["status"] == "building",
        "提交模型或缓存状态不一致",
    )
    require(bool(ids) and len(ids) == len(set(ids)), "提交批次为空或重复")
    require(
        all(row == job["by_id"][row["chunk_id"]] for row in group),
        "提交输入不符合当前计划",
    )
    stored = np.asarray(matrix, dtype="<f4")
    require(
        stored.shape == (len(group), job["scope"]["dimension"]), "待保存向量形状错误"
    )
    require(
        np.isfinite(stored).all()
        and (np.linalg.norm(stored.astype(np.float64), axis=1) > 0).all(),
        "待保存向量非法",
    )
    connection.execute("BEGIN IMMEDIATE")
    try:
        batch_id = connection.execute(
            "SELECT COALESCE(MAX(batch_id),0)+1 FROM batches"
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO batches VALUES (?,?,?,?,?,?,?)",
            (
                batch_id,
                canonical(ids),
                fingerprint(canonical(request_body(job["plan"]["profile"], group))),
                tokens,
                seconds,
                digest,
                digest,
            ),
        )
        for row, vector in zip(group, stored):
            blob = vector.tobytes()
            connection.execute(
                "INSERT INTO vectors VALUES (?,?,?,?,?,?,?,?)",
                (
                    row["chunk_id"],
                    job["ordinal"][row["chunk_id"]],
                    vector_identity(row, digest),
                    row["input_sha256"],
                    canonical(job["chunks"][row["chunk_id"]]),
                    blob,
                    sha256(blob).hexdigest(),
                    batch_id,
                ),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def finalize_cache(connection, job: dict) -> dict:
    snapshot = audit_cache(connection, job)
    require(not snapshot["missing"], "仍有未编码块，不能发布完成状态")
    header = {**snapshot["header"], "status": "encoded_complete"}
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "UPDATE metadata SET header_json=? WHERE id=1", (canonical(header),)
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return audit_cache(connection, job)


def show_plan(job: dict, snapshot: dict, max_batches: int | None = None):
    total = len(job["rows"])
    full = math.ceil(total / job["batch_size"])
    remaining = math.ceil(len(snapshot["missing"]) / job["batch_size"])
    planned = min(remaining, max_batches) if max_batches is not None else remaining
    print(
        f"[配置] Python：{sys.executable}；本地模型：{job['plan']['profile']['model']}；{job['plan']['profile']['endpoint']}"
    )
    print(
        f"[范围] {total}块；body-only；每批最多{job['batch_size']}条；float32/{job['scope']['dimension']}维。"
    )
    print(
        f"[缓存] {job['path']}；状态={snapshot['status']}；已提交并核验{snapshot['done']}块；待编码{len(snapshot['missing'])}块。"
    )
    print(
        f"[对照] 从零编码全部={full}次Embedding；复用当前缓存后剩余={remaining}次；本轮最多新增提交{planned}批。"
    )
    print(
        f"[条件计划] 本轮正常完成：Embedding {planned}次 + 模型digest元数据 {2 * planned}次 = HTTP {3 * planned}次；无自动重试。"
    )
    print(
        "[边界] 已完成复跑0 HTTP只做离线缓存校验，不证明此刻服务器模型未变；查询/入Milvus前仍须检查模型契约。"
    )
    print(
        "[保存] SQLite仅作检查点；每批原文/完整元数据/向量/批次账本一起提交；不读.env、不回退云端、不调用重排/聊天/Milvus。",
        flush=True,
    )


def run_job(
    job: dict,
    *,
    timeout: int = 180,
    max_batches: int | None = None,
    session_factory=None,
) -> dict:
    """持锁后才计算缺失；注入假Session可离线测试，不需要真实模型。"""
    require(
        max_batches is None or (type(max_batches) is int and max_batches > 0),
        "max-batches必须为正整数",
    )
    stats = {"embedding_requests": 0, "http_requests": 0}
    with writer_lock(job):
        existing = job["path"].exists()
        if existing:
            with connect_cache(job, writable=True) as connection:
                snapshot = audit_cache(connection, job)
                show_plan(job, snapshot, max_batches)
                if not snapshot["missing"]:
                    result = (
                        finalize_cache(connection, job)
                        if snapshot["status"] != "encoded_complete"
                        else snapshot
                    )
                    print(
                        "[缓存命中] 全部有效，无需创建HTTP客户端；本轮Embedding=0、元数据=0。"
                    )
                    return {
                        **stats,
                        "status": result["status"],
                        "done": result["done"],
                        "newly_encoded": 0,
                    }
        else:
            snapshot = inspect_cache(job)
            show_plan(job, snapshot, max_batches)
        if session_factory is None:
            import requests

            session_factory = requests.Session
        profile = job["plan"]["profile"]

        def get_digest(session):
            stats["http_requests"] += 1
            print(
                f"[元数据] 本轮HTTP第{stats['http_requests']}次：读取/api/tags核对模型digest。",
                flush=True,
            )
            return model_digest(
                model_entry(
                    http_json(
                        session, "GET", job["base_url"] + "/api/tags", timeout=15
                    ),
                    profile["model"],
                )
            )

        newly_encoded = 0
        with session_factory() as session:
            session.trust_env = False
            # 首批之前的读取兼作新缓存身份绑定；后续每批仍各读前/后一次。
            before = get_digest(session)
            if existing:
                require(
                    before == snapshot["header"]["server_model_digest"],
                    "服务器模型digest与缓存不同；拒绝续写，请另选新cache-dir",
                )
            if not existing:
                create_cache(job, before)
            with connect_cache(job, writable=True) as connection:
                audit_cache(connection, job)
                digest = before
                missing = snapshot["missing"]
                count = math.ceil(len(missing) / job["batch_size"])
                allowed = min(count, max_batches) if max_batches is not None else count
                for index in range(allowed):
                    group = missing[
                        index * job["batch_size"] : (index + 1) * job["batch_size"]
                    ]
                    ids = [row["chunk_id"] for row in group]
                    try:
                        if index:
                            before = get_digest(session)
                        require(before == digest, "批次前模型digest已变化，停止")
                        print(
                            f"[请求 {index + 1}/{allowed}] {len(group)}块；首块={ids[0]}；truncate=false。",
                            flush=True,
                        )
                        stats["embedding_requests"] += 1
                        stats["http_requests"] += 1
                        started = perf_counter()
                        data = http_json(
                            session,
                            "POST",
                            profile["endpoint"],
                            payload=request_body(profile, group),
                            timeout=timeout,
                        )
                        seconds = perf_counter() - started
                        matrix, _ = validate_embeddings(data, len(group), profile)
                        tokens = data.get("prompt_eval_count")
                        require(
                            tokens is None or (type(tokens) is int and tokens >= 0),
                            "服务Token统计不合法",
                        )
                        after = get_digest(session)
                        require(
                            before == after == digest,
                            "批次后模型digest已变化；当前批不提交",
                        )
                        commit_batch(
                            connection, job, group, matrix, digest, tokens, seconds
                        )
                    except BaseException:
                        print(
                            f"[本批中止] 块ID={', '.join(ids)}；提交状态以下次离线读回为准，已提交批次保留；本轮已尝试Embedding={stats['embedding_requests']}、HTTP={stats['http_requests']}。",
                            file=sys.stderr,
                            flush=True,
                        )
                        raise
                    newly_encoded += len(group)
                    print(
                        f"[已提交] {snapshot['done'] + newly_encoded}/{len(job['rows'])}块；本批服务Token={tokens if tokens is not None else '未知'}；请求耗时={seconds:.3f}s。",
                        flush=True,
                    )
                result = audit_cache(connection, job)
                if not result["missing"]:
                    result = finalize_cache(connection, job)
        print(
            f"[{'编码完成' if result['status'] == 'encoded_complete' else '阶段暂停'}] 已提交{result['done']}/{len(job['rows'])}块；本轮新增{newly_encoded}块；Embedding={stats['embedding_requests']}，HTTP={stats['http_requests']}。"
        )
        print(
            "[结论边界] digest前后观测一致与缓存指纹不代表语义质量；未改原语料ready、未入Milvus；当前仅支持同一语料/输入合同续跑。"
        )
        return {
            **stats,
            "status": result["status"],
            "done": result["done"],
            "newly_encoded": newly_encoded,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run", action="store_true", help="真实请求本地模型并创建/恢复缓存"
    )
    mode.add_argument(
        "--verify", action="store_true", help="纯离线核验已有缓存（可以尚未完成）"
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument(
        "--cache-dir", type=Path, help="默认按当前语料与输入合同指纹选择独立目录"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-batches", type=int, help="本轮最多新增提交几批，用于观察续跑"
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8 or not 1 <= args.timeout <= 600:
        parser.error("batch-size须为1～8，timeout须为1～600")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max-batches须为正整数")
    if args.verify and args.max_batches is not None:
        parser.error("verify不接受max-batches")
    job = prepare_job(args.corpus, args.batch_size, args.cache_dir)
    if args.run:
        try:
            run_job(job, timeout=args.timeout, max_batches=args.max_batches)
        except KeyboardInterrupt:
            print(
                "[中断] 已提交批次保留；重跑先离线核验，再决定当前批次是否需要重新编码。",
                file=sys.stderr,
            )
            return 130
        except Exception as error:
            # 不打印HTTP错误正文、向量或用户原文；保留真正的异常类型供诊断。
            print(
                f"[停止] {type(error).__name__}: {error if isinstance(error, (ValueError, sqlite3.Error)) else '请求/运行失败，请检查本地服务或网络'}；不自动重试、不回退云端、不删除缓存。",
                file=sys.stderr,
            )
            return 1
    else:
        snapshot = inspect_cache(job, required=args.verify)
        show_plan(job, snapshot, args.max_batches)
        print(
            f"[{'离线校验通过' if args.verify else '计划完成'}] 本轮实际HTTP=0；未创建HTTP客户端、未写缓存；{'部分缓存尚未编码完成' if snapshot['missing'] else '全部行已核验，尚不代表检索质量'}。"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        SyntaxError,
        OverflowError,
        sqlite3.Error,
    ) as error:
        print(
            f"[缓存检查失败] {error}；未请求模型，不自动修复或覆盖。", file=sys.stderr
        )
        raise SystemExit(2)
