"""第十二步：测试知识块的真实向量入库；默认只展示计划，不调用模型。"""

import argparse
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from math import isclose
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

import httpx
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document

from qa_platform.ingestion.railway import digest, load_records, split_records
from qa_platform.providers.cloud_embedding import API_URL, MODEL_NAME
from qa_platform.retrieval.dense import (
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
    embed_documents,
    validate_vectors,
)
from qa_platform.settings import ENV_PATH, PACKAGE_ROOT, PROJECT_ROOT
from qa_platform.storage.milvus_smoke import (
    VERIFIED_VERSIONS,
    installed_versions,
    require,
)


COLLECTION = "knowledge_chunks_v1"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "milvus_indexes" / "test_v1"
SCHEMA_VERSION = 1
TEXT_BYTES = 65535
METADATA_BYTES = 60000  # 保守预检上限，不静默删减来源、父记录或文档说明。
SCALAR_FIELDS = ["kb_id", "is_test", "review_status"]
OUTPUT_FIELDS = ["chunk_id", "text", *SCALAR_FIELDS, "metadata", "vector"]


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save_new_json(path: Path, value: dict) -> None:
    """完整写入临时文件后排他发布；已有文件不会被覆盖，避免半份ready标记。"""
    encoded = json_text(value) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".building-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def code_versions() -> dict[str, str]:
    """对真实业务模块取指纹，供跨进程构建与验证阶段核对。"""
    names = [
        "storage/milvus_build.py",
        "ingestion/railway.py",
        "retrieval/dense.py",
        "providers/cloud_embedding.py",
        "storage/milvus_smoke.py",
        "settings.py",
    ]
    return {
        name: sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest() for name in names
    }


def embedding_contract() -> dict:
    return {
        "model": MODEL_NAME,
        "endpoint": API_URL,
        "model_revision": None,
        "document_text_type": "document",
        "query_text_type": "query",
        "dtype": "float32",
        "metric_type": "COSINE",
        "index_type": "FLAT",
    }


def document_rows(documents: list[Document]) -> list[dict]:
    """在付费请求前检查文本、身份、元数据；完整保留原始来源身份。"""
    require(bool(documents), "没有可入库的知识块。")
    rows, seen = [], set()
    for document in documents:
        metadata = document.metadata
        chunk_id = metadata.get("chunk_id")
        require(
            isinstance(chunk_id, str) and 0 < len(chunk_id.encode()) <= 64,
            "chunk_id必须为不超过64字节的字符串。",
        )
        require(
            document.id == chunk_id and chunk_id not in seen,
            "块身份不一致或存在重复chunk_id。",
        )
        require(
            bool(document.page_content.strip())
            and len(document.page_content.encode()) <= TEXT_BYTES,
            "知识块为空或超过text字段字节上限。",
        )
        require(
            metadata.get("content_hash") == digest(document.page_content),
            "知识块正文与内容hash不一致。",
        )
        require(
            metadata.get("kb_id") == "test" and metadata.get("is_test") is True,
            "本步只允许测试知识库，必须保留is_test=True。",
        )
        require(
            metadata.get("review_status") == "test_data", "测试资料审核标记不一致。"
        )
        for field, limit in [("kb_id", 32), ("review_status", 32)]:
            require(
                isinstance(metadata[field], str)
                and 0 < len(metadata[field].encode()) <= limit,
                f"{field}超过字段上限。",
            )
        # 两种序列化都检查，避免SDK使用ASCII转义后字节数大幅增加。
        require(
            len(json_text(metadata).encode()) <= METADATA_BYTES,
            "元数据超过保守字节上限。",
        )
        require(
            len(json.dumps(metadata, ensure_ascii=True, allow_nan=False).encode())
            <= METADATA_BYTES,
            "ASCII转义后的元数据超过保守字节上限。",
        )
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": document.page_content,
                **{field: metadata[field] for field in SCALAR_FIELDS},
                "metadata": json.loads(json_text(metadata)),
            }
        )
        seen.add(chunk_id)
    return rows


def make_plan(documents: list[Document], args: argparse.Namespace) -> dict:
    rows = document_rows(documents)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "building",
        "created_at": now(),
        "collection": COLLECTION,
        "database": "knowledge.db",
        "kb_id": "test",
        "embedding": embedding_contract(),
        "chunking": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        },
        "batch_size": args.batch_size,
        "document_count": len(rows),
        "document_snapshot_sha256": sha256(json_text(rows).encode()).hexdigest(),
        "knowledge_versions": {
            row["metadata"]["source"]: row["metadata"]["document_version"]
            for row in rows
        },
        "packages": {
            name: version(name)
            for name in ["numpy", "httpx", "langchain-core", "langchain-text-splitters"]
        },
        "milvus_packages": installed_versions(),
        "code_sha256": code_versions(),
        "documents": rows,
    }


def load_plan(index_dir: Path) -> dict:
    plan = json.loads((index_dir / "build_plan.json").read_text())
    require(
        plan["schema_version"] == SCHEMA_VERSION and plan["collection"] == COLLECTION,
        "索引构建格式不兼容。",
    )
    require(
        plan["database"] == "knowledge.db" and plan["kb_id"] == "test",
        "数据库路径或知识库范围不匹配。",
    )
    require(
        plan["embedding"] == embedding_contract(),
        "Embedding模型、服务或编码约定已变化，拒绝混用。",
    )
    require(
        plan["code_sha256"] == code_versions(),
        "构建期间代码已变化，请使用新目录重新构建。",
    )
    require(
        plan["milvus_packages"] == installed_versions() == VERIFIED_VERSIONS,
        "Milvus依赖版本不匹配。",
    )
    documents = [
        Document(id=row["chunk_id"], page_content=row["text"], metadata=row["metadata"])
        for row in plan["documents"]
    ]
    require(
        document_rows(documents) == plan["documents"],
        "正文、显式字段与元数据副本不一致。",
    )
    require(len(documents) == plan["document_count"], "计划条数不一致。")
    require(
        sha256(json_text(plan["documents"]).encode()).hexdigest()
        == plan["document_snapshot_sha256"],
        "知识快照hash不一致。",
    )
    require(
        type(plan["batch_size"]) is int and 1 <= plan["batch_size"] <= MAX_BATCH_SIZE,
        "编码批量大小不合法。",
    )
    return plan


def vector_hash(vector: np.ndarray) -> str:
    return sha256(np.asarray(vector, dtype="<f4").tobytes()).hexdigest()


def write_index(index_dir: Path) -> None:
    """构建子进程：仅编码文档，不编码问题；密钥由父进程加载到环境中。"""
    plan = load_plan(index_dir)
    db_path = index_dir / "knowledge.db"
    require(
        not db_path.exists() and not db_path.is_symlink(),
        "数据库已存在，拒绝覆盖或重新编码。",
    )
    require(
        not (index_dir / "build_result.json").exists(), "已有写入结果，拒绝重复构建。"
    )
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    require(bool(api_key), "构建进程缺少DASHSCOPE_API_KEY；不会打印密钥。")
    documents = [
        Document(id=row["chunk_id"], page_content=row["text"], metadata=row["metadata"])
        for row in plan["documents"]
    ]
    from pymilvus import DataType, MilvusClient

    # 先确认本地SDK能导入、数据库能连接，再产生文档编码费用。
    client = MilvusClient(uri=str(db_path), timeout=20)
    try:
        require(not client.has_collection(COLLECTION), "集合已存在，拒绝覆盖。")
        with httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        ) as embedding_client:
            vectors = embed_documents(embedding_client, documents, plan["batch_size"])
        # FLOAT_VECTOR保存float32；转换后仍需排除溢出、非有限数和零向量。
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            vectors = vectors.astype(np.float32)
        validate_vectors(vectors, len(documents))
        dimension = int(vectors.shape[1])
        require(dimension >= 2, "实际Embedding维度不足2，不能建立该向量字段。")
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=TEXT_BYTES)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=32)
        schema.add_field("is_test", DataType.BOOL)
        schema.add_field("review_status", DataType.VARCHAR, max_length=32)
        schema.add_field("metadata", DataType.JSON)
        index = client.prepare_index_params()
        index.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_collection(
            collection_name=COLLECTION, schema=schema, index_params=index
        )
        rows = [
            {**row, "vector": vector.tolist()}
            for row, vector in zip(plan["documents"], vectors)
        ]
        result = client.insert(collection_name=COLLECTION, data=rows)
        require(result["insert_count"] == len(rows), "Milvus写入条数不正确。")
        client.flush(collection_name=COLLECTION)
        save_new_json(
            index_dir / "build_result.json",
            {
                "status": "written_not_verified",
                "written_at": now(),
                "dimension": dimension,
                "dimension_source": "embedding_response",
                "document_count": len(rows),
                "document_snapshot_sha256": plan["document_snapshot_sha256"],
                "vector_hash_encoding": "little_endian_float32",
                "vector_sha256": {
                    row["chunk_id"]: vector_hash(vector)
                    for row, vector in zip(rows, vectors)
                },
            },
        )
        print(
            f"[写入] PID={os.getpid()}；{len(rows)}块；实际维度={dimension}；float32；已flush，尚未标记ready。",
            flush=True,
        )
    finally:
        client.close()


def verify_index(index_dir: Path) -> None:
    """验证子进程：不读取密钥、不请求Embedding，核对全部记录与文档向量自检索。"""
    plan = load_plan(index_dir)
    result = json.loads((index_dir / "build_result.json").read_text())
    require(
        result["document_snapshot_sha256"] == plan["document_snapshot_sha256"],
        "写入结果与计划快照不一致。",
    )
    require(result["document_count"] == plan["document_count"], "写入结果条数不一致。")
    db_path = index_dir / "knowledge.db"
    require(
        db_path.is_dir() and not db_path.is_symlink(), "缺少新版Milvus Lite数据库目录。"
    )
    from pymilvus import MilvusClient

    client = MilvusClient(uri=str(db_path), timeout=20)
    try:
        require(bool(client.has_collection(COLLECTION)), "持久化集合不存在。")
        client.load_collection(collection_name=COLLECTION)
        require(
            int(client.get_collection_stats(COLLECTION)["row_count"])
            == plan["document_count"],
            "持久化记录总数不符。",
        )
        ids = [row["chunk_id"] for row in plan["documents"]]
        records = client.get(
            collection_name=COLLECTION, ids=ids, output_fields=OUTPUT_FIELDS
        )
        require(len(records) == len(ids), "未能找回全部主键。")
        by_id = {record["chunk_id"]: record for record in records}
        require(set(by_id) == set(ids), "持久化主键集合不一致。")
        for expected in plan["documents"]:
            actual = by_id[expected["chunk_id"]]
            require(
                all(actual[field] == value for field, value in expected.items()),
                "持久化原文、元数据或测试标记发生变化。",
            )
            vector = np.asarray(actual["vector"], dtype=np.float32)
            validate_vectors(vector[np.newaxis, :], 1, result["dimension"])
            require(
                vector_hash(vector) == result["vector_sha256"][expected["chunk_id"]],
                "持久化向量与写入向量指纹不一致。",
            )
        first_vector = by_id[ids[0]]["vector"]
        hits = client.search(
            collection_name=COLLECTION,
            data=[first_vector],
            anns_field="vector",
            limit=len(ids),
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["chunk_id"],
        )
        require(
            len(hits) == 1 and len(hits[0]) == len(ids),
            "文档向量自检索结果数量不正确。",
        )
        # 本SDK使用实际主键字段名；不能照抄演示集合的默认"id"。
        own_hits = [hit for hit in hits[0] if hit["chunk_id"] == ids[0]]
        require(
            len(own_hits) == 1
            and isclose(float(own_hits[0]["distance"]), 1.0, abs_tol=1e-5),
            "文档向量自检索未返回自身或分数异常。",
        )
        print(
            f"[验证] PID={os.getpid()}；全部{len(ids)}条原文、元数据和float32向量指纹一致。",
            flush=True,
        )
        print(
            "[验证] 已保存文档向量的自检索通过；未编码问题，不能据此宣称问句检索质量通过。",
            flush=True,
        )
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="测试知识库真实向量入库；默认只展示计划"
    )
    parser.add_argument(
        "--run", action="store_true", help="显式调用文档Embedding并新建索引"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="必须是不存在的新目录，绝不覆盖已有索引",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--_phase", choices=["build", "verify"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    index_dir = args.index_dir.expanduser().absolute()
    if args._phase:
        if not args.run:
            parser.error("内部阶段必须显式指定--run")
        try:
            phase = write_index if args._phase == "build" else verify_index
            phase(index_dir)
        except httpx.TimeoutException:
            print("[错误] Embedding请求超时；未标记ready。", file=sys.stderr)
            return 1
        except httpx.HTTPStatusError as exc:
            print(
                f"[错误] Embedding服务返回HTTP {exc.response.status_code}；不显示响应体。",
                file=sys.stderr,
            )
            return 1
        except httpx.RequestError:
            print("[错误] Embedding连接失败；不显示请求信息。", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[错误] {type(exc).__name__}：{exc}", file=sys.stderr)
            return 1
        return 0
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("必须满足chunk_size > 0且0 <= chunk_overlap < chunk_size")
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"batch-size必须在1到{MAX_BATCH_SIZE}之间")
    try:
        chunks = split_records(
            load_records("test"), args.chunk_size, args.chunk_overlap
        )
        plan = make_plan(chunks, args)
    except (OSError, ValueError, TypeError) as exc:
        print(f"[错误] 本地准备失败：{exc}", file=sys.stderr)
        return 1
    requests = (len(chunks) + args.batch_size - 1) // args.batch_size
    exists = index_dir.exists() or index_dir.is_symlink()
    print(f"[配置] Python：{sys.executable}")
    print(f"[配置] Embedding：{MODEL_NAME}；地址：{API_URL}")
    print(f"[索引] 测试知识块{len(chunks)}个；目标：{index_dir}")
    print(f"[计划] 文档Embedding {requests}次；问题Embedding/Reranker/聊天模型均0次。")
    print("[计划] 实测维度→float32→Milvus新集合→另一进程读回验证→发布ready清单。")
    print("[边界] 测试资料仍是虚构资料；不是现场维护依据。现有RAG尚未改用这个索引。")
    print(
        "[说明] 本脚本仅构建新索引；已存在目录（包括失败目录）不覆盖、不追加、不重新编码。"
    )
    if exists:
        print("[保护] 目标路径已存在；执行--run将被拒绝。")
    if not args.run:
        print(
            "[计划完成] 未读取密钥、未创建HTTP/Milvus客户端、未创建文件；加--run才执行。"
        )
        return 0
    if exists:
        print(
            "[错误] 请选择新的--index-dir；本次未读取密钥或发送请求。", file=sys.stderr
        )
        return 1
    if installed_versions() != VERIFIED_VERSIONS:
        print(f"[错误] Milvus依赖应为：{VERIFIED_VERSIONS}。", file=sys.stderr)
        return 1
    if not ENV_PATH.is_file():
        print(f"[错误] 配置文件不存在：{ENV_PATH}", file=sys.stderr)
        return 1
    load_dotenv(ENV_PATH, override=False)
    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        print("[错误] 未配置DASHSCOPE_API_KEY；不显示密钥。", file=sys.stderr)
        return 1
    created = False
    try:
        index_dir.mkdir(parents=True, exist_ok=False)
        created = True
        save_new_json(index_dir / "build_plan.json", plan)
        for phase, timeout in [("build", max(120, requests * 65 + 45)), ("verify", 45)]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_milvus.py"),
                    "--run",
                    "--_phase",
                    phase,
                    "--index-dir",
                    str(index_dir),
                ],
                check=False,
                timeout=timeout,
            )
            require(result.returncode == 0, f"{phase}阶段失败，索引不可用。")
        written = json.loads((index_dir / "build_result.json").read_text())
        manifest = {key: value for key, value in plan.items() if key != "documents"}
        manifest.update(
            status="ready",
            verified_at=now(),
            embedding={
                **plan["embedding"],
                "dimension": written["dimension"],
                "dimension_source": "embedding_response",
            },
            verification="cross_process_readback_and_document_vector_self_search",
        )
        save_new_json(index_dir / "manifest.json", manifest)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        if created:
            print(
                f"[保留] {index_dir}；未发布ready标记，不得作为可用索引。请另选新目录重试。",
                file=sys.stderr,
            )
        return 1
    print(f"[成功] 测试知识库真实向量入库完成：{index_dir / 'manifest.json'}")
    print("[下一步] 查询脚本将复用文档向量；本步尚未接入问题检索或替换原RAG。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
