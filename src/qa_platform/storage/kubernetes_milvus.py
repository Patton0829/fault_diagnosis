"""把已完成的Kubernetes编码缓存导入独立Milvus索引；不重新编码。

默认纯离线计划；--run仅请求一次模型digest元数据，然后跨进程写入/读回。
不读取.env，不做推理、不调用重排或聊天，不覆盖已有索引。
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isclose, isfinite
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from qa_platform.ingestion.embedding_plan import CORPUS, canonical, fingerprint, require
from qa_platform.providers.local_embedding import http_json, model_digest, model_entry
from qa_platform.settings import DATA_DIR, PACKAGE_ROOT, PROJECT_ROOT
from qa_platform.storage import embedding_cache
from qa_platform.storage.milvus_smoke import VERIFIED_VERSIONS, installed_versions


FORMAT_VERSION = 1
COLLECTION = "kubernetes_chunks_v1"
DATABASE = "knowledge.db"
DEFAULT_INDEX_DIR = DATA_DIR / "milvus_indexes" / "kubernetes_zh_v1"
TEXT_BYTES = 65535
METADATA_BYTES = 60000
SCALARS = {
    "kb_id": 32,
    "parent_id": 64,
    "snapshot_commit": 40,
    "review_status": 64,
}
OUTPUT_FIELDS = ["chunk_id", "text", *SCALARS, "metadata", "vector"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_new_json(path: Path, value: dict) -> None:
    """在自己的新目录内排他发布完整JSON，绝不覆盖已有文件。"""
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
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def code_versions() -> dict:
    names = [
        "storage/kubernetes_milvus.py",
        "storage/embedding_cache.py",
        "storage/milvus_smoke.py",
        "ingestion/embedding_plan.py",
        "providers/local_embedding.py",
        "settings.py",
    ]
    return {
        name: sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest() for name in names
    }


def read_cache(job: dict) -> dict:
    """同一个只读事务内先完整审计，再读取向量，避免跨快照拼接。"""
    import numpy as np

    embedding_cache.check_paths(job)
    require(job["path"].is_file(), "编码缓存不存在，请先完成build_k8s_embeddings")
    with embedding_cache.connect_cache(job) as connection:
        connection.execute("BEGIN")
        snapshot = embedding_cache.audit_cache(connection, job)
        require(
            snapshot["status"] == "encoded_complete" and not snapshot["missing"],
            "编码缓存尚未完成；本脚本不会替你编码缺失块",
        )
        saved = connection.execute("SELECT * FROM vectors ORDER BY ordinal").fetchall()
        records, vectors, hashes = [], [], {}
        for row in saved:
            chunk = json.loads(row["record_json"])
            text, metadata, chunk_id = (
                chunk["page_content"],
                chunk["metadata"],
                chunk["id"],
            )
            require(
                metadata["chunk_id"] == chunk_id
                and isinstance(chunk_id, str)
                and 0 < len(chunk_id.encode()) <= 64,
                "块主键不合法",
            )
            require(
                isinstance(text, str)
                and text.strip()
                and len(text.encode()) <= TEXT_BYTES,
                "块正文为空或超过Milvus字段字节上限；不会截断",
            )
            for field, limit in SCALARS.items():
                require(
                    isinstance(metadata.get(field), str)
                    and 0 < len(metadata[field].encode()) <= limit,
                    f"元数据字段{field}缺失或超出字节上限",
                )
            require(metadata["kb_id"] == "kubernetes_zh", "只允许独立Kubernetes语料")
            require(
                len(metadata["snapshot_commit"]) == 40
                and all(c in "0123456789abcdef" for c in metadata["snapshot_commit"]),
                "上游commit身份不合法",
            )
            require(
                max(
                    len(canonical(metadata).encode()),
                    len(
                        json.dumps(
                            metadata, ensure_ascii=True, allow_nan=False
                        ).encode()
                    ),
                )
                <= METADATA_BYTES,
                "完整元数据超过保守字节上限；不会静默删除来源字段",
            )
            records.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    **{field: metadata[field] for field in SCALARS},
                    "metadata": metadata,
                }
            )
            vectors.append(np.frombuffer(row["vector"], dtype="<f4").copy())
            hashes[chunk_id] = row["vector_sha"]
    return {
        "snapshot": snapshot,
        "records": records,
        "vectors": vectors,
        "hashes": hashes,
    }


def prepare_import(
    corpus: Path = CORPUS, cache_dir: Path | None = None
) -> tuple[dict, dict]:
    """只读冻结语料和已完成缓存；无SDK导入、HTTP或文件写入。"""
    job = embedding_cache.prepare_job(corpus, cache_dir=cache_dir)
    bundle = read_cache(job)
    profile = job["plan"]["profile"]
    digest = bundle["snapshot"]["header"]["server_model_digest"]
    records = bundle["records"]
    require(job["scope"]["dimension"] >= 2, "向量维度不足2")
    plan = {
        "format_version": FORMAT_VERSION,
        "collection": COLLECTION,
        "database": DATABASE,
        "kb_id": "kubernetes_zh",
        "document_count": len(records),
        "corpus_path": str(job["corpus_path"]),
        "cache_path": str(job["path"]),
        "cache_scope": job["scope"],
        "embedding": {
            **profile,
            "server_model_digest": digest,
            "actual_dimension": job["scope"]["dimension"],
            "dtype": "little_endian_float32",
            "document_mode": "body-only",
            "index_type": "FLAT",
            "metric_type": "COSINE",
            "status": "cached_vectors_validated_not_query_quality_verified",
        },
        "documents": records,
        "document_snapshot_sha256": fingerprint(canonical(records)),
        "vector_sha256": bundle["hashes"],
        "vector_snapshot_sha256": fingerprint(canonical(bundle["hashes"])),
        "milvus_packages": installed_versions(),
        "code_sha256": code_versions(),
    }
    return plan, bundle


def load_build(index_dir: Path) -> tuple[dict, dict]:
    saved = json.loads((index_dir / "build_plan.json").read_text())
    require(saved.get("format_version") == FORMAT_VERSION, "入库计划格式不兼容")
    plan, bundle = prepare_import(
        Path(saved["corpus_path"]), Path(saved["cache_path"]).parent
    )
    require(saved == plan, "语料、缓存、模型合同、依赖或代码已变化，拒绝继续构建")
    require(plan["milvus_packages"] == VERIFIED_VERSIONS, "Milvus依赖版本不符")
    return plan, bundle


def check_new_target(index_dir: Path) -> None:
    require(
        not index_dir.exists() and not index_dir.is_symlink(),
        "索引目标已存在（可能是有效库或失败现场）；不覆盖、不追加，请另选--index-dir",
    )


def check_target_scope(plan: dict, index_dir: Path) -> None:
    """禁止把新索引放进严格只读的缓存/语料包，避免污染后续校验。"""
    target = index_dir.resolve()
    for source in (Path(plan["cache_path"]).parent, Path(plan["corpus_path"])):
        require(
            not target.is_relative_to(source.resolve()),
            "index-dir不能位于编码缓存或冻结语料包内部；请选择独立索引目录",
        )


def check_live_digest(plan: dict, session_factory=None) -> dict:
    """只GET一次/api/tags；不发送正文，不创建任何推理请求。"""
    if session_factory is None:
        import requests

        session_factory = requests.Session
    with session_factory() as session:
        session.trust_env = False
        base = embedding_cache.local_endpoint_base(plan["embedding"])
        actual = model_digest(
            model_entry(
                http_json(session, "GET", base + "/api/tags", timeout=15),
                plan["embedding"]["model"],
            )
        )
    require(
        actual == plan["embedding"]["server_model_digest"],
        "服务器当前模型digest与缓存不同；未建索引，不自动重编码或混用",
    )
    return {"checked_at_utc": now(), "observed_digest": actual, "http_requests": 1}


def write_index(index_dir: Path) -> None:
    plan, bundle = load_build(index_dir)
    db_path = index_dir / DATABASE
    check_new_target(db_path)
    require(not (index_dir / "build_result.json").exists(), "已有写入结果，不重复插入")
    from pymilvus import DataType, MilvusClient

    client = MilvusClient(uri=str(db_path), timeout=30)
    try:
        require(not client.has_collection(COLLECTION), "目标集合已经存在")
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=TEXT_BYTES)
        schema.add_field(
            "vector", DataType.FLOAT_VECTOR, dim=plan["embedding"]["actual_dimension"]
        )
        for field, limit in SCALARS.items():
            schema.add_field(field, DataType.VARCHAR, max_length=limit)
        schema.add_field("metadata", DataType.JSON)
        index = client.prepare_index_params()
        index.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_collection(
            collection_name=COLLECTION, schema=schema, index_params=index
        )
        # 插入批量只改变本地写入分组，不改变向量或重新请求模型。
        inserted = 0
        for start in range(0, len(bundle["records"]), 32):
            rows = [
                {**row, "vector": vector.tolist()}
                for row, vector in zip(
                    bundle["records"][start : start + 32],
                    bundle["vectors"][start : start + 32],
                )
            ]
            result = client.insert(collection_name=COLLECTION, data=rows)
            require(result["insert_count"] == len(rows), "Milvus本批写入数量不符")
            inserted += len(rows)
        client.flush(collection_name=COLLECTION)
        save_new_json(
            index_dir / "build_result.json",
            {
                "status": "written_not_verified",
                "writer_pid": os.getpid(),
                "document_count": inserted,
                "document_snapshot_sha256": plan["document_snapshot_sha256"],
                "vector_snapshot_sha256": plan["vector_snapshot_sha256"],
            },
        )
        print(
            f"[写入] PID={os.getpid()}；复用{inserted}个缓存向量；已flush，尚未ready。",
            flush=True,
        )
    finally:
        client.close()


def verify_index(index_dir: Path) -> None:
    """独立进程重开，全部读回；用已有文档向量对照NumPy，不编码问题。"""
    import numpy as np

    plan, bundle = load_build(index_dir)
    written = json.loads((index_dir / "build_result.json").read_text())
    require(
        written["status"] == "written_not_verified"
        and written["writer_pid"] != os.getpid()
        and written["document_count"] == plan["document_count"]
        and all(
            written[key] == plan[key]
            for key in ("document_snapshot_sha256", "vector_snapshot_sha256")
        ),
        "写入回执不符或验证没有换进程",
    )
    db_path = index_dir / DATABASE
    require(db_path.is_dir() and not db_path.is_symlink(), "缺少Milvus Lite数据库目录")
    from pymilvus import DataType, MilvusClient

    client = MilvusClient(uri=str(db_path), timeout=30)
    try:
        require(client.has_collection(COLLECTION), "重开后集合丢失")
        description = client.describe_collection(collection_name=COLLECTION)
        fields = {field["name"]: field for field in description["fields"]}
        expected_types = {
            "chunk_id": DataType.VARCHAR,
            "text": DataType.VARCHAR,
            "vector": DataType.FLOAT_VECTOR,
            **{field: DataType.VARCHAR for field in SCALARS},
            "metadata": DataType.JSON,
        }
        require(
            set(fields) == set(expected_types)
            and all(
                fields[name]["type"] == kind for name, kind in expected_types.items()
            )
            and fields["chunk_id"].get("is_primary") is True
            and description["auto_id"] is False
            and description["enable_dynamic_field"] is False
            and int(fields["vector"]["params"]["dim"])
            == plan["embedding"]["actual_dimension"],
            "集合实际字段、主键或维度不符",
        )
        names = client.list_indexes(collection_name=COLLECTION)
        require(len(names) == 1, "实际向量索引数量不符")
        actual_index = client.describe_index(
            collection_name=COLLECTION, index_name=names[0]
        )
        require(
            actual_index["field_name"] == "vector"
            and actual_index["index_type"] == "FLAT"
            and actual_index["metric_type"] == "COSINE",
            "集合实际索引不是FLAT/COSINE",
        )
        client.load_collection(collection_name=COLLECTION)
        require(
            int(client.get_collection_stats(COLLECTION)["row_count"])
            == len(bundle["records"]),
            "数据库记录总数不符",
        )
        ids = [row["chunk_id"] for row in bundle["records"]]
        records = client.get(
            collection_name=COLLECTION, ids=ids, output_fields=OUTPUT_FIELDS
        )
        by_id = {row["chunk_id"]: row for row in records}
        require(len(records) == len(ids) and set(by_id) == set(ids), "读回主键集合不符")
        for expected in bundle["records"]:
            actual = by_id[expected["chunk_id"]]
            require(
                all(actual.get(key) == value for key, value in expected.items()),
                "读回原文或完整元数据不一致",
            )
            vector = np.asarray(actual["vector"], dtype="<f4")
            require(
                vector.shape == (plan["embedding"]["actual_dimension"],), "读回维度不符"
            )
            require(
                sha256(vector.tobytes()).hexdigest()
                == plan["vector_sha256"][expected["chunk_id"]],
                "读回向量指纹不符",
            )
        matrix = np.asarray(bundle["vectors"], dtype=np.float64)
        query = matrix[0]
        cosine = (
            matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        )
        hits = client.search(
            collection_name=COLLECTION,
            data=[query.tolist()],
            anns_field="vector",
            limit=len(ids),
            filter='kb_id == "kubernetes_zh"',
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["chunk_id"],
        )
        require(len(hits) == 1 and len(hits[0]) == len(ids), "自检索结果数不符")
        scores = {hit["chunk_id"]: float(hit["distance"]) for hit in hits[0]}
        require(set(scores) == set(ids), "自检索主键不符")
        require(all(np.isfinite(list(scores.values()))), "Milvus返回非有限分数")
        delta = max(
            abs(scores[key] - float(cosine[index])) for index, key in enumerate(ids)
        )
        require(
            delta <= 1e-5 and isclose(scores[ids[0]], 1.0, abs_tol=1e-5),
            "NumPy与Milvus余弦分数超出容差",
        )
        empty = client.search(
            collection_name=COLLECTION,
            data=[query.tolist()],
            anns_field="vector",
            limit=1,
            filter='kb_id == "test"',
            search_params={"metric_type": "COSINE", "params": {}},
        )
        require(len(empty) == 1 and not empty[0], "知识库过滤未隔离铁路test范围")
        save_new_json(
            index_dir / "verification.json",
            {
                "status": "verified",
                "reader_pid": os.getpid(),
                "writer_pid": written["writer_pid"],
                "document_count": len(ids),
                "max_cosine_difference": delta,
                "verified_at_utc": now(),
                "document_snapshot_sha256": plan["document_snapshot_sha256"],
                "vector_snapshot_sha256": plan["vector_snapshot_sha256"],
            },
        )
        print(
            f"[读回] PID={os.getpid()}；{len(ids)}条原文、完整元数据、float32向量指纹一致。",
            flush=True,
        )
        print(
            f"[对照] 同一文档向量：NumPy与Milvus全部分数最大差={delta:.3g}；自相似度≈1；跨库过滤为空。",
            flush=True,
        )
    finally:
        client.close()


def show_plan(plan: dict, index_dir: Path) -> None:
    print(f"[配置] Python：{sys.executable}；Milvus依赖：{plan['milvus_packages']}")
    print(
        f"[来源] 已完成SQLite缓存；{plan['document_count']}块；{plan['embedding']['actual_dimension']}维；float32。"
    )
    print(f"[目标] {index_dir}；集合={COLLECTION}；FLAT/COSINE；与原铁路索引分开。")
    print(
        f"[对照] 重新编码再入库：当前批量8时需{(plan['document_count'] + 7) // 8}次文档Embedding；复用缓存入库：文档Embedding=0。"
    )
    print(
        "[请求计划] --run只GET一次/api/tags核对digest；所有推理=0；Milvus本地操作另计，不是总网络通信为0。"
    )
    print(
        "[流程] 核对模型身份→新目录/写入进程→退出→独立读回进程→NumPy余弦对照及过滤→ready。"
    )
    print(
        "[边界] ready仅表示本次存储验收通过；保留原文未审核等标记，不改变原语料；未接入问句检索、BM25、重排或回答。",
        flush=True,
    )


def run_import(plan: dict, index_dir: Path) -> None:
    check_new_target(index_dir)  # 在任何HTTP之前保护已有目录。
    check_target_scope(plan, index_dir)
    require(
        plan["milvus_packages"] == VERIFIED_VERSIONS, "Milvus版本不符，不自动安装或升级"
    )
    observed = check_live_digest(plan)
    index_dir.mkdir(parents=True, exist_ok=False)
    try:
        save_new_json(index_dir / "build_plan.json", plan)
        save_new_json(index_dir / "model_observation.json", observed)
        for phase in ("write", "verify"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(PROJECT_ROOT / "scripts/build_k8s_milvus.py"),
                    "--run",
                    "--_phase",
                    phase,
                    "--index-dir",
                    str(index_dir),
                ],
                check=False,
                timeout=180,
            )
            require(result.returncode == 0, f"{phase}阶段失败，不发布ready")
        verified = json.loads((index_dir / "verification.json").read_text())
        written = json.loads((index_dir / "build_result.json").read_text())
        require(
            verified["status"] == "verified"
            and all(
                type(verified[key]) is int and verified[key] > 0
                for key in ("reader_pid", "writer_pid")
            )
            and verified["reader_pid"] != verified["writer_pid"]
            and verified["writer_pid"] == written["writer_pid"]
            and verified["document_count"] == plan["document_count"]
            and type(verified["max_cosine_difference"]) in (int, float)
            and isfinite(verified["max_cosine_difference"])
            and 0 <= verified["max_cosine_difference"] <= 1e-5
            and all(
                verified[key] == plan[key]
                for key in ("document_snapshot_sha256", "vector_snapshot_sha256")
            ),
            "缺少匹配的跨进程验证回执",
        )
        manifest = {key: value for key, value in plan.items() if key != "documents"}
        manifest.update(
            status="ready", verification=verified, model_observation=observed
        )
        save_new_json(index_dir / "manifest.json", manifest)
    except BaseException:
        print(
            f"[保留] {index_dir}；构建未正常完成，不自动删除或重试；以manifest.json及验收结果为准。",
            file=sys.stderr,
        )
        raise
    print(
        f"[成功] 独立Kubernetes索引已ready：{index_dir / 'manifest.json'}；Embedding/Reranker/聊天均0次。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--_phase", choices=("write", "verify"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    index_dir = args.index_dir.expanduser().absolute()
    if args._phase:
        require(args.run, "内部阶段需要--run")
        (write_index if args._phase == "write" else verify_index)(index_dir)
        return 0
    plan, _ = prepare_import(args.corpus, args.cache_dir)
    check_target_scope(plan, index_dir)
    show_plan(plan, index_dir)
    if not args.run:
        if index_dir.exists() or index_dir.is_symlink():
            print("[保护] 目标已存在，--run将拒绝执行，请另选新目录。")
        print(
            "[计划完成] 实际HTTP=0；未读取.env、未导入Milvus SDK、未创建客户端/文件；加--run才执行。"
        )
        return 0
    run_import(plan, index_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[中断] 未自动重试；缓存保留，已有目标目录不覆盖。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        message = (
            str(error)
            if isinstance(error, (ValueError, OSError))
            else "请检查本地服务、依赖或阶段输出；不显示HTTP响应正文"
        )
        print(f"[停止] {type(error).__name__}：{message}", file=sys.stderr)
        raise SystemExit(1)
