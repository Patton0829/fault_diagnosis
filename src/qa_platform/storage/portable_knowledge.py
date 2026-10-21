"""离线导出可迁移原文/向量快照；不复制Milvus运行库、不调用模型或改写冻结合同。"""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit
from uuid import uuid4

from qa_platform.settings import DATA_DIR


SCHEMA = "portable-kb/v1"
FILES = {"manifest.json", "records.jsonl", "vectors.npy"}
DEFAULT_INDEX = DATA_DIR / "milvus_indexes" / "kubernetes_zh_v1"
MAX_RECORDS, MAX_DIMENSION = 1000, 8192
MAX_RECORD_BYTES, MAX_MANIFEST_BYTES = 128 * 1024 * 1024, 256 * 1024
MAX_VECTOR_BYTES = MAX_RECORDS * MAX_DIMENSION * 4 + 10000
SCALARS = ("kb_id", "parent_id", "snapshot_commit", "review_status")
EMBEDDING_FIELDS = (
    "provider", "model", "server_model_digest", "actual_dimension", "document_mode",
    "document_query_prefix", "query_format", "query_instruction", "truncate", "dtype", "metric_type",
)
METADATA_FIELDS = {
    "chunk_id", "kb_id", "parent_id", "snapshot_commit", "snapshot_kind", "upstream_commit",
    "title", "sections", "section_path", "heading_path", "source_url", "source_format",
    "license", "attribution", "review_status", "is_test", "kubernetes_version", "kubernetes_release",
    "raw_sha256", "cleaned_sha256", "body_chars", "chunk_index", "chunk_count", "original_chunk_ids",
    "cleaned_char_start", "cleaned_char_end", "cleaned_line_numbers_are_source_lines", "offset_scope",
    "corpus_relative_path", "heading_context_ranges", "builder_version", "cleaner_version", "cleanup_version",
    "splitter_version", "parser_version", "strategy", "target_chars", "overlap_chars", "over_budget",
    "over_budget_reason", "is_navigation", "needs_heading_review", "ready_for_embedding", "token_limit_checked",
    "feature_states", "feature_states_scope", "version_constraints",
}
VERSION_FIELDS = {"min-kubernetes-server-version", "max-kubernetes-server-version"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _embedding(profile):
    require(isinstance(profile, dict) and set(profile) == set(EMBEDDING_FIELDS), "模型合同字段不兼容")
    require(profile["provider"] == "ollama" and profile["document_mode"] == "body-only"
            and profile["document_query_prefix"] is False and profile["truncate"] is False
            and profile["dtype"] == "little_endian_float32" and profile["metric_type"] == "COSINE",
            "只支持当前本地正文编码与COSINE合同")
    require(type(profile["actual_dimension"]) is int and 2 <= profile["actual_dimension"] <= MAX_DIMENSION,
            "向量维度不兼容")
    require(isinstance(profile["model"], str) and 0 < len(profile["model"]) <= 200
            and profile["model"] == profile["model"].strip() and _sha(profile["server_model_digest"]),
            "模型名称或digest无效")
    require(profile["query_format"] == "Instruct: {instruction}\nQuery: {query}"
            and isinstance(profile["query_instruction"], str) and 0 < len(profile["query_instruction"].strip()) <= 2000,
            "查询指令合同不兼容")


def _record(row):
    require(isinstance(row, dict) and set(row) == {"chunk_id", "text", "metadata", *SCALARS}, "记录字段不兼容")
    require(isinstance(row["chunk_id"], str) and 0 < len(row["chunk_id"].encode()) <= 64,
            "块ID不合法")
    require(isinstance(row["text"], str) and row["text"].strip() and len(row["text"].encode()) <= 65535,
            "块正文为空或超限")
    metadata = row["metadata"]
    require(isinstance(metadata, dict) and set(metadata).issubset(METADATA_FIELDS)
            and metadata.get("chunk_id") == row["chunk_id"], "来源元数据字段或身份无效")
    require(row["kb_id"] == "kubernetes_zh" and all(isinstance(row[key], str) and row[key]
            and metadata.get(key) == row[key] for key in SCALARS), "检索过滤身份无效")
    require(re.fullmatch(r"[0-9a-f]{40}", row["snapshot_commit"]) is not None, "快照身份无效")
    require(isinstance(metadata.get("title"), str) and metadata["title"].strip(), "公开文档标题缺失")
    source_url = metadata.get("source_url")
    require(isinstance(source_url, str), "公开来源URL缺失")
    parsed = urlsplit(source_url)
    prefix = f"/kubernetes/website/blob/{row['snapshot_commit']}/content/zh-cn/"
    require(parsed.scheme == "https" and parsed.netloc == "github.com"
            and parsed.path.startswith(prefix) and len(parsed.path) > len(prefix)
            and not parsed.query and not parsed.fragment and ".." not in parsed.path.split("/"),
            "公开来源必须为固定commit的Kubernetes中文文档")
    require(metadata.get("license") == "CC-BY-4.0" and row["review_status"] == "unreviewed_official_document",
            "来源许可或未审核标记不兼容")
    if "version_constraints" in metadata:
        conditions = metadata["version_constraints"]
        require(isinstance(conditions, dict) and set(conditions).issubset(VERSION_FIELDS)
                and all((isinstance(value, str) and 0 < len(value) <= 1000)
                        or (type(value) in {int, float} and math.isfinite(value) and value > 0)
                        for value in conditions.values()),
                "版本前提字段无效")
    require(len(canonical(metadata).encode()) <= 60000, "来源元数据超限")


def _portable_record(original):
    row = {key: original[key] for key in ("chunk_id", "text", *SCALARS)}
    row["metadata"] = {key: value for key, value in original["metadata"].items() if key in METADATA_FIELDS}
    front_matter = original["metadata"].get("front_matter", {})
    require(isinstance(front_matter, dict), "原始frontmatter格式无效")
    conditions = {key: value for key, value in front_matter.items() if key in VERSION_FIELDS}
    if conditions:
        row["metadata"]["version_constraints"] = conditions
    row = json.loads(canonical(row))  # 独立副本；绝不删除或改写源对象字段。
    _record(row)
    return row


def _matrix(matrix, count, dimension):
    import numpy as np

    require(isinstance(matrix, np.ndarray) and matrix.dtype == np.dtype("<f4")
            and matrix.shape == (count, dimension) and matrix.flags.c_contiguous,
            "向量必须为对齐记录的连续little-endian float32矩阵")
    require(np.isfinite(matrix).all(), "向量有非有限分量")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    require(np.isfinite(norms).all() and (norms > 0).all(), "向量范数无效")


def prepare_source(index_dir: Path = DEFAULT_INDEX, *, source_loader=None) -> dict:
    """只从已核验编码缓存取数；源Milvus验收回执为历史记录，不打开当前数据库。"""
    import numpy as np

    if source_loader is None:
        from qa_platform.storage.kubernetes_search import load_ready

        source_loader = load_ready
    plan, bundle = source_loader(index_dir)
    require(plan["documents"] == bundle["records"] and len(bundle["records"]) == plan["document_count"],
            "源记录与缓存顺序不一致")
    require(fingerprint(canonical(bundle["records"])) == plan["document_snapshot_sha256"], "源文档快照不一致")
    require(type(plan["document_count"]) is int and 1 <= plan["document_count"] <= MAX_RECORDS, "记录数超出当前范围")
    embedding = {key: plan["embedding"][key] for key in EMBEDDING_FIELDS}
    _embedding(embedding)
    original_matrix = np.asarray(bundle["vectors"])
    require(original_matrix.dtype.kind == "f" and original_matrix.dtype.itemsize == 4,
            "源向量必须为float32，不允许导出时静默降低精度")
    vectors = np.ascontiguousarray(original_matrix, dtype="<f4")
    _matrix(vectors, plan["document_count"], embedding["actual_dimension"])
    records, ids = [], []
    for ordinal, original in enumerate(bundle["records"]):
        row = _portable_record(original)
        chunk_id = row["chunk_id"]
        require(fingerprint(vectors[ordinal].tobytes()) == plan["vector_sha256"][chunk_id]
                == bundle["hashes"][chunk_id], "源向量与块ID指纹不匹配")
        ids.append(chunk_id)
        records.append(row)
    require(len(ids) == len(set(ids)), "源块ID重复")
    require(set(plan["vector_sha256"]) == set(bundle["hashes"]) == set(ids)
            and fingerprint(canonical(plan["vector_sha256"])) == plan["vector_snapshot_sha256"],
            "源向量指纹集合与历史快照不符")
    require(_sha(plan["document_snapshot_sha256"]) and _sha(plan["vector_snapshot_sha256"]), "历史来源指纹无效")
    license_bytes = (Path(plan["corpus_path"]) / "LICENSE").read_bytes()
    require(0 < len(license_bytes) <= 65536, "原许可证缺失或超限")
    return {"index_dir": Path(index_dir), "source_plan": plan, "records": records, "vectors": vectors,
            "embedding": embedding, "license": {"text": license_bytes.decode("utf-8"), "sha256": fingerprint(license_bytes)},
            "source_roots": (Path(index_dir), Path(plan["corpus_path"]), Path(plan["cache_path"]).parent)}


def _write(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def export_bundle(prepared: dict, output: Path) -> dict:
    """仅新目录发布；失败现场保留，完成读回校验才发布正式清单。"""
    import numpy as np

    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "目标已存在，不覆盖、不追加；已有包请--verify")
    resolved = output.resolve()
    require(all(not resolved.is_relative_to(Path(root).resolve()) for root in prepared["source_roots"]),
            "不能把导出目录放进原索引、语料或编码缓存内")
    records, matrix = prepared["records"], prepared["vectors"]
    source_plan = prepared["source_plan"]
    require(fingerprint(canonical(source_plan["documents"])) == source_plan["document_snapshot_sha256"]
            and records == [_portable_record(row) for row in source_plan["documents"]],
            "准备后的正文或来源记录已变化")
    require(prepared["embedding"] == {key: source_plan["embedding"][key] for key in EMBEDDING_FIELDS},
            "准备后的模型合同已变化")
    _embedding(prepared["embedding"])
    _matrix(matrix, len(records), prepared["embedding"]["actual_dimension"])
    require(1 <= len(records) <= MAX_RECORDS, "记录数量无效")
    for row in records:
        _record(row)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write(output / "records.jsonl", ("".join(canonical(row) + "\n" for row in records)).encode())
    with (output / "vectors.npy").open("xb") as stream:
        np.save(stream, matrix, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema_version": SCHEMA, "status": "payload_verified", "kb_id": "kubernetes_zh",
        "document_count": len(records), "chunk_ids": [row["chunk_id"] for row in records],
        "embedding": prepared["embedding"], "license": prepared["license"],
        "original_document_snapshot_sha256": prepared["source_plan"]["document_snapshot_sha256"],
        "original_vector_snapshot_sha256": prepared["source_plan"]["vector_snapshot_sha256"],
        "vector_sha256": {row["chunk_id"]: fingerprint(matrix[i].tobytes()) for i, row in enumerate(records)},
        "files": {name: {"bytes": (output / name).stat().st_size,
                         "sha256": fingerprint((output / name).read_bytes())} for name in ("records.jsonl", "vectors.npy")},
    }
    _write(output / ".manifest.pending.json", (canonical(manifest) + "\n").encode())
    load_bundle(output, manifest_name=".manifest.pending.json")
    require(not (output / "manifest.json").exists() and not (output / "manifest.json").is_symlink(), "正式清单已存在")
    (output / ".manifest.pending.json").rename(output / "manifest.json")
    return manifest


def load_bundle(directory: Path, *, manifest_name: str = "manifest.json") -> dict:
    """只读便携包，无原语料/SQLite/Milvus/接口文档依赖；不允许pickle对象。"""
    import numpy as np

    directory = Path(directory)
    require(manifest_name in {"manifest.json", ".manifest.pending.json"}, "清单文件名不合法")
    require(directory.is_dir() and not directory.is_symlink(), "包目录缺失或是软链接")
    expected_files = FILES - {"manifest.json"} | {manifest_name}
    require({path.name for path in directory.iterdir()} == expected_files, "包文件集合不完整或有额外文件")
    for name, limit in ((manifest_name, MAX_MANIFEST_BYTES), ("records.jsonl", MAX_RECORD_BYTES), ("vectors.npy", MAX_VECTOR_BYTES)):
        path = directory / name
        require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit, "包文件缺失、是软链接或大小超限")
    manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
    require(isinstance(manifest, dict) and set(manifest) == {
        "schema_version", "status", "kb_id", "document_count", "chunk_ids", "embedding", "license",
        "original_document_snapshot_sha256", "original_vector_snapshot_sha256", "vector_sha256", "files",
    }, "清单字段不兼容")
    require(manifest["schema_version"] == SCHEMA and manifest["status"] == "payload_verified"
            and manifest["kb_id"] == "kubernetes_zh", "包类型或状态不兼容")
    count = manifest["document_count"]
    require(type(count) is int and 1 <= count <= MAX_RECORDS, "记录数无效")
    _embedding(manifest["embedding"])
    require(_sha(manifest["original_document_snapshot_sha256"]) and _sha(manifest["original_vector_snapshot_sha256"]), "历史指纹无效")
    license_data = manifest["license"]
    require(isinstance(license_data, dict) and set(license_data) == {"text", "sha256"}
            and isinstance(license_data["text"], str) and 0 < len(license_data["text"].encode()) <= 65536
            and fingerprint(license_data["text"]) == license_data["sha256"], "许可证不完整或指纹不符")
    ids = manifest["chunk_ids"]
    require(isinstance(ids, list) and len(ids) == count and all(isinstance(value, str) for value in ids)
            and len(ids) == len(set(ids)), "块顺序身份无效")
    require(isinstance(manifest["files"], dict) and set(manifest["files"]) == {"records.jsonl", "vectors.npy"}, "负载文件名不兼容")
    for name in ("records.jsonl", "vectors.npy"):
        value = manifest["files"][name]
        raw = (directory / name).read_bytes()
        require(isinstance(value, dict) and set(value) == {"bytes", "sha256"}
                and type(value["bytes"]) is int and value["bytes"] == len(raw)
                and fingerprint(raw) == value["sha256"], "负载大小或SHA256不一致")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(records) == count, "原文记录数量不符")
    for row in records:
        _record(row)
    require([row["chunk_id"] for row in records] == ids, "原文顺序与向量映射身份不符")
    # mmap先限制头部/文件大小，再按合同核验shape；不按不可信shape申请大矩阵。
    matrix = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False, max_header_size=10000)
    _matrix(matrix, count, manifest["embedding"]["actual_dimension"])
    hashes = manifest["vector_sha256"]
    require(isinstance(hashes, dict) and set(hashes) == set(ids), "逐块向量指纹集合不符")
    require(fingerprint(canonical(hashes)) == manifest["original_vector_snapshot_sha256"], "历史向量快照指纹不符")
    for i, chunk_id in enumerate(ids):
        require(_sha(hashes[chunk_id]) and fingerprint(matrix[i].tobytes()) == hashes[chunk_id], "向量行与块ID指纹不符")
    return {"manifest": manifest, "records": records, "vectors": matrix}


def verify_bundle(directory: Path) -> dict:
    import numpy as np

    bundle = load_bundle(directory)
    manifest, matrix = bundle["manifest"], bundle["vectors"].astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    scores = matrix @ matrix[0] / (norms * norms[0])
    require(np.isfinite(scores).all() and abs(scores[0] - 1) <= 1e-5, "文档向量自相似校验失败")
    return {"document_count": manifest["document_count"], "dimension": manifest["embedding"]["actual_dimension"],
            "model": manifest["embedding"]["model"], "self_cosine": float(scores[0]),
            "network_requests": 0, "new_embeddings": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="零模型导出/验证可迁移知识库，不复制Milvus运行库")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--verify", type=Path)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            require(args.output_dir is None, "--verify不接受输出目录")
            print("[离线验证]", verify_bundle(args.verify))
            print("[边界] 只验证文件、身份映射及文档向量自相似，不证明问句检索质量；未读源语料或连接服务。")
            return 0
        prepared = prepare_source(args.index_dir)
        print(f"[范围] {len(prepared['records'])}块；{prepared['embedding']['actual_dimension']}维；float32；来源为已校验编码缓存。")
        print("[产物] records.jsonl + vectors.npy + manifest.json（含模型输入合同、原许可证与文件指纹）。")
        print("[边界] 不复制Milvus运行库；不改原索引/缓存；不保存构建路径、服务地址或密钥配置。正文仍须分享前自行检查。")
        if not args.run:
            print("[计划完成] 未写文件、未读.env、未建HTTP/Milvus客户端；全部模型/网络请求为0。加--run才导出。")
            return 0
        output = args.output_dir or DATA_DIR / "portable_bundles" / f"kubernetes-{uuid4().hex[:12]}"
        export_bundle(prepared, output)
        print("[导出并读回通过]", verify_bundle(output))
        print(f"[目录] {output}；可复制到新位置后--verify，校验不再读取原语料或缓存。")
        print("[尚未完成] 未重建Milvus、未切换问答服务或生成完整应用镜像；历史digest不证明当前服务器模型未变。")
        return 0
    except Exception:
        print("[未完成] 原始合同或包校验未通过，或输出已存在；未覆盖、不回退、不自动重试；若有部分新文件则保留现场。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
