"""第十一步：用合成向量验证Milvus Lite写入、跨进程持久化、搜索与过滤。"""

import argparse
from importlib.metadata import PackageNotFoundError, version
from math import isclose
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from qa_platform.settings import PROJECT_ROOT


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "milvus_smoke"
# 固定本轮验证的组合，避免误用不同存储格式的Lite版本。
VERIFIED_VERSIONS = {"pymilvus": "2.6.17", "milvus-lite": "3.2.2"}
COLLECTION = "synthetic_smoke_v1"
DIMENSION = 4
QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]
ROWS = [
    {
        "id": "demo-a",
        "vector": [1.0, 0.0, 0.0, 0.0],
        "text": "合成样本A：与查询向量同方向。",
        "kb_id": "smoke_a",
        "is_test": True,
        "embedding_model": "synthetic-demo-v1",
    },
    {
        "id": "demo-b",
        "vector": [0.8, 0.6, 0.0, 0.0],
        "text": "合成样本B：与查询向量部分同方向。",
        "kb_id": "smoke_a",
        "is_test": True,
        "embedding_model": "synthetic-demo-v1",
    },
    {
        "id": "demo-c",
        "vector": [0.0, 1.0, 0.0, 0.0],
        "text": "合成样本C：与查询向量正交。",
        "kb_id": "smoke_b",
        "is_test": True,
        "embedding_model": "synthetic-demo-v1",
    },
]
OUTPUT_FIELDS = ["text", "kb_id", "is_test", "embedding_model"]


def installed_versions() -> dict[str, str | None]:
    """只查询包信息；计划模式不导入SDK，也不会启动本地数据库。"""
    versions = {}
    for package in VERIFIED_VERSIONS:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def require(condition: bool, message: str) -> None:
    """运行时检查不能依赖assert，因为python -O会移除assert。"""
    if not condition:
        raise ValueError(message)


def write_demo(db_path: Path) -> None:
    """只写全新数据库，不复用、不清空已有库。"""
    require(
        not db_path.exists() and not db_path.is_symlink(),
        "数据库路径已存在，拒绝覆盖或追加。",
    )
    from pymilvus import DataType, MilvusClient

    client = MilvusClient(uri=str(db_path), timeout=20)
    try:
        require(not client.has_collection(COLLECTION), "演示集合已经存在，拒绝覆盖。")
        # Collection类似表，Schema显式规定字段；不让拼错的字段自动混入数据。
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIMENSION)
        # VARCHAR的max_length按UTF-8字节计算，不是中文字数。
        schema.add_field("text", DataType.VARCHAR, max_length=512)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=32)
        schema.add_field("is_test", DataType.BOOL)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=64)
        index = client.prepare_index_params()
        index.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_collection(
            collection_name=COLLECTION, schema=schema, index_params=index
        )
        result = client.insert(collection_name=COLLECTION, data=ROWS)
        require(result["insert_count"] == len(ROWS), "写入数量不符合预期。")
        client.flush(collection_name=COLLECTION)
        print(
            f"[写入进程] PID={os.getpid()}；已写入{len(ROWS)}条合成记录并flush。",
            flush=True,
        )
    finally:
        client.close()


def verify_demo(db_path: Path) -> None:
    """新进程只重开已有库验证，不重新插入；避免把进程内缓存误当成持久化。"""
    # Lite 3.x把.db路径作为数据目录，而不是旧版的单个SQLite文件。
    require(
        db_path.is_dir() and not db_path.is_symlink(),
        "数据库目录不存在，不能进行重开验证。",
    )
    from pymilvus import MilvusClient

    client = MilvusClient(uri=str(db_path), timeout=20)
    try:
        require(client.has_collection(COLLECTION), "重开后找不到演示集合。")  # type: ignore
        client.load_collection(collection_name=COLLECTION)
        stats = client.get_collection_stats(collection_name=COLLECTION)
        require(int(stats["row_count"]) == len(ROWS), "重开后记录总数不正确。")
        records = client.get(
            collection_name=COLLECTION,
            ids=[row["id"] for row in ROWS],
            output_fields=["id", "vector", *OUTPUT_FIELDS],
        )
        require(len(records) == len(ROWS), "无法按主键找回全部记录。")
        by_id = {record["id"]: record for record in records}
        require(set(by_id) == {row["id"] for row in ROWS}, "读回的主键不一致。")
        for expected in ROWS:
            actual = by_id[expected["id"]]
            require(
                all(actual[field] == expected[field] for field in OUTPUT_FIELDS),
                "读回的文本或元数据不一致。",
            )
            require(len(actual["vector"]) == DIMENSION, "读回向量的维度不一致。")
            require(
                all(
                    isclose(float(a), b, abs_tol=1e-6)
                    for a, b in zip(actual["vector"], expected["vector"])
                ),
                "读回向量的数值不一致。",
            )
        print(
            f"[读取进程] PID={os.getpid()}；重开后找回{len(records)}条记录，文本、向量与元数据一致。",
            flush=True,
        )

        results = client.search(
            collection_name=COLLECTION,
            data=[QUERY_VECTOR],
            anns_field="vector",
            limit=3,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=OUTPUT_FIELDS,
        )
        require(len(results) == 1 and len(results[0]) == 3, "搜索结果数量不正确。")
        require(
            [hit["id"] for hit in results[0]] == [row["id"] for row in ROWS],
            "余弦相似度排序不正确。",
        )
        for rank, (hit, expected_score) in enumerate(
            zip(results[0], [1.0, 0.8, 0.0]), 1
        ):
            score = float(hit["distance"])
            require(
                isclose(score, expected_score, abs_tol=1e-5),
                "余弦分数不符合合成向量的预期。",
            )
            print(
                f"[搜索 {rank}] {hit['id']} | COSINE={score:.4f} | {hit['entity']['text']}",
                flush=True,
            )

        filtered = client.search(
            collection_name=COLLECTION,
            data=[QUERY_VECTOR],
            anns_field="vector",
            limit=3,
            filter='kb_id == "smoke_b"',
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=OUTPUT_FIELDS,
        )
        require(
            len(filtered) == 1 and [hit["id"] for hit in filtered[0]] == ["demo-c"],
            "元数据过滤结果不正确。",
        )
        print(
            "[过滤] kb_id=smoke_b时只返回demo-c；过滤限定候选范围，不会提高相似度。",
            flush=True,
        )
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Milvus Lite本地烟测；默认只展示计划，零模型请求"
    )
    parser.add_argument(
        "--run", action="store_true", help="创建独立演示目录并执行写入、重开与搜索"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="每次运行在此目录下新建独立子目录，不复用旧库",
    )
    # 两个内部阶段由主进程依次启动，让读写发生在真正不同的Python进程中。
    parser.add_argument("--_phase", choices=["write", "verify"], help=argparse.SUPPRESS)
    parser.add_argument("--_db", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._phase:
        if not args.run or args._db is None:
            parser.error("内部阶段需要--run与数据库路径")
        try:
            versions = installed_versions()
            require(versions == VERIFIED_VERSIONS, "依赖版本与本脚本验证组合不一致。")
            phase = write_demo if args._phase == "write" else verify_demo
            phase(args._db)
        except Exception as exc:
            print(f"[错误] {type(exc).__name__}：{exc}", file=sys.stderr)
            return 1
        return 0
    if args._db is not None:
        parser.error("数据库路径仅供内部阶段使用；请通过--data-dir选择演示目录")

    data_dir = args.data_dir.expanduser().resolve()
    versions = installed_versions()
    print(f"[配置] Python：{sys.executable}")
    for package, expected in VERIFIED_VERSIONS.items():
        print(
            f"[依赖] {package}：{versions[package] or '未安装'}；本脚本验证版本：{expected}"
        )
    print(f"[目录] {data_dir}；--run后每次新建run-*子目录，保留已有库。")
    print(f"[计划] {len(ROWS)}条合成记录；{DIMENSION}维；FLAT索引；COSINE度量。")
    print("[计划] 写入进程退出后，再启动独立读取进程，检查记录、搜索排序和元数据过滤。")
    print(
        "[边界] 向量是人工构造的几何样本，不代表文本语义；不读取.env，不调用任何模型API。"
    )
    print(
        "[边界] 尚未把真实知识库或RAG接入Milvus，也未验证生产部署、并发或异常断电恢复。"
    )
    if not args.run:
        print(
            "[计划完成] 未导入Milvus SDK、未启动数据库、未创建目录或文件；加--run才执行。"
        )
        return 0
    if versions != VERIFIED_VERSIONS:
        print("[错误] 请先在当前解释器环境安装本脚本验证的依赖组合：", file=sys.stderr)
        print(
            f'{sys.executable} -m pip install "pymilvus==2.6.17" "milvus-lite==3.2.2"',
            file=sys.stderr,
        )
        return 1

    run_dir = None
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=data_dir))
        db_path = run_dir / "smoke.db"
        print(f"[数据库] {db_path}", flush=True)
        for phase in ("write", "verify"):
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "check_milvus.py"),
                    "--run",
                    "--_phase",
                    phase,
                    "--_db",
                    str(db_path),
                ],
                timeout=45,
                check=False,
            )
            require(result.returncode == 0, f"{phase}子进程未成功完成。")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        if run_dir is not None:
            print(
                f"[保留] 本次目录未删除：{run_dir}；再次运行会创建新目录，不会覆盖它。",
                file=sys.stderr,
            )
        return 1
    print("[成功] Milvus Lite跨进程持久化、向量搜索、元数据过滤验证通过；零模型请求。")
    print(f"[保留] 演示数据库：{db_path}；4维合成集合不能直接混入真实Embedding向量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
