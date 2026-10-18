"""使用逐字节保留的旧规则验证历史语料，不修改旧manifest或放宽校验。

冻结脚本使用原来的同目录导入，因此放在独立只读子进程中运行。
子进程仅调用纯读的verify_directory，不调用main、不读上游、不访问模型。
-I/-B用于隔离导入和禁止写pyc，不是操作系统只读沙箱。
"""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


SNAPSHOT_DIR = Path(__file__).parent / "_snapshots" / "k8s_prepare_v1"
SNAPSHOT_HASHES = {
    "check_k8s_documents.py": "a531e08d98b0cfac1d1e7a1c1d013259302f2d6d2646688de6365c4271cc9330",
    "check_k8s_chunking.py": "6dfe08402002cf3157705470a783924a9a9e481132e88b7de342f7ad1e3daba0",
    "check_k8s_chunk_cleanup.py": "969b592f5f45a2ef9c5105c344bf81e5043e9504b0eca3a7d9dcfc000fc5351d",
    "prepare_k8s_corpus.py": "8944b26185dcbdf86fe5aa53b2b1959edb2d074236d4959f32309b1f975acfcc",
}
_CHILD_CODE = """
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from prepare_k8s_corpus import verify_directory
try:
    result = verify_directory(Path(sys.argv[2]), sys.argv[3])
except Exception as error:
    print(json.dumps({"error": str(error)}, ensure_ascii=False))
    raise SystemExit(2)
print(json.dumps({"manifest": result}, ensure_ascii=False))
"""


def verify_snapshot() -> None:
    """执行旧代码前，先检查本地源码快照与迁移时记录的四个指纹一致。"""
    for name, expected in SNAPSHOT_HASHES.items():
        actual = sha256((SNAPSHOT_DIR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"历史规则快照已变化，拒绝执行：{name}")


def verify_legacy_directory(
    directory: Path, manifest_name: str = "manifest.json"
) -> dict:
    manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
    if (
        manifest.get("builder_version") != "k8s-prepare-v1"
        or manifest.get("rule_sha256") != SNAPSHOT_HASHES
    ):
        raise ValueError(
            "该历史包的规则版本不在已保留快照中；目前仅兼容现有v2所用规则。"
            "旧v1仍保留，但本次迁移没有补造其缺失的历史校验器"
        )
    verify_snapshot()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _CHILD_CODE,
                str(SNAPSHOT_DIR.resolve()),
                str(directory.resolve()),
                manifest_name,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("历史规则离线校验超时；未修改语料或清单") from error
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as error:
        raise ValueError(
            "历史校验进程未返回有效结果；请检查指定Python环境中的依赖"
        ) from error
    if completed.returncode != 0 or "manifest" not in result:
        raise ValueError(f"历史规则校验未通过：{result.get('error', '子进程异常退出')}")
    if result["manifest"] != manifest:
        raise ValueError("历史校验期间manifest发生变化")
    return result["manifest"]
