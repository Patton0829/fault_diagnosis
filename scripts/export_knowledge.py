"""兼容入口：零模型可迁移知识库导出与离线验证。"""

from importlib import import_module
from pathlib import Path
import runpy
import sys

_source = str(Path(__file__).resolve().parents[1] / "src")
if _source not in sys.path:
    sys.path.insert(0, _source)

if __name__ == "__main__":
    runpy.run_module("qa_platform.storage.portable_knowledge", run_name="__main__")
else:
    sys.modules[__name__] = import_module("qa_platform.storage.portable_knowledge")
