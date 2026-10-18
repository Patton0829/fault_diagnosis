"""兼容入口：实现已迁至 qa_platform.rag.milvus_demo，原运行命令继续有效。"""

from importlib import import_module
from pathlib import Path
import runpy
import sys

_source = str(Path(__file__).resolve().parents[1] / "src")
if _source not in sys.path:
    sys.path.insert(0, _source)

if __name__ == "__main__":
    runpy.run_module("qa_platform.rag.milvus_demo", run_name="__main__")
else:
    sys.modules[__name__] = import_module("qa_platform.rag.milvus_demo")
