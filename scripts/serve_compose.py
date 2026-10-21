"""运行入口：独立Compose基础服务；默认只展示零操作计划。"""

from importlib import import_module
from pathlib import Path
import runpy
import sys

_source = str(Path(__file__).resolve().parents[1] / "src")
if _source not in sys.path:
    sys.path.insert(0, _source)

if __name__ == "__main__":
    runpy.run_module("qa_platform.deployment", run_name="__main__")
else:
    sys.modules[__name__] = import_module("qa_platform.deployment")
