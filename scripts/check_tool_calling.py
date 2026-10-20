"""运行入口：实现位于qa_platform.providers.tool_calling。"""

from importlib import import_module
from pathlib import Path
import sys

_source = str(Path(__file__).resolve().parents[1] / "src")
if _source not in sys.path:
    sys.path.insert(0, _source)

if __name__ == "__main__":
    raise SystemExit(import_module("qa_platform.providers.tool_calling").main())
