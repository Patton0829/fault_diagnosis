"""运行入口：本地Kubernetes检索学习台，业务实现位于src。"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa_platform.retrieval_lab import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
