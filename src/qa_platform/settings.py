"""统一项目路径；仅计算路径，不读取.env、不初始化任何服务客户端。

当前仓库按src布局运行，知识库和生成数据保留原位置。
环境变量密钥暂留models/.env，迁移不读取、不复制、不提交其内容。
"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
ENV_PATH = PROJECT_ROOT / "models" / ".env"
MODEL_DOCS_DIR = PROJECT_ROOT / "docs" / "model_apis"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
DATA_DIR = PROJECT_ROOT / "data"
WEB_PAGE = PROJECT_ROOT / "web" / "data_lab.html"
