"""本地配置只在后端读取，API 不返回凭据。"""
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL,make_url

# 本文件位于 backend/infra/，仓库根要往上三层：infra -> backend -> 仓库根
ROOT = Path(__file__).resolve().parents[2]
CONTENT_ROOT = ROOT.parent / "模型知识库资料"
load_dotenv(ROOT / ".env")
DATA_ROOT = Path(os.getenv('MODEL_ATLAS_DATA_ROOT', str(ROOT / 'data'))).resolve()
DATA_ROOT.mkdir(parents=True,exist_ok=True)


def model_setting(name, default=""):
    return os.getenv(name) or default


MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "model_rag")
DATABASE_URL = URL.create(
    "mysql+pymysql",
    username=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    database=MYSQL_DATABASE,
    query={"charset": "utf8mb4"},
)
if os.getenv('DATABASE_URL'):
    DATABASE_URL = make_url(os.environ['DATABASE_URL'])

AUTH_SECRET_KEY = os.getenv('AUTH_SECRET_KEY', '')
AUTH_EXPIRE_HOURS = int(os.getenv('AUTH_EXPIRE_HOURS', '24'))
AUTH_COOKIE_SECURE = os.getenv('AUTH_COOKIE_SECURE', 'false').lower() == 'true'
ENABLE_API_DOCS = os.getenv('ENABLE_API_DOCS', 'false').lower() == 'true'
MAINTAINER_USER_IDS = json.loads(os.getenv('MAINTAINER_USER_IDS', '[]'))
if not isinstance(MAINTAINER_USER_IDS, list) or any(not isinstance(item, str) for item in MAINTAINER_USER_IDS):
    raise ValueError('MAINTAINER_USER_IDS需要JSON字符串数组')

EMBEDDING_MODEL = model_setting("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_BASE = model_setting("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_KEY = model_setting("EMBEDDING_API_KEY")
LLM_MODEL = model_setting("LLM_MODEL_NAME", "deepseek-ai/DeepSeek-V4-Flash")
LLM_BASE = model_setting("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_KEY = model_setting("LLM_API_KEY")
RERANK_MODEL = model_setting("RERANK_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B")
RERANK_BASE = model_setting("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
RERANK_KEY = model_setting("RERANK_API_KEY")
