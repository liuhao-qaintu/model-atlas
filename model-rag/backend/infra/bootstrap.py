"""首次配置 .env 后运行：只创建指定的新数据库，不修改已有数据库。"""
import re

from sqlalchemy import create_engine, text

from backend.infra.config import DATABASE_URL, MYSQL_DATABASE


def initialize_database():
    if not re.fullmatch(r"[A-Za-z0-9_]+", MYSQL_DATABASE):
        raise ValueError("数据库名仅允许英文、数字和下划线")
    server = create_engine(DATABASE_URL.set(database=None), pool_pre_ping=True)
    try:
        with server.begin() as connection:
            connection.execute(text(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    finally:
        server.dispose()
    print(f"数据库 {MYSQL_DATABASE} 已就绪；后端启动时会创建表并同步模型卡。")


if __name__ == "__main__":
    initialize_database()
