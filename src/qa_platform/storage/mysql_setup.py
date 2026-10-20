"""显式初始化本机MySQL；root密码只在交互终端使用，应用只用专用账号。"""

import argparse
from dataclasses import dataclass, field
from getpass import getpass
import os
from pathlib import Path
import secrets
import sys

from qa_platform.settings import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "models" / "database.env"
DATABASE = "fault_diagnosis"
APP_USER = "fault_qa"
NOTICE = "MySQL操作未完成；请检查本机服务、专用配置和表结构。不会回退内存、重置账号或自动重试。"


class SetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatabaseConfig:
    password: str = field(repr=False)


def read_config(path: Path = CONFIG_PATH) -> DatabaseConfig:
    """只接受本项目固定的本机专用账号，不将root配置交给服务。"""
    from dotenv import dotenv_values

    if not path.is_file():
        raise SetupError("未找到models/database.env；请先在本机运行setup_mysql.py --bootstrap。")
    try:
        values = dotenv_values(path, interpolate=False)
        expected = {"DB_HOST": "127.0.0.1", "DB_PORT": "3306", "DB_NAME": DATABASE, "DB_USER": APP_USER}
        if any(values.get(key) != value for key, value in expected.items()):
            raise ValueError
        password = values.get("DB_PASSWORD")
        if not password or not password.strip():
            raise ValueError
        return DatabaseConfig(password)
    except (ValueError, OSError):
        raise SetupError("数据库配置无效；仅支持127.0.0.1:3306/fault_diagnosis及fault_qa专用账号。") from None


def make_engine(config: DatabaseConfig):
    """URL.create避免密码中的特殊字符被误解析；不在此处连接或建表。"""
    from sqlalchemy import URL, create_engine

    return create_engine(
        URL.create("mysql+pymysql", username=APP_USER, password=config.password,
                   host="127.0.0.1", port=3306, database=DATABASE, query={"charset": "utf8mb4"}),
        pool_pre_ping=True, pool_recycle=1800, echo=False, hide_parameters=True,
        connect_args={"connect_timeout": 3, "read_timeout": 5, "write_timeout": 5},
    )


def _reserve_config(path: Path, config: DatabaseConfig) -> None:
    """独占创建0600文件；不覆盖现有凭据；只写生成的专用密码。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ("# 本机自动生成；不要提交或分享。root密码不保存。\n"
               f"DB_HOST=127.0.0.1\nDB_PORT=3306\nDB_NAME={DATABASE}\nDB_USER={APP_USER}\n"
               f"DB_PASSWORD={config.password}\n")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def bootstrap(root_password: str, path: Path = CONFIG_PATH) -> None:
    """新建固定命名空间，不覆盖旧库/旧用户；失败保留局部资源供人工处理。"""
    from sqlalchemy import URL, create_engine, text
    from sqlalchemy.exc import SQLAlchemyError

    from qa_platform.storage.sql_conversations import Base, check_schema

    if path.exists() or path.is_symlink():
        raise SetupError("配置已存在；不覆盖、不重新初始化。可运行--check检查现有配置。")
    root_engine = create_engine(
        URL.create("mysql+pymysql", username="root", password=root_password,
                   host="127.0.0.1", port=3306, query={"charset": "utf8mb4"}),
        echo=False, hide_parameters=True,
        connect_args={"connect_timeout": 3, "read_timeout": 5, "write_timeout": 5},
    )
    config = DatabaseConfig(secrets.token_urlsafe(30) + "aA1!")
    reserved = False
    try:
        with root_engine.connect() as connection:
            database_exists = connection.scalar(text(
                "SELECT COUNT(*) FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=:name"
            ), {"name": DATABASE})
            user_exists = connection.scalar(text("SELECT COUNT(*) FROM mysql.user WHERE User=:name"),
                                            {"name": APP_USER})
            if database_exists or user_exists:
                raise SetupError("fault_diagnosis数据库或fault_qa账号已存在；已停止，不覆盖、不删除、不重置密码。")
            # 先安全保存专用凭据：之后DDL失败仍保留文件，避免遗失已创建账号的密码。
            _reserve_config(path, config)
            reserved = True
            connection.exec_driver_sql("CREATE DATABASE `fault_diagnosis` CHARACTER SET utf8mb4")
            connection.exec_driver_sql("USE `fault_diagnosis`")
            # 数据库刚创建且为空；root连接最初没选库，不能用缓存的default_schema做反射检查。
            Base.metadata.create_all(connection, checkfirst=False)
            connection.exec_driver_sql("CREATE USER 'fault_qa'@'127.0.0.1' IDENTIFIED BY %s", (config.password,))
            connection.exec_driver_sql(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON `fault_diagnosis`.* TO 'fault_qa'@'127.0.0.1'"
            )
            connection.commit()
        app_engine = make_engine(config)
        try:
            check_schema(app_engine)
        finally:
            app_engine.dispose()
    except (SQLAlchemyError, OSError):
        suffix = ("已保留专用配置及可能创建的局部资源；MySQL DDL非整体原子操作，需人工核查，勿反复初始化。"
                  if reserved else "未开始创建库表。")
        raise SetupError(f"{NOTICE}{suffix}") from None
    except Exception as exc:
        if isinstance(exc, SetupError):
            raise
        raise SetupError(f"{NOTICE}可能已有局部资源，请人工核查。") from None
    finally:
        root_engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="本机MySQL会话存储：默认仅计划，--bootstrap交互初始化，--check只读检查")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print("[目标] 127.0.0.1:3306；新库fault_diagnosis；专用账号fault_qa；SQLAlchemy + PyMySQL。")
    print("[表] qa_conversations会话；qa_turns保存已校验回答、引用、计数及最近三轮摘要。")
    if not args.bootstrap and not args.check:
        print("[计划] --bootstrap在本机安全输入root密码，生成独立应用密码；root密码不保存、不显示。")
        print("[保护] 旧库/同名账号/配置存在即停止；新配置0600且被Git忽略，应用只有项目库CRUD权限。")
        print("[计划完成] 未读配置或密码，未导入SQLAlchemy，未连接数据库、建表、写文件或请求模型。")
        return 0
    try:
        if args.bootstrap:
            if not sys.stdin.isatty():
                raise SetupError("请在本机交互终端运行--bootstrap；不接受管道密码，请勿将密码发到聊天。")
            password = getpass("本机MySQL root密码（不回显、不保存）：")
            try:
                bootstrap(password)
            finally:
                password = None  # 缩短引用存活期；不声称可清除Python进程内存副本。
            print("[成功] 新库、两张表和专用账号已创建；应用账号只读表结构检查通过。")
            print("[配置] models/database.env（仅专用账号；不要截图、复制或提交此文件）。")
        else:
            from qa_platform.storage.sql_conversations import check_schema

            engine = make_engine(read_config())
            try:
                check_schema(engine)
            finally:
                engine.dispose()
            print("[成功] 专用账号连接与两表只读检查通过；未写记录、建表或请求模型。")
    except Exception as exc:
        print(f"[未完成] {str(exc) if isinstance(exc, SetupError) else NOTICE}")
        return 1
    print("[下一步] 停止旧服务后运行serve_retrieval_lab.py --port 8766 --storage mysql。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
