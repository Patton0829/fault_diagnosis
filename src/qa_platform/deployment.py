"""Compose基础服务学习入口；应用仍在Conda，不搬迁冻结知识库或原MySQL。"""

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess

from qa_platform.settings import PROJECT_ROOT


COMPOSE_FILE = PROJECT_ROOT / "deploy" / "compose.yaml"
LOCAL_DIR = PROJECT_ROOT / "deploy" / ".local"
PROJECT_NAME = "fault-diagnosis-learning"
FILES = {"mysql_root_password", "mysql_app_password", "redis_password", "redis.conf"}
PASSWORD_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}aA1!\Z")


class DeploymentError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeSecrets:
    mysql_password: str = field(repr=False)
    redis_password: str = field(repr=False)


def _redis_config(password: str) -> str:
    return ("bind 0.0.0.0\nport 6379\nprotected-mode yes\nsave \"\"\nappendonly no\n"
            "maxmemory 64mb\nmaxmemory-policy allkeys-lru\nrequirepass " + password + "\n")


def _write_new(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(local_dir: Path = LOCAL_DIR) -> None:
    """只生成新学习环境的凭据；旧配置/部分现场均不覆盖，不访问Docker。"""
    if local_dir.exists() or local_dir.is_symlink():
        raise DeploymentError("deploy/.local已存在；不覆盖、不重置密码。已有完整配置请直接--up/--check。")
    try:
        local_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        generated = {name: secrets.token_urlsafe(32) + "aA1!" for name in FILES if name != "redis.conf"}
        for name, value in generated.items():
            _write_new(local_dir / name, value + "\n")
        _write_new(local_dir / "redis.conf", _redis_config(generated["redis_password"]))
    except OSError:
        raise DeploymentError("生成配置未完成；已存在的局部文件保留，不自动清理或重新生成，请人工检查。") from None


def read_config(local_dir: Path = LOCAL_DIR) -> RuntimeSecrets:
    """只读取Compose自己的应用凭据；不读取本机models配置或容器root密码。"""
    try:
        if not local_dir.is_dir() or local_dir.is_symlink() or local_dir.stat().st_mode & 0o777 != 0o700:
            raise ValueError
        if {path.name for path in local_dir.iterdir()} != FILES:
            raise ValueError
        for name in FILES:
            path = local_dir / name
            if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o600:
                raise ValueError
        app = (local_dir / "mysql_app_password").read_text(encoding="utf-8").rstrip("\n")
        redis = (local_dir / "redis_password").read_text(encoding="utf-8").rstrip("\n")
        if not PASSWORD_PATTERN.fullmatch(app) or not PASSWORD_PATTERN.fullmatch(redis) or app == redis:
            raise ValueError
        if (local_dir / "redis.conf").read_text(encoding="utf-8") != _redis_config(redis):
            raise ValueError
        return RuntimeSecrets(app, redis)
    except (ValueError, OSError, UnicodeError):
        raise DeploymentError("Compose独立配置缺失或无效；请检查deploy/.local，不读取旧密码、不回退本机服务。") from None


def compose(action: str, *, local_dir: Path = LOCAL_DIR, runner=None) -> None:
    """只操作固定学习项目；不使用shell，不传密码参数，不删除数据卷。"""
    if action not in {"up", "down"}:
        raise DeploymentError("只允许启动或停止固定Compose学习项目。")
    if local_dir != LOCAL_DIR:
        raise DeploymentError("Compose命令只使用仓库固定配置；测试请替换执行器，不连接真实服务。")
    if action == "up":
        read_config(local_dir)
    command = ["docker", "compose", "--project-name", PROJECT_NAME, "--file", str(COMPOSE_FILE)]
    command += ["up", "-d", "--wait", "--wait-timeout", "90"] if action == "up" else ["down"]
    try:
        completed = (runner or subprocess.run)(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180, check=False,
        )
        if completed.returncode != 0:
            raise ValueError
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise DeploymentError(
            "Compose操作未完成；请检查Docker Desktop、镜像下载、端口及服务健康状态。"
            "可能已有部分容器/卷，保留供检查；未删除卷、未切换原应用或回退本机服务。"
        ) from None


def open_services(local_dir: Path = LOCAL_DIR):
    """只检查独立容器的应用账号与缓存；不建表、不写记录或请求模型。"""
    from qa_platform.storage.mysql_setup import DatabaseConfig, make_engine
    from qa_platform.storage.sql_conversations import check_schema
    from qa_platform.storage.retrieval_cache import RetrievalCache

    config = read_config(local_dir)
    engine, cache = None, None
    try:
        engine = make_engine(DatabaseConfig(config.mysql_password, port=13306))
        cache = RetrievalCache(port=16379, password=config.redis_password)
        check_schema(engine)
        cache.ping()
        return engine, cache
    except Exception:
        if engine is not None:
            engine.dispose()
        if cache is not None:
            cache.close()
        raise DeploymentError("Compose数据库或Redis只读检查未通过；未请求模型、建表或回退到本机原服务。") from None


def check(local_dir: Path = LOCAL_DIR) -> dict:
    from sqlalchemy import text

    engine, cache = open_services(local_dir)
    try:
        with engine.connect() as connection:
            conversations = connection.scalar(text("SELECT COUNT(*) FROM qa_conversations"))
            turns = connection.scalar(text("SELECT COUNT(*) FROM qa_turns"))
        return {"mysql_port": 13306, "redis_port": 16379, "conversations": conversations, "turns": turns}
    except Exception:
        raise DeploymentError("Compose记录只读检查未完成；未写数据或回退到原数据库。") from None
    finally:
        cache.close()
        engine.dispose()


def require_exclusive_app() -> None:
    """新端口不等于新Milvus：要求先手动退出原应用，避免两进程共用Lite目录。"""
    for port in (8766, 8767):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", port))
        except OSError:
            raise DeploymentError("8766或8767已有应用；请先手动停止旧应用，避免两个进程共用Milvus Lite。不会自动结束它。") from None


def serve(local_dir: Path = LOCAL_DIR) -> None:
    require_exclusive_app()
    from qa_platform.retrieval_lab import create_app
    from qa_platform.storage.sql_conversations import SQLConversationStore
    import uvicorn

    engine, cache = open_services(local_dir)
    try:
        print("[学习环境] http://127.0.0.1:8767/qa；连接Docker独立数据库，不迁移旧历史；应用仍在Conda。")
        uvicorn.run(create_app(conversation_store=SQLConversationStore(engine), storage_backend="mysql",
                               retrieval_cache=cache, cache_backend="redis", deployment_mode=True),
                    host="127.0.0.1", port=8767, access_log=False)
    finally:
        cache.close()
        engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose基础服务学习：仅MySQL/Redis容器，应用仍在Conda")
    modes = parser.add_mutually_exclusive_group()
    for action in ("prepare", "up", "check", "serve", "down"):
        modes.add_argument("--" + action, action="store_true")
    args = parser.parse_args()
    print("[范围] 独立Compose项目fault-diagnosis-learning；MySQL13306/Redis16379；本机原服务不变。")
    if not any(vars(args).values()):
        print("[步骤] --prepare生成独立配置 → --up启动基础服务 → --check只读检查。")
        print("[应用] --serve需先手动停止8766旧应用；在Conda启动8767，不代表应用镜像已完成。")
        print("[停止] --down保留MySQL卷和独立凭据；无删卷、迁移或重置密码操作。")
        print("[计划完成] 未读取密钥或配置、导入数据库/Redis SDK、执行Docker命令、写文件或调用模型。")
        return 0
    try:
        if args.prepare:
            prepare()
            print("[完成] 独立配置已生成至deploy/.local（0700/0600）；不显示密码，不提交Git。")
        elif args.up:
            compose("up")
            print("[完成] 独立基础服务已通过健康检查；未启动新问答进程，也未改变原应用。")
        elif args.check:
            print("[只读通过]", check())
        elif args.serve:
            serve()
        else:
            compose("down")
            print("[停止] 学习环境基础容器已停止；MySQL数据卷及凭据保留，未删除问答记录。")
    except Exception as exc:
        print("[未完成]", str(exc) if isinstance(exc, DeploymentError) else "学习环境操作失败，未回退或自动重试；已有资源保留。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
