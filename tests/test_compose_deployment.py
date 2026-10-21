"""Isolated Compose deployment contracts; never contact Docker, databases or models."""

import builtins
from pathlib import Path
import re
import runpy
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from qa_platform import deployment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "deploy" / "compose.yaml"


@pytest.fixture(autouse=True)
def prohibit_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Compose tests must not contact Docker, databases or models")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def compose_config():
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_compose_only_contains_infrastructure_with_pinned_official_images(compose_config):
    assert compose_config["name"] == "fault-diagnosis-learning"
    services = compose_config["services"]
    assert set(services) == {"mysql", "redis"}
    for name in services:
        image = services[name]["image"]
        assert re.fullmatch(rf"{name}:\d+\.\d+\.\d+(?:-alpine)?", image)
        assert "latest" not in image
        assert services[name]["restart"] == "no"
        assert services[name]["healthcheck"]["test"][0] == "CMD-SHELL"


def test_compose_ports_are_isolated_and_never_public(compose_config):
    services = compose_config["services"]
    assert services["mysql"]["ports"] == ["127.0.0.1:13306:3306"]
    assert services["redis"]["ports"] == ["127.0.0.1:16379:6379"]
    for service in services.values():
        assert not service.get("privileged", False)
        assert service.get("network_mode") != "host"
        assert service.get("pid") != "host"
        assert not service.get("cap_add")
    assert "docker.sock" not in COMPOSE_PATH.read_text(encoding="utf-8")


def test_compose_secrets_are_private_local_files_not_password_values(compose_config):
    expected = {"mysql_root_password", "mysql_app_password", "redis_password", "redis_config"}
    assert set(compose_config["secrets"]) == expected
    for secret in compose_config["secrets"].values():
        assert set(secret) == {"file"}
        assert secret["file"].startswith("./.local/")
        assert ".." not in Path(secret["file"]).parts
    environment = compose_config["services"]["mysql"]["environment"]
    assert environment["MYSQL_ROOT_PASSWORD_FILE"] == "/run/secrets/mysql_root_password"
    assert environment["MYSQL_PASSWORD_FILE"] == "/run/secrets/mysql_app_password"
    assert environment["MYSQL_ROOT_HOST"] == "localhost"
    assert "MYSQL_ROOT_PASSWORD" not in environment and "MYSQL_PASSWORD" not in environment
    assert not compose_config["services"]["redis"].get("environment")
    redis_check = compose_config["services"]["redis"]["healthcheck"]["test"][1]
    assert "redis-cli -h 127.0.0.1 -p 6379" in redis_check
    assert "--host" not in redis_check and "--port" not in redis_check


def test_compose_uses_new_named_mysql_volume_and_ephemeral_redis(compose_config):
    assert set(compose_config["volumes"]) == {"mysql_data"}
    mysql_volumes = compose_config["services"]["mysql"]["volumes"]
    assert mysql_volumes[0] == "mysql_data:/var/lib/mysql"
    assert mysql_volumes[1] == {
        "type": "bind", "source": "./mysql-init.sql",
        "target": "/docker-entrypoint-initdb.d/01-qa-schema.sql",
        "read_only": True, "bind": {"create_host_path": False},
    }
    redis = compose_config["services"]["redis"]
    assert not redis.get("volumes")
    assert any(item.startswith("/data:") for item in redis["tmpfs"])
    assert "models/" not in COMPOSE_PATH.read_text(encoding="utf-8")


def test_mysql_container_schema_matches_runtime_columns_and_primary_keys():
    pytest.importorskip("sqlalchemy")
    from qa_platform.storage.sql_conversations import Base

    sql = (PROJECT_ROOT / "deploy" / "mysql-init.sql").read_text(encoding="utf-8")
    for table in Base.metadata.sorted_tables:
        body = re.search(rf"CREATE TABLE `{table.name}` \((.*?)\) ENGINE=InnoDB;", sql, re.S)
        assert body is not None
        actual = re.findall(r"^\s+`(\w+)`\s+", body.group(1), re.M)
        assert actual == [column.name for column in table.columns]
        primary = ", ".join(f"`{column.name}`" for column in table.primary_key.columns)
        assert f"PRIMARY KEY ({primary})" in body.group(1)
    assert "FOREIGN KEY (`conversation_id`) REFERENCES `qa_conversations` (`id`)" in sql


def test_mysql_container_init_does_not_touch_native_users_or_grant_ddl():
    sql = (PROJECT_ROOT / "deploy" / "mysql-init.sql").read_text(encoding="utf-8")
    statements = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON" in statements
    assert "REVOKE ALL PRIVILEGES ON" in statements
    assert "'fault_qa'@'%'" in statements
    assert not re.search(r"\b(?:DROP|TRUNCATE|ALTER USER|CREATE USER)\b", statements, re.I)
    assert "root" not in statements and "127.0.0.1" not in statements


def test_default_plan_has_no_config_reads_imported_clients_or_commands(monkeypatch, capsys):
    forbidden = Mock(side_effect=AssertionError("The default plan must remain zero I/O"))
    for name in ("prepare", "read_config", "compose", "open_services", "check", "serve"):
        monkeypatch.setattr(deployment, name, forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("sqlalchemy", "pymysql", "redis", "docker", "uvicorn")):
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["serve_compose.py"])
    assert deployment.main() == 0
    assert "计划完成" in capsys.readouterr().out
    forbidden.assert_not_called()


def test_script_entry_default_plan_does_not_import_service_clients(monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("sqlalchemy", "pymysql", "redis", "docker", "uvicorn")):
            raise AssertionError("The default script imported a service client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["serve_compose.py"])
    monkeypatch.delitem(sys.modules, "qa_platform.deployment", raising=False)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(PROJECT_ROOT / "scripts" / "serve_compose.py"), run_name="__main__")
    assert caught.value.code == 0
    assert "计划完成" in capsys.readouterr().out


def test_prepare_creates_only_independent_private_files_without_reading_existing_secrets(tmp_path, monkeypatch):
    local_dir = tmp_path / ".local"
    forbidden = Mock(side_effect=AssertionError("Prepare must not read old secrets or execute Docker"))
    with monkeypatch.context() as guard:
        guard.setattr(Path, "read_text", forbidden)
        guard.setattr(Path, "read_bytes", forbidden)
        guard.setattr(subprocess, "run", forbidden)
        deployment.prepare(local_dir)
    forbidden.assert_not_called()
    assert local_dir.stat().st_mode & 0o777 == 0o700
    assert {path.name for path in local_dir.iterdir()} == deployment.FILES
    passwords = []
    for path in local_dir.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
        assert not path.is_symlink()
        if path.name.endswith("password"):
            password = path.read_text(encoding="utf-8").rstrip("\n")
            assert deployment.PASSWORD_PATTERN.fullmatch(password)
            passwords.append(password)
    assert len(passwords) == len(set(passwords)) == 3
    config = deployment.read_config(local_dir)
    assert config.mysql_password in passwords and config.redis_password in passwords
    assert all(password not in repr(config) for password in passwords)
    redis_config = (local_dir / "redis.conf").read_text(encoding="utf-8")
    assert 'save ""' in redis_config and "appendonly no" in redis_config
    assert "maxmemory 64mb" in redis_config and "maxmemory-policy allkeys-lru" in redis_config
    assert "protected-mode yes" in redis_config


@pytest.mark.parametrize("existing", ["directory", "file", "symlink"])
def test_prepare_refuses_existing_target_without_overwriting(tmp_path, existing):
    local_dir = tmp_path / ".local"
    target = tmp_path / "synthetic-existing-file"
    target.write_text("ORIGINAL_SYNTHETIC_CONTENT", encoding="utf-8")
    if existing == "directory":
        local_dir.mkdir()
    elif existing == "file":
        local_dir.write_text("ORIGINAL_SYNTHETIC_CONTENT", encoding="utf-8")
    else:
        local_dir.symlink_to(target)
    with pytest.raises(deployment.DeploymentError):
        deployment.prepare(local_dir)
    assert target.read_text(encoding="utf-8") == "ORIGINAL_SYNTHETIC_CONTENT"
    if existing == "file":
        assert local_dir.read_text(encoding="utf-8") == "ORIGINAL_SYNTHETIC_CONTENT"
    elif existing == "directory":
        assert list(local_dir.iterdir()) == []
    else:
        assert local_dir.is_symlink()


def test_prepare_partial_failure_keeps_files_and_redacts_exception(tmp_path, monkeypatch):
    local_dir = tmp_path / ".local"
    original_write = deployment._write_new
    calls = 0

    def fail_second(path, content):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("SYNTHETIC_SECRET_MUST_NOT_PRINT")
        original_write(path, content)

    monkeypatch.setattr(deployment, "_write_new", fail_second)
    with pytest.raises(deployment.DeploymentError) as caught:
        deployment.prepare(local_dir)
    assert "SYNTHETIC_SECRET_MUST_NOT_PRINT" not in str(caught.value)
    assert "保留" in str(caught.value) and len(list(local_dir.iterdir())) == 1
    with pytest.raises(deployment.DeploymentError):
        deployment.prepare(local_dir)
    assert calls == 2  # No automatic rewrite, cleanup or retry.


def test_read_config_never_reads_root_password_or_native_credentials(tmp_path, monkeypatch):
    local_dir = tmp_path / ".local"
    deployment.prepare(local_dir)
    original_read = Path.read_text
    read_paths = []

    def guarded_read(path, *args, **kwargs):
        assert path.parent == local_dir
        assert path.name != "mysql_root_password"
        read_paths.append(path.name)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    config = deployment.read_config(local_dir)
    assert config.mysql_password != config.redis_password
    assert set(read_paths) == {"mysql_app_password", "redis_password", "redis.conf"}


@pytest.mark.parametrize("mutation", [
    "extra_file", "missing_file", "directory_mode", "file_mode", "file_symlink",
    "directory_symlink", "invalid_password", "same_password", "redis_config",
])
def test_read_config_rejects_partial_insecure_or_mismatched_configuration(tmp_path, mutation):
    local_dir = tmp_path / ".local"
    deployment.prepare(local_dir)
    config = deployment.read_config(local_dir)
    if mutation == "extra_file":
        (local_dir / "unexpected").write_text("synthetic", encoding="utf-8")
    elif mutation == "missing_file":
        (local_dir / "redis.conf").unlink()
    elif mutation == "directory_mode":
        local_dir.chmod(0o755)
    elif mutation == "file_mode":
        (local_dir / "mysql_app_password").chmod(0o644)
    elif mutation == "file_symlink":
        app = local_dir / "mysql_app_password"
        outside = tmp_path / "synthetic-password"
        outside.write_bytes(app.read_bytes())
        outside.chmod(0o600)
        app.unlink()
        app.symlink_to(outside)
    elif mutation == "directory_symlink":
        actual = tmp_path / "actual"
        local_dir.rename(actual)
        local_dir.symlink_to(actual, target_is_directory=True)
    elif mutation == "invalid_password":
        (local_dir / "mysql_app_password").write_text("BAD_SYNTHETIC_SECRET", encoding="utf-8")
    elif mutation == "same_password":
        (local_dir / "mysql_app_password").write_text(config.redis_password + "\n", encoding="utf-8")
    else:
        with (local_dir / "redis.conf").open("a", encoding="utf-8") as stream:
            stream.write("protected-mode no\n")
    with pytest.raises(deployment.DeploymentError) as caught:
        deployment.read_config(local_dir)
    assert config.mysql_password not in str(caught.value)
    assert config.redis_password not in str(caught.value)
    assert "BAD_SYNTHETIC_SECRET" not in str(caught.value)


@pytest.mark.parametrize("action", ["up", "down"])
def test_compose_commands_are_fixed_no_shell_no_secrets_and_never_delete_volumes(tmp_path, monkeypatch, action):
    local_dir = tmp_path / ".local"
    deployment.prepare(local_dir)
    config = deployment.read_config(local_dir)
    before = {path.name: path.read_bytes() for path in local_dir.iterdir()}
    monkeypatch.setattr(deployment, "LOCAL_DIR", local_dir)
    runner = Mock(return_value=SimpleNamespace(returncode=0))
    deployment.compose(action, local_dir=local_dir, runner=runner)
    runner.assert_called_once()
    args, options = runner.call_args
    command = args[0]
    assert command[:7] == [
        "docker", "compose", "--project-name", "fault-diagnosis-learning",
        "--file", str(deployment.COMPOSE_FILE), action,
    ]
    assert command[7:] == (["-d", "--wait", "--wait-timeout", "90"] if action == "up" else [])
    assert "-v" not in command and "--volumes" not in command
    assert options == {"cwd": deployment.PROJECT_ROOT, "capture_output": True,
                       "text": True, "timeout": 180, "check": False}
    assert "shell" not in options
    assert config.mysql_password not in repr(runner.call_args)
    assert config.redis_password not in repr(runner.call_args)
    assert {path.name: path.read_bytes() for path in local_dir.iterdir()} == before


def test_compose_down_does_not_require_or_remove_secret_files(tmp_path, monkeypatch):
    local_dir = tmp_path / ".local"
    monkeypatch.setattr(deployment, "LOCAL_DIR", local_dir)
    read = Mock(side_effect=AssertionError("Stopping must not require secrets"))
    monkeypatch.setattr(deployment, "read_config", read)
    runner = Mock(return_value=SimpleNamespace(returncode=0))
    deployment.compose("down", local_dir=local_dir, runner=runner)
    read.assert_not_called()
    assert not local_dir.exists()
    assert runner.call_args.args[0][-1] == "down"


@pytest.mark.parametrize("action", ["restart", "rm", "down -v", "build"])
def test_compose_rejects_unapproved_commands_before_execution(tmp_path, action):
    runner = Mock()
    with pytest.raises(deployment.DeploymentError):
        deployment.compose(action, local_dir=tmp_path, runner=runner)
    runner.assert_not_called()


def test_compose_rejects_redirected_configuration_before_execution(tmp_path):
    runner = Mock()
    with pytest.raises(deployment.DeploymentError):
        deployment.compose("up", local_dir=tmp_path, runner=runner)
    runner.assert_not_called()


@pytest.mark.parametrize("failure", ["returncode", "os_error", "timeout"])
def test_compose_failure_is_sanitized_no_retry_or_fallback(tmp_path, monkeypatch, failure):
    local_dir = tmp_path / ".local"
    deployment.prepare(local_dir)
    monkeypatch.setattr(deployment, "LOCAL_DIR", local_dir)
    secret = "SYNTHETIC_DOCKER_ERROR_MUST_NOT_PRINT"
    runner = Mock(return_value=SimpleNamespace(returncode=1, stdout=secret, stderr=secret))
    if failure == "os_error":
        runner.side_effect = OSError(secret)
    elif failure == "timeout":
        runner.side_effect = subprocess.TimeoutExpired(secret, 180, output=secret, stderr=secret)
    with pytest.raises(deployment.DeploymentError) as caught:
        deployment.compose("up", local_dir=local_dir, runner=runner)
    assert secret not in str(caught.value)
    assert "保留" in str(caught.value)
    runner.assert_called_once()
    assert local_dir.exists()


def fake_services(monkeypatch):
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("redis")
    from qa_platform.storage import mysql_setup, retrieval_cache, sql_conversations

    engine = Mock()
    cache = Mock()
    make_engine = Mock(return_value=engine)
    make_cache = Mock(return_value=cache)
    schema = Mock()
    monkeypatch.setattr(mysql_setup, "make_engine", make_engine)
    monkeypatch.setattr(retrieval_cache, "RetrievalCache", make_cache)
    monkeypatch.setattr(sql_conversations, "check_schema", schema)
    monkeypatch.setattr(deployment, "read_config", Mock(return_value=deployment.RuntimeSecrets("MYSQL_SYNTHETIC", "REDIS_SYNTHETIC")))
    return engine, cache, make_engine, make_cache, schema


def test_open_services_uses_isolated_ports_and_application_credentials(monkeypatch, tmp_path):
    engine, cache, make_engine, make_cache, schema = fake_services(monkeypatch)
    assert deployment.open_services(tmp_path) == (engine, cache)
    config = make_engine.call_args.args[0]
    assert config.port == 13306 and config.password == "MYSQL_SYNTHETIC"
    make_cache.assert_called_once_with(port=16379, password="REDIS_SYNTHETIC")
    schema.assert_called_once_with(engine)
    cache.ping.assert_called_once_with()
    engine.dispose.assert_not_called()
    cache.close.assert_not_called()


@pytest.mark.parametrize("stage", ["schema", "redis"])
def test_open_services_failure_disposes_without_native_fallback(monkeypatch, tmp_path, stage):
    engine, cache, make_engine, make_cache, schema = fake_services(monkeypatch)
    if stage == "schema":
        schema.side_effect = RuntimeError("MYSQL_SYNTHETIC")
    else:
        cache.ping.side_effect = RuntimeError("REDIS_SYNTHETIC")
    with pytest.raises(deployment.DeploymentError) as caught:
        deployment.open_services(tmp_path)
    assert "MYSQL_SYNTHETIC" not in str(caught.value) and "REDIS_SYNTHETIC" not in str(caught.value)
    assert make_engine.call_count == make_cache.call_count == 1
    engine.dispose.assert_called_once_with()
    cache.close.assert_called_once_with()


def test_database_port_extension_preserves_native_default_and_only_changes_explicit_compose(monkeypatch):
    import sqlalchemy
    from qa_platform.storage.mysql_setup import DatabaseConfig, make_engine

    create = Mock(return_value=object())
    monkeypatch.setattr(sqlalchemy, "create_engine", create)
    for config, expected_port in ((DatabaseConfig("SYNTHETIC_PASSWORD"), 3306),
                                  (DatabaseConfig("SYNTHETIC_PASSWORD", port=13306), 13306)):
        make_engine(config)
        url = create.call_args.args[0]
        assert url.port == expected_port and url.host == "127.0.0.1"
        assert url.username == "fault_qa" and url.database == "fault_diagnosis"
        assert url.password == "SYNTHETIC_PASSWORD" and "SYNTHETIC_PASSWORD" not in repr(config)


@pytest.mark.parametrize("port", [3307, 6379, "13306", True])
def test_database_configuration_rejects_other_ports(port):
    from qa_platform.storage.mysql_setup import DatabaseConfig, SetupError

    with pytest.raises(SetupError):
        DatabaseConfig("SYNTHETIC_PASSWORD", port=port)


def test_redis_explicit_container_connection_uses_auth_and_zero_retry_without_eager_io():
    from qa_platform.storage.retrieval_cache import RetrievalCache

    fake = Mock()
    fake.ping.return_value = True
    factory = Mock(return_value=fake)
    instance = RetrievalCache(client_factory=factory, port=16379, password="REDIS_SYNTHETIC")
    factory.assert_not_called()
    instance.ping()
    options = factory.call_args.kwargs
    assert options["host"] == "127.0.0.1" and options["port"] == 16379 and options["db"] == 0
    assert options["password"] == "REDIS_SYNTHETIC" and options["retry"]._retries == 0
    assert options["retry_on_timeout"] is False and options["retry_on_error"] == []
    instance.close()
    fake.close.assert_called_once_with()


@pytest.mark.parametrize("options", [
    {"port": 16379}, {"port": 16379, "password": " "},
    {"port": 16379, "password": 123}, {"port": 16380, "password": "SYNTHETIC"},
])
def test_redis_container_configuration_rejects_missing_auth_or_other_ports(options):
    from qa_platform.storage.retrieval_cache import CacheError, RetrievalCache

    factory = Mock()
    with pytest.raises(CacheError):
        RetrievalCache(client_factory=factory, **options)
    factory.assert_not_called()


def test_check_only_counts_rows_and_closes_resources(monkeypatch, tmp_path):
    engine, cache = Mock(), Mock()
    connection = Mock()
    connection.scalar.side_effect = [2, 7]
    engine.connect.return_value.__enter__ = Mock(return_value=connection)
    engine.connect.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(deployment, "open_services", Mock(return_value=(engine, cache)))
    result = deployment.check(tmp_path)
    assert result == {"mysql_port": 13306, "redis_port": 16379, "conversations": 2, "turns": 7}
    assert [str(call.args[0]) for call in connection.scalar.call_args_list] == [
        "SELECT COUNT(*) FROM qa_conversations", "SELECT COUNT(*) FROM qa_turns",
    ]
    connection.execute.assert_not_called()
    cache.close.assert_called_once_with()
    engine.dispose.assert_called_once_with()


def test_serve_refuses_shared_milvus_app_before_opening_any_services(monkeypatch, tmp_path):
    lock = Mock(side_effect=deployment.DeploymentError("Original application still running"))
    opening = Mock()
    monkeypatch.setattr(deployment, "require_exclusive_app", lock)
    monkeypatch.setattr(deployment, "open_services", opening)
    with pytest.raises(deployment.DeploymentError):
        deployment.serve(tmp_path)
    lock.assert_called_once_with()
    opening.assert_not_called()


@pytest.mark.parametrize("occupied", [None, 8766, 8767])
def test_exclusive_app_checks_both_local_ports_without_ending_processes(monkeypatch, occupied):
    bound = []
    probes = []
    kill = Mock(side_effect=AssertionError("The port probe must not kill an existing application"))
    monkeypatch.setattr(deployment.os, "kill", kill)

    class FakeSocket:
        def __init__(self):
            self.events = []
            probes.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.events.append(("close",))
            return False

        def setsockopt(self, level, option, value):
            assert (level, option, value) == (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.events.append(("setsockopt", level, option, value))

        def bind(self, address):
            assert self.events == [("setsockopt", socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]
            self.events.append(("bind", address))
            bound.append(address)
            if address[1] == occupied:
                raise OSError("SYNTHETIC_BUSY_PORT")

    factory = Mock(side_effect=lambda *_args: FakeSocket())
    monkeypatch.setattr(socket, "socket", factory)
    if occupied is None:
        deployment.require_exclusive_app()
        assert bound == [("127.0.0.1", 8766), ("127.0.0.1", 8767)]
    else:
        with pytest.raises(deployment.DeploymentError) as caught:
            deployment.require_exclusive_app()
        assert "SYNTHETIC_BUSY_PORT" not in str(caught.value)
        assert "手动停止" in str(caught.value)
        assert bound[-1] == ("127.0.0.1", occupied)
    assert all(call.args == (socket.AF_INET, socket.SOCK_STREAM) for call in factory.call_args_list)
    assert all(probe.events[-1] == ("close",) for probe in probes)
    kill.assert_not_called()


@pytest.mark.parametrize("time_wait_port", [8766, 8767])
def test_exclusive_app_allows_stopped_server_time_wait_without_reusing_active_listeners(monkeypatch, time_wait_port):
    """Simulate TIME_WAIT offline: address reuse must be enabled before either bind."""
    import errno

    bound = []

    class TimeWaitSocket:
        def __init__(self):
            self.reuse_address = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def setsockopt(self, level, option, value):
            # SO_REUSEPORT would allow sharing a listener and must never be used here.
            assert (level, option, value) == (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.reuse_address = True

        def bind(self, address):
            if address[1] == time_wait_port and not self.reuse_address:
                raise OSError(errno.EADDRINUSE, "SYNTHETIC_TIME_WAIT")
            bound.append(address)

    monkeypatch.setattr(socket, "socket", Mock(side_effect=lambda *_args: TimeWaitSocket()))
    deployment.require_exclusive_app()
    assert bound == [("127.0.0.1", 8766), ("127.0.0.1", 8767)]


def test_serve_uses_local_8767_mysql_redis_and_closes_resources(monkeypatch, tmp_path):
    from qa_platform import retrieval_lab
    from qa_platform.storage import sql_conversations
    import uvicorn

    engine, cache, app, store = Mock(), Mock(), object(), object()
    exclusive = Mock()
    create = Mock(return_value=app)
    make_store = Mock(return_value=store)
    run = Mock()
    monkeypatch.setattr(deployment, "require_exclusive_app", exclusive)
    monkeypatch.setattr(deployment, "open_services", Mock(return_value=(engine, cache)))
    monkeypatch.setattr(retrieval_lab, "create_app", create)
    monkeypatch.setattr(sql_conversations, "SQLConversationStore", make_store)
    monkeypatch.setattr(uvicorn, "run", run)
    deployment.serve(tmp_path)
    exclusive.assert_called_once_with()
    make_store.assert_called_once_with(engine)
    create.assert_called_once_with(conversation_store=store, storage_backend="mysql",
                                   retrieval_cache=cache, cache_backend="redis", deployment_mode=True)
    run.assert_called_once_with(app, host="127.0.0.1", port=8767, access_log=False)
    cache.close.assert_called_once_with()
    engine.dispose.assert_called_once_with()


@pytest.mark.parametrize("action", ["prepare", "up", "check", "serve", "down"])
def test_cli_dispatches_only_explicit_action_without_secret_output(monkeypatch, capsys, action):
    callbacks = {name: Mock(return_value={"turns": 0}) for name in ("prepare", "compose", "check", "serve")}
    for name, callback in callbacks.items():
        monkeypatch.setattr(deployment, name, callback)
    monkeypatch.setattr(sys, "argv", ["serve_compose.py", "--" + action])
    assert deployment.main() == 0
    selected = "compose" if action in {"up", "down"} else action
    callbacks[selected].assert_called_once_with(*([action] if selected == "compose" else []))
    for name, callback in callbacks.items():
        if name != selected:
            callback.assert_not_called()
    assert "SYNTHETIC_SECRET" not in capsys.readouterr().out


def test_cli_does_not_print_unexpected_exception_details(monkeypatch, capsys):
    monkeypatch.setattr(deployment, "prepare", Mock(side_effect=RuntimeError("SYNTHETIC_SECRET_MUST_NOT_PRINT")))
    monkeypatch.setattr(sys, "argv", ["serve_compose.py", "--prepare"])
    assert deployment.main() == 1
    output = capsys.readouterr().out
    assert "SYNTHETIC_SECRET_MUST_NOT_PRINT" not in output
    assert "未完成" in output


def test_cli_rejects_multiple_actions_before_any_execution(monkeypatch):
    prepare, compose = Mock(), Mock()
    monkeypatch.setattr(deployment, "prepare", prepare)
    monkeypatch.setattr(deployment, "compose", compose)
    monkeypatch.setattr(sys, "argv", ["serve_compose.py", "--prepare", "--up"])
    with pytest.raises(SystemExit) as caught:
        deployment.main()
    assert caught.value.code == 2
    prepare.assert_not_called()
    compose.assert_not_called()
