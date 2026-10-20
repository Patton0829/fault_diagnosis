"""MySQL configuration/bootstrap safety tests with synthetic secrets and no DB."""

import builtins
from pathlib import Path
import runpy
import socket
import sys
from unittest.mock import Mock

import pytest

pytest.importorskip("sqlalchemy")

import sqlalchemy

from qa_platform.storage import mysql_setup as setup
from qa_platform.storage import sql_conversations as sql


SECRET = "SYNTHETIC_PASSWORD_MUST_NOT_PRINT"


@pytest.fixture(autouse=True)
def prohibit_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Database setup tests must not connect to a real server")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def config_file(tmp_path, **changes):
    values = {"DB_HOST": "127.0.0.1", "DB_PORT": "3306", "DB_NAME": "fault_diagnosis",
              "DB_USER": "fault_qa", "DB_PASSWORD": SECRET, **changes}
    path = tmp_path / "synthetic-database.env"
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    return path


def test_default_plan_never_reads_keys_config_imports_sqlalchemy_or_connects(monkeypatch, capsys):
    forbidden = Mock(side_effect=AssertionError("Default plan must be zero I/O"))
    for name in ("read_config", "make_engine", "bootstrap", "getpass"):
        monkeypatch.setattr(setup, name, forbidden)
    original_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        if name.startswith("sqlalchemy"):
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_guard)
    monkeypatch.setattr(sys, "argv", ["setup_mysql.py"])
    assert setup.main() == 0
    assert "计划完成" in capsys.readouterr().out
    forbidden.assert_not_called()


def test_script_entry_default_plan_is_also_read_only(monkeypatch, capsys):
    original_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        if name.startswith("sqlalchemy"):
            raise AssertionError("Default entry imported SQLAlchemy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_guard)
    monkeypatch.setattr(sys, "argv", ["setup_mysql.py"])
    monkeypatch.delitem(sys.modules, "qa_platform.storage.mysql_setup", raising=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "setup_mysql.py"
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(script), run_name="__main__")
    assert caught.value.code == 0
    assert "计划完成" in capsys.readouterr().out


def test_config_repr_and_engine_url_preserve_special_password_without_echo(tmp_path, monkeypatch):
    password = "Synthetic@p:/#?${NOT_INTERPOLATED}"
    config = setup.read_config(config_file(tmp_path, DB_PASSWORD=password))
    assert config.password == password and password not in repr(config)
    create = Mock(return_value=object())
    monkeypatch.setattr(sqlalchemy, "create_engine", create)
    setup.make_engine(config)
    url = create.call_args.args[0]
    options = create.call_args.kwargs
    assert url.password == password and url.username == "fault_qa"
    assert url.host == "127.0.0.1" and url.port == 3306 and url.database == "fault_diagnosis"
    assert password not in str(url)
    assert options["echo"] is False and options["hide_parameters"] is True
    assert options["connect_args"] == {"connect_timeout": 3, "read_timeout": 5, "write_timeout": 5}


@pytest.mark.parametrize("changes", [
    {"DB_USER": "root"}, {"DB_HOST": "remote.invalid"}, {"DB_PORT": "3307"},
    {"DB_NAME": "another_database"}, {"DB_PASSWORD": ""},
])
def test_config_rejects_root_remote_or_unexpected_namespace(tmp_path, changes):
    with pytest.raises(setup.SetupError) as caught:
        setup.read_config(config_file(tmp_path, **changes))
    assert SECRET not in str(caught.value)


def test_config_reservation_is_private_exclusive_and_never_overwrites(tmp_path):
    path = tmp_path / "models" / "database.env"
    setup._reserve_config(path, setup.DatabaseConfig(SECRET))
    assert path.stat().st_mode & 0o777 == 0o600
    original = path.read_bytes()
    assert b"DB_USER=fault_qa" in original and b"root" not in original.split(b"DB_USER=")[1]
    with pytest.raises(FileExistsError):
        setup._reserve_config(path, setup.DatabaseConfig("OTHER_SYNTHETIC_PASSWORD"))
    assert path.read_bytes() == original


def test_bootstrap_refuses_existing_config_before_creating_engine(tmp_path, monkeypatch):
    path = tmp_path / "existing.env"
    path.write_text("Synthetic existing content", encoding="utf-8")
    create = Mock(side_effect=AssertionError("Existing configuration must stop before engine creation"))
    monkeypatch.setattr(sqlalchemy, "create_engine", create)
    with pytest.raises(setup.SetupError):
        setup.bootstrap(SECRET, path)
    assert path.read_text(encoding="utf-8") == "Synthetic existing content"
    create.assert_not_called()


def test_bootstrap_refuses_symlink_config_without_following_or_overwriting(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.write_text("Synthetic target", encoding="utf-8")
    link = tmp_path / "database.env"
    link.symlink_to(target)
    create = Mock(side_effect=AssertionError("Symlink configuration must stop before engine creation"))
    monkeypatch.setattr(sqlalchemy, "create_engine", create)
    with pytest.raises(setup.SetupError):
        setup.bootstrap(SECRET, link)
    assert target.read_text(encoding="utf-8") == "Synthetic target"
    create.assert_not_called()


@pytest.mark.parametrize("existing", ["database", "account"])
def test_bootstrap_existing_namespace_never_creates_or_reserves_anything(tmp_path, monkeypatch, existing):
    path = tmp_path / "models" / "database.env"
    connection = Mock()
    connection.scalar.side_effect = [1, 0] if existing == "database" else [0, 1]
    manager = Mock()
    manager.__enter__ = Mock(return_value=connection)
    manager.__exit__ = Mock(return_value=False)
    engine = Mock()
    engine.connect.return_value = manager
    monkeypatch.setattr(sqlalchemy, "create_engine", Mock(return_value=engine))
    reserve = Mock(side_effect=AssertionError("Existing database/account must stop before reserving config"))
    monkeypatch.setattr(setup, "_reserve_config", reserve)
    with pytest.raises(setup.SetupError) as caught:
        setup.bootstrap(SECRET, path)
    assert SECRET not in str(caught.value) and not path.exists()
    assert connection.scalar.call_count == 2
    connection.exec_driver_sql.assert_not_called()
    reserve.assert_not_called()
    engine.dispose.assert_called_once_with()


def test_bootstrap_cli_requires_interactive_terminal_before_password_or_connection(monkeypatch, capsys):
    forbidden = Mock(side_effect=AssertionError("Non-interactive mode must not read password or connect"))
    monkeypatch.setattr(sys, "argv", ["setup_mysql.py", "--bootstrap"])
    monkeypatch.setattr(sys, "stdin", Mock(isatty=lambda: False))
    monkeypatch.setattr(setup, "getpass", forbidden)
    monkeypatch.setattr(setup, "bootstrap", forbidden)
    assert setup.main() == 1
    assert "不接受管道密码" in capsys.readouterr().out
    forbidden.assert_not_called()


def synthetic_root(monkeypatch):
    connection = Mock()
    connection.scalar.side_effect = [0, 0]
    manager = Mock()
    manager.__enter__ = Mock(return_value=connection)
    manager.__exit__ = Mock(return_value=False)
    engine = Mock()
    engine.connect.return_value = manager
    monkeypatch.setattr(sqlalchemy, "create_engine", Mock(return_value=engine))
    return engine, connection


def test_bootstrap_fresh_namespace_uses_bound_app_password_and_crud_only(tmp_path, monkeypatch):
    path = tmp_path / "database.env"
    root, connection = synthetic_root(monkeypatch)
    create_tables = Mock()
    monkeypatch.setattr(sql.Base.metadata, "create_all", create_tables)
    check = Mock()
    monkeypatch.setattr(sql, "check_schema", check)
    app = Mock()
    make_app = Mock(return_value=app)
    monkeypatch.setattr(setup, "make_engine", make_app)
    setup.bootstrap(SECRET, path)
    saved = setup.read_config(path)
    assert saved.password != SECRET and saved.password.endswith("aA1!")
    assert SECRET not in path.read_text(encoding="utf-8")
    statements = connection.exec_driver_sql.call_args_list
    assert statements[0].args == ("CREATE DATABASE `fault_diagnosis` CHARACTER SET utf8mb4",)
    assert statements[1].args == ("USE `fault_diagnosis`",)
    assert statements[2].args == ("CREATE USER 'fault_qa'@'127.0.0.1' IDENTIFIED BY %s", (saved.password,))
    assert statements[3].args == (
        "GRANT SELECT, INSERT, UPDATE, DELETE ON `fault_diagnosis`.* TO 'fault_qa'@'127.0.0.1'",
    )
    create_tables.assert_called_once_with(connection, checkfirst=False)
    check.assert_called_once_with(app)
    connection.commit.assert_called_once_with()
    root.dispose.assert_called_once_with()
    app.dispose.assert_called_once_with()


def test_bootstrap_partial_ddl_failure_keeps_generated_config_and_sanitizes_error(tmp_path, monkeypatch):
    path = tmp_path / "database.env"
    root, connection = synthetic_root(monkeypatch)
    connection.exec_driver_sql.side_effect = sqlalchemy.exc.SQLAlchemyError(SECRET)
    with pytest.raises(setup.SetupError) as caught:
        setup.bootstrap(SECRET, path)
    assert "已保留" in str(caught.value) and SECRET not in str(caught.value)
    assert path.is_file() and setup.read_config(path).password != SECRET
    assert connection.exec_driver_sql.call_count == 1  # No cleanup DROP or automatic retry.
    root.dispose.assert_called_once_with()
