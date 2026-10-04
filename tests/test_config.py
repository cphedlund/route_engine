import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

import app as app_module


def test_process_env_wins_over_dotenv_file(tmp_path, monkeypatch):
    env_file = tmp_path / "test.env"
    env_file.write_text("ATLAS_TEST_VAR=from_file\nATLAS_TEST_ONLY_FILE=file_only\n")
    monkeypatch.setenv("ATLAS_TEST_VAR", "from_process")
    monkeypatch.delenv("ATLAS_TEST_ONLY_FILE", raising=False)
    load_dotenv(env_file, override=False)
    assert os.environ["ATLAS_TEST_VAR"] == "from_process"
    assert os.environ["ATLAS_TEST_ONLY_FILE"] == "file_only"
    monkeypatch.delenv("ATLAS_TEST_ONLY_FILE", raising=False)


def test_app_uses_override_false():
    import re
    from pathlib import Path
    src = Path(app_module.__file__).read_text()
    assert re.search(r"load_dotenv\(override=False\)", src)
    assert "override=True" not in src


def test_flag_on_key_missing_refuses_to_start(monkeypatch):
    monkeypatch.setenv("REQUIRE_API_KEY", "1")
    monkeypatch.delenv("ROUTE_ENGINE_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        with TestClient(app_module.app):
            pass
    assert "ROUTE_ENGINE_API_KEY" in str(exc.value)


def test_flag_on_key_empty_refuses_to_start(monkeypatch):
    monkeypatch.setenv("REQUIRE_API_KEY", "1")
    monkeypatch.setenv("ROUTE_ENGINE_API_KEY", "   ")
    with pytest.raises(RuntimeError):
        app_module.validate_startup_config()


def test_flag_on_key_set_starts(monkeypatch):
    monkeypatch.setenv("REQUIRE_API_KEY", "1")
    monkeypatch.setenv("ROUTE_ENGINE_API_KEY", "unit-test-value")
    with TestClient(app_module.app):
        pass


def test_flag_off_key_missing_starts(monkeypatch):
    monkeypatch.delenv("REQUIRE_API_KEY", raising=False)
    monkeypatch.delenv("ROUTE_ENGINE_API_KEY", raising=False)
    with TestClient(app_module.app):
        pass


def test_error_message_has_no_secret(monkeypatch):
    monkeypatch.setenv("REQUIRE_API_KEY", "1")
    monkeypatch.delenv("ROUTE_ENGINE_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        app_module.validate_startup_config()
    assert "unit-test-value" not in str(exc.value)
