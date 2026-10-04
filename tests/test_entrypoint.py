"""Guard the production start target.

Railway runs railway.json's deploy.startCommand. If the module:attr it names stops
importing (as when main.py was deleted while Railway still defaulted to main:app),
these tests fail in CI instead of crash-looping production.
"""
import importlib
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


def _start_command() -> str:
    config = json.loads((ROOT / "railway.json").read_text())
    return config["deploy"]["startCommand"]


def test_app_serves_openapi():
    from app import app

    assert TestClient(app).get("/openapi.json").status_code == 200


def test_railway_start_target_imports():
    match = re.search(r"\buvicorn\s+([\w.]+):(\w+)", _start_command())
    assert match, "startCommand must be 'uvicorn <module>:<attr> ...'"
    module, attr = match.groups()
    assert hasattr(importlib.import_module(module), attr)


def test_railway_binds_port_and_healthcheck_is_public():
    config = json.loads((ROOT / "railway.json").read_text())["deploy"]
    assert "$PORT" in config["startCommand"] or "${PORT" in config["startCommand"]
    from app import app

    assert TestClient(app).get(config["healthcheckPath"]).status_code == 200
