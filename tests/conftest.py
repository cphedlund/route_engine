import os
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# No test may reach Mapbox with a real token. app.py calls load_dotenv(override=False),
# so an empty MAPBOX_TOKEN set here, before app is imported, keeps .env from supplying one.
# Scripts outside pytest get the same guarantee by running with MAPBOX_TOKEN= set.
os.environ["MAPBOX_TOKEN"] = ""


def _network_blocked(*args, **kwargs):
    raise requests.ConnectionError("network disabled in tests; stub requests.get explicitly")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Tests that need an HTTP response monkeypatch requests.get themselves; that later
    # setattr replaces this guard for the duration of the test.
    monkeypatch.setenv("MAPBOX_TOKEN", "")
    monkeypatch.setattr(requests, "get", _network_blocked)
    monkeypatch.setattr(requests.Session, "request", _network_blocked)
