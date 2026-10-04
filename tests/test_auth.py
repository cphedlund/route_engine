"""Every endpoint except / and /health must reject requests without a key.

API_KEY and MAKE_INGRESS_KEY are read at import time, so the tests patch the
module attributes to throwaway values. A new route that is not listed here
fails test_every_route_is_classified, forcing a decision about its auth.
"""
import pytest
from fastapi.testclient import TestClient

import app as appmod

PUBLIC = {("GET", "/"), ("GET", "/health")}

# (method, path template, concrete path, json body)
API_KEY_PROTECTED = [
    ("GET", "/routes/{route_id}/map.pdf", "/routes/any-route/map.pdf", None),
    ("POST", "/start_search", "/start_search", {"query": "hike"}),
    ("POST", "/more_results", "/more_results", {"session_id": "x", "n": 3}),
]

# Make ingress uses its own key (X-Make-Key), not X-API-Key.
MAKE_KEY_PROTECTED = [
    ("POST", "/make/translate_and_search", "/make/translate_and_search", {"query": "hike"}),
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "API_KEY", "test-api-key")
    monkeypatch.setattr(appmod, "MAKE_INGRESS_KEY", "test-make-key")
    return TestClient(appmod.app)


def _call(client, method, path, body, headers=None):
    return client.request(method, path, json=body, headers=headers or {})


def test_every_route_is_classified():
    documented = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    known = PUBLIC | {(m, t) for m, t, _, _ in API_KEY_PROTECTED + MAKE_KEY_PROTECTED}
    for route in appmod.app.routes:
        if route.path in documented:
            continue
        for method in getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}:
            assert (method, route.path) in known, f"unclassified endpoint {method} {route.path}"


@pytest.mark.parametrize("method,path", sorted(PUBLIC))
def test_public_endpoints_need_no_key(client, method, path):
    assert _call(client, method, path, None).status_code == 200


@pytest.mark.parametrize("method,template,path,body", API_KEY_PROTECTED)
def test_api_key_endpoints_reject_missing_key(client, method, template, path, body):
    assert _call(client, method, path, body).status_code == 401


@pytest.mark.parametrize("method,template,path,body", API_KEY_PROTECTED)
def test_api_key_endpoints_reject_wrong_key(client, method, template, path, body):
    assert _call(client, method, path, body, {"X-API-Key": "wrong"}).status_code == 401


@pytest.mark.parametrize("method,template,path,body", MAKE_KEY_PROTECTED)
def test_make_endpoint_rejects_missing_key(client, method, template, path, body):
    assert _call(client, method, path, body).status_code == 401


@pytest.mark.parametrize("method,template,path,body", MAKE_KEY_PROTECTED)
def test_make_endpoint_rejects_api_key_alone(client, method, template, path, body):
    assert _call(client, method, path, body, {"X-API-Key": "test-api-key"}).status_code == 401


def test_start_search_accepts_valid_key(client):
    r = _call(client, "POST", "/start_search", {"query": "hike"}, {"X-API-Key": "test-api-key"})
    assert r.status_code == 200
