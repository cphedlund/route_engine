import pymupdf
import pytest
import requests
from fastapi.testclient import TestClient

import app as appmod
import atlas_mapbox
import pdf_maps

KEY = "test-api-key"
OVERLAY_ROUTE_ID = "a1a107f84261f78e"
OVERLAY_ROUTE_NAME = "Bernal Hill Loop"
FALLBACK_ROUTE_ID = "fa333080a1500ea4"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "API_KEY", KEY)
    monkeypatch.setattr(appmod, "_MAP_PDF_CACHE", appmod.OrderedDict())
    monkeypatch.setattr(appmod, "_MAP_PDF_CACHE_BYTES", 0)
    return TestClient(appmod.app)


def _get(client, rid, key=KEY):
    headers = {"X-API-Key": key} if key else {}
    return client.get(f"/routes/{rid}/map.pdf", headers=headers)


def test_requires_api_key(client):
    assert _get(client, OVERLAY_ROUTE_ID, key=None).status_code == 401
    assert _get(client, OVERLAY_ROUTE_ID, key="wrong").status_code == 401


def test_unknown_route_404(client):
    r = _get(client, "does-not-exist")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["detail"] == "Route not found"


def test_route_ids_are_classified_as_expected():
    assert OVERLAY_ROUTE_ID in appmod._RAW_BY_ID
    assert FALLBACK_ROUTE_ID in appmod._RAW_BY_ID
    coords, _ = pdf_maps.atlas_scc.parse_gpx(appmod._RAW_BY_ID[FALLBACK_ROUTE_ID]["_path"])
    assert pdf_maps.pick_sheet(coords) is None


def test_overlay_pdf_ok(client):
    r = _get(client, OVERLAY_ROUTE_ID)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"] == 'attachment; filename="santa-teresa-bernal-hill-loop.pdf"'
    assert r.headers["cache-control"].startswith("private")
    assert r.headers["x-map-mode"] == "overlay"
    assert r.content[:4] == b"%PDF"
    with pymupdf.open(stream=r.content, filetype="pdf") as doc:
        assert doc.page_count >= 1
        assert OVERLAY_ROUTE_NAME in doc[0].get_text()
    again = _get(client, OVERLAY_ROUTE_ID)
    assert again.status_code == 200 and again.content == r.content


def test_lookup_by_route_name(client):
    r = _get(client, "bernal-hill-loop")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"


def test_fallback_without_token_is_clean_503(client, monkeypatch):
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    called = []
    monkeypatch.setattr(atlas_mapbox.requests, "get", lambda *a, **k: called.append(a))
    r = _get(client, FALLBACK_ROUTE_ID)
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/json")
    assert "MAPBOX_TOKEN" in r.json()["detail"]
    assert not called
    assert FALLBACK_ROUTE_ID not in appmod._MAP_PDF_CACHE


def test_fallback_network_error_is_clean_503(client, monkeypatch):
    monkeypatch.setenv("MAPBOX_TOKEN", "pk.offline-test")

    def boom(*a, **k):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(atlas_mapbox.requests, "get", boom)
    r = _get(client, FALLBACK_ROUTE_ID)
    assert r.status_code == 503
    assert "pk.offline-test" not in r.text
    assert "Mapbox request failed" in r.json()["detail"]


def _fake_png():
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 40), False)
    pix.set_rect(pix.irect, (200, 220, 200))
    return pix.tobytes("png")


class _FakeResp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def test_fallback_with_stubbed_mapbox_returns_pdf(client, monkeypatch):
    monkeypatch.setenv("MAPBOX_TOKEN", "pk.offline-test")
    png = _fake_png()
    monkeypatch.setattr(atlas_mapbox.requests, "get", lambda *a, **k: _FakeResp(png))
    r = _get(client, FALLBACK_ROUTE_ID)
    assert r.status_code == 200
    assert r.headers["x-map-mode"] == "fallback"
    assert r.content[:4] == b"%PDF"
    with pymupdf.open(stream=r.content, filetype="pdf") as doc:
        assert doc.page_count >= 1


def test_overlay_failure_falls_back_to_mapbox(client, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("overlay broke")

    monkeypatch.setattr(pdf_maps.atlas_scc, "render_scc_pdf", broken)
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    r = _get(client, OVERLAY_ROUTE_ID)
    assert r.status_code == 503

    monkeypatch.setenv("MAPBOX_TOKEN", "pk.offline-test")
    monkeypatch.setattr(atlas_mapbox.requests, "get", lambda *a, **k: _FakeResp(_fake_png()))
    r = _get(client, OVERLAY_ROUTE_ID)
    assert r.status_code == 200
    assert r.headers["x-map-mode"] == "fallback"


def test_unexpected_error_is_json_500_without_internals(client, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("secret internal path /x/y")

    monkeypatch.setattr(pdf_maps, "render_route_map", broken)
    r = _get(client, OVERLAY_ROUTE_ID)
    assert r.status_code == 500
    assert r.json() == {"detail": "Map render failed"}


def test_cache_is_bounded_by_bytes(client, monkeypatch):
    monkeypatch.setattr(appmod, "MAP_PDF_CACHE_MAX_BYTES", 20)
    monkeypatch.setattr(pdf_maps, "render_route_map", lambda *a, **k: (b"%PDF-123456", "overlay", "p"))
    assert _get(client, OVERLAY_ROUTE_ID).status_code == 200
    assert _get(client, FALLBACK_ROUTE_ID).status_code == 200
    assert appmod._MAP_PDF_CACHE_BYTES <= 20
    assert list(appmod._MAP_PDF_CACHE) == [FALLBACK_ROUTE_ID]
