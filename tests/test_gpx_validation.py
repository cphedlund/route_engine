import math
from pathlib import Path

import pytest

import gpx_loader as L

ROOT = Path(__file__).resolve().parent.parent
LAT0, LNG0 = 37.30, -121.90


def _loop(n=40, ele=100.0, step_deg=0.0004, lat0=LAT0, lng0=LNG0):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / (n - 1)
        pts.append((lat0 + step_deg * math.sin(a), lng0 + step_deg * math.cos(a), ele + i * 0.1))
    return pts


def _gpx_xml(tracks):
    body = ""
    for t in tracks:
        seg = "".join(
            f'<trkpt lat="{la}" lon="{lo}">' + (f"<ele>{e}</ele>" if e is not None else "") + "</trkpt>"
            for la, lo, e in t
        )
        body += f"<trk><name>t</name><trkseg>{seg}</trkseg></trk>"
    return f'<?xml version="1.0"?><gpx version="1.1" creator="t" xmlns="http://www.topografix.com/GPX/1/1">{body}</gpx>'


def test_valid_track_passes():
    v = L.validate_track(_loop())
    assert v["ok"] and v["reason"] == "" and v["distance_miles"] > 0.1


def test_too_few_points():
    assert L.validate_track(_loop()[:5])["reason"] == "too_few_points"


def test_nan_coordinate():
    pts = _loop()
    pts[3] = (float("nan"), pts[3][1], 100.0)
    assert L.validate_track(pts)["reason"] == "nan_coordinate"


def test_missing_elevation():
    pts = [(a, b, None) for a, b, _ in _loop()]
    assert L.validate_track(pts)["reason"] == "missing_elevation"


def test_bad_elevation():
    pts = _loop()
    pts[5] = (pts[5][0], pts[5][1], 50000.0)
    assert L.validate_track(pts)["reason"] == "bad_elevation"


def test_teleport_jump():
    a = _loop(20)
    b = _loop(20, lat0=LAT0 + 0.1)
    assert L.validate_track(a + b)["reason"] == "teleport_jump"


def test_outside_region_rejected():
    assert L.validate_track(_loop(lat0=34.0, lng0=-118.0))["reason"] == "outside_region"


def test_outside_county_flagged_not_rejected():
    v = L.validate_track(_loop(lat0=37.51, lng0=-121.87))
    assert v["ok"] and v["warnings"] == ["outside_county"]


def test_zero_length():
    pts = [(LAT0, LNG0, 100.0)] * 20
    assert L.validate_track(pts)["reason"] == "zero_length"


def test_impossible_distance(monkeypatch):
    monkeypatch.setattr(L, "MAX_ROUTE_MILES", 0.01)
    assert L.validate_track(_loop())["reason"] == "impossible_distance"


def test_7903_mile_failure_mode_is_rejected(tmp_path):
    tracks = []
    for i in range(60):
        tracks.append([(36.95 + 0.008 * (i % 60), -122.1 + 0.015 * (i % 40) + 0.0001 * j, None) for j in range(12)])
    (tmp_path / "export.gpx").write_text(_gpx_xml(tracks))
    naive = L._extract_points(__import__("gpxpy").parse((tmp_path / "export.gpx").read_text()))
    naive_miles = sum(L.haversine_miles(a[:2], b[:2]) for a, b in zip(naive, naive[1:]))
    assert naive_miles > 50
    v = L.validate_track(naive)
    assert not v["ok"] and v["reason"] in ("missing_elevation", "teleport_jump")


def test_loader_rejects_and_counts(tmp_path):
    (tmp_path / "good.gpx").write_text(_gpx_xml([_loop()]))
    (tmp_path / "short.gpx").write_text(_gpx_xml([_loop()[:4]]))
    (tmp_path / "noele.gpx").write_text(_gpx_xml([[(a, b, None) for a, b, _ in _loop()]]))
    (tmp_path / "broken.gpx").write_text("not xml")
    routes = L.load_routes_from_gpx_dir(str(tmp_path))
    assert [r["name"] for r in routes] == ["good"]
    rep = L.LAST_LOAD_REPORT
    assert rep["scanned"] == 4 and rep["loaded"] == 1 and rep["rejected_count"] == 3
    assert rep["rejected_by_reason"] == {"missing_elevation": 1, "parse_error": 1, "too_few_points": 1}


def test_quarantined_export_exists_and_is_rejected():
    f = ROOT / "data" / "quarantine" / "export.gpx"
    assert f.exists() and not (ROOT / "data" / "gpx" / "export.gpx").exists()
    import gpxpy
    pts = L._extract_points(gpxpy.parse(f.read_text(encoding="utf-8", errors="ignore")))
    assert not L.validate_track(pts)["ok"]


def test_real_data_has_no_implausible_routes():
    import app
    assert all(r["distance_miles"] <= L.MAX_ROUTE_MILES for r in app._RAW_GPX_ROUTES)
    assert len(app._RAW_GPX_ROUTES) == 255
