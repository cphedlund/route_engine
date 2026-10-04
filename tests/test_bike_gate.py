import math

import pytest
from shapely.geometry import LineString
from shapely.strtree import STRtree

import osm_layers
from osm_layers import BICYCLE_NO_MAX_RUN_M, max_bicycle_no_run_m

LAT0, LNG0 = 37.2, -121.8
M_PER_DEG_LAT = osm_layers._EARTH_RADIUS_M * math.pi / 180
TRACK_M = 300.0


def _pt(m):
    return (LAT0 + m / M_PER_DEG_LAT, LNG0)


def _track():
    return [_pt(m) for m in range(0, int(TRACK_M) + 1, 25)]


def _layer(no_segments, gaps=()):
    cuts = sorted({0.0, TRACK_M, *[x for s in no_segments for x in s], *[x for g in gaps for x in g]})
    no_set = {tuple(s) for s in no_segments}
    gap_set = {tuple(g) for g in gaps}
    layer = osm_layers.OSMLayer("trails_test", "none.geojson")
    for a, b in zip(cuts, cuts[1:]):
        if (a, b) in gap_set:
            continue
        pa, pb = _pt(a), _pt(b)
        layer.geometries.append(LineString([(pa[1], pa[0]), (pb[1], pb[0])]))
        layer.properties.append({"highway": "path", "bicycle": "no"} if (a, b) in no_set else {"highway": "path"})
    layer.index = STRtree(layer.geometries)
    return layer


def _enrich(monkeypatch, no_segments, gaps=()):
    monkeypatch.setattr(osm_layers, "TRAILS", _layer(no_segments, gaps))
    return osm_layers.enrich_route(_track())


def test_threshold_constant():
    assert BICYCLE_NO_MAX_RUN_M == 50.0
    assert not hasattr(osm_layers, "BICYCLE_NO_MAX_FRACTION")


def test_40m_run_is_legal(monkeypatch):
    out = _enrich(monkeypatch, [(105.0, 145.0)])
    assert out["osm_bicycle_no_max_run_m"] == pytest.approx(40.0, abs=0.1)
    assert out["osm_bicycle_legal"] is True


def test_60m_run_is_illegal(monkeypatch):
    out = _enrich(monkeypatch, [(105.0, 165.0)])
    assert out["osm_bicycle_no_max_run_m"] == pytest.approx(60.0, abs=0.1)
    assert out["osm_bicycle_legal"] is False


def test_two_30m_runs_separated_by_legal_track_are_legal(monkeypatch):
    out = _enrich(monkeypatch, [(55.0, 85.0), (155.0, 185.0)])
    assert out["osm_bicycle_no_max_run_m"] == pytest.approx(30.0, abs=0.1)
    assert out["osm_bicycle_legal"] is True


def test_exactly_50m_run_is_illegal(monkeypatch):
    out = _enrich(monkeypatch, [(105.0, 155.0)])
    assert out["osm_bicycle_no_max_run_m"] == 50.0
    assert out["osm_bicycle_legal"] is False


def test_unsnapped_gap_between_no_samples_bridges_the_run(monkeypatch):
    out = _enrich(monkeypatch, [(74.0, 83.0), (197.0, 205.0)], gaps=[(83.0, 197.0)])
    assert out["osm_bicycle_no_max_run_m"] == pytest.approx(130.0, abs=0.1)
    assert out["osm_bicycle_legal"] is False


def test_unsnapped_samples_outside_a_run_do_not_extend_it(monkeypatch):
    out = _enrich(monkeypatch, [(74.0, 83.0)], gaps=[(83.0, TRACK_M)])
    assert out["osm_bicycle_no_max_run_m"] == pytest.approx(30.0, abs=0.1)
    assert out["osm_bicycle_legal"] is True


def test_no_snapped_way_defaults_to_legal(monkeypatch):
    out = _enrich(monkeypatch, [], gaps=[(0.0, TRACK_M)])
    assert out["osm_bicycle_no_max_run_m"] == 0.0
    assert out["osm_bicycle_legal"] is True


def test_run_length_boundary_pure():
    samples = [_pt(10.0 * i) for i in range(12)]
    s = ["ok"] * 3 + ["no"] * 5 + ["ok"] * 4
    assert max_bicycle_no_run_m(samples, s) == pytest.approx(BICYCLE_NO_MAX_RUN_M, abs=1e-6)
    s = ["ok"] * 3 + ["no"] * 4 + ["ok"] * 5
    assert max_bicycle_no_run_m(samples, s) == pytest.approx(40.0, abs=1e-6)
    s = ["ok", "no", "no", None, None, "no", "ok", None, "no", "ok", "ok", "ok"]
    assert max_bicycle_no_run_m(samples, s) == pytest.approx(50.0, abs=1e-6)
