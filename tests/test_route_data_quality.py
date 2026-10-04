import pytest


@pytest.fixture(scope="module")
def raw():
    import app
    return app._RAW_GPX_ROUTES


def test_park_assignment_marker_is_consistent(raw):
    for r in raw:
        a = r["osm_park_assignment"]
        assert a in ("start_point", "footprint", "unassigned"), r["name"]
        if a == "unassigned":
            assert r["osm_park_name"] == "", r["name"]
        else:
            assert r["osm_park_name"], r["name"]
        if a == "footprint":
            assert r["osm_park_overlap_pct"] >= 50.0, r["name"]


def test_footprint_fills_known_quicksilver_route(raw):
    by = {r["name"]: r for r in raw}
    assert by["Big Loop @ Quicksilver"]["osm_park_name"] == "Almaden Quicksilver County Park"
    assert by["Big Loop @ Quicksilver"]["osm_park_assignment"] == "footprint"
    assert by["Canal Trail"]["osm_park_assignment"] == "unassigned"


def test_bike_legal_is_not_blanket_false_at_quicksilver(raw):
    qs = [r for r in raw if r["osm_park_name"] == "Almaden Quicksilver County Park"]
    assert qs
    assert any(r["osm_bicycle_legal"] for r in qs)
    by = {r["name"]: r for r in raw}
    assert by["English Camp Loop"]["osm_bicycle_legal"] is True
    assert by["Big Loop @ Quicksilver"]["osm_bicycle_legal"] is False


def test_bike_flag_matches_max_run(raw):
    for r in raw:
        assert r["osm_bicycle_legal"] == (r["osm_bicycle_no_max_run_m"] < 50.0), r["name"]
