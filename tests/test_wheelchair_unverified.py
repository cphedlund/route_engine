import pytest

from tests.benchmark import harness

UNVERIFIED_UNKNOWN_IDS = {
    "270bf04821300c5d",
    "423f74902dd28a48",
    "81986ce0261a6b86",
    "8fd04735ae9e86c4",
    "a20b399f209bf9f5",
    "b44257a6bcebb7e8",
    "c1d1aad6a8f1bc69",
    "ce6e59b9f1d6bda3",
    "d9dfd6188a37265e",
    "f2cf98fef0307774",
}

QUERIES = [
    "wheelchair accessible walk",
    "wheelchair accessible 1 mile",
    "wheelchair accessible 2 mile paved path",
    "wheelchair accessible 5 mile",
    "paved stroller friendly loop",
    "wheelchair accessible near Hellyer County Park",
    "wheelchair accessible Joseph D. Grant",
    "wheelchair accessible Coyote Lake",
    "ada accessible lake walk",
]


def test_known_ids_are_still_gpx_paved_osm_unknown():
    for rid in UNVERIFIED_UNKNOWN_IDS:
        r = harness.route_by_id(rid)
        assert r is not None, rid
        assert r.surface_type == "paved" and r.osm_surface == "unknown", rid
        assert harness.route_facts(r)["paved"] is False, rid


@pytest.mark.parametrize("query", QUERIES)
def test_unverified_routes_never_in_wheelchair_results(query):
    res = harness.run_query(query, pages=3)
    assert res["error"] is None
    bad = [r["route_id"] for r in res["routes"] if r["route_id"] in UNVERIFIED_UNKNOWN_IDS]
    assert not bad, bad
