import app
from engine import distance_miss_notice

HELLYER = "d4ae72c2f780d9a7"


def _search(query, batch_size=5):
    body = app.StartSearchBody(query=query, preferences=app.Preferences(), batch_size=batch_size, new_search=True)
    return app._start_search_core(body)


def test_notice_when_closest_route_misses_by_more_than_5pct():
    assert distance_miss_notice({"target_miles": 15.0}, [12.89]) == (
        "These routes differ from your requested 15 mi by more than 5% (closest: 12.9 mi)."
    )


def test_no_notice_when_any_returned_route_within_5pct():
    assert distance_miss_notice({"target_miles": 15.0}, [12.89, 14.5]) is None


def test_no_notice_without_target():
    assert distance_miss_notice({}, [3.0]) is None


def test_start_search_carries_notice():
    resp = _search("15 mile wheelchair accessible flat walk with dog")
    assert HELLYER in [r["route_id"] for r in resp["routes"]]
    assert "differ from your requested 15 mi by more than 5%" in (resp["notice"] or "")


def test_more_results_carries_notice():
    first = _search("15 mile wheelchair accessible flat walk with dog", batch_size=1)
    more = app.more_results(app.MoreResultsIn(session_id=first["session_id"], n=3), None)
    if more["routes"]:
        assert "more than 5%" in (more["notice"] or "")
    assert app.read_session_token(more["session_id"])["distance_explicit"] is True


def test_with_distance_notice_appends_on_more_results_batch():
    notice = app._with_distance_notice("Earlier note.", [{"route_id": HELLYER}], {"target_miles": 15.0}, True)
    assert notice.startswith("Earlier note. These routes differ")


def test_no_notice_without_explicit_distance():
    assert app._with_distance_notice(None, [{"route_id": HELLYER}], {"target_miles": 15.0}, False) is None
    resp = _search("30 minute walk")
    assert "more than 5%" not in (resp["notice"] or "")
