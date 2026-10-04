import pytest

from tests.benchmark import harness as H

CASES = H.load_cases()


def _marks(case):
    marks = [getattr(pytest.mark, case.get("category", "uncategorized").replace("-", "_"))]
    if "regression" in (case.get("tags") or []):
        marks.append(pytest.mark.regression)
    return marks


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["id"], marks=_marks(c)) for c in CASES],
)
def test_case(case, benchmark_results):
    result = H.evaluate_case(case)
    benchmark_results[case["id"]] = result
    assert result["passed"], "; ".join(result["failures"])


def test_case_ids_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


def test_minimum_case_count():
    assert len(CASES) >= 50


def test_both_known_bugs_are_covered():
    tags = {t for c in CASES for t in (c.get("tags") or [])}
    assert {"bug-false-empty", "bug-flat-elevation"} <= tags
