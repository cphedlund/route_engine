import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402

CASES = yaml.safe_load((Path(__file__).parent / "golden.yaml").read_text())
XFAIL_FILE = Path(__file__).parent / "golden_xfail.yaml"
XFAIL = yaml.safe_load(XFAIL_FILE.read_text()) if XFAIL_FILE.exists() else {}
TOL = 0.02


def _match(actual, expected):
    if expected == "ANY":
        return actual is not None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) <= max(TOL, 0.005 * abs(float(expected)))
    return actual == expected


def check(case, prefs):
    errs = []
    for label in ("hard", "soft"):
        for k, v in (case.get(label) or {}).items():
            if not _match(prefs.get(k), v):
                errs.append(f"{label} {k}: expected {v!r}, got {prefs.get(k)!r}")
    for k in case.get("absent") or []:
        if prefs.get(k) is not None:
            errs.append(f"absent {k}: got {prefs.get(k)!r}")
    return errs


def _params():
    out = []
    for c in CASES:
        marks = []
        if c["id"] in XFAIL:
            marks.append(pytest.mark.xfail(strict=True, reason=XFAIL[c["id"]]))
        out.append(pytest.param(c, id=c["id"], marks=marks))
    return out


def test_ids_unique_and_count():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 60


@pytest.mark.parametrize("case", _params())
def test_golden(case):
    prefs = app.translate_query_rules(case["query"])
    errs = check(case, prefs)
    assert not errs, f"{case['query']!r}: " + "; ".join(errs)
