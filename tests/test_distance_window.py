import pytest

from engine import distance_window_half_width
from tests.benchmark.harness import distance_window


@pytest.mark.parametrize(
    "target, lo, hi",
    [(2.0, 1.0, 3.0), (3.0, 2.0, 4.0), (5.0, 4.0, 6.0), (6.0, 5.0, 7.0), (8.0, 6.8, 9.2), (15.0, 12.75, 17.25)],
)
def test_window_is_max_15pct_or_1mi(target, lo, hi):
    half = distance_window_half_width(target)
    assert target - half == pytest.approx(lo)
    assert target + half == pytest.approx(hi)
    assert distance_window(target) == pytest.approx([lo, hi])


def test_relaxed_window_only_widens():
    for t in (2.0, 5.0, 15.0):
        assert distance_window_half_width(t, relax_level=2) >= distance_window_half_width(t)
