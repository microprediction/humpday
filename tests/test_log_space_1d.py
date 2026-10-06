"""to_log_space_1d keeps its endpoints, is monotone and continuous, on every kind of interval (#395)"""

import pytest

from humpday.objectives.transforms import to_log_space_1d

INTERVALS = [
    (1e-3, 10.0),
    (2.0, 3.0),
    (-10.0, -1.0),
    (-1e5, -1e-4),
    (-100.0, 100.0),
    (-100.0, 1.0),
    (-1.0, 100.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, 1e6),
    (-1e-9, 1e-9),
]
GRID = [i / 2000 for i in range(2001)]


@pytest.mark.parametrize("low,high", INTERVALS)
def test_endpoints_are_exact(low, high):
    assert to_log_space_1d(0, low, high) == low
    assert to_log_space_1d(1, low, high) == high


@pytest.mark.parametrize("low,high", INTERVALS)
def test_monotone_and_in_range(low, high):
    values = [to_log_space_1d(u, low, high) for u in GRID]
    assert all(low <= v <= high for v in values)
    assert all(a <= b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("low,high", INTERVALS)
@pytest.mark.parametrize("join", [0.475, 0.525])
def test_continuous_at_the_joins(low, high, join):
    eps = 1e-12
    left = to_log_space_1d(join - eps, low, high)
    right = to_log_space_1d(join + eps, low, high)
    assert abs(right - left) <= 1e-6 * (high - low)


def test_the_reported_cases():
    assert to_log_space_1d(0.475, -100, 100) == pytest.approx(-2.0)
    assert -10 < to_log_space_1d(0.8, -10, -1) < -1
    assert -100 < to_log_space_1d(0.8, -100, 1) < 1
    assert -1 < to_log_space_1d(0.5, -1, 0) < 0
    assert to_log_space_1d(1, -100, 100) == 100
