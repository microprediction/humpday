"""An interpolation set describes the points it was measured at (#390).

PRIMA's optimizers keep XPT as offsets from a base point and FVAL as the objective at those
offsets. The two have to be the same points. They were not: the offset was stored as proposed
while the value was measured at `clip(xbase + offset, 0, 1)`, so once the base sat near a bound
the model was fitted to coordinates nothing had been evaluated at.

With xbase at [0.9, 0.9] and rho 0.5, the positive first-axis entry claimed [1.4, 0.9] and was
measured at [1.0, 0.9] -- an error of 0.4 in the data the quadratic fit is built from, before
any question of conditioning. All three optimizers restart from non-central bases, so this is
reached in ordinary use, and a later base shift carries the wrong coordinate along rather than
repairing it.
"""

from __future__ import annotations

import pytest

from humpday import _array as _A
from humpday.optimizers.prima_algorithms import (
    PRIMA_BOBYQA,
    PRIMA_NEWUOA,
    PRIMA_UOBYQA,
)

# (class, initialiser, npt) for each optimizer, and a base close enough to the bound that the
# clip bites on at least one axis.
CASES = [
    (PRIMA_NEWUOA, "_initialize_newuoa_points", 5),
    (PRIMA_UOBYQA, "_initialize_interpolation_points", 6),
]
BASES = [[0.9, 0.9], [0.05, 0.5], [0.5, 0.97]]


def _quadratic(x):
    return float(x[0]) ** 2 + 2.0 * float(x[1]) ** 2


@pytest.mark.parametrize(
    "cls,method,npt", CASES, ids=lambda v: getattr(v, "__name__", str(v))
)
@pytest.mark.parametrize("base", BASES, ids=str)
def test_every_stored_offset_is_where_the_value_came_from(cls, method, npt, base):
    xbase = _A.asarray(base)
    seen: list = []

    def counted(x):
        seen.append([float(v) for v in x])
        return _quadratic(x)

    opt = cls(counted, 100, 2)
    XPT, FVAL = opt._drive_gen(getattr(opt, method)(xbase, 0.5, npt, 2))

    assert len(FVAL) == len(seen)
    for k in range(len(FVAL)):
        for i in range(2):
            stored = float(xbase[i]) + float(XPT[k][i])
            assert stored == pytest.approx(seen[k][i], abs=1e-14), (
                f"{cls.__name__} point {k} coordinate {i}: the set says {stored}, "
                f"the objective was called at {seen[k][i]}"
            )


@pytest.mark.parametrize("base", BASES, ids=str)
def test_bobyqa_stores_what_it_measured(base):
    """BOBYQA clamps its steps to [xl, xu] before stepping, so it was closer to right than the
    other two -- but the final clip could still move a point it had already recorded."""
    xbase = _A.asarray(base)
    seen: list = []

    def counted(x):
        seen.append([float(v) for v in x])
        return _quadratic(x)

    opt = PRIMA_BOBYQA(counted, 100, 2)
    xl = _A.zeros(2)
    xu = _A.ones(2)
    XPT, FVAL = opt._drive_gen(opt._initialize_bobyqa_points(xbase, 0.5, 5, 2, xl, xu))

    assert len(FVAL) == len(seen)
    for k in range(len(FVAL)):
        for i in range(2):
            stored = float(xbase[i]) + float(XPT[k][i])
            assert stored == pytest.approx(seen[k][i], abs=1e-14)


def test_the_points_stay_inside_the_cube():
    """The clip is not the defect -- evaluating outside [0, 1]^n would be. Both hold now."""
    xbase = _A.asarray([0.9, 0.9])
    seen: list = []

    def counted(x):
        seen.append([float(v) for v in x])
        return _quadratic(x)

    opt = PRIMA_NEWUOA(counted, 100, 2)
    opt._drive_gen(opt._initialize_newuoa_points(xbase, 0.5, 5, 2))
    for point in seen:
        assert all(0.0 <= v <= 1.0 for v in point), point
