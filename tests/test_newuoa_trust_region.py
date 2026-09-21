"""NEWUOA's trust-region step never increases the model (#352).

The dogleg fallback solved the indefinite Newton system and accepted its
stationary point whenever it lay inside the radius. A stationary point of an
indefinite quadratic is a saddle or a maximum: on g = 0.1, H = -1, rho = 0.5 the
returned step [0.1] raised the model from 0 to +0.005 while the boundary step
-0.5 lowers it to -0.175.
"""

import math

import pytest

from humpday import _array as _A
from humpday.optimizers.prima_algorithms import PRIMA_NEWUOA


def _model(g, H, d):
    Hd = _A.linalg.matvec(H, d)
    return float(_A.dot(g, d)) + 0.5 * float(_A.dot(d, Hd))


def _cauchy_value(g, H, rho):
    """Exact minimum of the model along -g inside the ball."""
    g_norm = float(_A.norm(g))
    gHg = float(_A.dot(g, _A.linalg.matvec(H, g)))
    if gHg <= 0:
        t = rho / g_norm
    else:
        t = min(g_norm * g_norm / gHg, rho / g_norm)
    return _model(g, H, -t * g)


def _solver():
    return PRIMA_NEWUOA(lambda x: 0.0, 20, 2)


CASES = [
    # (g, H, rho, tag)
    ([0.1], [[-1.0]], 0.5, "audit: 1-D negative curvature"),
    ([1.0, 0.0], [[2.0, 0.0], [0.0, 3.0]], 1.0, "positive definite, Newton inside"),
    ([3.0, 4.0], [[1.0, 0.0], [0.0, 1.0]], 0.5, "positive definite, Newton outside"),
    ([1.0, 1.0], [[1.0, 0.0], [0.0, -1.0]], 0.7, "indefinite saddle"),
    ([1.0, 0.5], [[-2.0, 0.0], [0.0, -1.0]], 0.3, "negative definite"),
    ([1.0, 1.0], [[1.0, 1.0], [1.0, 1.0]], 0.4, "singular PSD"),
    ([0.0, 1.0], [[1.0, 0.0], [0.0, 0.0]], 0.4, "singular, zero curvature along -g"),
]


@pytest.mark.parametrize("g,H,rho,tag", CASES, ids=[c[3] for c in CASES])
def test_step_is_feasible_and_achieves_cauchy_decrease(g, H, rho, tag):
    g = _A.asarray(g)
    for method in ("_solve_trust_region_newuoa", "_dogleg_method"):
        d = getattr(_solver(), method)(g, H, rho, len(g))
        assert float(_A.norm(d)) <= rho * (1 + 1e-12), tag
        q = _model(g, H, d)
        assert q <= 0.0, (tag, method, q)
        assert q <= _cauchy_value(g, H, rho) + 1e-12, (tag, method, q)


def test_audit_case_takes_the_boundary_step():
    g, H, rho = _A.asarray([0.1]), [[-1.0]], 0.5
    d = _solver()._solve_trust_region_newuoa(g, H, rho, 1)
    assert list(d) == pytest.approx([-0.5])
    assert _model(g, H, d) == pytest.approx(-0.175)


def test_positive_definite_newton_step_is_still_taken():
    g, H, rho = _A.asarray([1.0, 0.0]), [[2.0, 0.0], [0.0, 3.0]], 1.0
    d = _solver()._solve_trust_region_newuoa(g, H, rho, 2)
    assert list(d) == pytest.approx([-0.5, 0.0])


def test_dogleg_uses_newton_only_after_positive_definiteness():
    # Same gradient, one sign flip in H: the SPD case walks toward Newton,
    # the indefinite case stops at the Cauchy point.
    g, rho = _A.asarray([1.0, 1.0]), 10.0
    d_spd = _solver()._dogleg_method(g, [[1.0, 0.0], [0.0, 2.0]], rho, 2)
    assert list(d_spd) == pytest.approx([-1.0, -0.5])
    d_ind = _solver()._dogleg_method(g, [[1.0, 0.0], [0.0, -2.0]], rho, 2)
    assert math.isclose(
        _model(g, [[1.0, 0.0], [0.0, -2.0]], d_ind),
        _cauchy_value(g, [[1.0, 0.0], [0.0, -2.0]], rho),
        rel_tol=1e-12,
    )
