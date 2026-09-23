"""The projected-gradient norm is scipy's, and the polish stops on the right quantity (#407).

`_proj_grad_sup_norm` is ||P(x - g) - x||_inf, the quantity the polish compares against pgtol.
It clipped only when a variable sat exactly on a bound, so everywhere inside the cube it
returned the raw gradient: at x = [0.01, 0.5] with g = [100, 0] it said 100, where the projected
step is 0.01. Four orders too large, against a stopping tolerance of 1e-5.
"""

from __future__ import annotations

import pytest

from humpday.optimizers.scipy_algorithms import LBFGSB


def _polisher(n_dim=2):
    return LBFGSB(lambda x: 0.0, n_trials=20, n_dim=n_dim)


# (x, gradient, expected) computed from scipy's projgr: a positive component is bounded by the
# distance down to the lower bound, a negative one by the distance up to the upper bound.
CASES = [
    ([0.01, 0.5], [100.0, 0.0], 0.01),  # the issue's counterexample
    ([0.5, 0.5], [0.2, 0.0], 0.2),  # interior, small gradient: unchanged
    ([0.0, 0.5], [5.0, 0.0], 0.0),  # on the lower bound, pointing out
    ([1.0, 0.5], [-5.0, 0.0], 0.0),  # on the upper bound, pointing out
    (
        [0.0, 0.5],
        [-5.0, 0.0],
        1.0,
    ),  # on the lower bound, pointing in: capped by the span
    ([0.9, 0.5], [-100.0, 0.0], 0.1),  # near the upper bound, pointing up
    ([0.5, 0.5], [0.0, 0.0], 0.0),  # converged
]


@pytest.mark.parametrize("x,grad,expected", CASES, ids=lambda v: str(v))
def test_the_projected_gradient_is_the_projected_step(x, grad, expected):
    assert _polisher()._proj_grad_sup_norm(x, grad) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.parametrize("x,grad,expected", CASES, ids=lambda v: str(v))
def test_it_equals_the_definition_it_is_named_after(x, grad, expected):
    """Computed the long way -- project x - g back into the box and measure how far x moved --
    so the test checks the formula against its meaning, not against itself."""
    projected = [min(1.0, max(0.0, xi - gi)) for xi, gi in zip(x, grad)]
    by_definition = max(abs(p - xi) for p, xi in zip(projected, x))
    assert _polisher()._proj_grad_sup_norm(x, grad) == pytest.approx(
        by_definition, abs=1e-12
    )


def test_the_two_loop_recursion_scales_by_the_newest_curvature():
    """H0 = gamma I with gamma = s.y / y.y, as scipy does, rather than the identity.

    With one curvature pair the direction is exactly -gamma * g, which is checkable by hand.
    """
    opt = _polisher(n_dim=2)
    s = [[0.5, 0.0]]
    y = [[2.0, 0.0]]
    grad = [1.0, 1.0]
    # gamma = (0.5*2) / (2*2) = 0.25
    direction = opt._lbfgs_two_loop(grad, s, y)
    # The first coordinate also carries the rank-one correction; the second sees gamma alone.
    assert direction[1] == pytest.approx(-0.25, rel=1e-12)


def test_an_empty_memory_is_steepest_descent():
    opt = _polisher(n_dim=3)
    grad = [1.0, -2.0, 0.5]
    assert opt._lbfgs_two_loop(grad, [], []) == pytest.approx([-1.0, 2.0, -0.5])
