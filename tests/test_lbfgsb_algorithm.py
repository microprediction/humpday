"""L-BFGS-B is the bound-constrained algorithm, checked against scipy's (#407).

The polish used to take an unconstrained two-loop direction, zero the components pointing out
of an active bound, and backtrack. That is projected L-BFGS. Byrd-Lu-Nocedal-Zhu is a different
algorithm: it minimises the model along the piecewise projected-gradient path to find the
generalised Cauchy point -- which is how the active set is chosen -- and then minimises over
whatever remains free.

The difference shows up wherever a bound is active at the solution, which is where the earlier
version would stop at a corner the Cauchy point walks past.
"""

from __future__ import annotations

import pytest

from humpday.optimizers.base import BaseOptimizer
from humpday.optimizers.scipy_algorithms import LBFGSB

np = pytest.importorskip("numpy")
scipy_optimize = pytest.importorskip("scipy.optimize")


def _polisher(n_dim=2):
    return LBFGSB(lambda x: 0.0, n_trials=20, n_dim=n_dim)


def test_the_cauchy_point_pins_a_variable_the_gradient_pushes_out():
    """With no curvature yet the model is the identity, so the path is plain projected
    steepest descent: the first coordinate hits its bound and stops being free."""
    xcp, free, _ = BaseOptimizer._lbfgsb_cauchy(
        [0.5, 0.5], [1.0, 0.0], 1.0, [], [], [0.0, 0.0], [1.0, 1.0]
    )
    assert xcp[0] == pytest.approx(0.0)
    assert free == [1]


def test_a_variable_already_on_its_bound_is_not_walked_as_a_breakpoint():
    """Its breakpoint is zero, so it joins the active set immediately. Walking it adds its
    gradient to the directional derivative as though the path had travelled along it, which is
    what made the first version of this stop one step early."""
    xcp, free, _ = BaseOptimizer._lbfgsb_cauchy(
        [0.0, 0.0], [1.0, -0.6], 1.0, [], [], [0.0, 0.0], [1.0, 1.0]
    )
    assert xcp[0] == pytest.approx(0.0)
    assert xcp[1] > 0.0, "the second variable can still move up and should have"


def test_the_compact_representation_reproduces_the_scaling():
    theta, W, Minv = BaseOptimizer._lbfgsb_compact([[0.5, 0.0]], [[2.0, 0.0]])
    assert theta == pytest.approx(4.0)  # y.y / s.y = 4 / 1
    assert len(W) == 2  # [Y, theta S]
    assert Minv[0][0] == pytest.approx(-1.0)  # -D = -(s.y)


# Quadratics whose unconstrained minimum lies outside the box, so bounds are active at the
# solution and the Cauchy point has work to do.
BOUND_ACTIVE = {
    "one active": lambda x: (x[0] + 0.5) ** 2 + (x[1] - 0.3) ** 2,
    "both active": lambda x: (x[0] + 0.2) ** 2 + (x[1] + 0.7) ** 2 + 0.3 * x[0] * x[1],
    "coupled": lambda x: (
        10 * (x[0] - 0.9) ** 2 + (x[0] - x[1]) ** 2 + 0.5 * (x[1] - 0.1) ** 2
    ),
    "interior": lambda x: (x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2,
}


@pytest.mark.parametrize("name", sorted(BOUND_ACTIVE))
def test_it_reaches_what_scipy_reaches(name):
    f = BOUND_ACTIVE[name]
    x0 = [0.3, 0.7]
    from humpday import _array as _A

    _A.seed(0)
    opt = LBFGSB(f, 2000, 2)
    opt.best_x = _A.asarray(x0)
    opt.best_value = f(x0)
    opt._drive_gen(opt._lbfgs_polish_gen(start=x0, start_value=f(x0)))

    ref = scipy_optimize.minimize(
        lambda z: f(list(z)), np.array(x0), method="L-BFGS-B", bounds=[(0, 1)] * 2
    )
    assert opt.best_value <= ref.fun + 1e-6 * max(abs(ref.fun), 1.0), (
        f"{name}: humpday {opt.best_value:.8e}, scipy {ref.fun:.8e}"
    )


def test_the_projected_gradient_is_scipys():
    """At an interior point the bound distance caps the step; on a bound pointing out it is
    zero. Both were wrong before: the norm returned the raw gradient inside the cube."""
    p = _polisher()
    assert p._proj_grad_sup_norm([0.01, 0.5], [100.0, 0.0]) == pytest.approx(0.01)
    assert p._proj_grad_sup_norm([0.0, 0.5], [5.0, 0.0]) == pytest.approx(0.0)
    assert p._proj_grad_sup_norm([0.5, 0.5], [0.2, 0.0]) == pytest.approx(0.2)
