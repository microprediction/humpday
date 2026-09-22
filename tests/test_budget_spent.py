"""An optimizer given a budget uses it, and more of it buys no less (#330).

The cap in #345 is the other half of this: n_trials bounds what an optimizer may spend, and
nothing made it spend what it asked for. Several converged, returned, and handed the remainder
back -- LBFGSB used eleven evaluations of five thousand on the sphere, and its answer at a
budget of 5000 was the answer it had at 200, because the extra 4,989 were never requested.

A local method stopping when it has converged is correct; stopping and *returning* is what
wastes the budget. The remedy already used by HillClimbing, DifferentialEvolution and
SimulatedAnnealing is to start again somewhere else and keep the best point seen.
"""

from __future__ import annotations

import pytest

from humpday import _array as _A
from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

# Optimizers whose budget this file pins. It is not the whole roster: the rest of #330 is
# unfixed, and a test that fails for known reasons teaches nobody anything. Add a name here
# when its restart layer lands.
SPENDERS = ["LBFGSB", "Powell", "CoordinateDescent", "PatternSearch"]

BUDGETS = [200, 1000, 5000]


def _sphere(x):
    return sum((float(v) - 0.5) ** 2 for v in x)


def _rastriginish(x):
    import math

    z = [10.0 * (float(v) - 0.5) for v in x]
    return 10.0 * len(z) + sum(v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in z)


OBJECTIVES = {"sphere": _sphere, "rastriginish": _rastriginish}


def _run(name, objective, budget, n_dim=2, seed=0):
    calls = {"n": 0}

    def counted(x):
        calls["n"] += 1
        return objective(x)

    _A.seed(seed)
    value, _ = PURE_OPTIMIZERS[name](counted, budget, n_dim).optimize()
    return value, calls["n"]


@pytest.mark.parametrize("name", SPENDERS)
@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("objective", sorted(OBJECTIVES))
def test_the_budget_is_actually_spent(name, budget, objective):
    """Within a gradient of the budget: a pass that cannot afford one is not started."""
    _, used = _run(name, OBJECTIVES[objective], budget)
    n_dim = 2
    floor = budget - (2 * n_dim + 2)
    assert used >= floor, (
        f"{name} used {used} of {budget} on {objective}, leaving {budget - used} unspent"
    )


@pytest.mark.parametrize("name", SPENDERS)
@pytest.mark.parametrize("objective", sorted(OBJECTIVES))
def test_more_budget_is_never_worse(name, objective):
    """Not merely "not worse": the point of spending it is that it buys something.

    Monotonicity is the weaker claim that holds for every seed. It is asserted across the
    budgets rather than pairwise so a plateau between two adjacent budgets is allowed while a
    method that ignores the budget entirely is not.
    """
    results = [_run(name, OBJECTIVES[objective], b)[0] for b in BUDGETS]
    for smaller, larger, before, after in zip(
        BUDGETS, BUDGETS[1:], results, results[1:]
    ):
        assert after <= before * (1 + 1e-9) + 1e-30, (
            f"{name} on {objective} did worse with {larger} evaluations "
            f"({after:.6e}) than with {smaller} ({before:.6e})"
        )
    # And it bought something -- unless the smallest budget already solved the problem, which
    # Powell does on both of these: there is no improving on an exact zero, and demanding one
    # would be asking the test to fail for the best possible reason.
    if results[0] > 1e-30:
        assert results[-1] < results[0] * (1 - 1e-9), (
            f"{name} on {objective} got nothing from {BUDGETS[-1]} evaluations that it did not "
            f"already have at {BUDGETS[0]}: {results[0]:.6e} -> {results[-1]:.6e}"
        )
