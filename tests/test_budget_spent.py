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
    """Monotonicity holds everywhere: a larger budget cannot produce a worse answer.

    It is asserted across the budgets rather than pairwise so a plateau between two adjacent
    budgets is allowed. What is not allowed is going backwards, which is what a method that
    re-seeds and forgets its best point does.
    """
    results = [_run(name, OBJECTIVES[objective], b)[0] for b in BUDGETS]
    for smaller, larger, before, after in zip(
        BUDGETS, BUDGETS[1:], results, results[1:]
    ):
        assert after <= before * (1 + 1e-9) + 1e-30, (
            f"{name} on {objective} did worse with {larger} evaluations "
            f"({after:.6e}) than with {smaller} ({before:.6e})"
        )


@pytest.mark.parametrize("name", SPENDERS)
def test_more_budget_buys_something_somewhere(name):
    """And the budget buys something -- on at least one of the objectives.

    Not on every one. A local method on Rastrigin can find a good basin in its first two
    hundred evaluations and never beat it however many restarts follow, which is a property of
    the landscape rather than a failure to spend: LBFGSB sits at 0.99496 there, one Rastrigin
    step above the optimum, at every budget. Demanding improvement everywhere would make this
    test fail for the best possible reason, and demanding it nowhere would let an optimizer
    ignore the budget entirely.
    """
    improved = []
    contested = []
    for objective, func in OBJECTIVES.items():
        first = _run(name, func, BUDGETS[0])[0]
        last = _run(name, func, BUDGETS[-1])[0]
        if first <= 1e-30:
            continue  # already solved at the smallest budget; nothing left to win
        contested.append(objective)
        if last < first * (1 - 1e-9):
            improved.append(objective)

    if not contested:
        return  # solved everything at the smallest budget, which Powell does on both of these

    assert improved, (
        f"{name} got nothing from {BUDGETS[-1]} evaluations that it did not already have at "
        f"{BUDGETS[0]}, on any of {contested}"
    )
