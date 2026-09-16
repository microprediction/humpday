"""n_trials is a hard cap on objective calls, in every mode and both languages (#345).

Population methods used to initialise a whole population without looking at the
budget: DifferentialEvolution made 10 calls for a budget of 1, GeneticAlgorithm
40 at ten dimensions. The cap now lives in the shared driver, so no algorithm
can exceed it, and a partially initialised population simply returns the best
of the points it was allowed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from humpday import _array as _A
from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS
from humpday.optimizers.base import BaseOptimizer

BUDGETS = [0, 1, 2, 5, 10]
DIMS = [1, 2, 10]
NODE = shutil.which("node")
RUNNER = Path(__file__).parent / "js_budget_cap_runner.js"


def _sphere(x):
    return float(sum(float(v) * float(v) for v in x))


def _counting():
    calls = []

    def f(x):
        calls.append(1)
        return _sphere(x)

    return f, calls


@pytest.mark.parametrize("name", sorted(PURE_OPTIMIZERS))
@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("n_dim", DIMS)
def test_optimize_never_exceeds_the_budget(name, budget, n_dim):
    _A.seed(0)
    f, calls = _counting()
    opt = PURE_OPTIMIZERS[name](f, budget, n_dim)
    best_value, best_x = opt.optimize()
    assert len(calls) <= budget
    assert opt.evaluations == len(calls)
    if budget > 0:
        assert best_value == _sphere(best_x)


@pytest.mark.parametrize("name", sorted(PURE_OPTIMIZERS))
@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("n_dim", [1, 10])
def test_asktell_scalar_never_exceeds_the_budget(name, budget, n_dim):
    _A.seed(0)
    opt = PURE_OPTIMIZERS[name](_sphere, budget, n_dim)
    calls = 0
    while True:
        x = opt.suggest_next()
        if x is None:
            break
        calls += 1
        opt.receive_update(_sphere(x))
    assert calls <= budget
    assert opt.evaluations == calls
    assert opt.is_done()


@pytest.mark.parametrize("name", sorted(PURE_OPTIMIZERS))
@pytest.mark.parametrize("budget", BUDGETS)
def test_asktell_batch_never_exceeds_the_budget(name, budget):
    _A.seed(0)
    opt = PURE_OPTIMIZERS[name](_sphere, budget, 10)
    calls = 0
    while True:
        xs = opt.suggest_batch()
        if xs is None:
            break
        calls += len(xs)
        opt.tell_batch([_sphere(x) for x in xs])
    assert calls <= budget
    assert opt.evaluations == calls


@pytest.mark.parametrize("bad", [-1, 2.5, -0.5])
def test_bad_budgets_are_rejected(bad):
    with pytest.raises(ValueError, match="n_trials"):
        BaseOptimizer(_sphere, bad, 2)


def test_integral_float_budget_is_accepted():
    assert BaseOptimizer(_sphere, 5.0, 2).n_trials == 5


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_javascript_roster_never_exceeds_the_budget():
    result = subprocess.run(
        [NODE, str(RUNNER)], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    names = {r["name"] for r in rows}
    assert len(names) >= 20, names
    over = [
        r for r in rows if r["calls"] > r["budget"] or r["evaluations"] != r["calls"]
    ]
    assert not over, over[:10]
