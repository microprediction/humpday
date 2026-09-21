"""minimize() spends exactly the budget it was given and reports what it did (#346, #347).

Timing probes used to be four extra calls outside the budget whose values were
discarded, nfev was hard-coded to maxiter, and success was hard-coded to True.
"""

import math

import pytest

from humpday import _array as _A
from humpday import minimize


def _counted(f):
    seen = []

    def g(x):
        v = f(x)
        seen.append(v)
        return v

    return g, seen


def _centered(x):
    return sum((float(c) - 0.5) ** 2 for c in x)


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 10])
def test_auto_selection_never_exceeds_the_budget(budget):
    _A.seed(1)
    f, seen = _counted(_centered)
    r = minimize(f, bounds=[(0, 1)] * 2, options={"maxiter": budget})
    assert len(seen) <= budget
    assert r.nfev == len(seen)


@pytest.mark.parametrize("budget", [2, 3, 5])
def test_timing_observations_are_kept(budget):
    # The centre is the optimum of _centered, and timing probes the centre:
    # a result worse than 0 would mean those observations were discarded.
    _A.seed(123)
    f, seen = _counted(_centered)
    r = minimize(f, bounds=[(0, 1)] * 2, options={"maxiter": budget})
    assert r.fun == 0.0
    assert r.fun == min(seen)
    assert r.success


def test_a_budget_of_one_is_one_call_by_the_optimizer():
    f, seen = _counted(_centered)
    r = minimize(f, bounds=[(0, 1)] * 2, options={"maxiter": 1})
    assert len(seen) == 1
    assert r.nfev == 1
    assert r.eval_time_measured is None  # no probes fit


def test_explicit_method_reports_actual_calls():
    _A.seed(123)
    f, seen = _counted(_centered)
    r = minimize(f, bounds=[(0, 1)] * 2, method="Powell", options={"maxiter": 1000})
    assert r.nfev == len(seen)
    assert r.nfev < 1000  # Powell terminates early on a quadratic
    assert r.fun == min(seen)


def test_returned_point_is_the_best_observed():
    _A.seed(7)
    f, seen = _counted(_centered)
    r = minimize(f, bounds=[(0, 1)] * 2, method="RandomSearch", options={"maxiter": 30})
    assert r.fun == min(seen)
    assert math.isclose(_centered(r.x), r.fun, rel_tol=1e-12, abs_tol=1e-15)


def test_zero_budget_is_not_a_success():
    f, seen = _counted(_centered)
    r = minimize(f, bounds=[(0, 1)] * 2, method="RandomSearch", options={"maxiter": 0})
    assert len(seen) == 0
    assert r.nfev == 0
    assert r.success is False
    assert r.fun == float("inf")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_no_finite_value_is_not_a_success(bad):
    r = minimize(
        lambda x: bad,
        bounds=[(0, 1)] * 2,
        method="RandomSearch",
        options={"maxiter": 5},
    )
    assert r.nfev == 5
    assert r.success is False
    assert r.fun == float("inf")


def test_a_probe_that_raises_keeps_its_place_on_the_ledger():
    calls = []

    def flaky(x):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("second call fails")
        return _centered(x)

    _A.seed(3)
    r = minimize(flaky, bounds=[(0, 1)] * 2, options={"maxiter": 10})
    assert len(calls) <= 10
    assert r.nfev == len(calls)
    assert r.success


def test_auto_timing_off_spends_everything_on_the_search():
    f, seen = _counted(_centered)
    r = minimize(
        f,
        bounds=[(0, 1)] * 2,
        method="RandomSearch",
        options={"maxiter": 7, "auto_timing": False},
    )
    assert len(seen) == 7
    assert r.nfev == 7
