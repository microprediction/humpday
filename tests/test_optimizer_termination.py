"""Every optimizer must TERMINATE and respect its evaluation budget.

This is a regression guard for a class of bug where an optimizer's main loop is
gated on its own progress: e.g. FireflyAlgorithm only called evaluate() when one
firefly was strictly brighter than another, so once the swarm collapsed onto a
single point (identical, hence equal-intensity, after clipping to a cube corner)
or the objective was flat, no evaluation ever happened and `while
self.evaluations < budget` spun forever. That wedged a downstream study for 3.5
days before it was noticed.

Two invariants are checked for every optimizer in the registry:
  1. optimize() returns within a wall-clock bound on adversarial objectives
     (flat, corner-optimum, plateau-with-well) that provoke degenerate states.
  2. optimize() never exceeds the n_trials budget (arguments propagate and cap
     the work actually done).
"""

import sys

try:
    import pytest
except ImportError:  # allow running the __main__ self-check without pytest
    pytest = None

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS, create_optimizer_function

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None

# SIGALRM is POSIX-only; on Windows we skip the wall-clock half of the suite.
try:
    import signal

    _HAS_ALARM = hasattr(signal, "SIGALRM")
except Exception:  # noqa: BLE001
    _HAS_ALARM = False

ALL = sorted(PURE_OPTIMIZERS)

# Objectives engineered to drive optimizers into degenerate states. A correct
# optimizer must still terminate on every one of them.
ADVERSARIAL = {
    "flat": lambda x: 1.0,  # every point equal -> no "improvement" anywhere
    "corner": lambda x: float(
        sum((xi - 1.0) ** 2 for xi in x)
    ),  # optimum at a cube corner
    "plateau_well": lambda x: (
        -1.0 if max(abs(xi - 0.5) for xi in x) <= 0.2 else 0.0
    ),  # huge flat plateau around a small well
    "sphere": lambda x: float(sum((xi - 0.3) ** 2 for xi in x)),  # benign control
}


class _Timeout(Exception):
    pass


def _run_with_timeout(name, objective, n_trials, n_dim, seconds):
    """Run one optimizer; raise _Timeout if it does not return in `seconds`."""
    import random

    random.seed(0)
    if np is not None:
        np.random.seed(0)
    fn = create_optimizer_function(PURE_OPTIMIZERS[name])

    if not _HAS_ALARM:
        return fn(objective, n_dim, n_trials=n_trials, with_count=True)

    old = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(seconds)
    try:
        return fn(objective, n_dim, n_trials=n_trials, with_count=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _check_terminates(name, oname):
    objective = ADVERSARIAL[oname]
    try:
        best_value, best_x, evals = _run_with_timeout(
            name, objective, n_trials=200, n_dim=3, seconds=30
        )
    except _Timeout:
        raise AssertionError(
            f"{name} did not terminate within 30s on the '{oname}' objective "
            f"(likely a progress-gated loop that stops consuming budget)"
        )
    assert best_x is not None
    # never blow the budget (polish stages may add a bounded margin)
    assert evals <= 200 * 1.25 + 10, f"{name} used {evals} evals for a budget of 200"


def _check_budget(name):
    objective = ADVERSARIAL["sphere"]
    for n_trials in (40, 120):
        _bv, _bx, evals = _run_with_timeout(
            name, objective, n_trials=n_trials, n_dim=4, seconds=30
        )
        assert evals <= n_trials * 1.25 + 10, (
            f"{name} used {evals} evals for budget {n_trials} (budget not respected)"
        )


if pytest is not None:

    @pytest.mark.parametrize("name", ALL)
    @pytest.mark.parametrize("oname", list(ADVERSARIAL))
    def test_optimizer_terminates(name, oname):
        _check_terminates(name, oname)

    @pytest.mark.parametrize("name", ALL)
    def test_optimizer_respects_budget(name):
        _check_budget(name)


def _main():
    failures = []
    for name in ALL:
        for oname in ADVERSARIAL:
            try:
                _check_terminates(name, oname)
            except AssertionError as e:
                failures.append(str(e))
        try:
            _check_budget(name)
        except AssertionError as e:
            failures.append(str(e))
        print(f"  {name:24s} ok")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"\nall {len(ALL)} optimizers terminate + respect budget")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
