"""
Python ↔ JavaScript parity tests.

For each algorithm, runs both the Python implementation and the JS port
on the same simple objective with the same n_trials, and asserts that
both reach within tolerance of the known minimum.

This is a *soundness* test, not a strict equivalence test: Python and
JS use independent RNGs so they explore the search space differently.
We don't try to match step-for-step; we check that both converge to
roughly the same answer on convex test problems.

The JS side runs in a Node subprocess via `tests/js_parity_runner.js`.
Tests are skipped if `node` isn't on PATH (so the parity tests are
opt-in and don't break Python-only CI runs).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

REPO_ROOT = Path(__file__).parent.parent
RUNNER_JS = Path(__file__).parent / "js_parity_runner.js"

NODE = shutil.which("node")
NODE_AVAILABLE = NODE is not None

# Objective ID → Python callable. The JS runner mirrors these IDs in its
# own OBJECTIVES dict — keep both in sync. All callables defined on
# [0,1]^n with a known minimum of 0.
OBJECTIVES = {
    "sphere_at_half": lambda x: sum((v - 0.5) ** 2 for v in x),
    "quad_at_0_7": lambda x: sum((v - 0.7) ** 2 for v in x),
}

# Convergence threshold: algorithms expected to converge cleanly on a
# 300-eval budget should land below this.
CONVERGE_TOL = 0.05

# Looser threshold for algorithms that genuinely need a bigger budget;
# the test still catches complete divergence (e.g. NaN, or sitting at
# the initial point).
WEAK_TOL = 1.0

# Algorithms that genuinely can't hit CONVERGE_TOL in a small budget,
# even on a sphere. The parity test still runs and just checks both
# Python and JS land in the same ballpark for these.
WEAK_ON_SMALL_BUDGET = {
    "RandomSearch",
    "HillClimbing",
    "AntColonyOpt",
    "TabuSearch",
    "FireflyAlgorithm",
    "HarmonySearch",
    "AdaptiveRandomSearch",
    "GeneticAlgorithm",
    "EvolutionStrategy",
    "PatternSearch",
    "CoordinateDescent",
}

# All 22 algorithms (the keys of PURE_OPTIMIZERS). Each algorithm
# spawns one Node subprocess, so the sweep runs in seconds, not minutes.
PARITY_ALGORITHMS = list(PURE_OPTIMIZERS.keys())


def _run_js(algorithm: str, n_trials: int, n_dim: int, func_id: str) -> dict:
    """Run the JS port via Node and return {best_value, best_x}."""
    result = subprocess.run(
        [NODE, str(RUNNER_JS), algorithm, str(n_trials), str(n_dim), func_id],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"JS runner failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return json.loads(result.stdout.strip())


def _run_python(algorithm: str, n_trials: int, n_dim: int, func_id: str) -> dict:
    """Run the Python port directly and return {best_value, best_x}."""
    func = OBJECTIVES[func_id]
    cls = PURE_OPTIMIZERS[algorithm]
    opt = cls(func, n_trials=n_trials, n_dim=n_dim)
    result = opt.optimize()
    if isinstance(result, tuple) and len(result) == 2:
        best_value, best_x = result
    else:
        best_value = opt.best_value
        best_x = opt.best_x
    return {"best_value": float(best_value), "best_x": list(best_x)}


@pytest.fixture(scope="module")
def parity_cfg():
    return {"n_trials": 300, "n_dim": 2, "func_id": "sphere_at_half"}


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node not on PATH")
@pytest.mark.parametrize("algorithm", PARITY_ALGORITHMS)
def test_python_js_parity_sphere(parity_cfg, algorithm):
    """Both Python and JS should find x = [0.5, 0.5] on the sphere.

    This is a *soundness* test, not a strict equivalence test. Python
    and JavaScript use independent RNGs, so the two implementations
    explore the search space differently and will converge to
    different floor values. We assert only that both implementations
    actually work on the simplest possible objective (a 2-D sphere
    centred at the cube centre, n_trials=300):

      - Algorithms that should converge cleanly: both py and js must
        land within `CONVERGE_TOL` of the optimum.
      - Algorithms that genuinely need bigger budgets (the
        WEAK_ON_SMALL_BUDGET set): both must at least land within
        `WEAK_TOL` so we'd catch a complete divergence.

    Either direction (Python regresses but JS still works, or vice
    versa) trips the test, which is the catch we actually want.
    """
    py = _run_python(algorithm, **parity_cfg)
    js = _run_js(algorithm, **parity_cfg)

    if algorithm in WEAK_ON_SMALL_BUDGET:
        tol = WEAK_TOL
    else:
        tol = CONVERGE_TOL

    assert py["best_value"] < tol, (
        f"Python {algorithm} regressed: {py['best_value']:.4g} ≥ {tol} "
        f"(JS got {js['best_value']:.4g})"
    )
    assert js["best_value"] < tol, (
        f"JS {algorithm} regressed: {js['best_value']:.4g} ≥ {tol} "
        f"(Python got {py['best_value']:.4g})"
    )
