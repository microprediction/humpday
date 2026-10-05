"""
Python ↔ JavaScript parity tests.

For each algorithm, runs both the Python implementation and the JS port
on a shifted sphere with the same n_trials and the same PCG32 seeds, and
asserts that both get below a per-algorithm bar and stay within budget.
The bars are measured, not guessed: see RANDOM_SAMPLING_BAR and TOLERANCE.

This is a *soundness* test, not a strict equivalence test: the bit-exact
twins are checked point for point by test_transition_vectors.py, and the
other ports draw some of their randomness from Math.random, so the two
sides need not take the same path. They must both actually search.

The JS side runs in a Node subprocess via `tests/js_parity_runner.js`.
Tests are skipped if `node` isn't on PATH (so the parity tests are
opt-in and don't break Python-only CI runs).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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
def _rosenbrock_unit(x):
    a = 4 * x[0] - 2
    b = 4 * x[1] - 2
    return (1 - a) ** 2 + 100 * (b - a * a) ** 2


# Optimum of `sphere_shifted`; identical to SHIFT in js_parity_runner.js. The
# sphere used to be centred at 0.5, the cube centre, which is where most of the
# ports take their first point: an optimizer that evaluated the centre once and
# stopped scored exactly 0 there (#403). Every coordinate here sits 0.19 to 0.31
# from the centre and off any grid node. The test uses the first two.
SHIFT = (0.2473, 0.7718, 0.1859, 0.6934, 0.8146)

OBJECTIVES = {
    "sphere_shifted": lambda x: sum((v - SHIFT[i]) ** 2 for i, v in enumerate(x)),
    "quad_at_0_7": lambda x: sum((v - 0.7) ** 2 for v in x),
    "rosenbrock_unit": _rosenbrock_unit,
}

# Quality bars for `test_python_js_parity_sphere`: sphere_shifted, n_dim=2,
# n_trials=300. Reference points for the objective on [0,1]^2:
#
#   maximum (the corner (1, 0))                         1.162
#   value at the cube centre, where most ports start    0.138
#   best of 300 uniform random points, 200k draws:
#     median 7.4e-4, 99th percentile 4.9e-3, 99.9th percentile 7.35e-3
#
# RANDOM_SAMPLING_BAR is that 99.9th percentile. Every algorithm must beat it,
# so an optimizer that starts at the centre and stops there (0.138) fails, and
# a single uniform random point gets under it 2.3% of the time.
RANDOM_SAMPLING_BAR = 7.35e-3

# The bar for an algorithm whose worst result over seeds 0-49, on both ports,
# was below 3e-11. Seven orders of magnitude under the centre value.
CONVERGED = 1e-8

# Algorithms that do search but do not converge in 300 evaluations. Each bar is
# ten times the worst result measured over seeds 0-49 on either port, rounded
# up, and never above RANDOM_SAMPLING_BAR. Worst measured values in brackets.
#
# The last four are at RANDOM_SAMPLING_BAR because ten times their worst result
# is above it: at this budget the tests can only say they are no worse than
# random sampling, which on a 2-D sphere is what they measure as.
TOLERANCE = {
    "HillClimbing": 5e-6,  # [4.5e-7]
    "AntColonyOpt": 1e-3,  # [8.4e-5]
    "HarmonySearch": 2e-3,  # [1.3e-4]
    "EvolutionStrategy": RANDOM_SAMPLING_BAR,  # [6.7e-4]
    "GridSearch": RANDOM_SAMPLING_BAR,  # [8.0e-4, deterministic: nearest node of a 17x17 grid]
    "GeneticAlgorithm": RANDOM_SAMPLING_BAR,  # [1.2e-3]
    "RandomSearch": RANDOM_SAMPLING_BAR,  # [5.8e-3; it is the baseline]
}

# JS ports measured below the bar their Python twin meets. The Python side is
# still held to the bar; the JS side xfails while it falls short and fails the
# test once it stops falling short, so the entry gets removed.
JS_BELOW_BAR = {
    # Worst 2.5e-3, median 6.1e-4, best 2.4e-5 over seeds 0-49, against 7.7e-34
    # for Python: the JS port does not have Powell's line search (#78, #402).
    "Powell": "JS Powell lacks Brent's line search and does not converge (#78)",
}

# Seeds for the portable PCG32 stream, the same stream on both sides. The bars
# above were measured over seeds 0-49, so any seed is a fair draw. One, because
# pure-Python BayesianOpt takes about 40s for 300 evaluations, and the pure-backend
# CI job runs this file.
PARITY_SEEDS = (0,)


def _tolerance(algorithm: str) -> float:
    return TOLERANCE.get(algorithm, CONVERGED)


# All 22 algorithms (the keys of PURE_OPTIMIZERS). Each algorithm
# spawns one Node subprocess, so the sweep runs in seconds, not minutes.
PARITY_ALGORITHMS = list(PURE_OPTIMIZERS.keys())


def _run_python(
    algorithm: str, n_trials: int, n_dim: int, func_id: str, seed: int | None = None
) -> dict:
    """Run the Python port directly and return {best_value, best_x, evaluations}.

    With a seed, the run draws from the portable PCG32 stream, the same stream
    `js_parity_runner.js` gives the JS port for that seed."""
    from humpday import _array as A

    func = OBJECTIVES[func_id]
    cls = PURE_OPTIMIZERS[algorithm]
    if seed is not None:
        A.use_portable_rng(seed)
    try:
        opt = cls(func, n_trials=n_trials, n_dim=n_dim)
        result = opt.optimize()
    finally:
        A.use_legacy_rng()
    if isinstance(result, tuple) and len(result) == 2:
        best_value, best_x = result
    else:
        best_value = opt.best_value
        best_x = opt.best_x
    return {
        "best_value": float(best_value),
        "best_x": list(best_x),
        "evaluations": opt.evaluations,
    }


def _run_js_batch(
    algorithm: str,
    n_trials: int,
    n_dim: int,
    func_id: str,
    n_runs: int,
    seed: int | None = None,
) -> list[dict]:
    """Run `n_runs` independent JS instances of the algorithm in one Node
    subprocess and return the list of {best_value, best_x, evaluations} dicts.
    With a seed, run i uses the portable stream seeded with seed + i."""
    args = [NODE, str(RUNNER_JS), algorithm, str(n_trials), str(n_dim), func_id]
    args.append(str(n_runs))
    if seed is not None:
        args.append(str(seed))
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"JS runner failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    runs = [json.loads(line) for line in result.stdout.strip().splitlines() if line]
    errors = [r["error"] for r in runs if "error" in r]
    if errors:
        raise RuntimeError(f"JS {algorithm} raised: {errors[0]}")
    return runs


def _run_python_batch(
    algorithm: str, n_trials: int, n_dim: int, func_id: str, n_runs: int
) -> list[dict]:
    """Run the Python port `n_runs` times back-to-back."""
    return [_run_python(algorithm, n_trials, n_dim, func_id) for _ in range(n_runs)]


@pytest.fixture(scope="module")
def parity_cfg():
    return {"n_trials": 300, "n_dim": 2, "func_id": "sphere_shifted"}


def _check_runs(label: str, algorithm: str, runs: list[dict], n_trials: int) -> None:
    """Assert every run stayed within budget and got below the algorithm's bar."""
    tol = _tolerance(algorithm)
    for seed, r in zip(PARITY_SEEDS, runs):
        assert r["evaluations"] <= n_trials, (
            f"{label} {algorithm} seed {seed}: {r['evaluations']} evaluations, "
            f"budget {n_trials}"
        )
        assert r["best_value"] < tol, (
            f"{label} {algorithm} seed {seed}: {r['best_value']:.4g} >= {tol:g}"
        )


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node not on PATH")
@pytest.mark.parametrize("algorithm", PARITY_ALGORITHMS)
def test_python_js_parity_sphere(parity_cfg, algorithm):
    """Both ports must find the minimum of a 2-D shifted sphere in 300 evaluations.

    For each seed in PARITY_SEEDS, the Python port and the JS port must each
    use no more than n_trials evaluations and get below the algorithm's bar
    (`_tolerance`): CONVERGED for an algorithm that converges, the measured
    entry in TOLERANCE for one that does not, and never looser than
    RANDOM_SAMPLING_BAR, which an optimizer that does not search fails.

    Either direction (Python regresses but JS still works, or vice versa)
    trips the test. A JS port in JS_BELOW_BAR xfails on its own shortfall
    and fails the test once it no longer falls short.
    """
    n_trials = parity_cfg["n_trials"]
    py = [_run_python(algorithm, **parity_cfg, seed=s) for s in PARITY_SEEDS]
    js = _run_js_batch(
        algorithm, **parity_cfg, n_runs=len(PARITY_SEEDS), seed=PARITY_SEEDS[0]
    )
    assert len(js) == len(PARITY_SEEDS)

    _check_runs("Python", algorithm, py, n_trials)

    if algorithm not in JS_BELOW_BAR:
        _check_runs("JS", algorithm, js, n_trials)
        return
    try:
        _check_runs("JS", algorithm, js, n_trials)
    except AssertionError as shortfall:
        pytest.xfail(f"{JS_BELOW_BAR[algorithm]}: {shortfall}")
    pytest.fail(
        f"JS {algorithm} now meets its bar on every seed; remove it from JS_BELOW_BAR"
    )


class _NoSearch:
    """An optimizer that evaluates one point and stops: the failure #403 found
    the old bars could not see. `point` is "centre", "corner" or "random"."""

    def __init__(self, point: str):
        self.point = point

    def __call__(self, objective, n_trials, n_dim):
        from humpday import _array as A
        from humpday.optimizers.base import BaseOptimizer

        point = self.point

        class NoSearch(BaseOptimizer):
            def optimize(self):
                if point == "centre":
                    x = A.full(self.n_dim, 0.5)
                elif point == "corner":
                    x = A.zeros(self.n_dim)
                else:
                    x = A.random_uniform(self.n_dim)
                self.evaluate(x)
                return self.best_value, self.best_x

        return NoSearch(objective, n_trials, n_dim)


@pytest.mark.parametrize("point", ["centre", "corner", "random"])
@pytest.mark.parametrize("algorithm", PARITY_ALGORITHMS)
def test_parity_gate_fails_an_optimizer_that_does_not_search(
    parity_cfg, algorithm, point, monkeypatch
):
    """Mutation check: replace an algorithm with one that evaluates a single
    point, on both sides, and the parity test must fail rather than pass or
    xfail. This is the check the old 1.0 bar on a sphere whose maximum is 0.5
    passed for 11 algorithms (#403)."""
    from humpday.optimizers import alloptimizers

    monkeypatch.setitem(alloptimizers.PURE_OPTIMIZERS, algorithm, _NoSearch(point))
    # Stand in for Node with the same stub, so this runs without it.
    monkeypatch.setattr(
        sys.modules[__name__],
        "_run_js_batch",
        lambda alg, n_trials, n_dim, func_id, n_runs, seed: [
            _run_python(alg, n_trials, n_dim, func_id, seed=seed + i)
            for i in range(n_runs)
        ],
    )
    with pytest.raises(AssertionError, match=">="):
        test_python_js_parity_sphere(parity_cfg, algorithm)


# How many independent (python, js) match-ups to run per algorithm in
# the win-rate test. 20 runs keeps each algorithm under ~5s and gives
# enough statistical power that the per-algorithm pass/fail call is
# stable across runs.
WINRATE_RUNS = 20

# Acceptable win-rate window. Under H0 (equivalent ports), each port's
# win count is Binomial(WINRATE_RUNS, 0.5). Demanding ≥ 4 wins per side
# out of 20 (i.e. ≤ p=0.6% per tail) is strict enough to surface real
# port divergence but not so strict that honest stochastic variation
# trips it. The test fails only on near-sweeps — the catch this is for.
WINRATE_MIN_WINS_PER_SIDE = 4

# Known divergent ports — the win-rate test trips on these with N=20 runs
# on the 2-D Rosenbrock objective. They xfail with strict=False, so:
#
#   - When a port is genuinely divergent (the common case), pytest
#     reports XFAIL and the test passes.
#   - When a port has been fixed and the test now passes, pytest
#     reports XPASS (highlighted in the summary) so we know to delete
#     the entry — but it does NOT fail the test, which keeps CI stable
#     for borderline algorithms that occasionally cross the threshold.
#
# As of 2026-05-30, only the trust-region family remains divergent.
# LBFGSB, BayesianOpt, AntColonyOpt, and HillClimbing all closed their
# gaps after the recent algorithm rewrites (#188 SA polish, #189 DE polish,
# ACOR port, Rechenberg/HillClimbing port) and are no longer divergent
# (all XPASS in the 2026-05-30 run).
#
# PRIMA_BOBYQA was at 88× worse on Rosenbrock; the JS port was rewritten
# to mirror the Python port (FD-gradient fallback when the model fit is
# singular). Both ports now converge to within 2% of each other, but the
# win-rate test still trips because both implementations are
# deterministic — every paired matchup has the same winner.
#
# The remaining divergent set (PRIMA trio + Powell) is the trust-region family, where the two
# ports disagree numerically rather than structurally. It was identified against a two-dimensional
# Elo sweep that has since been deleted; `humpday/data/ratings.json` is the current record, and it
# rates those four only on the Python side, so this set is a standing claim about the JS ports
# rather than something the table now measures.
KNOWN_DIVERGENT_PORTS = {
    "PRIMA_UOBYQA",
    "PRIMA_NEWUOA",
    "PRIMA_BOBYQA",
    "Powell",
}


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node not on PATH")
@pytest.mark.slow
@pytest.mark.parametrize(
    "algorithm",
    [
        pytest.param(
            a,
            marks=pytest.mark.xfail(
                strict=False,
                reason="Known port divergence; see KNOWN_DIVERGENT_PORTS note",
            ),
        )
        if a in KNOWN_DIVERGENT_PORTS
        else a
        for a in PARITY_ALGORITHMS
    ],
)
def test_python_js_winrate_rosenbrock(algorithm):
    """Head-to-head: on `WINRATE_RUNS` independent randomised matchups,
    Python and JS should each win some.

    A pair of equivalent implementations on the same objective with
    independent RNGs gives a 50/50 win rate in expectation. If one side
    sweeps (e.g. Python wins 12/12), the two ports have genuinely
    diverged — that's the catch this test is for.

    We use a Rosenbrock objective (rather than the sphere from
    `test_python_js_parity_sphere`) because Rosenbrock's ill-conditioned
    valley discriminates between algorithms much better. On the sphere
    almost every algorithm hits the floor regardless of port.
    """
    cfg = {
        "n_trials": 300,
        "n_dim": 2,
        "func_id": "rosenbrock_unit",
        "n_runs": WINRATE_RUNS,
    }
    py = _run_python_batch(algorithm, **cfg)
    js = _run_js_batch(algorithm, **cfg)

    assert len(py) == WINRATE_RUNS, (
        f"Python returned {len(py)} runs, expected {WINRATE_RUNS}"
    )
    assert len(js) == WINRATE_RUNS, (
        f"JS returned {len(js)} runs, expected {WINRATE_RUNS}"
    )

    py_vals = [r["best_value"] for r in py]
    js_vals = [r["best_value"] for r in js]

    # Pair-wise comparison: i-th Python run vs i-th JS run in the order
    # they happened. Each run is independently randomly initialized, so
    # this is a random pairing of independent samples — the canonical
    # head-to-head setup. (Sorting before comparison would amplify any
    # systematic floor-value difference between the ports and trip the
    # test even when both converge to numerical precision.)
    py_wins = sum(1 for p, j in zip(py_vals, js_vals) if p < j)
    js_wins = sum(1 for p, j in zip(py_vals, js_vals) if j < p)

    py_median = sorted(py_vals)[len(py_vals) // 2]
    js_median = sorted(js_vals)[len(js_vals) // 2]
    print(
        f"\n{algorithm}: py_wins={py_wins} js_wins={js_wins}  "
        f"(py median {py_median:.4g}, js median {js_median:.4g})"
    )

    # Both ports converging to numerical precision counts as a pass:
    # there's nothing to test if neither side has meaningful headroom
    # to lose. The win-rate check is only informative when at least one
    # side has visible residual error.
    if py_median < 1e-8 and js_median < 1e-8:
        return

    # Only fail on a near-sweep. Under H0 of equivalent ports, each
    # side's win count is Binomial(WINRATE_RUNS, 0.5); seeing fewer than
    # WINRATE_MIN_WINS_PER_SIDE wins out of 12 has p < 1.9% under H0,
    # i.e. real divergence, not stochastic noise.
    decisive = py_wins + js_wins
    if decisive == 0:
        return
    assert py_wins >= WINRATE_MIN_WINS_PER_SIDE, (
        f"{algorithm}: Python lost {py_wins}/{decisive} decisive runs — "
        f"JS port appears to dominate (py median {py_median:.4g}, "
        f"js median {js_median:.4g})"
    )
    assert js_wins >= WINRATE_MIN_WINS_PER_SIDE, (
        f"{algorithm}: JS lost {js_wins}/{decisive} decisive runs — "
        f"Python port appears to dominate (py median {py_median:.4g}, "
        f"js median {js_median:.4g})"
    )
