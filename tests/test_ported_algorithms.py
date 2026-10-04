"""
Tests for the algorithms that have been ported to use `humpday._array`
instead of direct numpy.

Each ported algorithm must:

1. Get below a measured bar on a shifted sphere (see TOLERANCE) under whichever
   shim backend is active.
2. Stay within its declared evaluation budget.
3. Return `best_x` whose elements all lie in [0, 1].

The harness force-imports the pure-Python backend via
`HUMPDAY_FORCE_PURE_ARRAY=1` for one set of tests, then runs the same
algorithms again against the default (numpy) backend. Both must pass.

Doing the force-pure path requires careful subprocess work because the
shim's backend selection happens at import time — once `humpday._array`
is loaded, the choice is sticky for the process. We run pure-backend
tests in a subprocess for isolation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

# The four algorithms ported in this PR. Adding a new ported algorithm?
# Add its class name to this list and confirm tests still pass under both
# backends.
PORTED = [
    ("evolutionary_algorithms", "RandomSearch"),
    ("evolutionary_algorithms", "HillClimbing"),
    ("evolutionary_algorithms", "SimulatedAnnealing"),
    ("evolutionary_algorithms", "HarmonySearch"),
    ("evolutionary_algorithms", "FireflyAlgorithm"),
    ("evolutionary_algorithms", "ParticleSwarm"),
    ("evolutionary_algorithms", "DifferentialEvolution"),
    ("evolutionary_algorithms", "GeneticAlgorithm"),
    ("evolutionary_algorithms", "EvolutionStrategy"),
    ("search_algorithms", "Rechenberg"),
    ("search_algorithms", "CoordinateDescent"),
    ("search_algorithms", "PatternSearch"),
    ("scipy_algorithms", "NelderMead"),
    ("scipy_algorithms", "Powell"),
    ("scipy_algorithms", "LBFGSB"),
    ("evolutionary_algorithms", "AntColonyOpt"),
    ("evolutionary_algorithms", "CMAEvolutionStrategy"),
    ("evolutionary_algorithms", "BayesianOpt"),
    ("prima_algorithms", "PRIMA_UOBYQA"),
    ("prima_algorithms", "PRIMA_NEWUOA"),
    ("prima_algorithms", "PRIMA_BOBYQA"),
]


# Both tests below minimise a shifted sphere in 5-D with 200 evaluations. Its
# optimum is SHIFT, the same one tests/test_js_parity.py uses: the sphere used to
# be centred at 0.5, where most of these algorithms take their first point, so an
# optimizer that evaluated the centre once and stopped scored 0 (#403).
SHIFT = (0.2473, 0.7718, 0.1859, 0.6934, 0.8146)
N_DIM = 5
N_TRIALS = 200
# Seeds for the portable PCG32 stream. The bars below were measured over seeds
# 0-49, so any seed is a fair draw; one keeps BayesianOpt's pure-Python run (22s
# at 200 evaluations, in the pure-backend CI job) from being paid three times.
SEEDS = (0,)

# Reference points for this objective on [0,1]^5:
#
#   maximum (the corner (1, 0, 0, 1, 0))                 2.969
#   value at the cube centre                             0.373
#   best of 200 uniform random points, 200k draws:
#     median 0.054, 99th percentile 0.127, 99.9th percentile 0.159
#
# RANDOM_SAMPLING_BAR is that 99.9th percentile. No algorithm's bar is looser,
# so an optimizer that stops where it started fails, and a single uniform random
# point gets under it 3.5% of the time.
RANDOM_SAMPLING_BAR = 0.159

# The bar for an algorithm whose worst result over seeds 0-49 (portable PCG32
# stream, numpy and pure backends alike) was below 5e-13.
CONVERGED = 1e-8

# Algorithms that search but do not converge in 200 evaluations. Each bar is ten
# times the worst result measured over seeds 0-49, rounded up, and never above
# RANDOM_SAMPLING_BAR. Worst measured values in brackets.
#
# The bottom five are at RANDOM_SAMPLING_BAR because ten times their worst result
# is above it. NelderMead's median is 3.5e-7, but one seed in ten ends above 0.018
# on this sphere, which for a simplex method is a shortfall rather than noise.
TOLERANCE = {
    "CoordinateDescent": 5e-6,  # [2.9e-7]
    "PatternSearch": 1e-4,  # [6.4e-6]
    "HillClimbing": 2e-4,  # [1.2e-5]
    "Rechenberg": 0.05,  # [2.5e-3]
    "HarmonySearch": 0.1,  # [0.010]
    "AntColonyOpt": RANDOM_SAMPLING_BAR,  # [0.025]
    "NelderMead": RANDOM_SAMPLING_BAR,  # [0.061]
    "EvolutionStrategy": RANDOM_SAMPLING_BAR,  # [0.077]
    "GeneticAlgorithm": RANDOM_SAMPLING_BAR,  # [0.11]
    "RandomSearch": RANDOM_SAMPLING_BAR,  # [0.14; it is the baseline]
}


def _tolerance(cls_name: str) -> float:
    return TOLERANCE.get(cls_name, CONVERGED)


def _shifted_sphere(x):
    return float(sum((xi - SHIFT[i]) ** 2 for i, xi in enumerate(x)))


def _run_seeded(cls, n_trials, seed):
    from humpday import _array as A

    A.use_portable_rng(seed)
    try:
        opt = cls(_shifted_sphere, n_trials=n_trials, n_dim=N_DIM)
        best_value, best_x = opt.optimize()
    finally:
        A.use_legacy_rng()
    return opt, best_value, best_x


def _check_quality(label, cls_name, seed, evaluations, n_trials, best_value):
    """The budget and quality gate shared by the numpy and pure-backend tests."""
    assert evaluations <= n_trials, (
        f"{label} {cls_name} seed {seed}: {evaluations} evals, budget {n_trials}"
    )
    tol = _tolerance(cls_name)
    assert best_value < tol, (
        f"{label} {cls_name} seed {seed}: {best_value:.4g} >= {tol:g}"
    )


@pytest.mark.parametrize("module,cls_name", PORTED)
def test_numpy_backend(module, cls_name):
    """Default numpy backend — what every existing user runs."""
    mod = __import__(f"humpday.optimizers.{module}", fromlist=[cls_name])
    cls = getattr(mod, cls_name)

    for seed in SEEDS:
        opt, best_value, best_x = _run_seeded(cls, N_TRIALS, seed)
        assert len(best_x) == N_DIM
        assert all(0.0 <= float(xi) <= 1.0 for xi in best_x), (
            f"{cls_name} returned out-of-bound best_x: {list(best_x)}"
        )
        _check_quality(
            "numpy", cls_name, seed, opt.evaluations, opt.n_trials, best_value
        )


def _no_search(point):
    """An optimizer that evaluates one point and stops: the failure the old bars,
    1.0 and 1.5 on a sphere whose maximum was 1.25, could not see (#403)."""
    from humpday import _array as A
    from humpday.optimizers.base import BaseOptimizer

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

    return NoSearch


@pytest.mark.parametrize("point", ["centre", "corner", "random"])
@pytest.mark.parametrize("module,cls_name", PORTED)
def test_gate_fails_an_optimizer_that_does_not_search(
    module, cls_name, point, monkeypatch
):
    """Mutation check: replace the algorithm with one that evaluates a single
    point, and the quality gate (shared with the pure-backend test) must fail."""
    mod = __import__(f"humpday.optimizers.{module}", fromlist=[cls_name])
    monkeypatch.setattr(mod, cls_name, _no_search(point))
    with pytest.raises(AssertionError, match=">="):
        test_numpy_backend(module, cls_name)


def test_pure_backend_works_for_ported_algorithms(tmp_path):
    """Run the same algorithms in a fresh subprocess with the pure backend
    forced via the env var. Confirms each one completes an optimization without
    any direct numpy call, within budget, and gets below the same bar as on the
    numpy backend (the portable PCG32 stream makes the two backends take the
    same path)."""

    script = textwrap.dedent("""
        import json, sys
        from humpday import _array as A
        assert A.BACKEND == "pure", f"expected pure, got {A.BACKEND}"

        from humpday.optimizers.evolutionary_algorithms import (
            RandomSearch, HillClimbing, SimulatedAnnealing, HarmonySearch,
            FireflyAlgorithm,
            ParticleSwarm, DifferentialEvolution, GeneticAlgorithm,
            EvolutionStrategy,
            AntColonyOpt, CMAEvolutionStrategy, BayesianOpt,
        )
        from humpday.optimizers.search_algorithms import (
            Rechenberg, CoordinateDescent, PatternSearch,
        )
        from humpday.optimizers.scipy_algorithms import NelderMead, Powell, LBFGSB
        from humpday.optimizers.prima_algorithms import (
            PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA,
        )

        SHIFT, N_DIM, SEEDS, BUDGET = json.loads(sys.argv[1])

        def sphere(x):
            return float(sum((xi - SHIFT[i]) ** 2 for i, xi in enumerate(x)))

        results = {}
        ALGORITHMS = [
            RandomSearch, HillClimbing, SimulatedAnnealing, HarmonySearch,
            FireflyAlgorithm,
            ParticleSwarm, DifferentialEvolution, GeneticAlgorithm,
            EvolutionStrategy,
            AntColonyOpt, CMAEvolutionStrategy, BayesianOpt,
            Rechenberg, CoordinateDescent, PatternSearch,
            NelderMead, Powell, LBFGSB,
            PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA,
        ]
        for cls in ALGORITHMS:
            n_trials = BUDGET[cls.__name__]
            runs = []
            for seed in SEEDS:
                A.use_portable_rng(seed)
                opt = cls(sphere, n_trials=n_trials, n_dim=N_DIM)
                best_value, best_x = opt.optimize()
                runs.append({
                    "seed": seed,
                    "n_trials": n_trials,
                    "best_value": float(best_value),
                    "best_x_len": len(best_x),
                    "best_x_type": type(best_x).__name__,
                    "evaluations": opt.evaluations,
                    "in_bounds": all(0.0 <= float(xi) <= 1.0 for xi in best_x),
                })
            results[cls.__name__] = runs
        print(json.dumps(results))
    """)

    # Every algorithm gets the numpy test's budget except BayesianOpt. Its
    # pure-Python Gaussian-process fit costs 22s for 200 evaluations against 1s
    # for 100, so it runs 100 here. It still converges at 100 (worst 2.1e-13 over
    # seeds 0-49), so it is held to the same CONVERGED bar.
    budget = {name: N_TRIALS for _, name in PORTED}
    budget["BayesianOpt"] = 100

    env = dict(os.environ)
    env["HUMPDAY_FORCE_PURE_ARRAY"] = "1"
    # Make sure subprocess sees the in-tree humpday, not whatever might be
    # `pip install`-ed system-wide.
    env["PYTHONPATH"] = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            json.dumps([SHIFT, N_DIM, list(SEEDS), budget]),
        ],
        env=env,
        capture_output=True,
        text=True,
        # Measured at about 6s on an M-series laptop, most of it BayesianOpt and
        # PRIMA_UOBYQA. 180s leaves room for slower CI runners.
        timeout=180,
    )

    assert completed.returncode == 0, (
        f"pure-backend run failed:\nstdout: {completed.stdout}\n"
        f"stderr: {completed.stderr}"
    )

    results = json.loads(completed.stdout.strip().splitlines()[-1])
    expected = {name for _, name in PORTED}
    assert set(results) == expected, (
        f"missing or extra results: expected {expected}, got {set(results)}"
    )
    for name, runs in results.items():
        assert len(runs) == len(SEEDS)
        for r in runs:
            assert r["best_x_len"] == N_DIM, f"{name}: best_x len {r['best_x_len']}"
            assert r["best_x_type"] == "_Vec", (
                f"{name}: best_x was {r['best_x_type']!r}, expected pure-backend _Vec"
            )
            assert r["in_bounds"], f"{name} returned out-of-bound best_x"
            _check_quality(
                "pure",
                name,
                r["seed"],
                r["evaluations"],
                r["n_trials"],
                r["best_value"],
            )
