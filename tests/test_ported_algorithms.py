"""
Tests for the algorithms that have been ported to use `humpday._array`
instead of direct numpy.

Each ported algorithm must:

1. Produce a sensible result (close to the known minimum) under whichever
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
    ("evolutionary_algorithms", "TabuSearch"),
    ("evolutionary_algorithms", "FireflyAlgorithm"),
    ("search_algorithms", "AdaptiveRandomSearch"),
    ("search_algorithms", "CoordinateDescent"),
    ("search_algorithms", "PatternSearch"),
]


@pytest.mark.parametrize("module,cls_name", PORTED)
def test_numpy_backend(module, cls_name):
    """Default numpy backend — what every existing user runs."""
    mod = __import__(f"humpday.optimizers.{module}", fromlist=[cls_name])
    cls = getattr(mod, cls_name)

    def sphere(x):
        return float(sum((xi - 0.5) ** 2 for xi in x))

    opt = cls(sphere, n_trials=200, n_dim=5)
    best_value, best_x = opt.optimize()

    # Budget must be respected.
    assert opt.evaluations <= opt.n_trials, (
        f"{cls_name} did {opt.evaluations} evals, budget was {opt.n_trials}"
    )
    # best_x is a numpy ndarray under numpy backend.
    assert len(best_x) == 5
    assert all(0.0 <= float(xi) <= 1.0 for xi in best_x), (
        f"{cls_name} returned out-of-bound best_x: {list(best_x)}"
    )
    # Sphere at the centre is exactly 0. With 200 trials in 5-D any decent
    # algorithm should find a value below 1.0 (much weaker than convergence
    # — this just guards against an algorithm returning the worst point).
    assert best_value < 1.0, f"{cls_name} barely improved: {best_value}"


def test_pure_backend_works_for_ported_algorithms(tmp_path):
    """Run the same four algorithms in a fresh subprocess with the pure
    backend forced via the env var. Confirms each one can complete an
    optimization without any direct numpy call."""

    script = textwrap.dedent("""
        import json, sys
        from humpday import _array as A
        assert A.BACKEND == "pure", f"expected pure, got {A.BACKEND}"

        from humpday.optimizers.evolutionary_algorithms import (
            RandomSearch, HillClimbing, SimulatedAnnealing, HarmonySearch,
            TabuSearch, FireflyAlgorithm,
        )
        from humpday.optimizers.search_algorithms import (
            AdaptiveRandomSearch, CoordinateDescent, PatternSearch,
        )

        def sphere(x):
            return float(sum((xi - 0.5) ** 2 for xi in x))

        results = {}
        ALGORITHMS = [
            RandomSearch, HillClimbing, SimulatedAnnealing, HarmonySearch,
            TabuSearch, FireflyAlgorithm,
            AdaptiveRandomSearch, CoordinateDescent, PatternSearch,
        ]
        for cls in ALGORITHMS:
            opt = cls(sphere, n_trials=200, n_dim=5)
            best_value, best_x = opt.optimize()
            results[cls.__name__] = {
                "best_value": float(best_value),
                "best_x_len": len(best_x),
                "best_x_type": type(best_x).__name__,
                "evaluations": opt.evaluations,
                "in_bounds": all(0.0 <= float(xi) <= 1.0 for xi in best_x),
            }
        print(json.dumps(results))
    """)

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
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, (
        f"pure-backend run failed:\nstdout: {completed.stdout}\n"
        f"stderr: {completed.stderr}"
    )

    import json

    results = json.loads(completed.stdout.strip().splitlines()[-1])
    expected = {name for _, name in PORTED}
    assert set(results) == expected, (
        f"missing or extra results: expected {expected}, got {set(results)}"
    )
    for name, r in results.items():
        assert r["evaluations"] <= 200, f"{name}: {r['evaluations']} > 200"
        assert r["best_x_len"] == 5, f"{name}: best_x len {r['best_x_len']}"
        assert r["best_x_type"] == "_Vec", (
            f"{name}: best_x was {r['best_x_type']!r}, expected pure-backend _Vec"
        )
        assert r["in_bounds"], f"{name} returned out-of-bound best_x"
        assert r["best_value"] < 1.0, (
            f"{name} barely improved under pure backend: {r['best_value']}"
        )
