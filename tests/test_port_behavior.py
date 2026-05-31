"""
Characterisation harness for the 21 algorithm pairs.

This test isn't a pass/fail gate — it's a record of where each port
stands relative to (a) the known global optimum of the test problem
and (b) the other port. Used to:

  1. Target ports for improvement (those farthest from optimum).
  2. Verify that a "fix" actually improves the port (compare before/after
     numbers from this test).

For each (algorithm, objective, n_trials) cell, runs the algorithm
N times in each language and records:
  - median best value
  - distance to the known global optimum (relative error)
  - which port wins the head-to-head

Run with:
    pytest tests/test_port_behavior.py -m slow -s
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

# Test problems. Each is defined on [0,1]^n with a known minimum value.
# JS runner mirrors these in tests/js_parity_runner.js — keep in sync.
PROBLEMS = {
    "sphere_at_half": {
        "py": lambda x: sum((v - 0.5) ** 2 for v in x),
        "opt_value": 0.0,
        "description": "convex, centred",
    },
    "rosenbrock_unit": {
        # 2-D Rosenbrock, mapped [0,1]^2 → [-2,2]^2 via 4x-2.
        # Minimum at (a,b) = (1,1), i.e. (x,y) = (0.75, 0.75).
        "py": lambda x: (
            (1 - (4 * x[0] - 2)) ** 2
            + 100 * ((4 * x[1] - 2) - (4 * x[0] - 2) ** 2) ** 2
        ),
        "opt_value": 0.0,
        "description": "ill-conditioned valley",
    },
}

# Per-algorithm budgets — smaller for sphere, larger for Rosenbrock.
N_RUNS = 12  # match-ups per cell
BUDGETS = [100, 300]


def _run_python(algorithm: str, func, n_trials: int, n_dim: int) -> float:
    """Run the Python port and return best_value."""
    cls = PURE_OPTIMIZERS[algorithm]
    opt = cls(func, n_trials=n_trials, n_dim=n_dim)
    result = opt.optimize()
    if isinstance(result, tuple) and len(result) == 2:
        return float(result[0])
    return float(opt.best_value)


def _run_js_batch(
    algorithm: str, n_trials: int, n_dim: int, func_id: str, n_runs: int
) -> list[float]:
    """Run the JS port `n_runs` times in one Node call; return list of best_values."""
    result = subprocess.run(
        [
            NODE,
            str(RUNNER_JS),
            algorithm,
            str(n_trials),
            str(n_dim),
            func_id,
            str(n_runs),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"JS exit {result.returncode}: {result.stderr.strip()}")
    out = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        d = json.loads(line)
        if "error" in d:
            raise RuntimeError(f"JS error: {d['error']}")
        out.append(float(d["best_value"]))
    return out


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.slow
def test_characterise_all_ports():
    """Walk every (algorithm, problem, budget) cell and record the
    median Python value, median JS value, distance to known optimum,
    and head-to-head win counts. Always passes; the value is the
    printed table — keep stdout (-s) when running."""
    rows = []

    for problem_id, problem in PROBLEMS.items():
        opt_value = problem["opt_value"]
        py_func = problem["py"]

        for n_trials in BUDGETS:
            for algorithm in sorted(PURE_OPTIMIZERS.keys()):
                # Python: N runs in-process.
                py_vals = [
                    _run_python(algorithm, py_func, n_trials, n_dim=2)
                    for _ in range(N_RUNS)
                ]
                # JS: N runs in one Node subprocess.
                try:
                    js_vals = _run_js_batch(
                        algorithm, n_trials, n_dim=2, func_id=problem_id, n_runs=N_RUNS
                    )
                except RuntimeError as e:
                    js_vals = []
                    js_error = str(e)
                else:
                    js_error = None

                py_med = _median(py_vals)
                js_med = _median(js_vals) if js_vals else float("inf")
                py_to_opt = py_med - opt_value
                js_to_opt = js_med - opt_value

                py_wins = (
                    sum(1 for p, j in zip(py_vals, js_vals) if p < j)
                    if js_vals
                    else None
                )
                js_wins = (
                    sum(1 for p, j in zip(py_vals, js_vals) if j < p)
                    if js_vals
                    else None
                )

                rows.append(
                    {
                        "problem": problem_id,
                        "n_trials": n_trials,
                        "algorithm": algorithm,
                        "py_median": py_med,
                        "js_median": js_med,
                        "py_to_opt": py_to_opt,
                        "js_to_opt": js_to_opt,
                        "py_wins": py_wins,
                        "js_wins": js_wins,
                        "js_error": js_error,
                    }
                )

    # Print a readable table per problem.
    for problem_id in PROBLEMS:
        for n_trials in BUDGETS:
            cell_rows = [
                r
                for r in rows
                if r["problem"] == problem_id and r["n_trials"] == n_trials
            ]
            print(f"\n=== {problem_id}, n_trials={n_trials}, n_dim=2 ===")
            print(
                f"{'algorithm':<24}  {'py median':>11}  {'js median':>11}  "
                f"{'py wins':>7}  {'js wins':>7}  {'ratio (js/py)':>14}"
            )
            for r in sorted(cell_rows, key=lambda r: r["js_to_opt"] - r["py_to_opt"]):
                if r["js_error"]:
                    print(f"  {r['algorithm']:<22}  JS ERROR: {r['js_error'][:80]}")
                    continue
                if r["py_median"] > 1e-12:
                    ratio = r["js_median"] / r["py_median"]
                    ratio_s = f"{ratio:>13.2f}x"
                else:
                    ratio_s = "n/a (py≈0)"
                print(
                    f"  {r['algorithm']:<22}  {r['py_median']:>11.4g}  "
                    f"{r['js_median']:>11.4g}  {r['py_wins']:>7}  "
                    f"{r['js_wins']:>7}  {ratio_s:>14}"
                )

    # Persist for later regression comparisons.
    out_path = REPO_ROOT / "benchmarks" / "port_characterisation.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "n_runs": N_RUNS}, f, indent=2)
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
