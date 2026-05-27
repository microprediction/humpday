"""
Reference-alignment tests for the HumpDay Python ports.

For each algorithm pair (HumpDay, trusted reference), this runs both
on the same set of test problems and records:
  - final objective values
  - number of function evaluations
  - distance to the known global optimum

The goal is **alignment**: HumpDay's port should behave indistinguishably
from the reference within floating-point reason. Today most algorithms
deviate substantially — those gaps are what this test makes visible.

The test always *passes* (it's a characterisation harness, not a gate).
The point is the printed table and the persisted snapshot at
`benchmarks/reference_alignment.json`. Run with stdout visible:

    pytest tests/test_reference_alignment.py -m reference -s

Each reference dependency auto-skips if not installed. Install them all
with `pip install humpday[reference]` (defined in pyproject.toml).
"""

from __future__ import annotations

import importlib
import json
import math
import time
from pathlib import Path

import pytest

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

REPO_ROOT = Path(__file__).parent.parent
N_RUNS = 4

# Cap n_trials per reference. Some references (e.g. skopt's gp_minimize)
# scale cubically with n_calls; running them at the full HumpDay budget
# makes the harness take an hour. Use a smaller budget for those.
REFERENCE_BUDGET_OVERRIDE = {
    "BayesianOpt": 50,  # skopt GP fits cubically in n_calls
}


# ---------- objectives (defined on [0, 1]^n with known optima) ----------


def _sphere(x):
    """Convex quadratic, minimum 0 at [0.5, ..., 0.5]."""
    return sum((v - 0.5) ** 2 for v in x)


def _rosenbrock_unit(x):
    """2-D Rosenbrock mapped to [0,1]^2 via 4xi - 2. Minimum at (0.75, 0.75)."""
    a = 4 * x[0] - 2
    b = 4 * x[1] - 2
    return (1 - a) ** 2 + 100 * (b - a * a) ** 2


def _ackley(x):
    """Ackley centred at [0.5, 0.5]. Minimum 0 at [0.5, 0.5]."""
    n = len(x)
    s = [10 * (v - 0.5) for v in x]
    return (
        -20 * math.exp(-0.2 * math.sqrt(sum(v * v for v in s) / n))
        - math.exp(sum(math.cos(2 * math.pi * v) for v in s) / n)
        + 20
        + math.e
    )


PROBLEMS = {
    "sphere": {"func": _sphere, "opt": 0.0, "x_opt": [0.5, 0.5]},
    "rosenbrock": {"func": _rosenbrock_unit, "opt": 0.0, "x_opt": [0.75, 0.75]},
    "ackley": {"func": _ackley, "opt": 0.0, "x_opt": [0.5, 0.5]},
}


# ---------- helpers ----------


def _try_import(name):
    try:
        return importlib.import_module(name)
    except (ImportError, RuntimeError):
        # RuntimeError covers e.g. PDFO compiled against numpy 1.x raising
        # when imported under numpy 2.x.
        return None


def _can_load_pdfo():
    """Defensive PDFO probe — its top-level import doesn't trigger the
    numpy-2.x crash, but `from .gethuge import gethuge` inside the
    solver call does. Run a tiny call here so the failure surfaces at
    skip-time, not test-time."""
    try:
        import numpy as np
        from pdfo import newuoa

        newuoa(
            lambda x: float(sum(x)),
            np.array([0.0, 0.0]),
            options={"maxfev": 5, "rhobeg": 0.1, "rhoend": 1e-2},
        )
        return True
    except Exception:
        return False


def _run_humpday(algorithm: str, func, n_trials: int, n_dim: int):
    cls = PURE_OPTIMIZERS[algorithm]
    opt = cls(func, n_trials=n_trials, n_dim=n_dim)
    res = opt.optimize()
    if isinstance(res, tuple) and len(res) == 2:
        best_value = float(res[0])
    else:
        best_value = float(opt.best_value)
    return {"best_value": best_value, "evals": int(opt.evaluations)}


# ---------- reference adapters ----------
# Each adapter returns {best_value, evals} on the same problem the HumpDay
# port saw. References themselves can use different conventions (some
# count evals differently); we report what the reference reports.


def _ref_scipy_neldermead(func, n_trials, n_dim):
    from scipy.optimize import minimize

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    x0 = [0.5] * n_dim
    r = minimize(
        wrapped,
        x0,
        method="Nelder-Mead",
        options={"maxfev": n_trials, "xatol": 1e-8, "fatol": 1e-8},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_powell(func, n_trials, n_dim):
    from scipy.optimize import minimize

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    x0 = [0.5] * n_dim
    r = minimize(
        wrapped,
        x0,
        method="Powell",
        options={"maxfev": n_trials, "xtol": 1e-8, "ftol": 1e-8},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_de(func, n_trials, n_dim):
    from scipy.optimize import differential_evolution

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    bounds = [(0, 1)] * n_dim
    r = differential_evolution(
        wrapped,
        bounds,
        maxiter=max(1, n_trials // (10 * n_dim)),
        popsize=10,
        tol=1e-8,
        seed=0,
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_dual_annealing(func, n_trials, n_dim):
    from scipy.optimize import dual_annealing

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    bounds = [(0, 1)] * n_dim
    r = dual_annealing(wrapped, bounds, maxiter=max(1, n_trials // 10), seed=0)
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_skopt_gp(func, n_trials, n_dim):
    from skopt import gp_minimize

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    bounds = [(0.0, 1.0)] * n_dim
    r = gp_minimize(
        wrapped,
        bounds,
        n_calls=n_trials,
        random_state=0,
        n_initial_points=min(10, n_trials // 2),
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_cmaes(func, n_trials, n_dim):
    """CyberAgent `cmaes` reference. The library's `ask/tell` API expects
    exactly `population_size` solutions per `tell` call — partial
    generations (e.g. due to budget exhaustion) cause it to raise. We
    therefore complete whole generations only, and stop once the budget
    can't fit another full one."""
    import numpy as np
    from cmaes import CMA

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    es = CMA(
        mean=np.full(n_dim, 0.5), sigma=0.2, bounds=np.array([[0, 1]] * n_dim), seed=0
    )
    best = float("inf")
    while counter["n"] + es.population_size <= n_trials:
        sols = []
        for _ in range(es.population_size):
            x = es.ask()
            v = wrapped(x.tolist())
            sols.append((x, v))
            if v < best:
                best = v
        es.tell(sols)
    return {"best_value": float(best), "evals": counter["n"]}


def _ref_pybobyqa(func, n_trials, n_dim):
    import numpy as np
    import pybobyqa

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    x0 = np.full(n_dim, 0.5)
    bounds = (np.zeros(n_dim), np.ones(n_dim))
    r = pybobyqa.solve(
        wrapped,
        x0,
        bounds=bounds,
        maxfun=n_trials,
        seek_global_minimum=False,
        rhobeg=0.3,
        rhoend=1e-8,
        print_progress=False,
    )
    return {"best_value": float(r.f), "evals": counter["n"]}


def _ref_pdfo_newuoa(func, n_trials, n_dim):
    import numpy as np
    from pdfo import newuoa

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    r = newuoa(
        wrapped,
        np.full(n_dim, 0.5),
        options={"maxfev": n_trials, "rhobeg": 0.3, "rhoend": 1e-8},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_pdfo_uobyqa(func, n_trials, n_dim):
    import numpy as np
    from pdfo import uobyqa

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    r = uobyqa(
        wrapped,
        np.full(n_dim, 0.5),
        options={"maxfev": n_trials, "rhobeg": 0.3, "rhoend": 1e-8},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


# Algorithm -> (reference_label, reference_adapter, required_modules).
REFERENCES = {
    "NelderMead": ("scipy.optimize Nelder-Mead", _ref_scipy_neldermead, ["scipy"]),
    "Powell": ("scipy.optimize Powell", _ref_scipy_powell, ["scipy"]),
    "DifferentialEvolution": ("scipy differential_evolution", _ref_scipy_de, ["scipy"]),
    "SimulatedAnnealing": (
        "scipy dual_annealing",
        _ref_scipy_dual_annealing,
        ["scipy"],
    ),
    "BayesianOpt": ("scikit-optimize gp_minimize", _ref_skopt_gp, ["skopt"]),
    "CMAEvolutionStrategy": ("cmaes (CyberAgent)", _ref_cmaes, ["cmaes", "numpy"]),
    "PRIMA_BOBYQA": ("Py-BOBYQA", _ref_pybobyqa, ["pybobyqa", "numpy"]),
    "PRIMA_NEWUOA": ("PDFO newuoa", _ref_pdfo_newuoa, ["pdfo", "numpy"]),
    "PRIMA_UOBYQA": ("PDFO uobyqa", _ref_pdfo_uobyqa, ["pdfo", "numpy"]),
}


_PDFO_OK = None  # cached so we don't probe twice


def _all_installed(modules):
    global _PDFO_OK
    for m in modules:
        if _try_import(m) is None:
            return False
        if m == "pdfo":
            if _PDFO_OK is None:
                _PDFO_OK = _can_load_pdfo()
            if not _PDFO_OK:
                return False
    return True


# ---------- the characterisation test ----------


@pytest.mark.reference
def test_reference_alignment():
    """Print + persist a table of HumpDay-vs-reference final values for every
    algorithm where we have a reference adapter."""
    n_trials_default = 200
    n_dim = 2
    rows = []

    for algorithm, (ref_label, ref_fn, mods) in REFERENCES.items():
        if not _all_installed(mods):
            print(f"\n--- {algorithm}: SKIP ({', '.join(mods)} not installed) ---")
            continue
        n_trials = REFERENCE_BUDGET_OVERRIDE.get(algorithm, n_trials_default)
        print(f"\n=== {algorithm}  vs  {ref_label}  (n_trials={n_trials}) ===")
        for problem_id, problem in PROBLEMS.items():
            func = problem["func"]
            opt_value = problem["opt"]

            hd_vals = []
            for _ in range(N_RUNS):
                hd_vals.append(
                    _run_humpday(algorithm, func, n_trials, n_dim)["best_value"]
                )

            ref_vals = []
            for _ in range(N_RUNS):
                try:
                    ref_vals.append(ref_fn(func, n_trials, n_dim)["best_value"])
                except Exception as e:
                    print(f"    reference error on {problem_id}: {e}")
                    ref_vals.append(float("inf"))

            hd_med = sorted(hd_vals)[N_RUNS // 2]
            ref_med = sorted(ref_vals)[N_RUNS // 2]
            hd_gap = hd_med - opt_value
            ref_gap = ref_med - opt_value
            relative = (hd_gap + 1e-15) / (ref_gap + 1e-15)

            print(
                f"  {problem_id:<12}  hd={hd_med:>10.4g}  ref={ref_med:>10.4g}"
                f"  hd-to-opt={hd_gap:>10.4g}  ref-to-opt={ref_gap:>10.4g}"
                f"  hd/ref={relative:>9.2f}"
            )

            rows.append(
                {
                    "algorithm": algorithm,
                    "reference": ref_label,
                    "problem": problem_id,
                    "humpday_median": hd_med,
                    "reference_median": ref_med,
                    "humpday_to_opt": hd_gap,
                    "reference_to_opt": ref_gap,
                    "ratio_humpday_over_reference": relative,
                }
            )

    out = REPO_ROOT / "benchmarks" / "reference_alignment.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "rows": rows,
                "n_runs": N_RUNS,
                "n_trials": n_trials,
                "n_dim": n_dim,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
