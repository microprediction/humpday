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

Fairness of starting points
---------------------------
HumpDay's algorithms initialise from `0.3 + 0.4 * U[0, 1]^n` (a random
interior point in the unit cube). To compare apples-to-apples we draw
each trial's x0 from the **same distribution**, seeded by the trial
index, and use it on both sides — `_array.seed(seed)` for HumpDay
(plus the stdlib `random.seed`, which the evolutionary algorithms use)
and an explicit `x0` for any local-search reference that takes one.
Global-search references (DE, dual annealing, gp_minimize, cmaes) get
the same integer seed via their own RNG parameter.

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
import random
import time
from pathlib import Path

import pytest

import humpday._array as _humpday_array
from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

REPO_ROOT = Path(__file__).parent.parent
N_RUNS = 4

# Same distribution HumpDay's optimisers use internally. Draw `n_dim`
# coordinates in [0.3, 0.7] via a seeded `random.Random` so a given
# trial index produces a reproducible x0 across calls.
X0_LO, X0_HI = 0.3, 0.7


def _draw_x0(seed, n_dim):
    rng = random.Random(seed)
    return [X0_LO + (X0_HI - X0_LO) * rng.random() for _ in range(n_dim)]


def _seed_humpday(seed):
    """Seed every RNG HumpDay's algorithms might draw from."""
    _humpday_array.seed(seed)
    random.seed(seed)


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


def _run_humpday(algorithm: str, func, n_trials: int, n_dim: int, seed: int):
    _seed_humpday(seed)
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
# port saw, started from a seed-determined x0 drawn from the same
# distribution HumpDay uses. References themselves can use different
# conventions (some count evals differently); we report what the
# reference reports.


def _ref_scipy_neldermead(func, n_trials, n_dim, seed):
    from scipy.optimize import minimize

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    # Tolerances match HumpDay's NelderMead (xatol=fatol=1e-12). With
    # scipy's default 1e-4 the reference stops far below its potential;
    # at 1e-12 both implementations run until budget exhaustion or
    # genuine numerical convergence.
    r = minimize(
        wrapped,
        _draw_x0(seed, n_dim),
        method="Nelder-Mead",
        options={"maxfev": n_trials, "xatol": 1e-12, "fatol": 1e-12},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_powell(func, n_trials, n_dim, seed):
    from scipy.optimize import minimize

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    # Tolerances match HumpDay's Powell (ftol=1e-12) — see the
    # NelderMead adapter above for rationale.
    r = minimize(
        wrapped,
        _draw_x0(seed, n_dim),
        method="Powell",
        options={"maxfev": n_trials, "xtol": 1e-12, "ftol": 1e-12},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_lbfgsb(func, n_trials, n_dim, seed):
    from scipy.optimize import minimize

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    bounds = [(0.0, 1.0)] * n_dim
    # `maxfun` (not `maxiter`) caps function evaluations — L-BFGS-B
    # uses finite-difference gradient internally so one "iteration"
    # already eats ~n_dim evals. ftol=1e-12, gtol=1e-12 push the
    # solver to the same precision floor we use for NelderMead /
    # Powell (scipy's defaults stop ~1e-7 below the optimum).
    r = minimize(
        wrapped,
        _draw_x0(seed, n_dim),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxfun": n_trials, "ftol": 1e-12, "gtol": 1e-12},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_de(func, n_trials, n_dim, seed):
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
        seed=seed,
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_dual_annealing(func, n_trials, n_dim, seed):
    from scipy.optimize import dual_annealing

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    bounds = [(0, 1)] * n_dim
    r = dual_annealing(wrapped, bounds, maxiter=max(1, n_trials // 10), seed=seed)
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_skopt_gp(func, n_trials, n_dim, seed):
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
        random_state=seed,
        n_initial_points=min(10, n_trials // 2),
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_cmaes(func, n_trials, n_dim, seed):
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
        mean=np.asarray(_draw_x0(seed, n_dim)),
        sigma=0.2,
        bounds=np.array([[0, 1]] * n_dim),
        seed=seed,
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


def _ref_random_search(func, n_trials, n_dim, seed):
    """Uniform-sample baseline — draw n_trials i.i.d. samples from
    `U[0, 1]^n_dim` and return the best. Cheapest possible
    lower-bound: any algorithm worth using should beat this median.
    """
    rng = random.Random(seed)
    best = float("inf")
    n_evals = 0
    for _ in range(n_trials):
        x = [rng.random() for _ in range(n_dim)]
        v = func(x)
        n_evals += 1
        if v < best:
            best = v
    return {"best_value": float(best), "evals": n_evals}


def _ref_grid_search(func, n_trials, n_dim, seed):
    """Regular-grid baseline — `n_per_axis^n_dim` evaluations on a
    uniform Cartesian grid with bin-centred coordinates. Like
    `_ref_random_search`, this is included as a sanity floor and is
    deterministic in `n_trials`/`n_dim` (the `seed` is unused)."""
    n_per_axis = max(2, round(n_trials ** (1.0 / n_dim)))
    indices = [0] * n_dim
    best = float("inf")
    n_evals = 0
    while n_evals < n_trials:
        x = [(i + 0.5) / n_per_axis for i in indices]
        v = func(x)
        n_evals += 1
        if v < best:
            best = v
        d = n_dim - 1
        while d >= 0:
            indices[d] += 1
            if indices[d] < n_per_axis:
                break
            indices[d] = 0
            d -= 1
        if d < 0:
            break
    _ = seed
    return {"best_value": float(best), "evals": n_evals}


def _ref_oneplusone_es_decay(func, n_trials, n_dim, seed):
    """(1+1)-ES with a geometric sigma decay schedule — the natural
    reference for HillClimbing. Starts at sigma=0.1 and decays so the
    final sigma is ~1e-3, matching a "hill-climbing with shrinking
    perturbations" intuition."""
    rng = random.Random(seed)
    x = [_draw_x0(seed, n_dim)[i] for i in range(n_dim)]
    fx = func(x)
    n_evals = 1
    sigma_init = 0.1
    sigma_final = 1e-3
    decay = (sigma_final / sigma_init) ** (1.0 / max(1, n_trials - 1))
    sigma = sigma_init

    for _ in range(n_trials - 1):
        z = [rng.gauss(0, 1) for _ in range(n_dim)]
        x_new = [min(1.0, max(0.0, x[i] + sigma * z[i])) for i in range(n_dim)]
        fx_new = func(x_new)
        n_evals += 1
        if fx_new < fx:
            x, fx = x_new, fx_new
        sigma *= decay
    return {"best_value": float(fx), "evals": n_evals}


def _ref_oneplusone_es_oneFifth(func, n_trials, n_dim, seed):
    """(1+1)-ES with Rechenberg's 1/5-success-rule — the natural
    reference for Rechenberg. Sigma grows by 1.5× when the
    success rate over the last 10 trials exceeds 1/5, shrinks by 1/1.5
    otherwise."""
    rng = random.Random(seed)
    x = [_draw_x0(seed, n_dim)[i] for i in range(n_dim)]
    fx = func(x)
    n_evals = 1
    sigma = 0.1
    window = []
    window_size = 10

    for _ in range(n_trials - 1):
        z = [rng.gauss(0, 1) for _ in range(n_dim)]
        x_new = [min(1.0, max(0.0, x[i] + sigma * z[i])) for i in range(n_dim)]
        fx_new = func(x_new)
        n_evals += 1
        accepted = fx_new < fx
        if accepted:
            x, fx = x_new, fx_new
        window.append(accepted)
        if len(window) > window_size:
            window.pop(0)
        if len(window) >= window_size:
            rate = sum(window) / window_size
            if rate > 1 / 5:
                sigma *= 1.5
            elif rate < 1 / 5:
                sigma /= 1.5
    return {"best_value": float(fx), "evals": n_evals}


def _ref_scipy_powell_diagonal(func, n_trials, n_dim, seed):
    """scipy.optimize.minimize(method="Powell", direc=I) — Powell's
    direction-set method constrained to the cardinal directions. This
    is the textbook "coordinate descent with a line search per axis"
    and is the natural reference for HumpDay's CoordinateDescent."""
    import numpy as np
    from scipy.optimize import minimize

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    r = minimize(
        wrapped,
        _draw_x0(seed, n_dim),
        method="Powell",
        options={
            "maxfev": n_trials,
            "xtol": 1e-12,
            "ftol": 1e-12,
            "direc": np.eye(n_dim),
        },
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_direct(func, n_trials, n_dim, seed):
    """scipy.optimize.direct (DIRECT — DIviding RECTangles) — natural
    reference for HumpDay's PatternSearch. DIRECT is deterministic so
    `seed` is unused; we still take it to keep the adapter signature
    uniform. Bounds = [0,1]^n; maxfun caps the budget directly."""
    from scipy.optimize import direct

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    r = direct(wrapped, [(0.0, 1.0)] * n_dim, maxfun=n_trials)
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_mealpy(cls_path, func, n_trials, n_dim, seed, pop_size=20, kwargs=None):
    """Run a mealpy algorithm and return {best_value, evals}.

    `cls_path` is a string like "mealpy.swarm_based.PSO.OriginalPSO";
    we import lazily so a missing mealpy install just makes the
    corresponding `REFERENCES` entry skip cleanly.

    mealpy budgets total evaluations as `epoch * pop_size`, so we set
    `epoch = max(1, n_trials // pop_size)`. We also silence its
    INFO-level logging (~one line per epoch — drowns out the
    pytest -s view) and wrap the objective to count evaluations
    ourselves rather than relying on mealpy's `nfe_*` attributes
    (which differ between algorithm classes).
    """
    import importlib
    import logging

    import numpy as np
    from mealpy import FloatVar

    mod_path, _, cls_name = cls_path.rpartition(".")
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    epoch = max(1, n_trials // pop_size)
    problem = {
        "obj_func": wrapped,
        "bounds": FloatVar(lb=[0.0] * n_dim, ub=[1.0] * n_dim),
        "minmax": "min",
    }
    # Quiet mealpy's per-epoch log lines for the whole sweep.
    logging.getLogger("mealpy").setLevel(logging.WARNING)

    opt = cls(epoch=epoch, pop_size=pop_size, **(kwargs or {}))
    g_best = opt.solve(problem, seed=seed)
    return {"best_value": float(g_best.target.fitness), "evals": counter["n"]}


def _ref_mealpy_pso(func, n_trials, n_dim, seed):
    return _ref_mealpy(
        "mealpy.swarm_based.PSO.OriginalPSO", func, n_trials, n_dim, seed
    )


def _ref_mealpy_ga(func, n_trials, n_dim, seed):
    return _ref_mealpy(
        "mealpy.evolutionary_based.GA.BaseGA", func, n_trials, n_dim, seed
    )


def _ref_mealpy_firefly(func, n_trials, n_dim, seed):
    # Use FFA (Firefly Algorithm) — `mealpy.swarm_based.FA` is the
    # Fireworks Algorithm (different family). #176 picked the wrong
    # one; the snapshot's previous "Firefly" comparison was actually
    # humpday's Firefly vs mealpy's Fireworks.
    return _ref_mealpy(
        "mealpy.swarm_based.FFA.OriginalFFA", func, n_trials, n_dim, seed
    )


def _ref_mealpy_harmony(func, n_trials, n_dim, seed):
    return _ref_mealpy("mealpy.music_based.HS.OriginalHS", func, n_trials, n_dim, seed)


def _ref_mealpy_es(func, n_trials, n_dim, seed):
    return _ref_mealpy(
        "mealpy.evolutionary_based.ES.OriginalES", func, n_trials, n_dim, seed
    )


def _ref_mealpy_acor(func, n_trials, n_dim, seed):
    # ACOR uses sample_count instead of pop_size for the colony, but the
    # solve loop still does epoch × sample_count. Defaults are similar
    # enough that the standard pattern works.
    return _ref_mealpy(
        "mealpy.swarm_based.ACOR.OriginalACOR", func, n_trials, n_dim, seed
    )


def _ref_pybobyqa(func, n_trials, n_dim, seed):
    import numpy as np
    import pybobyqa

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    bounds = (np.zeros(n_dim), np.ones(n_dim))
    r = pybobyqa.solve(
        wrapped,
        np.asarray(_draw_x0(seed, n_dim)),
        bounds=bounds,
        maxfun=n_trials,
        seek_global_minimum=False,
        rhobeg=0.2,
        rhoend=1e-8,
        print_progress=False,
    )
    return {"best_value": float(r.f), "evals": counter["n"]}


def _ref_pdfo_newuoa(func, n_trials, n_dim, seed):
    import numpy as np
    from pdfo import newuoa

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    r = newuoa(
        wrapped,
        np.asarray(_draw_x0(seed, n_dim)),
        options={"maxfev": n_trials, "rhobeg": 0.2, "rhoend": 1e-8},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_pdfo_uobyqa(func, n_trials, n_dim, seed):
    import numpy as np
    from pdfo import uobyqa

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return func(list(x))

    r = uobyqa(
        wrapped,
        np.asarray(_draw_x0(seed, n_dim)),
        options={"maxfev": n_trials, "rhobeg": 0.2, "rhoend": 1e-8},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


# Algorithm -> (reference_label, reference_adapter, required_modules).
REFERENCES = {
    "NelderMead": ("scipy.optimize Nelder-Mead", _ref_scipy_neldermead, ["scipy"]),
    "Powell": ("scipy.optimize Powell", _ref_scipy_powell, ["scipy"]),
    "LBFGSB": ("scipy.optimize L-BFGS-B", _ref_scipy_lbfgsb, ["scipy"]),
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
    "ParticleSwarm": ("mealpy PSO", _ref_mealpy_pso, ["mealpy", "numpy"]),
    "GeneticAlgorithm": ("mealpy GA", _ref_mealpy_ga, ["mealpy", "numpy"]),
    "FireflyAlgorithm": ("mealpy FFA", _ref_mealpy_firefly, ["mealpy", "numpy"]),
    "HarmonySearch": ("mealpy HS", _ref_mealpy_harmony, ["mealpy", "numpy"]),
    "EvolutionStrategy": ("mealpy ES", _ref_mealpy_es, ["mealpy", "numpy"]),
    "AntColonyOpt": ("mealpy ACOR", _ref_mealpy_acor, ["mealpy", "numpy"]),
    "RandomSearch": ("uniform-sample baseline", _ref_random_search, []),
    "GridSearch": ("regular grid baseline", _ref_grid_search, []),
    "HillClimbing": (
        "(1+1)-ES sigma-decay schedule",
        _ref_oneplusone_es_decay,
        [],
    ),
    "Rechenberg": (
        "(1+1)-ES 1/5-success-rule (Rechenberg)",
        _ref_oneplusone_es_oneFifth,
        [],
    ),
    "CoordinateDescent": (
        "scipy Powell (direc=I)",
        _ref_scipy_powell_diagonal,
        ["scipy"],
    ),
    "PatternSearch": ("scipy.optimize.direct (DIRECT)", _ref_scipy_direct, ["scipy"]),
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
            for trial in range(N_RUNS):
                hd_vals.append(
                    _run_humpday(algorithm, func, n_trials, n_dim, seed=trial)[
                        "best_value"
                    ]
                )

            ref_vals = []
            for trial in range(N_RUNS):
                try:
                    ref_vals.append(
                        ref_fn(func, n_trials, n_dim, seed=trial)["best_value"]
                    )
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
