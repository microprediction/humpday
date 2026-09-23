"""
Reference-alignment tests for the HumpDay Python ports.

For each algorithm pair (HumpDay, trusted reference), this runs both
on the same set of test problems and records:
  - final objective values
  - number of function evaluations
  - distance to the known global optimum

The goal is **alignment**: HumpDay's port should behave indistinguishably
from the reference within floating-point reason. What this test makes
visible is where they do not.

Fairness of the comparison
--------------------------
Both sides get the same problem on the same feasible set from the same
distribution of starting points, and the harness checks rather than
asserts it:

* x0 is drawn from `U[0, 1]^n`, which is what `_A.random_uniform` gives
  and what twenty-seven of the thirty x0 sites in `humpday/optimizers/`
  use. It used to be `U[0.3, 0.7]^n` under a comment claiming the two
  were the same, so the reference started in the middle of the cube,
  near the optimum, and the port started anywhere in it.
* Every reference objective goes through `_in_cube`, which refuses a
  point outside `[0, 1]^n`. The scipy Nelder-Mead and Powell adapters
  passed no `bounds` and so were solving the unconstrained problem --
  sixty-three out-of-cube evaluations on Rosenbrock at seed 0, invisible
  because these objectives are defined out there too (#409).
* HumpDay gets `_array.seed(seed)` plus the stdlib `random.seed`, which
  the evolutionary algorithms use. Global-search references (DE, dual
  annealing, gp_minimize, cmaes) get the same integer seed via their own
  RNG parameter.

What the gate measures
----------------------
Two numbers per pair, and the second is the one that gates.

`hd/ref` is humpday's median gap over the reference's. It is the readable
one, and on a unimodal problem it is the right one. On a multimodal
problem it is not a statistic at all: the outcome there is bimodal --
a run either finds the global funnel or is trapped on the ring -- so a
median flips between the two modes, and the ratio of two such medians
flips by whatever separates them. At four runs, shifting the seed block
with the code untouched moved `CoordinateDescent/ackley` by a factor of
889 and `PatternSearch/ackley` by 460,000. `Rechenberg/ackley` read
519,288 and was carried in the ceiling table as the roster's worst
divergence; it loses about half its head-to-head pairings, which is to
say it matches its reference.

`lost` is the fraction of all (humpday run, reference run) pairings that
humpday loses, ties counted a half. It is bounded, it has no denominator
to blow up, two ports that both converge score 0.5, and across the roster
it moves by 0.05 to 0.32 under the same seed-block shift. A pair fails
when it loses more than its win ceiling; a large `hd/ref` fails only when
the pair is also behind head to head, since otherwise the ratio is the
artifact described above.

The printed table and the snapshot at `benchmarks/reference_alignment.json`
are still the point: the ceilings come from that snapshot rather than from
taste, so they say what the ports do today and stop it getting worse. Run
with stdout visible:

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
# Four runs was not enough to measure anything on a multimodal problem. Outcomes there are
# bimodal -- a run either finds the global funnel or is trapped on the ring -- so the median
# of four flips between the two modes on nothing but which seeds were drawn. Shifting the
# seed block with the code untouched moved `CoordinateDescent/ackley` by 889x and
# `PatternSearch/ackley` by 460,000x. At twenty-one runs those spreads are 1.8x and 3.3x.
N_RUNS = 21

# The distribution HumpDay's optimisers actually start from. This used to read
# `0.3 + 0.4 * U[0, 1]` with a comment claiming that was what they did; twenty-seven of the
# thirty x0 sites in `humpday/optimizers/` call `_A.random_uniform`, which is `U[0, 1]`. So
# every reference that takes an x0 was handed a start drawn from the middle 40% of each axis
# while humpday was dropped anywhere in the cube -- and with the optimum at (0.4127, 0.6831),
# the middle box is a much better place to begin. That is #387's defect one level up: an
# initial condition that encodes where the answer is. Six of the eleven pairs measured here
# moved when it was corrected, by up to a hundredfold, and the five whose references take no
# x0 did not move at all.
#
# Three sites (NelderMead's and Powell's seed points, CMA-ES's initial mean) do start from
# `0.3 + 0.4 * U`. That is the algorithm's own choice, made inside the algorithm, and it is
# not the harness's business to hand the same choice to the reference.
X0_LO, X0_HI = 0.0, 1.0


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


# Where the optima sit, and why not where they used to.
#
# These three had their minima at [0.5, 0.5] and [0.75, 0.75]. Both are places an optimizer can
# arrive at without searching: the PRIMA family starts at the centre of the cube, and 0.75 is a
# bin centre of any 14-bin grid, so a plain sweep of the old Rosenbrock returned exactly zero.
# The gate then read a refined multi-resolution GridSearch as 1e14 times worse than "a regular
# grid baseline", when what it had measured was that the baseline's spacing landed on the answer
# (#387, the same defect in the recorder's own suite).
#
# The offsets below are not round numbers in any base the algorithms use: not the centre, not a
# bin centre of a small grid, not a bisection point. An optimizer has to search for these.
_OPT = (0.4127, 0.6831)


def _sphere(x):
    """Convex quadratic, minimum 0 at _OPT."""
    return sum((v - _OPT[i % 2]) ** 2 for i, v in enumerate(x))


def _rosenbrock_unit(x):
    """2-D Rosenbrock on the cube, minimum 0 at _OPT.

    The map sends _OPT to Rosenbrock's own (1, 1) rather than sending the cube's centre there.
    """
    a = 4 * (x[0] - _OPT[0] + 0.5) - 2 + 1.0
    b = 4 * (x[1] - _OPT[1] + 0.5) - 2 + 1.0
    return (1 - a) ** 2 + 100 * (b - a * a) ** 2


def _ackley(x):
    """Ackley with its minimum 0 at _OPT."""
    n = len(x)
    s = [10 * (v - _OPT[i % 2]) for i, v in enumerate(x)]
    return (
        -20 * math.exp(-0.2 * math.sqrt(sum(v * v for v in s) / n))
        - math.exp(sum(math.cos(2 * math.pi * v) for v in s) / n)
        + 20
        + math.e
    )


PROBLEMS = {
    "sphere": {"func": _sphere, "opt": 0.0, "x_opt": list(_OPT)},
    "rosenbrock": {"func": _rosenbrock_unit, "opt": 0.0, "x_opt": list(_OPT)},
    "ackley": {"func": _ackley, "opt": 0.0, "x_opt": list(_OPT)},
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


class OutsideTheCube(AssertionError):
    """A reference asked for a point HumpDay's port is not allowed to visit."""


def _in_cube(func, n_dim, counter):
    """Wrap an objective so it counts calls and refuses points outside [0, 1]^n.

    Every HumpDay port clips to the unit cube. Some references were not told about it:
    scipy's Nelder-Mead and Powell adapters passed no `bounds`, so the reference was solving
    the unconstrained problem -- an easier one -- while the port solved the constrained one.
    It went unnoticed because these objectives are defined outside the cube too and simply
    return a value, so nothing complained; #409 counted 63 out-of-cube evaluations on
    Rosenbrock at seed 0, and points as far out as (-0.98, 0.60).

    A comparison across two different feasible sets is not a comparison, so this raises rather
    than clipping. Clipping would hide the next one.
    """

    def wrapped(x):
        xs = [float(v) for v in x]
        counter["n"] += 1
        for v in xs:
            if not (0.0 <= v <= 1.0):
                raise OutsideTheCube(
                    f"reference evaluated {xs}, outside [0, 1]^{n_dim}"
                )
        return func(xs)

    return wrapped


def _ref_scipy_neldermead(func, n_trials, n_dim, seed):
    from scipy.optimize import minimize

    counter = {"n": 0}
    wrapped = _in_cube(func, n_dim, counter)

    # Tolerances match HumpDay's NelderMead (xatol=fatol=1e-12). With
    # scipy's default 1e-4 the reference stops far below its potential;
    # at 1e-12 both implementations run until budget exhaustion or
    # genuine numerical convergence.
    r = minimize(
        wrapped,
        _draw_x0(seed, n_dim),
        method="Nelder-Mead",
        bounds=[(0.0, 1.0)] * n_dim,
        options={"maxfev": n_trials, "xatol": 1e-12, "fatol": 1e-12},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_powell(func, n_trials, n_dim, seed):
    from scipy.optimize import minimize

    counter = {"n": 0}
    wrapped = _in_cube(func, n_dim, counter)

    # Tolerances match HumpDay's Powell (ftol=1e-12) — see the
    # NelderMead adapter above for rationale.
    r = minimize(
        wrapped,
        _draw_x0(seed, n_dim),
        method="Powell",
        bounds=[(0.0, 1.0)] * n_dim,
        options={"maxfev": n_trials, "xtol": 1e-12, "ftol": 1e-12},
    )
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_scipy_lbfgsb(func, n_trials, n_dim, seed):
    from scipy.optimize import minimize

    counter = {"n": 0}
    wrapped = _in_cube(func, n_dim, counter)

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
    wrapped = _in_cube(func, n_dim, counter)

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
    wrapped = _in_cube(func, n_dim, counter)

    bounds = [(0, 1)] * n_dim
    r = dual_annealing(wrapped, bounds, maxiter=max(1, n_trials // 10), seed=seed)
    return {"best_value": float(r.fun), "evals": counter["n"]}


def _ref_skopt_gp(func, n_trials, n_dim, seed):
    from skopt import gp_minimize

    counter = {"n": 0}
    wrapped = _in_cube(func, n_dim, counter)

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
    wrapped = _in_cube(func, n_dim, counter)

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


def _ref_coord_descent_greedy(func, n_trials, n_dim, seed):
    """Textbook coordinate descent with greedy expansion per axis —
    fair reference for HumpDay's CoordinateDescent.

    The previous reference here was scipy.optimize.minimize with
    method="Powell" and direc=I — an algorithm-class mismatch because
    scipy's Powell uses golden-section line search per axis (much
    tighter convergence) while HumpDay implements greedy expansion
    (the classical Brent/textbook approach). Same algorithm family,
    different line-search policy — humpday looked artificially weak.

    This inline implementation matches HumpDay's algorithm class:
    greedy expansion per axis, halve the step on failed full sweeps,
    floor at 1e-12. No bells, no Powell direction update, no
    line-search subroutine — the most direct textbook reference.
    """
    rng = random.Random(seed)
    x = [_draw_x0(seed, n_dim)[i] for i in range(n_dim)]
    f = func(x)
    n_evals = 1
    step = 0.1
    step_min = 1e-12

    while n_evals < n_trials and step > step_min:
        improved = False
        for i in range(n_dim):
            for sign in (1, -1):
                if n_evals >= n_trials:
                    break
                xi_new = max(0.0, min(1.0, x[i] + sign * step))
                if abs(xi_new - x[i]) < 1e-15:
                    continue
                x_trial = x[:]
                x_trial[i] = xi_new
                f_trial = func(x_trial)
                n_evals += 1
                if f_trial >= f:
                    continue
                x, f = x_trial, f_trial
                improved = True
                # Greedy expansion in the same direction.
                while n_evals < n_trials:
                    xi_next = max(0.0, min(1.0, x[i] + sign * step))
                    if abs(xi_next - x[i]) < 1e-15:
                        break
                    x_trial2 = x[:]
                    x_trial2[i] = xi_next
                    f_trial2 = func(x_trial2)
                    n_evals += 1
                    if f_trial2 >= f:
                        break
                    x, f = x_trial2, f_trial2
                break  # don't try the other sign
        if not improved:
            step *= 0.5
    # rng is unused but kept for signature uniformity.
    _ = rng
    return {"best_value": float(f), "evals": n_evals}


def _ref_hooke_jeeves(func, n_trials, n_dim, seed):
    """Textbook Hooke-Jeeves pattern search (1961) — fair reference for
    HumpDay's PatternSearch.

    The previous reference here was scipy.optimize.direct (DIRECT), a
    deterministic *global* optimizer with provable convergence on
    multimodal landscapes. Hooke-Jeeves is a *local* pattern search —
    different algorithm class, different problem. The comparison was
    fundamentally unfair: HumpDay's PatternSearch couldn't approach
    DIRECT on Ackley regardless of tuning.

    This inline reference is canonical Hooke-Jeeves (1961):
      1. Exploratory move from a base — try ±step on each axis, keep
         improvements (first-improvement per coord).
      2. If exploratory improved, pattern move: extrapolate from base
         through the new point.
      3. Exploratory from the pattern point; accept if better.
      4. Halve step on failed sweep; floor at 1e-12.
    """
    rng = random.Random(seed)

    def explore(b, fb, step):
        for i in range(n_dim):
            for sign in (1, -1):
                xi_new = max(0.0, min(1.0, b[i] + sign * step))
                if abs(xi_new - b[i]) < 1e-15:
                    continue
                t = b[:]
                t[i] = xi_new
                ft = func(t)
                explore.n += 1
                if ft < fb:
                    b, fb = t, ft
                    break
        return b, fb

    explore.n = 0

    base = [_draw_x0(seed, n_dim)[i] for i in range(n_dim)]
    f_base = func(base)
    explore.n = 1
    step = 0.1
    step_min = 1e-12

    while explore.n < n_trials and step > step_min:
        x, f = explore(base[:], f_base, step)
        if explore.n >= n_trials:
            break
        if f < f_base:
            new_base = [
                max(0.0, min(1.0, x[i] + (x[i] - base[i]))) for i in range(n_dim)
            ]
            f_new_base = func(new_base)
            explore.n += 1
            x2, f2 = explore(new_base[:], f_new_base, step)
            if f2 < f:
                base, f_base = x2, f2
            else:
                base, f_base = x, f
        else:
            step *= 0.5
    _ = rng
    return {"best_value": float(f_base), "evals": explore.n}


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
    wrapped = _in_cube(func, n_dim, counter)

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
    wrapped = _in_cube(func, n_dim, counter)

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
    wrapped = _in_cube(func, n_dim, counter)

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
    wrapped = _in_cube(func, n_dim, counter)

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
        "coord descent + greedy expansion (textbook)",
        _ref_coord_descent_greedy,
        [],
    ),
    "PatternSearch": (
        "Hooke-Jeeves (1961)",
        _ref_hooke_jeeves,
        [],
    ),
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


# ---------- how far a port may lag its reference ----------

# The ratio printed as `hd/ref`: humpday's median gap to the optimum over the reference's, so
# 1.0 is parity and 2.0 is twice the remaining error. The default is what a port is expected to
# stay inside, and the exceptions below record the pairs that do not, so that a run tells the
# difference between a known deficiency and a new one.
#
# Ceilings are measured, not chosen: each is about twice the value recorded in
# benchmarks/reference_alignment.json, which gives a row room to move with a library version or
# a seed without letting it double. They are not targets. Five of sixty-three pairs need one,
# and each is a port that has genuinely not solved its problem, not a converged run a few ulps
# behind another -- the floor below takes care of those. #78 tracks the divergences.
DEFAULT_RATIO_CEILING = 3.0

# Below this, a port has solved the problem and the ratio stops meaning anything. The gaps
# being divided are then a few ulps apart and the +1e-15 guard in the denominator dominates:
# PRIMA_BOBYQA on Rosenbrock measures 1.26 on macOS and 27.10 on Linux, from gaps of 2.7e-16
# and 2.7e-14 against a reference that reached zero. Neither number describes a deficiency in
# the port -- both runs converged -- and a gate that fails on the difference between two
# converged runs reports the platform, not the code.
CONVERGED_GAP = 1e-10

# The ratio of medians is a badly conditioned statistic whenever both sides straddle the two
# modes, and no number of runs fixes that: at twenty-one runs `Rechenberg/ackley` still reads
# anywhere from 1.6e-05 to 105 depending only on which seeds were drawn, because it is dividing
# one nearly-converged median by another. The win rate is not. It is the fraction of all
# head-to-head pairings (every humpday run against every reference run) that humpday loses,
# ties counted a half, so it is bounded in [0, 1], has no denominator to blow up, and answers
# the question the gate is actually asking: does this port do what the reference does.
#
# 0.5 is indistinguishable. 1.0 is a port that loses every single pairing.
#
# Where the default sits is set by two measurements. A port that matches its reference reads up
# to 0.65 across the roster. The block-to-block spread of the statistic is about 0.3 and barely
# improves with more runs -- 0.37 at seven, 0.38 at twenty-one, 0.30 at thirty-one -- because
# it is the sampling variance of a rate, which falls as 1/sqrt(n). So 0.8 is above the noise a
# matching port makes, and well below the two real deficiencies this found the first time it
# ran: CoordinateDescent at 1.00 on the sphere and PatternSearch at 0.98.
#
# N_RUNS is not set for this statistic, which would be content with seven. It is set for the
# ratio printed beside it, which needs twenty-one.
DEFAULT_WIN_CEILING = 0.80

RATIO_CEILING = {
    # Re-measured after the gate stopped measuring itself: the reference's head start removed,
    # its feasible set matched to the port's, and twenty-one runs instead of four (#409). The
    # table went from thirteen entries topping out at 1.1e6 to these five, because most of what
    # it had been recording was the instrument.
    #
    # `Rechenberg/ackley` is the clearest case. It was carried at 1.1e6 against a measurement
    # of 519,288 and stood as the roster's worst divergence; the two implementations are the
    # same algorithm and it loses 0.63 of its head-to-head pairings, which is to say it is a
    # little behind and nothing like six orders behind. `DifferentialEvolution/rosenbrock` was
    # carried at 25,000 and measures 3.80. `CoordinateDescent/ackley` was carried at 1,100 and
    # needs no entry: the restart threshold that produced it is fixed, and it now loses none of
    # its pairings there.
    #
    # `Powell/rosenbrock` is new, and is the honest number rather than a regression. Its
    # reference used to run unconstrained -- sixty-three evaluations outside the cube on this
    # problem alone -- which made scipy's Powell look worse than it is. Bounded, it reaches
    # 0.034 where it used to reach 0.56, and humpday is 4.11 behind it.
    #
    # BayesianOpt was here at 162 on Ackley and 9.9 on Rosenbrock and is here no longer: its
    # acquisition function is now optimised rather than sampled ten times, and it matches or
    # beats gp_minimize on all three (0.68 / 1.46 / 0.00).
    (
        "BayesianOpt",
        "ackley",
    ): 162.0,  # measured 80.89, loses 0.90 -- a real one, see below
    ("Rechenberg", "ackley"): 53.0,  # measured 26.59, loses 0.63
    ("BayesianOpt", "rosenbrock"): 9.9,  # measured 4.94, loses 0.75
    ("Powell", "rosenbrock"): 8.2,  # measured 4.11, loses 0.59
    ("DifferentialEvolution", "rosenbrock"): 7.6,  # measured 3.80, loses 0.60
}


# Pairs that lose more head to head than the default allows. Measured from
# benchmarks/reference_alignment.json and rounded up a little, same as the ratio ceilings: a
# record of what the ports do, not a target.
#
# Empty, and it was not. The first run of this statistic put BayesianOpt at 0.90 on Ackley --
# it sat at 2.58, which is where a run trapped on the ring sits, against gp_minimize's 0.032.
# A rate that lopsided is not the sampling noise this statistic makes, and it was not: the
# acquisition function was being sampled ten times rather than optimised. Fixed, it loses 0.46
# there and no pair on the roster needs an entry.
#
# #408 still has the JavaScript twin replacing the GP with a nearest-neighbour heuristic
# outright, and #81 tracks the hyperparameters, which are still fixed where scikit-optimize
# fits them.
WIN_CEILING: dict[tuple[str, str], float] = {}


def ratio_ceiling(algorithm: str, problem: str) -> float:
    return RATIO_CEILING.get((algorithm, problem), DEFAULT_RATIO_CEILING)


def win_ceiling(algorithm: str, problem: str) -> float:
    return WIN_CEILING.get((algorithm, problem), DEFAULT_WIN_CEILING)


def head_to_head(hd_vals, ref_vals) -> float:
    """Fraction of all (humpday run, reference run) pairings humpday loses; ties count a half.

    Every run against every run rather than pairing by seed index, because the two sides draw
    from different RNG streams and seed 3 is not the same problem for both of them. Pairing
    arbitrary runs would manufacture differences; comparing the samples does not.

    Two ports that both converge to exactly the optimum score 0.5 here, which is what the
    `CONVERGED_GAP` floor exists to say about the ratio. This statistic needs no such floor.
    """
    worse = 0.0
    for a in hd_vals:
        for b in ref_vals:
            worse += 1.0 if a > b else (0.5 if a == b else 0.0)
    return worse / (len(hd_vals) * len(ref_vals))


# ---------- the characterisation test ----------


@pytest.mark.reference
def test_reference_alignment():
    """Print + persist a table of HumpDay-vs-reference final values for every
    algorithm where we have a reference adapter."""
    n_trials_default = 200
    n_dim = 2
    rows = []
    failures: dict = {}

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
            ref_errors = []
            for trial in range(N_RUNS):
                try:
                    ref_vals.append(
                        ref_fn(func, n_trials, n_dim, seed=trial)["best_value"]
                    )
                except Exception as e:
                    print(f"    reference error on {problem_id}: {e}")
                    ref_errors.append(f"{type(e).__name__}: {e}")
                    ref_vals.append(float("inf"))
            # A reference that raised used to be recorded as inf, which made its gap infinite and
            # the ratio zero: the comparison humpday most conclusively "won" was the one where
            # the thing it is measured against never ran. There is nothing to compare, so say so.
            failures.setdefault("reference did not run", [])
            if ref_errors:
                failures["reference did not run"].append(
                    f"{algorithm}/{problem_id}: {ref_errors[0]}"
                )

            hd_med = sorted(hd_vals)[N_RUNS // 2]
            ref_med = sorted(ref_vals)[N_RUNS // 2]
            hd_gap = hd_med - opt_value
            ref_gap = ref_med - opt_value
            relative = (hd_gap + 1e-15) / (ref_gap + 1e-15)
            lost = head_to_head(hd_vals, ref_vals)

            print(
                f"  {problem_id:<12}  hd={hd_med:>10.4g}  ref={ref_med:>10.4g}"
                f"  hd-to-opt={hd_gap:>10.4g}  ref-to-opt={ref_gap:>10.4g}"
                f"  hd/ref={relative:>9.2f}  lost={lost:>5.2f}"
            )

            # A pair fails when the port has not solved the problem and either loses more
            # than its win ceiling head to head, or sits more than its ratio ceiling behind
            # while also losing a majority.
            #
            # The `CONVERGED_GAP` floor governs both tests, because the win rate has no scale:
            # it flags a port that is consistently a hair behind exactly as hard as one that is
            # consistently a thousandfold behind. Powell on Ackley reaches 7.5e-11 against
            # scipy's 4.8e-12 and loses 0.82 of pairings, which is a real ordering and not a
            # deficiency -- both have solved it to ten decimal places.
            #
            # The ratio is subordinate to the win rate because a large ratio with the port
            # ahead head to head is the artifact this gate spent a while chasing: two medians
            # that landed in different modes. `Rechenberg/ackley` read 519,288 that way and
            # loses about half its pairings, which is to say it matches its reference.
            win_cap = win_ceiling(algorithm, problem_id)
            if lost > win_cap and hd_gap > CONVERGED_GAP:
                failures.setdefault("lagging the reference", []).append(
                    f"{algorithm}/{problem_id}: loses {lost:.2f} of head-to-head pairings, "
                    f"over its ceiling {win_cap:g}"
                )
            ceiling = ratio_ceiling(algorithm, problem_id)
            if lost > 0.5 and relative > ceiling and hd_gap > CONVERGED_GAP:
                failures.setdefault("lagging the reference", []).append(
                    f"{algorithm}/{problem_id}: hd/ref {relative:.2f} over its ceiling {ceiling:g}"
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
                    "ratio_ceiling": ceiling,
                    "head_to_head_lost": lost,
                    "win_ceiling": win_cap,
                    "converged": hd_gap <= CONVERGED_GAP,
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

    # The snapshot is written first, so a failing run still leaves the table it was judged on.
    assert rows, (
        "no reference adapter could run: install the comparisons with "
        "`pip install humpday[reference]`, or this test is watching nothing"
    )
    reported = {k: v for k, v in failures.items() if v}
    assert not reported, "\n".join(
        f"{heading}:\n  " + "\n  ".join(items) for heading, items in reported.items()
    )
