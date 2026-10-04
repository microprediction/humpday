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
* Every reference objective goes through `Metered`, which refuses a
  point outside `[0, 1]^n`. The scipy Nelder-Mead and Powell adapters
  passed no `bounds` and so were solving the unconstrained problem --
  sixty-three out-of-cube evaluations on Rosenbrock at seed 0, invisible
  because these objectives are defined out there too (#409). PDFO's
  NEWUOA and UOBYQA, unconstrained by design, see `objective(clip(x))`,
  which is what the port computes for every point it proposes.
* Both sides get the same budget in objective calls, enforced rather than
  requested. `Metered` stops a reference at call N+1 and its result is the
  best value it actually observed. Translating the budget into each
  library's units -- generations, iterations, epochs -- let differential
  evolution spend up to 433 calls where the port had 200 and mealpy's
  Firefly 4,020 (#404). Every run's call count is kept in the snapshot.
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

import functools
import importlib
import json
import math
import os
import random
import time
import warnings
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


class _Observed:
    """The objective as an optimizer sees it, with a record of what it actually returned.

    A run's claimed best value is only evidence if the objective returned it. This keeps the
    call count and the lowest value handed back, so `validate_run` can hold the claim against
    the observation rather than taking it on trust (#405).
    """

    def __init__(self, func):
        self.func = func
        self.n = 0
        self.best = math.inf

    def __call__(self, x):
        self.n += 1
        value = self.func(x)
        v = float(value)
        if v < self.best:
            self.best = v
        return value


def _run_humpday(algorithm: str, func, n_trials: int, n_dim: int, seed: int):
    _seed_humpday(seed)
    seen = _Observed(func)
    cls = PURE_OPTIMIZERS[algorithm]
    opt = cls(seen, n_trials=n_trials, n_dim=n_dim)
    res = opt.optimize()
    if isinstance(res, tuple) and len(res) == 2:
        best_value = float(res[0])
    else:
        best_value = float(opt.best_value)
    return {
        "best_value": best_value,
        "evals": seen.n,
        "reported_evals": int(opt.evaluations),
        "observed_best": seen.best,
    }


# ---------- what counts as a result ----------

# How far below the known minimum a value may sit and still be a rounding of it. Every objective
# here is exactly zero at its optimum; Ackley evaluates there to a few ulps either side of zero.
BELOW_OPTIMUM_TOLERANCE = 1e-12


def _same_value(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def validate_run(run: dict, problem: dict, budget: int) -> list[str]:
    """Everything wrong with one run's result, or an empty list if it is a result at all.

    The gate used to sort and take medians of whatever came back. A reference that returned
    `inf` made its gap infinite and the ratio zero; `NaN` made every comparison false, and false
    is a pass; a HumpDay value of -1 on a problem whose minimum is 0 was simply accepted. None
    of those is a measurement, so none of them may take part in one (#405).

    A run is a result when its value is finite, at least one objective call stands behind it,
    it spent no more than its budget, it is not below the known minimum, and -- where the run
    says what the objective returned -- it is a value the objective actually returned.
    """
    issues = []
    value = run.get("best_value")
    finite = isinstance(value, (int, float)) and math.isfinite(value)
    if not finite:
        issues.append(f"best value {value!r} is not a finite number")

    evals = run.get("evals")
    if not isinstance(evals, int) or evals < 1:
        issues.append(
            f"made {evals!r} objective calls, so no observation stands behind its value"
        )
    elif evals > budget:
        issues.append(f"made {evals} objective calls on a budget of {budget}")
    reported_evals = run.get("reported_evals")
    if reported_evals is not None and reported_evals != evals:
        issues.append(
            f"reports {reported_evals} evaluations but made {evals} objective calls"
        )

    if finite and value < problem["opt"] - BELOW_OPTIMUM_TOLERANCE:
        issues.append(
            f"best value {value!r} is below the known minimum {problem['opt']!r}"
        )

    observed = run.get("observed_best")
    if finite and observed is not None and not _same_value(value, observed):
        issues.append(
            f"claims {value!r} but the lowest value the objective returned was {observed!r}"
        )

    # A library's own report of its minimum, kept beside the value the harness observed. It may
    # be higher (some return their last point rather than their best); it may not be lower than
    # anything it was ever given.
    reported = run.get("reported_value")
    if reported is not None and finite:
        if not math.isfinite(reported):
            issues.append(f"the library reports {reported!r} as its minimum")
        elif reported < value - 1e-12 * max(1.0, abs(value)):
            issues.append(
                f"the library reports {reported!r}, below anything it evaluated ({value!r})"
            )

    if run.get("outside"):
        issues.append(f"evaluated {run['outside']} points outside the unit cube")
    return issues


def _json_number(v):
    """A float the snapshot can hold. Strict JSON has no NaN or Infinity, and the snapshot is
    written with `allow_nan=False` so that one cannot slip in as a bare token again."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else repr(v)


def _evidence(seed, run, issues):
    """One run as the snapshot keeps it: what it found, what it spent, and what was wrong."""
    return {
        "seed": seed,
        "value": run.get("best_value"),
        "evals": run.get("evals"),
        "stopped_at_budget": bool(run.get("stopped_at_budget")),
        "projected": run.get("projected", 0),
        "issues": issues,
    }


def _columns(side, runs):
    """A side's runs as seed-ordered columns, so the snapshot keeps every run without a line
    per field per run. The counts used to be thrown away (#404); a reader can now see what each
    side actually spent, and where the meter stopped a reference."""
    ran = [r for r in runs if "error" not in r]
    evals = sorted(r["evals"] for r in ran if isinstance(r["evals"], int))
    out = {
        f"{side}_values": [_json_number(r["value"]) for r in ran],
        f"{side}_evals": [r["evals"] for r in ran],
        f"{side}_evals_median": evals[len(evals) // 2] if evals else None,
    }
    if side == "reference":
        out["reference_stopped_at_budget"] = sum(r["stopped_at_budget"] for r in ran)
        if any(r["projected"] for r in ran):
            out["reference_projected"] = [r["projected"] for r in ran]
    errors = [
        {"seed": r["seed"], "error": r.get("error") or "; ".join(r["issues"])}
        for r in runs
        if "error" in r or r["issues"]
    ]
    if errors:
        out[f"{side}_invalid"] = errors
    return out


def _dump(obj, indent=0):
    """JSON with objects indented and lists of numbers kept on one line: the per-run columns
    would otherwise take a line per number. Strict: no NaN or Infinity (`allow_nan=False`)."""
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f"{pad}  {json.dumps(str(k))}: {_dump(v, indent + 1)}"
            for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(obj, list) and any(isinstance(v, (dict, list)) for v in obj):
        items = [f"{pad}  {_dump(v, indent + 1)}" for v in obj]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(obj, allow_nan=False)


# ---------- reference adapters ----------
# Each adapter runs its library on the same problem the HumpDay port saw, started from a
# seed-determined x0 drawn from the same distribution HumpDay uses, through a `Metered`
# objective that holds it to the same budget. It returns what the harness observed -- the
# lowest value the objective actually returned and the number of calls actually made -- with
# the library's own report of its minimum kept beside it for `validate_run` to check.


class OutsideTheCube(AssertionError):
    """A reference asked for a point HumpDay's port is not allowed to visit."""


class BudgetExhausted(Exception):
    """A reference asked for call N+1 of an N-call budget."""


class Metered:
    """The objective as a reference sees it: counted, capped at the budget, and recorded.

    The budget is the comparison. HumpDay's ports are hard-capped at `n_trials` objective calls
    by `Optimizer.optimize`; the references were told a budget in each library's own units and
    nothing held them to it, so at a budget of 200 scipy's differential_evolution spent up to 433
    calls on Ackley, dual_annealing up to 318 and mealpy's Firefly 4,020 -- twenty times the
    port's allowance -- and the harness, which counted every one of them, threw the counts away
    and compared final values (#404). A ratio across unequal budgets measures the budget.

    So the cap is enforced here, in the one place every call passes through, rather than in each
    library's vocabulary: call N+1 raises `BudgetExhausted` before the objective is evaluated,
    the adapter stops, and the result is the best value the reference actually observed in its
    N calls, including a local search interrupted part way. No library is trusted to stop
    itself. A reference that stops early of its own accord -- converged, or out of iterations --
    is recorded as having spent what it spent.

    `outside` says what to do with a point outside the unit cube:

    * "raise" (the default) refuses it. Every HumpDay port clips to the cube; scipy's Nelder-Mead
      and Powell adapters once passed no `bounds` and solved the unconstrained problem, which
      went unnoticed because these objectives are defined outside the cube too (#409). Clipping
      would hide the next one.
    * "project" evaluates the objective at the nearest point of the cube, which is what the port
      does to every point it proposes: `Optimizer.evaluate` computes `objective(clip(x))`. It is
      for the references that are unconstrained by design and so have no bounds to be given --
      PDFO's NEWUOA and UOBYQA -- and the count of projected proposals is kept with the result.

    `hard_cap=False` is for references whose objective is called from compiled code through
    f2py. An exception raised inside an f2py callback does not propagate: it kills the
    interpreter ("Fatal Python error: F2PySwapThreadLocalCallbackPtr"), which took the whole
    gate down the first time PDFO was actually installed alongside it. Those references are
    given their budget in their own units, and an overspend is then recorded and failed by
    `validate_run` instead of interrupted.
    """

    def __init__(self, func, n_dim, budget, *, outside="raise", hard_cap=True):
        self.func = func
        self.n_dim = n_dim
        self.budget = budget
        self.outside = outside
        self.hard_cap = hard_cap
        self.n = 0
        self.best_value = math.inf
        self.best_x = None
        self.projected = 0
        self.stopped_at_budget = False

    def __call__(self, x):
        xs = [float(v) for v in x]
        if self.hard_cap and self.n >= self.budget:
            self.stopped_at_budget = True
            raise BudgetExhausted(f"call {self.n + 1} on a budget of {self.budget}")
        self.n += 1
        if any(not (0.0 <= v <= 1.0) for v in xs):
            if self.outside == "project":
                self.projected += 1
                xs = [min(1.0, max(0.0, v)) for v in xs]
            else:
                raise OutsideTheCube(
                    f"reference evaluated {xs}, outside [0, 1]^{self.n_dim}"
                )
        value = float(self.func(xs))
        if value < self.best_value:
            self.best_value, self.best_x = value, xs
        return value

    def result(self, reported=None):
        return {
            "best_value": self.best_value,
            "evals": self.n,
            "budget": self.budget,
            "stopped_at_budget": self.stopped_at_budget,
            "reported_value": None if reported is None else float(reported),
            "projected": self.projected,
            "best_x": self.best_x,
        }


def _reference(outside="raise", hard_cap=True):
    """Turn `body(f, n_trials, n_dim, seed) -> reported minimum` into an adapter.

    The adapter keeps the signature every caller uses, `(func, n_trials, n_dim, seed)`, and
    returns `Metered.result`: what was observed, not what the library says it found.
    """

    def wrap(body):
        @functools.wraps(body)
        def adapter(func, n_trials, n_dim, seed):
            f = Metered(func, n_dim, n_trials, outside=outside, hard_cap=hard_cap)
            reported = None
            try:
                reported = body(f, n_trials, n_dim, seed)
            except BudgetExhausted:
                pass
            return f.result(reported)

        return adapter

    return wrap


@_reference()
def _ref_scipy_neldermead(f, n_trials, n_dim, seed):
    from scipy.optimize import minimize

    # Tolerances match HumpDay's NelderMead (xatol=fatol=1e-12). With
    # scipy's default 1e-4 the reference stops far below its potential;
    # at 1e-12 both implementations run until budget exhaustion or
    # genuine numerical convergence.
    r = minimize(
        f,
        _draw_x0(seed, n_dim),
        method="Nelder-Mead",
        bounds=[(0.0, 1.0)] * n_dim,
        options={"maxfev": n_trials, "xatol": 1e-12, "fatol": 1e-12},
    )
    return r.fun


@_reference()
def _ref_scipy_powell(f, n_trials, n_dim, seed):
    from scipy.optimize import minimize

    # Tolerances match HumpDay's Powell (ftol=1e-12) — see the
    # NelderMead adapter above for rationale.
    r = minimize(
        f,
        _draw_x0(seed, n_dim),
        method="Powell",
        bounds=[(0.0, 1.0)] * n_dim,
        options={"maxfev": n_trials, "xtol": 1e-12, "ftol": 1e-12},
    )
    return r.fun


@_reference()
def _ref_scipy_lbfgsb(f, n_trials, n_dim, seed):
    from scipy.optimize import minimize

    # `maxfun` is scipy's own cap, but not a hard one: with finite-difference gradients a
    # line search can run past it (#404 measured 225 calls on a budget of 200). The meter
    # stops it at the budget. ftol=1e-12, gtol=1e-12 push the solver to the same precision
    # floor we use for NelderMead / Powell (scipy's defaults stop ~1e-7 below the optimum).
    r = minimize(
        f,
        _draw_x0(seed, n_dim),
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * n_dim,
        options={"maxfun": n_trials, "ftol": 1e-12, "gtol": 1e-12},
    )
    return r.fun


@_reference()
def _ref_scipy_de(f, n_trials, n_dim, seed):
    from scipy.optimize import differential_evolution

    # The generations used to be `n_trials // (10 * n_dim)`, which forgot the initial
    # population and left `polish=True` to spend whatever it liked afterwards: 229 to 433
    # calls on a budget of 200 (#404). Now the generations, initial population included, take
    # half the budget and the L-BFGS-B polish has the rest, which is the split HumpDay's DE
    # makes between the same two stages; the meter stops the polish where the budget ends.
    popsize = 10
    population = popsize * n_dim
    generations = max(1, (n_trials // 2) // population - 1)
    r = differential_evolution(
        f,
        [(0, 1)] * n_dim,
        maxiter=generations,
        popsize=popsize,
        tol=1e-8,
        seed=seed,
        polish=True,
    )
    return r.fun


@_reference()
def _ref_scipy_dual_annealing(f, n_trials, n_dim, seed):
    from scipy.optimize import dual_annealing

    # `maxfun` is dual_annealing's own budget in calls; it used to get none, and `maxiter` set
    # from `n_trials // 10`, so its local searches spent from 108 to 318 calls on a budget of
    # 200 (#404). scipy checks `maxfun` between local searches, not inside them, so the meter
    # is what stops it.
    r = dual_annealing(f, [(0, 1)] * n_dim, maxfun=n_trials, seed=seed)
    return r.fun


@_reference()
def _ref_skopt_gp(f, n_trials, n_dim, seed):
    from skopt import gp_minimize

    r = gp_minimize(
        f,
        [(0.0, 1.0)] * n_dim,
        n_calls=n_trials,
        random_state=seed,
        n_initial_points=min(10, n_trials // 2),
    )
    return r.fun


@_reference()
def _ref_cmaes(f, n_trials, n_dim, seed):
    """CyberAgent `cmaes` reference.

    The library's `tell` wants exactly `population_size` solutions, so the last generation
    used to be dropped whenever it did not fit, leaving the reference 198 calls where the port
    had 200. Now it asks for the whole generation and evaluates as much of it as the budget
    allows; the meter stops it, and the best point observed counts whether or not that last
    generation was ever told.
    """
    import numpy as np
    from cmaes import CMA

    es = CMA(
        mean=np.asarray(_draw_x0(seed, n_dim)),
        sigma=0.2,
        bounds=np.array([[0, 1]] * n_dim),
        seed=seed,
    )
    while True:
        sols = []
        for _ in range(es.population_size):
            x = es.ask()
            sols.append((x, f(x.tolist())))
        es.tell(sols)


@_reference()
def _ref_random_search(func, n_trials, n_dim, seed):
    """Uniform-sample baseline — draw n_trials i.i.d. samples from
    `U[0, 1]^n_dim` and return the best. Cheapest possible
    lower-bound: any algorithm worth using should beat this median.
    """
    rng = random.Random(seed)
    best = float("inf")
    for _ in range(n_trials):
        x = [rng.random() for _ in range(n_dim)]
        v = func(x)
        if v < best:
            best = v
    return best


@_reference()
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
    return best


@_reference()
def _ref_oneplusone_es_decay(func, n_trials, n_dim, seed):
    """(1+1)-ES with a geometric sigma decay schedule — the natural
    reference for HillClimbing. Starts at sigma=0.1 and decays so the
    final sigma is ~1e-3, matching a "hill-climbing with shrinking
    perturbations" intuition."""
    rng = random.Random(seed)
    x = [_draw_x0(seed, n_dim)[i] for i in range(n_dim)]
    fx = func(x)
    sigma_init = 0.1
    sigma_final = 1e-3
    decay = (sigma_final / sigma_init) ** (1.0 / max(1, n_trials - 1))
    sigma = sigma_init

    for _ in range(n_trials - 1):
        z = [rng.gauss(0, 1) for _ in range(n_dim)]
        x_new = [min(1.0, max(0.0, x[i] + sigma * z[i])) for i in range(n_dim)]
        fx_new = func(x_new)
        if fx_new < fx:
            x, fx = x_new, fx_new
        sigma *= decay
    return fx


@_reference()
def _ref_oneplusone_es_oneFifth(func, n_trials, n_dim, seed):
    """(1+1)-ES with Rechenberg's 1/5-success-rule — the natural
    reference for Rechenberg. Sigma grows by 1.5× when the
    success rate over the last 10 trials exceeds 1/5, shrinks by 1/1.5
    otherwise."""
    rng = random.Random(seed)
    x = [_draw_x0(seed, n_dim)[i] for i in range(n_dim)]
    fx = func(x)
    sigma = 0.1
    window = []
    window_size = 10

    for _ in range(n_trials - 1):
        z = [rng.gauss(0, 1) for _ in range(n_dim)]
        x_new = [min(1.0, max(0.0, x[i] + sigma * z[i])) for i in range(n_dim)]
        fx_new = func(x_new)
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
    return fx


@_reference()
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
    return f


@_reference()
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
    return f_base


def _ref_mealpy(cls_path, f, n_trials, n_dim, seed, pop_size=20, kwargs=None):
    """Run a mealpy algorithm on a metered objective and return its reported minimum.

    `cls_path` is a string like "mealpy.swarm_based.PSO.OriginalPSO";
    we import lazily so a missing mealpy install just makes the
    corresponding `REFERENCES` entry skip cleanly.

    This used to set `epoch = n_trials // pop_size` on the theory that mealpy spends
    `epoch * pop_size` calls. It does not: at a budget of 200 the six adapters spent 170 to
    270, and Firefly, whose every epoch compares the swarm pairwise, spent 4,020 (#404). The
    epoch count is now only an upper bound -- one call per epoch at the least, so `n_trials`
    epochs cannot run out first -- and the meter stops the run at the budget. None of the six
    classes used here reads `self.epoch` inside `evolve`, so the bound does not change how they
    search, only when they are stopped.

    We silence mealpy's INFO-level logging (~one line per epoch, which
    drowns out the pytest -s view).
    """
    import importlib
    import logging

    from mealpy import FloatVar

    mod_path, _, cls_name = cls_path.rpartition(".")
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)

    problem = {
        "obj_func": f,
        "bounds": FloatVar(lb=[0.0] * n_dim, ub=[1.0] * n_dim),
        "minmax": "min",
        "log_to": None,
    }
    # Quiet mealpy's per-epoch log lines for the whole sweep.
    logging.getLogger("mealpy").setLevel(logging.WARNING)

    opt = cls(epoch=max(1, n_trials), pop_size=pop_size, **(kwargs or {}))
    g_best = opt.solve(problem, seed=seed)
    return g_best.target.fitness


@_reference()
def _ref_mealpy_pso(f, n_trials, n_dim, seed):
    return _ref_mealpy("mealpy.swarm_based.PSO.OriginalPSO", f, n_trials, n_dim, seed)


@_reference()
def _ref_mealpy_ga(f, n_trials, n_dim, seed):
    return _ref_mealpy("mealpy.evolutionary_based.GA.BaseGA", f, n_trials, n_dim, seed)


@_reference()
def _ref_mealpy_firefly(f, n_trials, n_dim, seed):
    # Use FFA (Firefly Algorithm) — `mealpy.swarm_based.FA` is the
    # Fireworks Algorithm (different family). #176 picked the wrong
    # one; the snapshot's previous "Firefly" comparison was actually
    # humpday's Firefly vs mealpy's Fireworks.
    return _ref_mealpy("mealpy.swarm_based.FFA.OriginalFFA", f, n_trials, n_dim, seed)


@_reference()
def _ref_mealpy_harmony(f, n_trials, n_dim, seed):
    return _ref_mealpy("mealpy.music_based.HS.OriginalHS", f, n_trials, n_dim, seed)


@_reference()
def _ref_mealpy_es(f, n_trials, n_dim, seed):
    return _ref_mealpy(
        "mealpy.evolutionary_based.ES.OriginalES", f, n_trials, n_dim, seed
    )


@_reference()
def _ref_mealpy_acor(f, n_trials, n_dim, seed):
    # ACOR uses sample_count instead of pop_size for the colony; the meter stops it at the
    # budget whatever its per-epoch cost.
    return _ref_mealpy("mealpy.swarm_based.ACOR.OriginalACOR", f, n_trials, n_dim, seed)


@_reference()
def _ref_pybobyqa(f, n_trials, n_dim, seed):
    import numpy as np
    import pybobyqa

    r = pybobyqa.solve(
        f,
        np.asarray(_draw_x0(seed, n_dim)),
        bounds=(np.zeros(n_dim), np.ones(n_dim)),
        maxfun=n_trials,
        seek_global_minimum=False,
        rhobeg=0.2,
        rhoend=1e-8,
        print_progress=False,
    )
    return r.f


# NEWUOA and UOBYQA are unconstrained methods with no bounds to be given, and PDFO calls the
# objective from Fortran through f2py, where a Python exception is fatal to the interpreter.
# Hence "project" -- the reference sees `objective(clip(x))`, exactly as the port does -- and no
# hard cap: PDFO's `maxfev` is a hard limit of its own, and `validate_run` fails the run if it
# ever is not.
@_reference(outside="project", hard_cap=False)
def _ref_pdfo_newuoa(f, n_trials, n_dim, seed):
    import numpy as np
    from pdfo import newuoa

    r = newuoa(
        f,
        np.asarray(_draw_x0(seed, n_dim)),
        options={"maxfev": n_trials, "rhobeg": 0.2, "rhoend": 1e-8},
    )
    return r.fun


@_reference(outside="project", hard_cap=False)
def _ref_pdfo_uobyqa(f, n_trials, n_dim, seed):
    import numpy as np
    from pdfo import uobyqa

    r = uobyqa(
        f,
        np.asarray(_draw_x0(seed, n_dim)),
        options={"maxfev": n_trials, "rhobeg": 0.2, "rhoend": 1e-8},
    )
    return r.fun


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


# ---------- which comparisons can run, and whether they had to ----------

# Set in the CI job. Locally a missing reference library is a skip, reported; in CI it is a
# failure, because that job is the one place the whole matrix is supposed to run. It used to be
# a skip there too, and the job's install did not include PDFO, so two PRIMA comparisons never
# ran in the environment configured to run them (#410).
STRICT_ENV = "HUMPDAY_REFERENCE_STRICT"

OK, ABSENT, PROBE_FAILED = "ok", "absent", "probe failed"


def strict() -> bool:
    return os.environ.get(STRICT_ENV, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def _probe_pdfo():
    """PDFO imports fine under NumPy 2 and fails at its first solver call, where `from
    .gethuge import gethuge` meets an extension compiled for NumPy 1.x. So call it."""
    import numpy as np
    from pdfo import newuoa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        newuoa(
            lambda x: float(sum(x)),
            np.array([0.0, 0.0]),
            options={"maxfev": 5, "rhobeg": 0.1, "rhoend": 1e-2},
        )


# A module whose import can succeed while the library is unusable gets a probe that uses it.
_PROBES = {"pdfo": _probe_pdfo}
_STATUS: dict = {}


def module_status(name: str) -> tuple:
    """`(OK, "")`, `(ABSENT, why)` or `(PROBE_FAILED, why)` for one module, cached.

    The two failures are told apart because they mean different things. Absent is a choice
    about what to install. Probe failed is an installation that is there and broken -- a
    compiled extension built for another NumPy, a dependency of the dependency missing --
    and used to be folded silently into the same skip (#410).
    """
    if name not in _STATUS:
        try:
            importlib.import_module(name)
            probe = _PROBES.get(name)
            if probe is not None:
                probe()
            _STATUS[name] = (OK, "")
        except ModuleNotFoundError as e:
            if e.name == name or name.startswith(f"{e.name}."):
                _STATUS[name] = (ABSENT, f"{name} is not installed")
            else:
                _STATUS[name] = (PROBE_FAILED, f"{name}: {type(e).__name__}: {e}")
        except Exception as e:
            _STATUS[name] = (PROBE_FAILED, f"{name}: {type(e).__name__}: {e}")
    return _STATUS[name]


def dependency_status(modules) -> tuple:
    """The first module that is not usable, or `(OK, "")`."""
    for m in modules:
        status = module_status(m)
        if status[0] != OK:
            return status
    return (OK, "")


def _all_installed(modules):
    return dependency_status(modules)[0] == OK


def require(modules):
    """Skip a test whose reference is unavailable -- or fail it, in the CI job (#410)."""
    status, why = dependency_status(modules)
    if status != OK:
        message = f"{status}: {why}"
        if strict():
            pytest.fail(f"reference expected in CI is unavailable -- {message}")
        pytest.skip(message)


# ---------- how far a port may lag its reference ----------

# The ratio printed as `hd/ref`: humpday's median gap to the optimum over the reference's, so
# 1.0 is parity and 2.0 is twice the remaining error. The default is what a port is expected to
# stay inside, and the exceptions below record the pairs that do not, so that a run tells the
# difference between a known deficiency and a new one.
#
# Ceilings are measured, not chosen: each is about twice the value recorded in
# benchmarks/reference_alignment.json, which gives a row room to move with a library version or
# a seed without letting it double. They are not targets. Four of sixty-six pairs need one,
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
    # beats gp_minimize on all three (0.68 / 1.71 / 0.00 at equal budgets).
    #
    # Re-measured again once both sides were held to the same number of objective calls
    # (#404). The references that used to overspend -- differential evolution, dual
    # annealing, L-BFGS-B, the mealpy six -- now stop at the budget, which moved their rows
    # but put no new pair over the default: humpday was already ahead of each of them, and
    # the extra calls had been flattering the references. Measured with NumPy 1.26.4, SciPy
    # 1.17.1, scikit-optimize 0.10.2, cmaes 0.13.1, mealpy 3.0.3, Py-BOBYQA 1.5.0, PDFO 2.2.0.
    #
    # `Powell/rosenbrock` measures 7.32 where it was recorded at 4.11, and the port has not
    # moved. The reference is never stopped by the meter; what changed is which of its values
    # counts. scipy's bounded Powell returns its final point, and on sixteen of twenty-one
    # seeds here it had evaluated a better one on the way (0.0081 against a returned 0.0226 at
    # seed 1). The harness now takes the best value each side actually observed, which is what
    # HumpDay's `best_value` has always been, so the reference's median gap fell from 0.034 to
    # 0.019. The ceiling follows the measurement at the usual factor of two.
    #
    # `PRIMA_UOBYQA/ackley` is new because PDFO is new to the run (#410): the CI job had never
    # installed it, so this pair had never been measured by the gate meant to measure it. It
    # is a real gap, not the instrument -- 6.2e-06 against PDFO's 5.1e-08, losing 0.81 of
    # pairings, as #410 found at 101x on another platform -- and is recorded here so that the
    # gate tells it apart from a new one, not because it is acceptable. #154 tracks it.
    ("Rechenberg", "ackley"): 53.0,  # measured 26.59, loses 0.63
    ("Powell", "rosenbrock"): 15.0,  # measured 7.32, loses 0.63
    ("DifferentialEvolution", "rosenbrock"): 7.6,  # measured 3.57, loses 0.58
    ("PRIMA_UOBYQA", "ackley"): 250.0,  # measured 122.93, loses 0.81 (#154)
}


# Pairs that lose more head to head than the default allows. Measured from
# benchmarks/reference_alignment.json and rounded up a little, same as the ratio ceilings: a
# record of what the ports do, not a target.
#
# The first run of this statistic put BayesianOpt at 0.90 on Ackley -- it sat at 2.58, which is
# where a run trapped on the ring sits, against gp_minimize's 0.032. A rate that lopsided is not
# the sampling noise this statistic makes, and it was not: the acquisition function was being
# sampled ten times rather than optimised. Fixed, it loses 0.46 there.
#
# The one entry is PRIMA_UOBYQA on Ackley, measured the first time PDFO was installed (#410),
# and explained above: a real gap, tracked by #154.
#
# #408 still has the JavaScript twin replacing the GP with a nearest-neighbour heuristic
# outright, and #81 tracks the hyperparameters, which are still fixed where scikit-optimize
# fits them.
WIN_CEILING: dict[tuple[str, str], float] = {
    ("PRIMA_UOBYQA", "ackley"): 0.95,  # measured 0.81 (#154)
}


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

    It refuses a value that is not finite. `NaN > b` and `NaN == b` are both false, so a NaN
    run used to count as a win for whichever side produced it, and `inf` on the reference side
    was a win for humpday. `validate_run` keeps such runs out; this is the second lock (#405).
    """
    for v in (*hd_vals, *ref_vals):
        if not math.isfinite(v):
            raise ValueError(f"head_to_head given a value that is not finite: {v!r}")
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

    def fail(heading, item):
        failures.setdefault(heading, []).append(item)

    # What ran and what did not, kept in the snapshot. A nonempty table used to be enough for
    # green, and six of the references are inline baselines with no dependencies at all, so
    # with every third-party library missing or broken the gate passed on those six (#410).
    coverage = {"strict": strict(), "compared": [], "absent": {}, "probe_failed": {}}
    for algorithm, (ref_label, ref_fn, mods) in REFERENCES.items():
        status, why = dependency_status(mods)
        if status != OK:
            key = "absent" if status == ABSENT else "probe_failed"
            coverage[key][algorithm] = why
            loud = "" if status == ABSENT else "  <-- installed but broken"
            print(f"\n--- {algorithm}: SKIP, {status}: {why}{loud} ---")
            continue
        coverage["compared"].append(algorithm)
        n_trials = REFERENCE_BUDGET_OVERRIDE.get(algorithm, n_trials_default)
        print(f"\n=== {algorithm}  vs  {ref_label}  (n_trials={n_trials}) ===")
        for problem_id, problem in PROBLEMS.items():
            func = problem["func"]
            opt_value = problem["opt"]
            pair = f"{algorithm}/{problem_id}"

            # Every run is checked before it can take part in a median or a pairing. One that
            # is not a result -- not finite, no call behind it, over budget, below the known
            # minimum, a value the objective never returned -- fails the pair, and is left out
            # of the statistics rather than counted as a win for the other side (#405).
            hd_vals, hd_runs = [], []
            for trial in range(N_RUNS):
                run = _run_humpday(algorithm, func, n_trials, n_dim, seed=trial)
                issues = validate_run(run, problem, n_trials)
                hd_runs.append(_evidence(trial, run, issues))
                if issues:
                    fail(
                        "humpday result invalid",
                        f"{pair} seed {trial}: {'; '.join(issues)}",
                    )
                else:
                    hd_vals.append(run["best_value"])

            # A reference that raised used to be recorded as inf, which made its gap infinite
            # and the ratio zero: the comparison humpday most conclusively "won" was the one
            # where the thing it is measured against never ran. A reference that *returned* inf
            # or NaN did the same thing by another route. Neither is a comparison.
            ref_vals, ref_runs = [], []
            for trial in range(N_RUNS):
                try:
                    run = ref_fn(func, n_trials, n_dim, seed=trial)
                except Exception as e:
                    print(f"    reference error on {problem_id}: {e}")
                    fail(
                        "reference did not run",
                        f"{pair} seed {trial}: {type(e).__name__}: {e}",
                    )
                    ref_runs.append(
                        {"seed": trial, "error": f"{type(e).__name__}: {e}"}
                    )
                    continue
                issues = validate_run(run, problem, n_trials)
                ref_runs.append(_evidence(trial, run, issues))
                if issues:
                    fail(
                        "reference result invalid",
                        f"{pair} seed {trial}: {'; '.join(issues)}",
                    )
                else:
                    ref_vals.append(run["best_value"])

            if not hd_vals or not ref_vals:
                print(
                    f"  {problem_id:<12}  no valid runs on one side; nothing to compare"
                )
                rows.append(
                    {
                        "algorithm": algorithm,
                        "reference": ref_label,
                        "problem": problem_id,
                        "budget": n_trials,
                        "humpday_valid_runs": len(hd_vals),
                        "reference_valid_runs": len(ref_vals),
                        **_columns("humpday", hd_runs),
                        **_columns("reference", ref_runs),
                    }
                )
                continue

            hd_med = sorted(hd_vals)[len(hd_vals) // 2]
            ref_med = sorted(ref_vals)[len(ref_vals) // 2]
            hd_gap = hd_med - opt_value
            ref_gap = ref_med - opt_value
            relative = (hd_gap + 1e-15) / (ref_gap + 1e-15)
            lost = head_to_head(hd_vals, ref_vals)

            hd_cols = _columns("humpday", hd_runs)
            ref_cols = _columns("reference", ref_runs)
            print(
                f"  {problem_id:<12}  hd={hd_med:>10.4g}  ref={ref_med:>10.4g}"
                f"  hd-to-opt={hd_gap:>10.4g}  ref-to-opt={ref_gap:>10.4g}"
                f"  hd/ref={relative:>9.2f}  lost={lost:>5.2f}"
                f"  evals hd={hd_cols['humpday_evals_median']}"
                f" ref={ref_cols['reference_evals_median']}"
                f" (stopped {ref_cols['reference_stopped_at_budget']}/{N_RUNS})"
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
                fail(
                    "lagging the reference",
                    f"{pair}: loses {lost:.2f} of head-to-head pairings, "
                    f"over its ceiling {win_cap:g}",
                )
            ceiling = ratio_ceiling(algorithm, problem_id)
            if lost > 0.5 and relative > ceiling and hd_gap > CONVERGED_GAP:
                fail(
                    "lagging the reference",
                    f"{pair}: hd/ref {relative:.2f} over its ceiling {ceiling:g}",
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
                    "budget": n_trials,
                    "humpday_valid_runs": len(hd_vals),
                    "reference_valid_runs": len(ref_vals),
                    **hd_cols,
                    **ref_cols,
                }
            )

    out = REPO_ROOT / "benchmarks" / "reference_alignment.json"
    out.parent.mkdir(exist_ok=True)
    snapshot = {
        "rows": [
            {
                k: (_json_number(v) if isinstance(v, float) else v)
                for k, v in row.items()
            }
            for row in rows
        ],
        "n_runs": N_RUNS,
        "seeds": list(range(N_RUNS)),
        "n_trials": n_trials_default,
        "n_trials_overrides": REFERENCE_BUDGET_OVERRIDE,
        "n_dim": n_dim,
        "coverage": coverage,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out, "w") as f:
        f.write(_dump(snapshot) + "\n")
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")

    # The snapshot is written first, so a failing run still leaves the table it was judged on.
    third_party = [a for a in coverage["compared"] if REFERENCES[a][2]]
    if not third_party:
        fail(
            "no third-party comparison ran",
            "only the inline baselines did, which compares the ports with nothing they were "
            "written from: install `pip install humpday[reference]`",
        )
    if coverage["strict"]:
        for key, label in (("absent", ABSENT), ("probe_failed", PROBE_FAILED)):
            for algorithm, why in coverage[key].items():
                fail(
                    "comparison expected in CI did not run",
                    f"{algorithm}: {label}: {why}",
                )
    assert not failures, "\n".join(
        f"{heading}:\n  " + "\n  ".join(items) for heading, items in failures.items()
    )
