"""Record the optimizer ratings table. One script, one artifact.

Replaces `record_elo.py` (two dimensions, sphere and Rosenbrock) and
`build_recommendation_grid.py` (nine analytic surfaces, no engineering problems). Those measured
either too little or the wrong thing, and their outputs drove different parts of the library.

The grid is dimension x budget x suite. See `humpday/ratings.py` for why those three and not
others, and why dimension is never interpolated.

Budgets are absolute rather than multiples of the dimension, because a caller knows how many
evaluations they can afford and not how many an interpolation set needs. The consequence is
deliberate: at n=100 a budget of 50 records what happens when the model-based methods cannot build
a model at all, which is a fact about that situation rather than a gap in the table.

Every run is capped in wall-clock time. Without that this grid does not terminate: UOBYQA at a
hundred variables wants an interpolation set of 5,151 points and solves the corresponding system in
pure Python. An optimizer that overruns the cap twice in a cell is disqualified there and recorded
under `timed_out`, which ranks it last in that cell. Being unable to return inside the allowance is
a fact about using the method at that size, not a gap in the table, but it is kept separate from
the ratings because it is a different kind of failure from losing.

Cells are independent, so they are sharded one file per cell under `benchmarks/ratings_cells/` and
run in parallel. Nothing is lost to a crash or a Ctrl-C, and a finished cell is skipped on the next
run, so this can be run repeatedly and in pieces:

    python benchmarks/record_ratings.py                      # everything missing, all cores
    python benchmarks/record_ratings.py --dims 50,100        # just the expensive end
    python benchmarks/record_ratings.py --problems 30        # deepen what is already there
    python benchmarks/record_ratings.py --merge              # rebuild the shipped table only
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import math
import multiprocessing
import os
import random
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humpday.ratings import SUITES  # noqa: E402

OUT = REPO_ROOT / "humpday" / "data" / "ratings.json"
SHARDS = REPO_ROOT / "benchmarks" / "ratings_cells"

# The old grid's spread, plus the rungs where worked engineering demos actually live.
DEFAULT_DIMS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 50, 100)

# Chosen against what the methods need before they can propose anything: 2n+1 for NEWUOA and
# BOBYQA, (n+1)(n+2)/2 for UOBYQA. 50 is below that from n=24 up, so it records the starved case.
# 200 is the practitioner default. 1000 makes UOBYQA viable to about n=44. 5000 sits just under the
# 5,151 it needs at n=100, the largest budget at which the three are still separable.
DEFAULT_BUDGETS = (50, 200, 1000, 5000)

# What an optimizer may spend, as a multiple of what the evaluations alone cost. A flat cap in
# seconds does not work here: at five variables the worked demos are real simulations and a
# five-thousand-evaluation run legitimately takes a minute, while the same run over an analytic
# surface takes under a second. A flat cap disqualifies the optimizer for the objective's cost.
# RandomSearch runs first on every problem and is almost pure sampling, so its wall clock measures
# the evaluations; everything else is allowed this multiple of it.
DEFAULT_OVERHEAD = 25.0
REFERENCE = "RandomSearch"


def DIM_CAP_OF(name: str) -> int:
    from humpday.eligibility import DIM_CAP

    return DIM_CAP.get(name, 0)


# A second floor under that allowance, in overhead permitted per evaluation. The multiple above is
# the wrong test on its own when the objective is an analytic surface costing microseconds: a flat
# ten seconds then disqualified BOBYQA and NEWUOA at sixteen and twenty-four variables, which is a
# configuration people run happily, for spending milliseconds per step on their interpolation set.
# What matters is overhead per evaluation, which is also what `humpday.eligibility` tiers on, so
# that is what is budgeted: twenty milliseconds a step, or a hundred seconds over five thousand.
SECONDS_PER_EVAL = 0.02

# Outer floor for tiny budgets, and a ceiling so one method cannot hold a core for an afternoon.
MIN_SECONDS, MAX_SECONDS = 5.0, 180.0

# Wall clock for a whole cell. A cell that runs out stops early with the problems it managed, and
# `humpday.ratings` already shrinks a thin cell's ranks toward the mean, so the table degrades by
# growing less confident rather than by going missing.
DEFAULT_CELL_SECONDS = 2700.0

# Overruns tolerated before an optimizer is dropped from a cell. One absorbs a straggler; the
# second is evidence, and paying the cap twenty-four times over to confirm it is waste.
TIMEOUT_STRIKES = 2

# Enough distinct engineering problems that a cell is not one landscape replayed. Where the fixed
# demos are thinner than this, the scalable stand-ins make up the difference.
MIN_ENGINEERING_VARIETY = 6


class _Overran(BaseException):
    """Not an `Exception`, deliberately.

    Twenty-five `except Exception` handlers sit between here and the running optimizer, and any one
    of them will treat a deadline as a bad function evaluation, swallow it and carry on. Inheriting
    from `BaseException` puts the abort past all of them, the same reasoning that makes
    `KeyboardInterrupt` what it is.
    """


@contextlib.contextmanager
def _deadline(seconds: float):
    """Abort whatever is running after `seconds`, wherever it is.

    A deadline checked inside the objective would not do: the methods at risk here spend their time
    between evaluations, fitting a surrogate or solving for an interpolation set, and would sail
    past it. SIGALRM interrupts the interpreter itself.

    The timer repeats rather than firing once, so a handler that does catch the abort buys a second
    and no more. Without that a single swallowed signal disarms the deadline for good, which is how
    a ten-second cap on UOBYQA at a hundred variables ran for seven minutes and was still going.
    """

    def fire(_signum, _frame):
        raise _Overran

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def surfaces_generator(n_dim: int, seed: int):
    """Analytic functions, freshly morphed each draw so no fixed landscape can be learned."""
    from humpday.objectives.stochastic_surfaces import create_fair_benchmark_run

    rnd = random.Random(seed)
    while True:
        with contextlib.redirect_stdout(io.StringIO()):
            produced = create_fair_benchmark_run(
                n_functions=6, seed=rnd.randrange(2**31)
            )
        suite = produced[0] if isinstance(produced, tuple) else produced
        yield from (list(suite.values()) if isinstance(suite, dict) else list(suite))


def engineering_generator(n_dim: int, seed: int):
    """Worked engineering problems at this dimension, topped up with scalable stand-ins.

    The demos are the better evidence and are used wherever they exist, but they are distributed by
    what someone wrote up rather than by dimension: fifteen at four variables, one at sixteen. A
    cell drawing on a single demo measures that one landscape however many times it is replayed, so
    the family below fills the pool out to `MIN_ENGINEERING_VARIETY`.
    """
    from humpday.objectives import physics_objectives

    fixed = [f for _n, f, d in physics_objectives() if d == n_dim]
    pool = list(fixed)
    if len(pool) < MIN_ENGINEERING_VARIETY:
        stand_ins = _scalable(n_dim, seed)
        pool += [next(stand_ins) for _ in range(MIN_ENGINEERING_VARIETY - len(pool))]

    rnd = random.Random(seed)
    while True:
        rnd.shuffle(pool)
        yield from pool


def _scalable(n_dim: int, seed: int):
    """Engineering structures parameterised by dimension, for where no fixed demo exists.

    A partial stand-in, measured rather than assumed: raced at twelve dimensions against the real
    demos these sit between them and the analytic surfaces, surfacing the direct-search methods the
    demos favour while still letting a trust-region method lead. Three structures cannot stand in
    for seventy-six problems. Read a rating from them as better than surfaces alone and weaker than
    the demos.
    """
    rnd = random.Random(seed)

    def packing(u):
        # Soft minimum over clearances. A hard min is the honest packing objective and useless to
        # race at size: the binding constraint is one pair, so at fifty variables it reads zero
        # coordinates at a random point and gives an optimizer nothing to follow.
        pts = [(u[2 * i], u[2 * i + 1]) for i in range(len(u) // 2)]
        gaps = []
        for cx, cy in pts:
            gaps += [cx, 1 - cx, cy, 1 - cy]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                gaps.append(
                    math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) / 2
                )
        lo, beta = min(gaps), 60.0
        return -(lo - math.log(sum(math.exp(-beta * (g - lo)) for g in gaps)) / beta)

    def dispatch(u):
        # Battery charge/discharge against a price curve, state of charge carried across steps.
        price = [
            1.0 + math.sin(2 * math.pi * k / max(len(u), 2)) for k in range(len(u))
        ]
        soc, revenue = 0.5, 0.0
        for k, x in enumerate(u):
            rate = 2.0 * x - 1.0
            soc -= rate * 0.1
            if soc < 0.0 or soc > 1.0:
                revenue -= 10.0 * abs(soc - min(max(soc, 0.0), 1.0))
                soc = min(max(soc, 0.0), 1.0)
            revenue += rate * price[k]
        return -revenue

    def descent(u):
        # Discretised brachistochrone: heights interact through the speed carried forward.
        ys = [1.0 - 0.9 * v for v in u]
        t, v, prev, dx = 0.0, 0.0, 1.0, 1.0 / max(len(ys), 1)
        for y in ys:
            drop = prev - y
            v = math.sqrt(max(v * v + 2 * 9.81 * drop, 1e-9))
            t += math.hypot(dx, drop) / v
            prev = y
        return t

    family = [packing, dispatch, descent]
    while True:
        rnd.shuffle(family)
        yield from family


GENERATORS = {"surfaces": surfaces_generator, "engineering": engineering_generator}


def _shard_path(n_dim: int, budget: int, suite: str) -> Path:
    return SHARDS / f"{n_dim}_{budget}_{suite}.json"


def _write_shard(path: Path, payload: dict) -> None:
    SHARDS.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)  # atomic: an interrupted write cannot truncate the record


def run_cell(
    n_dim: int,
    budget: int,
    suite: str,
    problems: int,
    seed: int,
    overhead: float,
    cell_seconds: float,
) -> dict:
    """Race every optimizer over `problems` draws from one suite, and rate them by Elo.

    Elo is computed over the pairwise outcomes within this cell alone. It is not comparable across
    cells and `humpday.ratings` never treats it as if it were.
    """
    from humpday.eligibility import min_trials, passes_dim, passes_trials
    from humpday.optimizers.adaptive_optimizer import (
        EloRatingSystem,
        normalize_performance,
    )
    from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS, pure_optimize

    # Race what `eligibility` would let `recommend` consider, and nothing else. Two reasons, and
    # the second is the important one. It is wasteful to pay the allowance twice a cell to
    # rediscover that a GP is cubic in observations at a hundred variables, or that UOBYQA cannot
    # fill a 5,151-point interpolation set out of a 5,000-evaluation budget -- both are arithmetic,
    # not open questions. And a rating for an optimizer that `recommend` can never return is
    # evidence about nothing: the table should describe the choice actually on offer.
    ineligible = {}
    contenders = []
    for name in PURE_OPTIMIZERS:
        if not passes_dim(name, n_dim):
            ineligible[name] = f"dimension cap {DIM_CAP_OF(name)}"
        elif not passes_trials(name, n_dim, budget):
            ineligible[name] = f"needs {min_trials(name, n_dim)} evaluations"
        else:
            contenders.append(name)
    if len(contenders) < 2 or REFERENCE not in contenders:
        return {
            "n_dim": n_dim,
            "budget": budget,
            "suite": suite,
            "problems": 0,
            "ratings": {},
            "ineligible": ineligible,
        }

    path = _shard_path(n_dim, budget, suite)
    shard = json.loads(path.read_text()) if path.exists() else {}
    done = int(shard.get("problems", 0))
    if done >= problems:
        return shard

    elo = EloRatingSystem()
    elo.ratings.update(shard.get("ratings", {}))
    strikes = dict(shard.get("strikes", {}))
    timed_out = dict(shard.get("timed_out", {}))

    generator = GENERATORS[suite](n_dim, seed + n_dim + budget)
    for _ in range(done):
        next(generator)  # resume the stream where the shard left off

    started = time.time()
    index = done - 1
    for index in range(done, problems):
        if time.time() - started > cell_seconds:
            break
        objective = next(generator)
        results = {}

        clock = time.time()
        try:
            with _deadline(MAX_SECONDS):
                results[REFERENCE] = pure_optimize(objective, REFERENCE, budget, n_dim)[
                    0
                ]
        except _Overran:
            break  # the objective alone exhausts the ceiling; nothing here is measurable
        except Exception:
            results[REFERENCE] = float("inf")
        seconds = overhead * (time.time() - clock)
        seconds = min(max(seconds, SECONDS_PER_EVAL * budget, MIN_SECONDS), MAX_SECONDS)

        for name in contenders:
            if name == REFERENCE or name in timed_out:
                continue
            try:
                with _deadline(seconds):
                    value, _ = pure_optimize(objective, name, budget, n_dim)
                results[name] = value
            except _Overran:  # BaseException, so this must precede the Exception clause
                strikes[name] = strikes.get(name, 0) + 1
                if strikes[name] >= TIMEOUT_STRIKES:
                    timed_out[name] = round(seconds, 1)
            except Exception:
                results[name] = float("inf")  # a failure is a loss, not an exclusion

        if len(results) > 1:
            names = list(results)
            scores = dict(zip(names, normalize_performance(list(results.values()))))
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    if scores[a] == scores[b]:
                        outcome = 0.5
                    else:
                        outcome = 1.0 if scores[a] > scores[b] else 0.0
                    elo.update_ratings(a, b, outcome)

        shard = {
            "n_dim": n_dim,
            "budget": budget,
            "suite": suite,
            "problems": index + 1,
            "ratings": dict(elo.ratings),
            "strikes": strikes,
            "timed_out": timed_out,
            "ineligible": ineligible,
            "seconds": round(time.time() - started, 1),
            "allowance": round(seconds, 1),
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        _write_shard(path, shard)
    if index + 1 < problems and shard:
        shard["stopped_early"] = f"{cell_seconds:.0f}s budget"
        _write_shard(path, shard)
    return shard


def _run_cell_job(job) -> str:
    key = f"{job[0]}/{job[1]}/{job[2]}"
    started = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            shard = run_cell(*job)
    except Exception as exc:  # one bad cell must not take the grid down
        return f"{key}: FAILED {type(exc).__name__}: {exc}"
    top = sorted(shard.get("ratings", {}).items(), key=lambda kv: -kv[1])[:3]
    out = (
        f"{key}: {shard.get('problems', 0)}p in {time.time() - started:.0f}s  "
        + ", ".join(f"{n} {r:.0f}" for n, r in top)
    )
    if not shard.get("problems"):
        return f"{key}: nothing eligible"
    if shard.get("timed_out"):
        out += f"  [timed out: {', '.join(sorted(shard['timed_out']))}]"
    return out


def merge() -> dict:
    """Collect the shards into the table that ships inside the package."""
    cells = {}
    for path in sorted(SHARDS.glob("*.json")):
        shard = json.loads(path.read_text())
        if not shard.get("problems"):
            continue  # a cell with no eligible field is not a cell
        cells[f"{shard['n_dim']}/{shard['budget']}/{shard['suite']}"] = {
            "ratings": shard.get("ratings", {}),
            "problems": shard.get("problems", 0),
            "timed_out": sorted(shard.get("timed_out", {})),
            "ineligible": dict(sorted(shard.get("ineligible", {}).items())),
        }
    table = {
        "cells": cells,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(table, indent=2, sort_keys=True))
    tmp.replace(OUT)
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", default=",".join(str(d) for d in DEFAULT_DIMS))
    ap.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS))
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--problems", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD)
    ap.add_argument("--cell-seconds", type=float, default=DEFAULT_CELL_SECONDS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument(
        "--merge", action="store_true", help="rebuild the shipped table and stop"
    )
    a = ap.parse_args()

    if a.merge:
        table = merge()
        print(f"{len(table['cells'])} cells -> {OUT.relative_to(REPO_ROOT)}")
        return 0

    jobs = [
        (n_dim, budget, suite, a.problems, a.seed, a.overhead, a.cell_seconds)
        for n_dim in (int(x) for x in a.dims.split(",") if x.strip())
        for budget in (int(x) for x in a.budgets.split(",") if x.strip())
        for suite in (s for s in a.suites.split(",") if s.strip())
    ]
    # Longest pole first, so the makespan is not one enormous cell starting last on one core --
    # but interleaved with the cheapest, so the table is populated and checkable within minutes
    # instead of only after the expensive end finishes. The biggest job still starts at t=0, which
    # is the part that governs how long the whole run takes.
    jobs.sort(key=lambda j: -(j[0] * j[1]))
    jobs = [
        j
        for pair in zip(
            jobs[: (len(jobs) + 1) // 2], reversed(jobs[(len(jobs) + 1) // 2 :])
        )
        for j in pair
    ] + ([jobs[len(jobs) // 2]] if len(jobs) % 2 else [])
    print(
        f"{len(jobs)} cells on {a.workers} workers, {a.overhead:.0f}x evaluation cost "
        f"per run, {a.cell_seconds:.0f}s per cell",
        flush=True,
    )

    if a.workers == 1:
        for job in jobs:
            print(_run_cell_job(job), flush=True)
    else:
        with multiprocessing.Pool(a.workers) as pool:
            # A pool worker is a daemon, which gets it reaped when the parent exits cleanly and
            # orphaned when the parent is signalled. Orphans are not merely untidy: they keep
            # writing shards from the code they were started with, so a run launched after an
            # edit quietly records a mixture of two versions while competing for the cores.
            def _stop(_signum, _frame):
                pool.terminate()
                raise SystemExit(143)

            for name in ("SIGTERM", "SIGINT", "SIGHUP"):
                signal.signal(getattr(signal, name), _stop)
            for index, line in enumerate(
                pool.imap_unordered(_run_cell_job, jobs), start=1
            ):
                print(f"[{index}/{len(jobs)}] {line}", flush=True)
                merge()

    table = merge()
    print(f"\n{len(table['cells'])} cells -> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
