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

Resuming a cell continues one recording rather than splicing two together. Each shard records the
seed, backend and package version it was recorded with, and a run that disagrees with an existing
shard on any of those stops and says so instead of extending it; delete the shard to rebuild the
cell. Shards carry the full stream position too, counting the problems drawn and skipped as well
as the ones rated, so the second half of a cell races problems the first half has not seen. Within
a cell, every optimizer run is seeded from its own coordinates (cell, problem, optimizer), so the
tournament does not depend on how the work was divided between workers.
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
import statistics
import subprocess
import sys
import time
import zlib
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

# What EloRatingSystem seeds every optimizer with in its constructor. A rating still exactly equal
# to it means the optimizer never completed a rated match.
INITIAL_RATING = 1500.0


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

# Multiple of the MEDIAN completed run in this cell.
#
# Expect this to make the expensive engineering cells THINNER rather than merely more accurate.
# Where an objective genuinely costs a converging optimizer tens of seconds, a fair allowance means
# twenty-three times that per problem, and the per-cell wall clock will stop the cell well short of
# the target. That is the intended trade: `humpday.ratings` already shrinks a cell's ranks toward
# the cross-suite mean by how many problems back it, so a thin honest cell is worth more than a
# deep one whose disqualifications are an artifact. Watch `problems` and `timed_out` together when
# reading a re-record. The slowest completed run was the obvious
# statistic and is the wrong one: it is an extreme, so a single slow optimizer ratchets the
# allowance for all twenty-three and the cell stops being affordable -- measured, one problem in
# 4/1000/engineering did not finish in nineteen minutes where the whole cell previously took
# sixty-one. The median tracks what the objective costs a typical converging optimizer and is not
# moved by one outlier, so the factor can be generous.
TYPICAL_FACTOR = 4.0

# Outer floor for tiny budgets, and a ceiling so one method cannot hold a core for an afternoon.
MIN_SECONDS, MAX_SECONDS = 5.0, 180.0

# Problems abandoned before a cell is given up on. A problem whose evaluations alone exceed the
# ceiling is skipped rather than raced; enough of those and the budget is simply beyond what this
# suite can be measured at, which is worth recording rather than grinding at.
MAX_SKIPS = 8

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


# How many exception messages per optimizer a shard keeps, so a table can be audited for what
# actually went wrong without growing without bound.
MAX_FAILURE_REASONS = 5


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _provenance(seed: int, overhead: float, cell_seconds: float) -> dict:
    """What a shard was recorded with. Two shards are only combined when this agrees.

    The parts that change the stream of problems and optimizer draws (seed, backend, package
    version) must match exactly; the wall-clock settings are recorded so a table can say what it
    was measured under, but a resume may change them.
    """
    import humpday
    from humpday import _array as _A

    return {
        "seed": int(seed),
        "backend": _A.BACKEND,
        "humpday": getattr(humpday, "__version__", "unknown"),
        "git": _git_revision(),
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "overhead": float(overhead),
        "cell_seconds": float(cell_seconds),
    }


_PROVENANCE_MUST_MATCH = ("seed", "backend", "humpday")


def _seed_run(
    seed: int, n_dim: int, budget: int, suite: str, draw: int, name: str
) -> None:
    """One deterministic seed per (cell, problem, optimizer) run.

    The optimizer's draws used to depend on whatever the worker process had done before, so the
    same cell recorded on a different worker schedule was a different tournament. Seeding every
    run from its coordinates makes a cell reproducible however the work is scheduled, and keeps
    the problems' randomness independent of the optimizers'.
    """
    import random as _random

    from humpday import _array as _A

    key = f"{seed}/{n_dim}/{budget}/{suite}/{draw}/{name}"
    value = zlib.crc32(key.encode()) & 0xFFFFFFFF
    _A.seed(value)
    _random.seed(value)


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
        pairwise_outcome,
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

    provenance = _provenance(seed, overhead, cell_seconds)
    if shard:
        # A resumed cell continues one recording; it must not splice a second one onto it. Anything
        # recorded without provenance predates this check and is rebuilt rather than extended.
        previous = shard.get("provenance")
        mismatch = [
            k
            for k in _PROVENANCE_MUST_MATCH
            if previous is None or previous.get(k) != provenance.get(k)
        ]
        if mismatch:
            raise ValueError(
                f"{path.name}: existing shard was recorded with different "
                f"{', '.join(mismatch)} ({previous}); delete it to rebuild the cell"
            )
        # The shard describes the recording, not the process that happened to finish it, so the
        # provenance written when the cell was started is the one it keeps.
        provenance = previous

    elo = EloRatingSystem()
    elo.ratings.update(shard.get("ratings", {}))
    strikes = dict(shard.get("strikes", {}))
    timed_out = dict(shard.get("timed_out", {}))
    played = dict(shard.get("played", {}))
    failures = dict(shard.get("failures", {}))
    failure_reasons = {k: list(v) for k, v in shard.get("failure_reasons", {}).items()}

    # Resume where the stream actually left off: every draw counts, including the problems that
    # were skipped as too costly, or the resumed cell races a different sequence of problems.
    drawn = int(shard.get("drawn", done))
    generator = GENERATORS[suite](n_dim, seed + n_dim + budget)
    for _ in range(drawn):
        next(generator)

    started = time.time()
    recorded, skipped = done, int(shard.get("skipped", 0))
    completed: list = [float(v) for v in shard.get("completed_seconds", [])]
    while recorded < problems:
        if time.time() - started > cell_seconds:
            break
        objective = next(generator)
        draw = drawn
        drawn += 1
        results = {}

        clock = time.time()
        try:
            _seed_run(seed, n_dim, budget, suite, draw, REFERENCE)
            with _deadline(MAX_SECONDS):
                results[REFERENCE] = pure_optimize(objective, REFERENCE, budget, n_dim)[
                    0
                ]
        except _Overran:
            # The evaluations alone exceed the ceiling, so nothing on this problem is measurable
            # at this budget. Skip the problem, not the cell: an engineering suite is a handful of
            # worked simulations of wildly different cost, and one cart-pole rollout that runs
            # forty milliseconds a step should not cost the other five their tournament.
            skipped += 1
            if skipped > MAX_SKIPS:
                break
            continue
        except Exception as exc:
            results[REFERENCE] = float("inf")
            failures[REFERENCE] = failures.get(REFERENCE, 0) + 1
            failure_reasons.setdefault(REFERENCE, [])
            if len(failure_reasons[REFERENCE]) < MAX_FAILURE_REASONS:
                failure_reasons[REFERENCE].append(f"{type(exc).__name__}: {exc}"[:200])
        # Allowance for every other optimizer on this problem.
        #
        # A random sampler is the right reference for an optimizer's own overhead and the wrong one
        # for an objective whose cost depends on where it is evaluated: the sampler never visits
        # the expensive basin the optimizer converges into. In one cell of the previous recording
        # eighteen of twenty-three optimizers overran against a reference that finished in 0.8
        # seconds, which is not eighteen overhead problems, it is one mis-calibrated allowance.
        #
        # So the reference is a floor on the estimate, not the whole of it: the allowance also
        # tracks the slowest run that has actually completed in this cell, which is a direct
        # measurement of what the objective costs an optimizer that is converging. It can only
        # grow, and the ceiling still bounds it.
        seconds = overhead * (time.time() - clock)
        seconds = max(seconds, SECONDS_PER_EVAL * budget, MIN_SECONDS)
        typical = statistics.median(completed) if completed else 0.0
        seconds = min(max(seconds, TYPICAL_FACTOR * typical), MAX_SECONDS)

        for name in contenders:
            if name == REFERENCE or name in timed_out:
                continue
            run_clock = time.time()
            try:
                _seed_run(seed, n_dim, budget, suite, draw, name)
                with _deadline(seconds):
                    value, _ = pure_optimize(objective, name, budget, n_dim)
                results[name] = value
                completed.append(time.time() - run_clock)
                del completed[:-200]  # a rolling tail is enough for a median
            except _Overran:  # BaseException, so this must precede the Exception clause
                strikes[name] = strikes.get(name, 0) + 1
                if strikes[name] >= TIMEOUT_STRIKES:
                    timed_out[name] = round(seconds, 1)
            except Exception as exc:
                results[name] = float("inf")  # a failure is a loss, not an exclusion
                failures[name] = failures.get(name, 0) + 1
                failure_reasons.setdefault(name, [])
                if len(failure_reasons[name]) < MAX_FAILURE_REASONS:
                    failure_reasons[name].append(f"{type(exc).__name__}: {exc}"[:200])

        # Rate on the values themselves. Normalising the round first meant one optimizer raising
        # an exception -- recorded as inf, so max_val is inf -- turned every score in the round
        # into NaN, and NaN compares false both ways, so the whole round was scored as a draw
        # except where the tie test also failed. See pairwise_outcome.
        if len(results) > 1:
            names = list(results)
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    outcome = pairwise_outcome(results[a], results[b])
                    if outcome is None:
                        continue  # two failures carry no evidence about each other
                    elo.update_ratings(a, b, outcome)
                    # Participation is what distinguishes an earned 1500 from a seeded one, so it
                    # is counted here, where a rated match actually happens, and persisted.
                    played[a] = played.get(a, 0) + 1
                    played[b] = played.get(b, 0) + 1

        shard = {
            "n_dim": n_dim,
            "budget": budget,
            "suite": suite,
            "problems": recorded + 1,
            # Only what raced. EloRatingSystem seeds every optimizer at 1500 on construction, so
            # `elo.ratings` carries an untouched initial rating for each one filtered out before
            # the tournament -- a fabricated number sitting mid-table, which is the thing this
            # whole artifact exists to stop shipping.
            "ratings": {n: r for n, r in elo.ratings.items() if n in contenders},
            "strikes": strikes,
            "timed_out": timed_out,
            # Rated matches per optimizer. `merge` needs this to tell an optimizer that played and
            # happens to sit at 1500 from one that never raced at all.
            "played": played,
            "failures": failures,
            "failure_reasons": failure_reasons,
            "skipped": skipped,
            # Draws taken from the generator, including the skipped ones, so a resumed cell picks
            # the stream up where it left off rather than replaying problems it already saw.
            "drawn": drawn,
            "provenance": provenance,
            "ineligible": ineligible,
            "seconds": round(time.time() - started, 1),
            "allowance": round(seconds, 1),
            "completed_seconds": [round(v, 3) for v in completed[-200:]],
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        _write_shard(path, shard)
        recorded += 1

    if recorded < problems and shard:
        # The last write happened before these last draws; without this a cell that stopped on the
        # skip limit would resume from the wrong point in the stream.
        shard["drawn"] = drawn
        shard["skipped"] = skipped
        shard["stopped_early"] = (
            f"{skipped} problems too costly to measure"
            if skipped > MAX_SKIPS
            else f"{cell_seconds:.0f}s budget"
        )
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
        if shard.get("ineligible") is not None and not shard.get("skipped"):
            return f"{key}: nothing eligible to race"
        return f"{key}: not measurable at this budget"
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
        # Drop every rating that was never earned. Three ways that happens, and the first fix
        # caught only one of them.
        #
        # An optimizer filtered out before the tournament is named under `ineligible`. One that
        # struck out on its first two problems is named under `timed_out` and never completed a
        # rated match either, so its entry is still the bare 1500.0 that EloRatingSystem seeds
        # every optimizer with in its constructor. Fifty such entries shipped in v0.24.0, where
        # `elo_by_dimension()` ranked a seeded 1500.0 above a PatternSearch that had earned
        # 1499.4 by losing.
        #
        # The third way needs no `timed_out` entry at all: a cell that stops early on the
        # wall-clock budget (#339) can end after an optimizer's first overrun, one strike short
        # of the two that would name it in `timed_out`, having still never returned a value.
        #
        # The test is participation, not the value of the rating. Equality with the seed catches
        # all three cases but is not the same question, and it answers wrongly for the optimizer
        # that played and came back to exactly 1500 -- win one, lose one, or draw every match in
        # a round of equal values. Such an optimizer has a real rating, earned across real
        # matches, and dropping it silently withdraws it from the grid. `run_cell` counts a match
        # for each name on each rated pair, so a positive count is the direct evidence.
        #
        # A timed-out optimizer that did complete some problems has a real, if partial, rating and
        # keeps it -- `_ranked` still places it last, and the revolt branch still needs it.
        excluded = set(shard.get("ineligible", {}))
        played = shard.get("played")
        if played is None:
            # A shard recorded before participation was tracked. Fall back to the old test, which
            # is right except for the exactly-1500 case it cannot see.
            def _earned(name: str, rating: float) -> bool:
                return rating != INITIAL_RATING
        else:

            def _earned(name: str, rating: float) -> bool:
                return played.get(name, 0) > 0

        kept = {
            n: r
            for n, r in shard.get("ratings", {}).items()
            if n not in excluded and _earned(n, r)
        }

        cells[f"{shard['n_dim']}/{shard['budget']}/{shard['suite']}"] = {
            "ratings": kept,
            "problems": shard.get("problems", 0),
            "timed_out": sorted(shard.get("timed_out", {})),
            "overruns": dict(sorted(shard.get("strikes", {}).items())),
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
