"""Record Elo ratings per problem dimension, incrementally.

`record_elo.py` runs one tournament in two dimensions on sphere and Rosenbrock variants. That is
the whole evidence base behind `humpday.suggest`, whose ordering varies with `n_dim` -- so the
ratings cover exactly one of the four dimension buckets it recommends for, and the bucket where the
default matters most (n_dim > 50) has none.

This records ratings **per dimension**, on stochastic surfaces rather than fixed landscapes.
`stochastic_surfaces.create_fair_benchmark_run` regenerates every instance with a fresh shift,
rotation (70% of runs), scale, noise level and modal frequency, so no optimizer can be tuned to a
known landscape and repeated runs are genuinely new evidence rather than the same match replayed.

Incremental by design. Each run loads the existing ratings for a dimension, plays more matches, and
writes them back, so the record improves in whatever increments you have time for:

    python benchmarks/record_elo_by_dimension.py --dims 2,10                 # a quick pass
    python benchmarks/record_elo_by_dimension.py --dims 50,100 --problems 20 # a longer one
    python benchmarks/record_elo_by_dimension.py                             # all buckets

Output: humpday/data/elo_by_dimension.json, read by humpday.suggest.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humpday.optimizers.adaptive_optimizer import (  # noqa: E402
    EloRatingSystem,
    run_algorithm_tournament,
)

OUT = REPO_ROOT / "humpday" / "data" / "elo_by_dimension.json"

# One per bucket boundary in suggest_pure, so every ordering it can return has evidence behind it.
# Dimensions where enough engineering demos live to make a tournament meaningful.
DEFAULT_DIMS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 24)


def scalable_physics(n_dim: int, seed: int):
    """Engineering objectives that take the dimension as a parameter.

    The fixed demos under ``example_applications`` stop at 24 dimensions (one reaches 60), so above
    that the physics half of the tournament has nothing to say, and recording only analytic surfaces
    there would rank optimizers on exactly the suite that misleads. These are the same structures
    with the structural constant turned up: pack n/2 circles rather than six, schedule n hours of
    battery dispatch rather than twenty-four, place n control points on a descent rather than eight.

    How good a stand-in are they? Partial, and measured rather than asserted. Raced at twelve
    dimensions against the real demos and the analytic surfaces:

        real demos         CoordinateDescent, PRIMA_BOBYQA, Rechenberg, Alloy
        this family        PRIMA_BOBYQA, PatternSearch, PRIMA_NEWUOA, CoordinateDescent
        analytic surfaces  PRIMA_NEWUOA, PRIMA_UOBYQA, PRIMA_BOBYQA, NelderMead

    They sit between the two. The direct-search methods that the real demos favour and the
    surfaces do not, PatternSearch and CoordinateDescent, show up here; but a trust-region method
    still leads, as on the surfaces. Three structures cannot stand in for seventy-six problems.
    Read a high-dimensional physics rating as better evidence than analytic surfaces alone and
    weaker evidence than the fixed demos, not as equivalent to them.
    """
    rnd = random.Random(seed)

    def packing(u):
        """Pack n/2 circles into the unit square, scored by a soft minimum over clearances.

        A hard `min` is the honest packing objective but it is useless to race at this size: with
        twenty-five circles the binding constraint is a single pair, so moving any other circle
        leaves the value untouched. Measured, the hard version reads 0 of 50 coordinates at a
        random point, which gives an optimizer nothing to follow. The soft minimum below keeps the
        structure -- still dominated by the tightest clearances, still ridged where the binding
        pair changes -- while letting every circle move the score.
        """
        pts = [(u[2 * i], u[2 * i + 1]) for i in range(len(u) // 2)]
        gaps = []
        for cx, cy in pts:
            gaps += [cx, 1 - cx, cy, 1 - cy]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx, dy = pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]
                gaps.append(math.hypot(dx, dy) / 2)
        beta = 60.0  # sharp enough that the tightest clearances dominate
        lo = min(gaps)
        soft = lo - math.log(sum(math.exp(-beta * (g - lo)) for g in gaps)) / beta
        return -soft

    def dispatch(u, price=None):
        # Charge/discharge a battery against a price curve, with state of charge carried across
        # steps and penalised at the bounds. Revenue is negated so lower is better.
        price = price or [
            1.0 + math.sin(2 * math.pi * k / max(len(u), 2)) for k in range(len(u))
        ]
        soc, revenue = 0.5, 0.0
        for k, x in enumerate(u):
            rate = 2.0 * x - 1.0  # [-1, 1]: discharge when positive
            soc -= rate * 0.1
            if soc < 0.0 or soc > 1.0:
                revenue -= 10.0 * abs(soc - min(max(soc, 0.0), 1.0))
                soc = min(max(soc, 0.0), 1.0)
            revenue += rate * price[k]
        return -revenue

    def descent(u):
        # Time down a piecewise-linear curve through n control heights: a discretised
        # brachistochrone, where the heights interact through the speed carried forward.
        ys = [1.0 - 0.9 * v for v in u]
        t, v = 0.0, 0.0
        dx = 1.0 / max(len(ys), 1)
        prev = 1.0
        for y in ys:
            drop = prev - y
            v = math.sqrt(max(v * v + 2 * 9.81 * drop, 1e-9))
            t += math.hypot(dx, drop) / v
            prev = y
        return t

    family = [packing, dispatch, descent]

    def cycle():
        while True:
            rnd.shuffle(family)
            yield from family

    return cycle()


def physics_generator(n_dim: int, seed: int):
    """Cycle the engineering demos that live at this dimension.

    These are the realistic half of the evidence. The analytic surfaces are smooth, and smoothness
    rewards local search: on morphed surfaces the trust-region trio takes first, second and third
    at every dimension recorded, and on the physics demos at the same dimensions they are displaced
    -- PRIMA_BOBYQA falls to fifth at d=4 and the other two out of the top five altogether. Ranking
    optimizers on surfaces alone measures how smooth the surfaces are.
    """
    from humpday.objectives import physics_objectives

    group = [f for _n, f, d in physics_objectives() if d == n_dim]
    if not group:
        return None
    rnd = random.Random(seed)

    def cycle():
        while True:
            rnd.shuffle(group)
            yield from group

    return cycle()


def stochastic_generator(n_dim: int, seed: int):
    """Yield freshly morphed objectives. Each problem is a new random surface."""
    from humpday.objectives.stochastic_surfaces import create_fair_benchmark_run

    rnd = random.Random(seed)
    while True:
        # The generator chatters on construction; it is not our stdout to spend.
        with contextlib.redirect_stdout(io.StringIO()):
            produced = create_fair_benchmark_run(
                n_functions=6, seed=rnd.randrange(2**31)
            )
        suite = produced[0] if isinstance(produced, tuple) else produced
        fns = list(suite.values()) if isinstance(suite, dict) else list(suite)
        yield from fns


def budget_for(n_dim: int, requested: int | None) -> int:
    """Evaluations to allow at this dimension.

    A fixed budget across dimensions quietly rigs the tournament. The model-based methods
    (BOBYQA, NEWUOA, UOBYQA) build an interpolation set of roughly 2n+1 points before they can
    propose anything, which is 201 points at n=100. Given a flat 100 evaluations they return
    almost immediately having built no model -- and then placed first, second and third on a
    recorded d=100 leaderboard, which is how this was noticed. Scaling with dimension lets every
    optimizer do the thing it exists to do before being scored.
    """
    return requested if requested else max(100, 10 * n_dim)


# Cost is dominated by the trust-region methods once the budget is large enough for them to build
# a model. At d=100 with 1000 evaluations PRIMA_BOBYQA and PRIMA_NEWUOA run for minutes each while
# the other twenty-one finish in under a second, so those buckets are recorded deliberately rather
# than swept up in a default run. A per-optimizer wall-clock cap would bound this, but it has to be
# enforced per optimizer to be fair, and run_algorithm_tournament drives them internally.
EXPENSIVE_ABOVE = 25


def load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {
        "by_dimension": {},
        "generator": "stochastic_surfaces.create_fair_benchmark_run",
    }


def _save(data: dict) -> None:
    data["recorded_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(OUT)  # atomic: a crash mid-write cannot leave a truncated record


def record(
    dims, n_problems: int, trials: int, seed: int, suite: str = "smooth"
) -> dict:
    """Play `n_problems` more matches per dimension, checkpointing after every one.

    A high-dimensional bucket runs for minutes, so nothing here is allowed to depend on reaching
    the end: the record is written after each individual problem, through a temporary file so an
    interrupted write cannot truncate it, and a failure in one dimension does not cost the others.
    """
    data = load()
    for n_dim in dims:
        key = str(n_dim)
        prior = data["by_dimension"].get(key, {}).get(suite, {})
        elo = EloRatingSystem()
        for name, rating in prior.get("ratings", {}).items():
            elo.ratings[name] = rating  # resume rather than restart
        played = prior.get("match_history_count", 0)
        budget = budget_for(n_dim, trials)
        print(
            f"n_dim={n_dim} [{suite}]: {n_problems} problems x {budget} trials "
            f"(resuming from {played})",
            flush=True,
        )

        if suite == "physics":
            generator = physics_generator(n_dim, seed + n_dim)
            if generator is None:
                # No fixed demo at this dimension: use the scalable engineering family rather
                # than leaving the dimension with analytic surfaces as its only evidence.
                print(
                    f"   no fixed demo at d={n_dim}; using scalable engineering family",
                    flush=True,
                )
                generator = scalable_physics(n_dim, seed + n_dim)
        else:
            generator = stochastic_generator(n_dim, seed + n_dim)
        for i in range(n_problems):
            try:
                elo = run_algorithm_tournament(
                    generator,
                    trials_per_problem=budget,
                    n_problems=1,
                    n_dim=n_dim,
                    elo_system=elo,
                )
            except Exception as exc:  # keep what has been paid for, then move on
                print(
                    f"   problem {i + 1} failed ({type(exc).__name__}: {exc}); "
                    f"keeping {played + i} matches",
                    flush=True,
                )
                break
            data["by_dimension"].setdefault(key, {})[suite] = {
                "ratings": dict(elo.ratings),
                "match_history_count": played + i + 1,
                "trials_per_problem": budget,
            }
            _save(data)

        top = sorted(elo.ratings.items(), key=lambda kv: -kv[1])[:3]
        print("   top: " + ", ".join(f"{n} {r:.0f}" for n, r in top), flush=True)
    print(f"\nrecord at {OUT.relative_to(REPO_ROOT)}", flush=True)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", default=",".join(str(d) for d in DEFAULT_DIMS))
    ap.add_argument("--problems", type=int, default=10)
    ap.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Fixed budget. Default scales with dimension: max(100, 10*n_dim).",
    )
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument(
        "--suite",
        choices=("physics", "smooth"),
        default="physics",
        help="physics: the engineering demos, at the dimension each problem actually has. "
        "smooth: randomly morphed analytic surfaces. Physics is the default because "
        "smooth surfaces reward local search and rank optimizers accordingly.",
    )
    a = ap.parse_args()
    random.seed(a.seed)
    record(
        [int(d) for d in a.dims.split(",") if d.strip()],
        a.problems,
        a.trials,
        a.seed,
        a.suite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
