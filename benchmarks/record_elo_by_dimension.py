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
DEFAULT_DIMS = (2, 5, 10, 25)  # 50 and 100 are affordable but slow; ask for them by name


def stochastic_generator(n_dim: int, seed: int):
    """Yield freshly morphed objectives. Each problem is a new random surface."""
    from humpday.objectives.stochastic_surfaces import create_fair_benchmark_run

    rnd = random.Random(seed)
    while True:
        # The generator chatters on construction; it is not our stdout to spend.
        with contextlib.redirect_stdout(io.StringIO()):
            produced = create_fair_benchmark_run(n_functions=6, seed=rnd.randrange(2**31))
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
    return {"by_dimension": {}, "generator": "stochastic_surfaces.create_fair_benchmark_run"}


def _save(data: dict) -> None:
    data["recorded_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(OUT)  # atomic: a crash mid-write cannot leave a truncated record


def record(dims, n_problems: int, trials: int, seed: int) -> dict:
    """Play `n_problems` more matches per dimension, checkpointing after every one.

    A high-dimensional bucket runs for minutes, so nothing here is allowed to depend on reaching
    the end: the record is written after each individual problem, through a temporary file so an
    interrupted write cannot truncate it, and a failure in one dimension does not cost the others.
    """
    data = load()
    for n_dim in dims:
        key = str(n_dim)
        prior = data["by_dimension"].get(key, {})
        elo = EloRatingSystem()
        for name, rating in prior.get("ratings", {}).items():
            elo.ratings[name] = rating  # resume rather than restart
        played = prior.get("match_history_count", 0)
        budget = budget_for(n_dim, trials)
        print(
            f"n_dim={n_dim}: {n_problems} problems x {budget} trials (resuming from {played})",
            flush=True,
        )

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
                print(f"   problem {i + 1} failed ({type(exc).__name__}: {exc}); "
                      f"keeping {played + i} matches", flush=True)
                break
            data["by_dimension"][key] = {
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
        "--trials", type=int, default=None,
        help="Fixed budget. Default scales with dimension: max(100, 10*n_dim).",
    )
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args()
    random.seed(a.seed)
    record([int(d) for d in a.dims.split(",") if d.strip()], a.problems, a.trials, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
