"""
Record current Elo ratings for all 22 HumpDay algorithms.

Usage:
    python benchmarks/record_elo.py

Runs `run_algorithm_tournament` across a mix of objective generators
(sphere variants + Rosenbrock variants) in 2 dimensions with a small
budget per algorithm, then writes the ratings to
`benchmarks/elo_ratings.json` for the perf column of GOALS.md.

Re-run this script whenever an algorithm's logic changes
substantially (a port, a fix, a tuning change) so the recorded
ratings stay current.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humpday.optimizers.adaptive_optimizer import (  # noqa: E402
    EloRatingSystem,
    rosenbrock_variants_generator,
    run_algorithm_tournament,
    sphere_variants_generator,
)


def main() -> int:
    # Determinism: seed both Python's random and (if numpy is around)
    # numpy's. We want the recorded benchmark to be reproducible.
    random.seed(20260526)
    try:
        import numpy as np
        np.random.seed(20260526)
    except ImportError:
        pass

    n_dim = 2
    trials_per_problem = 100
    n_problems_per_family = 15

    elo = EloRatingSystem()

    print(f"Sphere family — {n_problems_per_family} problems, {trials_per_problem} trials each, n_dim={n_dim}")
    elo = run_algorithm_tournament(
        sphere_variants_generator(n_dim=n_dim),
        trials_per_problem=trials_per_problem,
        n_problems=n_problems_per_family,
        n_dim=n_dim,
        elo_system=elo,
    )

    print(f"\nRosenbrock family — {n_problems_per_family} problems, {trials_per_problem} trials each")
    elo = run_algorithm_tournament(
        rosenbrock_variants_generator(n_dim=n_dim),
        trials_per_problem=trials_per_problem,
        n_problems=n_problems_per_family,
        n_dim=n_dim,
        elo_system=elo,
    )

    # Output: leaderboard + JSON
    print("\n" + "=" * 60)
    print("Final Elo ratings (higher is better)")
    print("=" * 60)
    for name, rating in elo.get_top_algorithms(n=22):
        print(f"  {name:28s}  {rating:7.1f}")

    out = REPO_ROOT / "benchmarks" / "elo_ratings.json"
    out.parent.mkdir(exist_ok=True)
    data = {
        "ratings": elo.ratings,
        "match_history_count": len(elo.match_history),
        "initial_rating": elo.initial_rating,
        "k_factor": elo.k_factor,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "n_dim": n_dim,
            "trials_per_problem": trials_per_problem,
            "n_problems_per_family": n_problems_per_family,
            "families": ["sphere", "rosenbrock"],
            "seed": 20260526,
        },
    }
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
