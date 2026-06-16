"""
Disguised-benchmark scoring driver — a fitness function for goal-based
development of optimization algorithms.

Score optimizers across the *disguised* example-application suite: every problem
is expanded into several instances by a seeded cube->cube diffeomorphism
(`example_demos.disguised_demos`), so the optimum sits somewhere different on
each instance and an algorithm cannot win by memorising locations — it has to
search.

Because the problems live on wildly different scales (a Δv of 0.19, a gearbox
weight of ~3000, a cluster energy of -9), raw best-values can't be averaged
across problems. We score scale-free, per instance:
  - rank the optimizers (1 = best best-value on that instance), then
  - average each optimizer's rank over all instances (lower = better), and
  - report a win rate (fraction of instances it ranks first).
Mean rank across the disguised suite is the single fitness an algorithm-
development loop can maximise (minimise) for a candidate optimizer.

    python papers/dfo_recommender/disguise_bench.py            # default panel
    python papers/dfo_recommender/disguise_bench.py --quick    # fast smoke
    python papers/dfo_recommender/disguise_bench.py --optimizers NelderMead,CMAEvolutionStrategy,DifferentialEvolution
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from example_demos import DEMOS, disguised_demos  # noqa: E402

from humpday.optimizers.alloptimizers import (  # noqa: E402
    PURE_OPTIMIZERS,
    pure_optimize,
)

# A representative cross-section (one local, trust-region, several population,
# Bayesian, annealing) — the default panel to rank a candidate against.
DEFAULT_PANEL = [
    "NelderMead",
    "Powell",
    "PRIMA_BOBYQA",
    "DifferentialEvolution",
    "ParticleSwarm",
    "CMAEvolutionStrategy",
    "BayesianOpt",
    "SimulatedAnnealing",
]

INF = float("inf")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass


def _best_value(algo: str, demo, n_trials: int, run_seed: int) -> float:
    """Best objective `algo` finds on `demo` in n_trials (lower = better)."""
    _set_seed(run_seed)  # reproducible, and seeds any stochastic objective
    try:
        f_best, _ = pure_optimize(demo.objective, algo, n_trials, demo.n_dim)
        return float(f_best)
    except Exception as e:  # noqa: BLE001
        print(f"    ! {algo} on {demo.name}: {e}", flush=True)
        return INF


def benchmark(optimizers, demos=None, seeds=(0, 1, 2), n_trials=120, base_demos=None):
    """Run every optimizer on every disguised instance.

    Returns {instance_name: {optimizer: best_value}}."""
    base = DEMOS if base_demos is None else base_demos
    instances = disguised_demos(base, seeds=tuple(seeds))
    results: dict[str, dict[str, float]] = {}
    for i, inst in enumerate(instances):
        row: dict[str, float] = {}
        for algo in optimizers:
            row[algo] = _best_value(algo, inst, n_trials, run_seed=1000 + i)
        results[inst.name] = row
        best = min(row.values())
        winner = min(row, key=row.get)
        print(
            f"  [{i + 1}/{len(instances)}] {inst.name:28s} best={best:.4g} ({winner})",
            flush=True,
        )
    return results


def leaderboard(results, optimizers):
    """Scale-free aggregation: per-instance ranks -> mean rank + win rate."""
    ranks = {a: [] for a in optimizers}
    wins = dict.fromkeys(optimizers, 0)
    for _, row in results.items():
        order = sorted(optimizers, key=lambda a: row.get(a, INF))
        # competition ranking with ties shared
        last_val, last_rank = None, 0
        for pos, a in enumerate(order, start=1):
            v = row.get(a, INF)
            r = pos if v != last_val else last_rank
            ranks[a].append(r)
            last_val, last_rank = v, r
        best = min(row.values()) if row else INF
        for a in optimizers:
            if row.get(a, INF) == best:
                wins[a] += 1
    n = max(1, len(results))
    table = [(a, mean(ranks[a]), wins[a] / n) for a in optimizers]
    table.sort(key=lambda t: t[1])
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--optimizers",
        default=",".join(DEFAULT_PANEL),
        help="comma-separated optimizer names (or 'all')",
    )
    ap.add_argument("--seeds", default="0,1,2", help="disguise seeds (comma-separated)")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument(
        "--demos", default="", help="comma-separated subset of base demo names"
    )
    ap.add_argument(
        "--quick", action="store_true", help="fast smoke: 6 demos, 2 seeds, 40 trials"
    )
    args = ap.parse_args()

    if args.optimizers == "all":
        optimizers = list(PURE_OPTIMIZERS.keys())
    else:
        optimizers = [a.strip() for a in args.optimizers.split(",") if a.strip()]
    optimizers = [a for a in optimizers if a in PURE_OPTIMIZERS]

    base = DEMOS
    if args.demos:
        wanted = {d.strip() for d in args.demos.split(",")}
        base = [d for d in DEMOS if d.name in wanted]
    seeds = tuple(int(s) for s in args.seeds.split(","))
    trials = args.trials

    if args.quick:
        base = base[:6]
        seeds = (0, 1)
        trials = 40

    print(
        f"Disguised benchmark: {len(base)} base demos x {len(seeds)} seeds "
        f"= {len(base) * len(seeds)} instances, {len(optimizers)} optimizers, "
        f"{trials} trials each\n"
    )
    results = benchmark(optimizers, base_demos=base, seeds=seeds, n_trials=trials)

    print(
        "\n=== Leaderboard (mean rank across disguised instances; lower = better) ==="
    )
    print(f"{'optimizer':<24}  {'mean rank':>9}  {'win rate':>8}")
    print("-" * 48)
    for a, mr, wr in leaderboard(results, optimizers):
        print(f"  {a:<22}  {mr:>9.3f}  {wr:>7.1%}")
    print(
        f"\n{len(results)} instances scored. Mean rank is the scale-free fitness "
        "an\nalgorithm-development loop maximises for a candidate optimizer."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
