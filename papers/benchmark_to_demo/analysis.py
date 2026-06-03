"""
Analysis backing the benchmark-to-demonstration transfer paper.

For each demo we have ported into `demos.py`:

  1. Run every eligible HumpDay algorithm on it with N_SEEDS seeds at the
     demo's suggested_n_trials. Record median best-value per (demo, algo).
  2. Compute the per-demo Borda rank for every algorithm.
  3. Compute Spearman ρ between (per-demo Borda) and (per-algo Borda from
     the benchmark recommendation grid at the matching (n_dim, n_trials)
     cell).
  4. Aggregate: does the grid's algorithm ranking predict the demo's?

Usage:
    python papers/benchmark_to_demo/analysis.py
    python papers/benchmark_to_demo/analysis.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

PAPER_DIR = Path(__file__).parent
REPO_ROOT = PAPER_DIR.parent.parent
GRID_PATH = REPO_ROOT / "benchmarks" / "recommendation_grid.json"
RESULTS_PATH = PAPER_DIR / "demo_results.json"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))

import numpy as np  # noqa: E402
from demos import DEMOS  # noqa: E402

from humpday import eligibility as E  # noqa: E402
from humpday.optimizers.alloptimizers import (  # noqa: E402
    PURE_OPTIMIZERS,
    pure_optimize,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_demos(n_seeds: int = 3) -> dict:
    """For every (demo, eligible algo, seed) tuple, run the optimizer once
    and record best_value + wall-clock. Returns the raw results dict."""
    results: dict[str, dict[str, dict[str, list[float]]]] = {}
    for demo in DEMOS:
        print(
            f"\n=== {demo.name}  (n_dim={demo.n_dim}, n_trials={demo.suggested_n_trials}) ==="
        )
        eligible = E.eligible(
            PURE_OPTIMIZERS.keys(),
            n_dim=demo.n_dim,
            n_trials=demo.suggested_n_trials,
            eval_time=None,
        )
        print(f"  eligible algorithms: {len(eligible)}")
        results[demo.name] = {
            "n_dim": demo.n_dim,
            "n_trials": demo.suggested_n_trials,
            "runs": {},
        }
        for algo in eligible:
            best_values: list[float] = []
            walls: list[float] = []
            for seed in range(n_seeds):
                _set_seed(seed)
                t0 = time.perf_counter()
                try:
                    v, _ = pure_optimize(
                        demo.objective,
                        algo,
                        demo.suggested_n_trials,
                        demo.n_dim,
                    )
                    best_values.append(float(v))
                    walls.append(time.perf_counter() - t0)
                except Exception as e:  # noqa: BLE001
                    print(f"    ! {algo} seed {seed}: {e}")
                    best_values.append(float("inf"))
                    walls.append(time.perf_counter() - t0)
            results[demo.name]["runs"][algo] = {
                "best_values": best_values,
                "walls": walls,
            }
            valid = [v for v in best_values if v != float("inf")]
            med = median(valid) if valid else float("inf")
            print(f"    {algo:24s}  median best = {med:>12.4g}")
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH.relative_to(REPO_ROOT)}")
    return results


def _per_demo_borda(results: dict) -> dict[str, dict[str, float]]:
    """For each demo, compute Borda mean-rank across seeds for every algo
    that ran. Lower is better (rank 1 = best on that seed)."""
    out: dict[str, dict[str, float]] = {}
    for demo_name, data in results.items():
        runs = data["runs"]
        algos = list(runs.keys())
        n_seeds = max(len(runs[a]["best_values"]) for a in algos)
        ranks: dict[str, list[int]] = {a: [] for a in algos}
        for seed in range(n_seeds):
            scored = [
                (runs[a]["best_values"][seed], a)
                for a in algos
                if seed < len(runs[a]["best_values"])
                and runs[a]["best_values"][seed] != float("inf")
            ]
            scored.sort()
            for i, (_, a) in enumerate(scored, start=1):
                ranks[a].append(i)
        out[demo_name] = {
            a: (mean(rs) if rs else float("inf")) for a, rs in ranks.items()
        }
    return out


def _grid_borda_at(cell: dict, algos: list[str]) -> dict[str, float]:
    """Return per-algorithm Borda from a grid cell, filtered to `algos`."""
    return {a: cell.get(a, {}).get("borda_score", float("inf")) for a in algos}


def _snap(grid_cells: dict, n_dim: int, n_trials: int) -> dict | None:
    """Same snap-to-nearest-cell logic as humpday.eligibility."""
    candidates = []
    for key in grid_cells:
        d, t = key.split("/")
        candidates.append((int(d), int(t), key))
    feasible = [(d, t, k) for d, t, k in candidates if d <= n_dim and t <= n_trials]
    if feasible:
        feasible.sort(key=lambda x: (abs(x[0] - n_dim), abs(x[1] - n_trials)))
        return grid_cells[feasible[0][2]]
    candidates.sort(key=lambda x: abs(x[0] - n_dim) + abs(x[1] - n_trials))
    return grid_cells[candidates[0][2]] if candidates else None


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank-based correlation. Handles ties via average ranking."""
    assert len(xs) == len(ys)
    n = len(xs)
    if n < 2:
        return float("nan")

    def average_ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=vals.__getitem__)
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = average_ranks(xs)
    ry = average_ranks(ys)
    mx = mean(rx)
    my = mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def transfer_table(results: dict) -> None:
    """Table 1: per-demo Spearman ρ between demo Borda and grid Borda."""
    grid = json.loads(GRID_PATH.read_text())
    cells = grid["cells"]
    demo_borda = _per_demo_borda(results)

    print("\n# Table 1: Benchmark→demo Borda correlation\n")
    print(
        f"{'demo':>20s}  {'n_dim':>6s}  {'n_trials':>10s}  "
        f"{'algos':>6s}  {'Spearman ρ':>12s}  {'p < ?':>8s}"
    )
    print(
        f"{'-' * 20:>20s}  {'-' * 6:>6s}  {'-' * 10:>10s}  "
        f"{'-' * 6:>6s}  {'-' * 12:>12s}  {'-' * 8:>8s}"
    )

    all_demo_ranks: list[float] = []
    all_grid_ranks: list[float] = []

    for demo_name, data in results.items():
        n_dim = data["n_dim"]
        n_trials = data["n_trials"]
        d_borda = demo_borda[demo_name]
        cell = _snap(cells, n_dim, n_trials)
        if cell is None:
            print(f"{demo_name:>20s}  no grid cell")
            continue
        algos_common = [
            a
            for a in d_borda
            if d_borda[a] != float("inf")
            and a in cell
            and cell[a].get("borda_score", float("inf")) != float("inf")
        ]
        if len(algos_common) < 3:
            continue
        xs = [d_borda[a] for a in algos_common]
        ys = [cell[a]["borda_score"] for a in algos_common]
        rho = spearman(xs, ys)
        # Rough significance: rho needs to exceed ~2/sqrt(n) for p<0.05.
        threshold = 2 / (len(algos_common) ** 0.5)
        sig = "0.05" if abs(rho) > threshold else "—"
        print(
            f"{demo_name:>20s}  {n_dim:>6d}  {n_trials:>10d}  "
            f"{len(algos_common):>6d}  {rho:>12.3f}  {sig:>8s}"
        )

        all_demo_ranks.extend(xs)
        all_grid_ranks.extend(ys)

    if len(all_demo_ranks) >= 4:
        agg = spearman(all_demo_ranks, all_grid_ranks)
        print(
            f"\n  Pooled across all demos: ρ = {agg:.3f}  (n = {len(all_demo_ranks)})"
        )


def per_demo_top_pick_table(results: dict) -> None:
    """Table 2: best algorithm per demo vs recommendation grid's pick."""
    grid_path = GRID_PATH
    print("\n# Table 2: Per-demo winner vs grid recommendation\n")
    print(
        f"{'demo':>20s}  {'demo winner':>24s}  {'grid recommends':>24s}  {'agree?':>7s}"
    )
    print(f"{'-' * 20:>20s}  {'-' * 24:>24s}  {'-' * 24:>24s}  {'-' * 7:>7s}")
    demo_borda = _per_demo_borda(results)
    for demo_name, data in results.items():
        d_borda = demo_borda[demo_name]
        if not d_borda:
            continue
        finite = {a: b for a, b in d_borda.items() if b != float("inf")}
        if not finite:
            continue
        winner = min(finite, key=finite.get)
        rec = E.recommend(
            n_dim=data["n_dim"],
            n_trials=data["n_trials"],
            eval_time=None,  # we don't know — use quality-only
            grid_path=grid_path,
        )
        match = "✓" if rec == winner else "—"
        print(f"{demo_name:>20s}  {winner:>24s}  {rec:>24s}  {match:>7s}")


SECTIONS = {
    "run": lambda _: run_demos(),
    "transfer": transfer_table,
    "top_pick": per_demo_top_pick_table,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--section", choices=list(SECTIONS) + ["all"], default="all")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force a fresh run instead of loading demo_results.json",
    )
    args = parser.parse_args()

    if args.section == "run" or args.rerun or not RESULTS_PATH.exists():
        results = run_demos(n_seeds=args.seeds)
    else:
        results = json.loads(RESULTS_PATH.read_text())

    if args.section in ("all", "transfer"):
        transfer_table(results)
    if args.section in ("all", "top_pick"):
        per_demo_top_pick_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
