"""
Analysis backing the "Default algorithm selection for derivative-free
black-box optimization" paper. Single script that regenerates every
table and figure from `benchmarks/recommendation_grid.json` so the
paper's claims stay in sync with the underlying data.

Usage:
    python paper/dfo_recommender/analysis.py
    python paper/dfo_recommender/analysis.py --section oracle_gap
    python paper/dfo_recommender/analysis.py --section wallclock
    python paper/dfo_recommender/analysis.py --section loo

The output goes to:
  * stdout (formatted tables for inclusion in the LaTeX as verbatim)
  * paper/dfo_recommender/figures/*.{png,pdf} for figures
  * paper/dfo_recommender/tables/*.tex for the actual LaTeX tables
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

PAPER_DIR = Path(__file__).parent
REPO_ROOT = PAPER_DIR.parent.parent
GRID_PATH = REPO_ROOT / "benchmarks" / "recommendation_grid.json"
FIGURES = PAPER_DIR / "figures"
TABLES = PAPER_DIR / "tables"

sys.path.insert(0, str(REPO_ROOT))
from humpday import eligibility as E  # noqa: E402
from humpday.optimizers.alloptimizers import suggest_pure  # noqa: E402

EVAL_TIMES = [
    ("1µs", 1e-6),
    ("10µs", 1e-5),
    ("100µs", 1e-4),
    ("1ms", 1e-3),
    ("10ms", 1e-2),
    ("1s", 1.0),
]


# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------


def load_grid() -> dict:
    return json.loads(GRID_PATH.read_text())


def algos_with_runs(cell: dict) -> list[str]:
    return [
        n for n, e in cell.items() if e.get("runs") and not e.get("skipped_too_slow")
    ]


def borda_oracle(cell: dict) -> str | None:
    scored = [
        (e.get("borda_score", float("inf")), n)
        for n, e in cell.items()
        if not e.get("skipped_too_slow")
        and e.get("borda_score", float("inf")) != float("inf")
    ]
    if not scored:
        return None
    scored.sort()
    return scored[0][1]


def cost_aware_oracle(
    cell: dict, n_trials: int, eval_time: float, overhead_budget: float = 1.0
) -> str | None:
    """Best Borda among algorithms whose recorded mean_wall is within
    `overhead_budget` × (n_trials × eval_time). At expensive eval_times this
    converges to `borda_oracle`; at cheap eval_times it strips out heavy
    algorithms before ranking."""
    user_baseline = n_trials * eval_time
    best = None
    for a, e in cell.items():
        if e.get("skipped_too_slow"):
            continue
        if (
            user_baseline > 0
            and e.get("mean_wall", 0.0) > overhead_budget * user_baseline
        ):
            continue
        borda = e.get("borda_score", float("inf"))
        if borda == float("inf"):
            continue
        if best is None or borda < best[0]:
            best = (borda, a)
    return best[1] if best else None


# -----------------------------------------------------------------------------
# § 3.1 Oracle gap — recommender vs oracle vs baselines
# -----------------------------------------------------------------------------


def oracle_gap_table(grid: dict) -> None:
    """Table 1: recommender vs naive oracle vs three baselines."""
    cells = grid["cells"]

    # Fixed-best baseline: the single algorithm with the best average Borda
    # across cells where it has data.
    algo_bordas: dict[str, list[float]] = defaultdict(list)
    for cell in cells.values():
        for n, e in cell.items():
            s = e.get("borda_score", float("inf"))
            if s != float("inf"):
                algo_bordas[n].append(s)
    fixed_best = min(algo_bordas, key=lambda n: mean(algo_bordas[n]))

    results = {
        name: {"match": 0, "regret_borda": [], "ratio_best": [], "n": 0}
        for name in ("recommender", "fixed_best", "suggest_pure", "random")
    }
    for ck, cell in cells.items():
        n_dim, n_trials = map(int, ck.split("/"))
        o = borda_oracle(cell)
        if o is None:
            continue
        o_borda = cell[o]["borda_score"]
        o_med = cell[o].get("median_best", float("inf"))

        for _, et in EVAL_TIMES:
            eligible_names = E.eligible(E.TIER.keys(), n_dim, n_trials, et)
            if not eligible_names:
                continue
            picks = {
                "recommender": E.recommend(n_dim, n_trials, et, grid_path=GRID_PATH),
                "fixed_best": fixed_best
                if fixed_best in eligible_names
                else "RandomSearch",
                "suggest_pure": next(
                    (a for a in suggest_pure(n_dim, n_trials) if a in eligible_names),
                    "RandomSearch",
                ),
                "random": eligible_names[len(eligible_names) // 2],
            }
            for name, pick in picks.items():
                r = results[name]
                r["n"] += 1
                r["match"] += int(pick == o)
                pe = cell.get(pick, {})
                if pe.get("borda_score", float("inf")) != float("inf"):
                    r["regret_borda"].append(pe["borda_score"] - o_borda)
                p_med = pe.get("median_best", float("inf"))
                if p_med != float("inf") and o_med > 0:
                    r["ratio_best"].append(p_med / o_med)

    print("# Table 1: Recommender vs oracle vs baselines\n")
    print(f"Fixed-best baseline algorithm: {fixed_best}\n")
    print(
        f"{'strategy':>14s}  {'match':>8s}  {'mean reg':>10s}  {'med reg':>10s}  {'med ratio':>12s}  {'n':>4s}"
    )
    print(
        f"{'-' * 14:>14s}  {'-' * 8:>8s}  {'-' * 10:>10s}  {'-' * 10:>10s}  {'-' * 12:>12s}  {'-' * 4:>4s}"
    )
    for name in ("recommender", "fixed_best", "suggest_pure", "random"):
        r = results[name]
        if r["n"] == 0:
            continue
        print(
            f"{name:>14s}  {r['match'] / r['n']:>8.1%}  "
            f"{mean(r['regret_borda']):>10.2f}  "
            f"{median(r['regret_borda']):>10.2f}  "
            f"{median(r['ratio_best']):>12.2e}  "
            f"{r['n']:>4d}"
        )


# -----------------------------------------------------------------------------
# § 3.2 Wall-clock-aware oracle — the 100µs gap finding
# -----------------------------------------------------------------------------


def wallclock_table(grid: dict, overhead_budget: float = 1.0) -> None:
    """Table 2: recommender matches naive vs cost-aware oracle per eval_time."""
    cells = grid["cells"]

    print("\n# Table 2: Recommender vs naive oracle vs wall-clock-aware oracle")
    print(f"# (overhead_budget = {overhead_budget}× user wall-clock)\n")
    print(
        f"{'eval_time':>10s}  {'naive match':>12s}  {'cost-aware':>12s}  "
        f"{'oracles ≠':>10s}  {'n':>4s}"
    )
    print(
        f"{'-' * 10:>10s}  {'-' * 12:>12s}  {'-' * 12:>12s}  "
        f"{'-' * 10:>10s}  {'-' * 4:>4s}"
    )

    for label, et in EVAL_TIMES:
        naive = cost = diff = n = 0
        for ck, cell in cells.items():
            n_dim, n_trials = map(int, ck.split("/"))
            naive_o = borda_oracle(cell)
            cost_o = cost_aware_oracle(cell, n_trials, et, overhead_budget)
            if naive_o is None or cost_o is None:
                continue
            pick = E.recommend(n_dim, n_trials, et, grid_path=GRID_PATH)
            n += 1
            if pick == naive_o:
                naive += 1
            if pick == cost_o:
                cost += 1
            if cost_o != naive_o:
                diff += 1
        if n == 0:
            continue
        print(
            f"{label:>10s}  {naive / n:>12.0%}  {cost / n:>12.0%}  "
            f"{diff / n:>10.0%}  {n:>4d}"
        )


# -----------------------------------------------------------------------------
# § 3.3 Leave-one-objective-out cross-validation
# -----------------------------------------------------------------------------


def borda_excluding(
    cell: dict, objectives: list[str], seeds: list[int], excluded: str
) -> dict[str, float]:
    """Recompute per-algorithm Borda mean-rank using all (objective, seed)
    runs in `cell` except those where objective == excluded."""
    candidates = algos_with_runs(cell)
    ranks_by_algo: dict[str, list[int]] = {a: [] for a in candidates}
    for obj in objectives:
        if obj == excluded:
            continue
        for seed in seeds:
            run_key = f"{obj}/{seed}"
            scored: list[tuple[float, str]] = []
            for a in candidates:
                run = cell[a].get("runs", {}).get(run_key)
                if run is None or run["best"] == float("inf"):
                    continue
                scored.append((run["best"], a))
            if not scored:
                continue
            scored.sort()
            prev_val: float | None = None
            prev_rank = 0
            for i, (v, a) in enumerate(scored, start=1):
                if prev_val is not None and v == prev_val:
                    rank = prev_rank
                else:
                    rank = i
                    prev_val = v
                    prev_rank = i
                ranks_by_algo[a].append(rank)
    return {a: (mean(rs) if rs else float("inf")) for a, rs in ranks_by_algo.items()}


def median_best_on_objective(
    cell: dict, algo: str, objective: str, seeds: list[int]
) -> float:
    runs = cell.get(algo, {}).get("runs", {})
    vals = [
        runs[f"{objective}/{s}"]["best"]
        for s in seeds
        if f"{objective}/{s}" in runs
        and runs[f"{objective}/{s}"]["best"] != float("inf")
    ]
    return median(vals) if vals else float("inf")


def loo_table(grid: dict) -> None:
    """Table 3: leave-one-objective-out cross-validation of the recommender."""
    cells = grid["cells"]
    objectives = grid["meta"]["objectives"]
    seeds = list(range(grid["meta"]["n_seeds"]))

    print("\n# Table 3: Leave-one-objective-out reproducibility\n")
    print(
        f"{'held-out objective':>22s}  {'LOO match':>10s}  "
        f"{'med ratio':>10s}  {'mean ratio':>12s}  {'n':>4s}"
    )
    print(
        f"{'-' * 22:>22s}  {'-' * 10:>10s}  {'-' * 10:>10s}  "
        f"{'-' * 12:>12s}  {'-' * 4:>4s}"
    )

    all_match = 0
    all_n = 0
    all_ratios: list[float] = []

    for held in objectives:
        matches = 0
        ratios: list[float] = []
        n = 0
        for cell in cells.values():
            new_borda = borda_excluding(cell, objectives, seeds, held)
            if not new_borda or all(v == float("inf") for v in new_borda.values()):
                continue
            loo_pick = min(new_borda, key=new_borda.get)
            scored = [
                (median_best_on_objective(cell, a, held, seeds), a)
                for a in algos_with_runs(cell)
            ]
            scored = [(s, a) for s, a in scored if s != float("inf")]
            if not scored:
                continue
            scored.sort()
            oracle, oracle_med = scored[0][1], scored[0][0]
            loo_med = median_best_on_objective(cell, loo_pick, held, seeds)
            if loo_med == float("inf") or oracle_med == 0:
                continue
            n += 1
            if loo_pick == oracle:
                matches += 1
            ratios.append(loo_med / oracle_med)
        if not ratios:
            continue
        all_match += matches
        all_n += n
        all_ratios.extend(ratios)
        print(
            f"{held:>22s}  {matches / n:>10.0%}  {median(ratios):>10.2e}  "
            f"{mean(ratios):>12.2e}  {n:>4d}"
        )
    print(
        f"{'OVERALL':>22s}  {all_match / all_n:>10.0%}  "
        f"{median(all_ratios):>10.2e}  {mean(all_ratios):>12.2e}  {all_n:>4d}"
    )


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

SECTIONS = {
    "oracle_gap": oracle_gap_table,
    "wallclock": wallclock_table,
    "loo": loo_table,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=list(SECTIONS) + ["all"],
        default="all",
        help="Which paper section to regenerate. Default: all.",
    )
    args = parser.parse_args()

    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    grid = load_grid()
    sections = list(SECTIONS) if args.section == "all" else [args.section]
    for s in sections:
        SECTIONS[s](grid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
