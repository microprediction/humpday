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
    return [n for n, e in cell.items()
            if e.get("runs") and not e.get("skipped_too_slow")]


def borda_oracle(cell: dict) -> str | None:
    scored = [(e.get("borda_score", float("inf")), n) for n, e in cell.items()
              if not e.get("skipped_too_slow")
              and e.get("borda_score", float("inf")) != float("inf")]
    if not scored:
        return None
    scored.sort()
    return scored[0][1]


def cost_aware_oracle(cell: dict, n_trials: int, eval_time: float,
                      overhead_budget: float = 1.0) -> str | None:
    """Best Borda among algorithms whose recorded mean_wall is within
    `overhead_budget` × (n_trials × eval_time). At expensive eval_times this
    converges to `borda_oracle`; at cheap eval_times it strips out heavy
    algorithms before ranking."""
    user_baseline = n_trials * eval_time
    best = None
    for a, e in cell.items():
        if e.get("skipped_too_slow"):
            continue
        if user_baseline > 0 and e.get("mean_wall", 0.0) > overhead_budget * user_baseline:
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
                "fixed_best": fixed_best if fixed_best in eligible_names else "RandomSearch",
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

    print(f"# Table 1: Recommender vs oracle vs baselines\n")
    print(f"Fixed-best baseline algorithm: {fixed_best}\n")
    print(f"{'strategy':>14s}  {'match':>8s}  {'mean reg':>10s}  {'med reg':>10s}  {'med ratio':>12s}  {'n':>4s}")
    print(f"{'-' * 14:>14s}  {'-' * 8:>8s}  {'-' * 10:>10s}  {'-' * 10:>10s}  {'-' * 12:>12s}  {'-' * 4:>4s}")
    for name in ("recommender", "fixed_best", "suggest_pure", "random"):
        r = results[name]
        if r["n"] == 0:
            continue
        print(f"{name:>14s}  {r['match'] / r['n']:>8.1%}  "
              f"{mean(r['regret_borda']):>10.2f}  "
              f"{median(r['regret_borda']):>10.2f}  "
              f"{median(r['ratio_best']):>12.2e}  "
              f"{r['n']:>4d}")


# -----------------------------------------------------------------------------
# § 3.2 Wall-clock-aware oracle — the 100µs gap finding
# -----------------------------------------------------------------------------

def wallclock_table(grid: dict, overhead_budget: float = 1.0) -> None:
    """Table 2: recommender matches naive vs cost-aware oracle per eval_time."""
    cells = grid["cells"]

    print(f"\n# Table 2: Recommender vs naive oracle vs wall-clock-aware oracle")
    print(f"# (overhead_budget = {overhead_budget}× user wall-clock)\n")
    print(f"{'eval_time':>10s}  {'naive match':>12s}  {'cost-aware':>12s}  "
          f"{'oracles ≠':>10s}  {'n':>4s}")
    print(f"{'-' * 10:>10s}  {'-' * 12:>12s}  {'-' * 12:>12s}  "
          f"{'-' * 10:>10s}  {'-' * 4:>4s}")

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
        print(f"{label:>10s}  {naive / n:>12.0%}  {cost / n:>12.0%}  "
              f"{diff / n:>10.0%}  {n:>4d}")


# -----------------------------------------------------------------------------
# § 3.3 Leave-one-objective-out cross-validation
# -----------------------------------------------------------------------------

def borda_excluding(cell: dict, objectives: list[str], seeds: list[int],
                    excluded: str) -> dict[str, float]:
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


def median_best_on_objective(cell: dict, algo: str, objective: str,
                             seeds: list[int]) -> float:
    runs = cell.get(algo, {}).get("runs", {})
    vals = [runs[f"{objective}/{s}"]["best"]
            for s in seeds
            if f"{objective}/{s}" in runs
            and runs[f"{objective}/{s}"]["best"] != float("inf")]
    return median(vals) if vals else float("inf")


def loo_table(grid: dict) -> None:
    """Table 3: leave-one-objective-out cross-validation of the recommender."""
    cells = grid["cells"]
    objectives = grid["meta"]["objectives"]
    seeds = list(range(grid["meta"]["n_seeds"]))

    print(f"\n# Table 3: Leave-one-objective-out reproducibility\n")
    print(f"{'held-out objective':>22s}  {'LOO match':>10s}  "
          f"{'med ratio':>10s}  {'mean ratio':>12s}  {'n':>4s}")
    print(f"{'-' * 22:>22s}  {'-' * 10:>10s}  {'-' * 10:>10s}  "
          f"{'-' * 12:>12s}  {'-' * 4:>4s}")

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
            scored = [(median_best_on_objective(cell, a, held, seeds), a)
                      for a in algos_with_runs(cell)]
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
        print(f"{held:>22s}  {matches / n:>10.0%}  {median(ratios):>10.2e}  "
              f"{mean(ratios):>12.2e}  {n:>4d}")
    print(f"{'OVERALL':>22s}  {all_match / all_n:>10.0%}  "
          f"{median(all_ratios):>10.2e}  {mean(all_ratios):>12.2e}  {all_n:>4d}")


# -----------------------------------------------------------------------------
# § 4 Soft cost-weighted Borda
# -----------------------------------------------------------------------------
# Motivated by the 100µs gap (§ 3.2): the current recommender applies the
# overhead tier as a hard binary filter, which means at the tier-2
# threshold (100µs) it admits PRIMA_BOBYQA / PRIMA_NEWUOA and picks one by
# Borda even when a lighter tier-1 algorithm would deliver comparable
# solution quality with much smaller wall-clock cost.
#
# The soft variant replaces the hard tier filter with a continuous penalty:
#
#   adjusted_borda(a) = borda(a) + λ · log(1 + mean_wall(a) / user_baseline)
#
# where user_baseline = n_trials · eval_time is the wall-clock the user
# expects to spend if the algorithm itself were free. λ=0 reproduces the
# current recommender; λ→∞ picks the algorithm with the smallest overhead
# regardless of Borda quality. The dimensional-cap and minimum-trials
# filters still apply on top of the soft penalty.

import math


def soft_cost_pick(cell: dict, n_dim: int, n_trials: int, eval_time: float,
                   lam: float) -> str | None:
    """Pick the algorithm with the lowest adjusted_borda among algorithms
    that pass the dim cap and min-trials filters at this cell. Tier filter
    is intentionally not applied — that's what the soft penalty replaces."""
    user_baseline = max(n_trials * eval_time, 1e-15)
    best = None
    for a, e in cell.items():
        if e.get("skipped_too_slow"):
            continue
        if not E.passes_dim(a, n_dim) or not E.passes_trials(a, n_dim, n_trials):
            continue
        borda = e.get("borda_score", float("inf"))
        if borda == float("inf"):
            continue
        overhead = e.get("mean_wall", 0.0)
        penalty = lam * math.log(1.0 + overhead / user_baseline)
        adjusted = borda + penalty
        if best is None or adjusted < best[0]:
            best = (adjusted, a)
    return best[1] if best else None


LAMBDA_SWEEP = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def soft_borda_sweep(grid: dict, overhead_budget: float = 1.0) -> None:
    """Table 4: λ sweep of the soft cost-weighted Borda recommender,
    evaluated against both naive and cost-aware oracles at each eval_time."""
    cells = grid["cells"]
    print(f"\n# Table 4: λ sweep for soft cost-weighted Borda")
    print(f"# (penalty = λ · log(1 + mean_wall / (n_trials · eval_time)))\n")
    print(f"# Each cell shows (naive_match% / cost_aware_match%) at (eval_time, λ).")
    print(f"# λ=0 reproduces the current recommender. "
          f"Cost-aware oracle uses overhead_budget={overhead_budget}.\n")

    headers = ["eval_time"] + [f"λ={lam}" for lam in LAMBDA_SWEEP]
    widths = [10] + [12 for _ in LAMBDA_SWEEP]
    print("  " + "  ".join(f"{h:>{w}s}" for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))

    per_lambda_cost_match: dict[float, list[float]] = {lam: [] for lam in LAMBDA_SWEEP}

    for label, et in EVAL_TIMES:
        row = [label]
        for lam in LAMBDA_SWEEP:
            naive_m = cost_m = n = 0
            for ck, cell in cells.items():
                n_dim, n_trials = map(int, ck.split("/"))
                naive_o = borda_oracle(cell)
                cost_o = cost_aware_oracle(cell, n_trials, et, overhead_budget)
                if naive_o is None or cost_o is None:
                    continue
                pick = soft_cost_pick(cell, n_dim, n_trials, et, lam)
                if pick is None:
                    continue
                n += 1
                if pick == naive_o:
                    naive_m += 1
                if pick == cost_o:
                    cost_m += 1
            if n == 0:
                row.append("—")
                continue
            row.append(f"{naive_m / n:>4.0%} / {cost_m / n:>4.0%}")
            per_lambda_cost_match[lam].append(cost_m / n)
        print("  " + "  ".join(f"{c:>{w}s}" for c, w in zip(row, widths)))

    # λ* — best mean cost-aware match across eval_time buckets.
    print()
    print(f"  {'λ':>8s}  {'mean cost-aware match':>22s}")
    print(f"  {'-' * 8:>8s}  {'-' * 22:>22s}")
    best_lam: float | None = None
    best_mean = -1.0
    for lam in LAMBDA_SWEEP:
        if not per_lambda_cost_match[lam]:
            continue
        m = mean(per_lambda_cost_match[lam])
        marker = "  ← best" if m > best_mean else ""
        if m > best_mean:
            best_mean = m
            best_lam = lam
        print(f"  {lam:>8.1f}  {m:>22.1%}{marker}")
    print(f"\n  λ* = {best_lam} (mean cost-aware match {best_mean:.1%})")

    # Example switches at λ*.
    if best_lam and best_lam > 0:
        print(f"\n# Example switches at λ* = {best_lam} (eval_time=10µs):\n")
        print(f"  {'cell':>10s}  {'λ=0 pick':>22s}  {'λ=λ* pick':>22s}  "
              f"{'cost-aware oracle':>22s}")
        shown = 0
        for ck, cell in cells.items():
            if shown >= 8:
                break
            n_dim, n_trials = map(int, ck.split("/"))
            old = soft_cost_pick(cell, n_dim, n_trials, 1e-5, 0.0)
            new = soft_cost_pick(cell, n_dim, n_trials, 1e-5, best_lam)
            if old == new or new is None:
                continue
            co = cost_aware_oracle(cell, n_trials, 1e-5, overhead_budget)
            if co is None:
                continue
            print(f"  {ck:>10s}  {str(old):>22s}  {str(new):>22s}  {str(co):>22s}")
            shown += 1


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

SECTIONS = {
    "oracle_gap": oracle_gap_table,
    "wallclock": wallclock_table,
    "loo": loo_table,
    "soft_borda": soft_borda_sweep,
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
