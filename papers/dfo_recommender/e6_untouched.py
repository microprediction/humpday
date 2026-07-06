"""E6 — Strictly-untouched out-of-sample race.

Same protocol as e2_hardening.py, but on an EXPLICIT list of demos verified
(via git archaeology of the demo suite at every selection-era commit:
3e43e16, 9fb904b, aed5f02, HEAD) to have been used by NO selection or
evaluation step of any simplex/algo_dev experiment. This closes the
contamination found in E2: the suite grew between the warm run and the E2
run, so E2's runtime exclusion (computed against the newer list) let three
warm-selection demos (espresso_dialin, facility_location, gear_ratios) into
its "held-out" set.

Field: panel {NM, DE, CMA, ngCMA} + centroid + surrogate + unstructured.
Crash-safe per-instance; resume by re-running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
from e2_hardening import (  # noqa: E402
    INF,
    atomic_dump,
    load_centroid,
    load_surrogate,
    load_unstructured_best,
    run_discovered,
)
from example_demos import DEMOS, disguise_demo  # noqa: E402
from rankcorr import run_opt  # noqa: E402

# Verified untouched at 2026-07-04 (see docstring). 29 demos, dims 2-90.
UNTOUCHED = [
    "battery_dispatch_72d",
    "diet_problem",
    "economic_dispatch_valve",
    "ev_fleet_charging",
    "fir_filter_design",
    "free_kick",
    "groundwater_remediation",
    "heat_exchanger_network",
    "index_tracking",
    "inventory_policy",
    "lennard_jones_cluster_90d",
    "microgrid_dispatch",
    "multi_exponential_fit",
    "pool",
    "protein_fold_toy",
    "radiation_therapy",
    "reservoir_release",
    "revenue_pricing",
    "robot_arm",
    "rocket_landing",
    "rocket_landing_24d",
    "rocket_landing_36d",
    "rocket_landing_48d",
    "rocket_landing_60d",
    "satellite_phasing",
    "traffic_signal",
    "vaccine_allocation",
    "water_distribution",
    "wind_farm_40d",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="60,120,240,480")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/e6_untouched.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    budgets = [int(b) for b in a.budgets.split(",")]
    names = list(UNTOUCHED)
    if a.quick:
        seeds, budgets, names = [0], [120], names[:3]

    discovered = {"centroid": load_centroid(), "surrogate": load_surrogate()}
    try:
        uns, _ = load_unstructured_best()
        discovered["unstructured"] = uns
    except Exception as e:  # noqa: BLE001
        print(f"  (unstructured optimizer unavailable: {e})", flush=True)
    panel = ["NelderMead", "DifferentialEvolution", "CMAEvolutionStrategy", "ngCMA"]
    field = panel + list(discovered)

    by_name = {d.name: d for d in DEMOS}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"demos not found: {missing}")
    held = sorted((by_name[n] for n in names), key=lambda d: d.n_dim)
    print(
        f"untouched demos ({len(held)}, dims {held[0].n_dim}-{held[-1].n_dim})\n",
        flush=True,
    )

    rows = []
    done = set()
    if os.path.exists(a.out):
        try:
            rows = json.load(open(a.out)).get("rows", [])
            done = {(r["budget"], r["demo"], r["seed"]) for r in rows}
        except Exception:  # noqa: BLE001
            pass

    total = len(budgets) * len(held) * len(seeds)
    c = 0
    for budget in budgets:
        for dm in held:
            for s in seeds:
                c += 1
                if (budget, dm.name, s) in done:
                    continue
                inst = disguise_demo(dm, s)
                obj, nd = inst.objective, dm.n_dim
                vals = {}
                for o in panel:
                    vals[o] = run_opt(o, obj, nd, budget, 9000 + s)
                for o, opt in discovered.items():
                    vals[o] = run_discovered(opt, obj, nd, budget, 9000 + s)
                ranks = {
                    o: 1 + sum(1 for x in field if vals[x] < vals[o] - 1e-12)
                    for o in field
                }
                rows.append(
                    {
                        "budget": budget,
                        "demo": dm.name,
                        "n": nd,
                        "seed": s,
                        "vals": {
                            o: (None if vals[o] >= INF else vals[o]) for o in field
                        },
                        "ranks": ranks,
                    }
                )
                print(
                    f"[{c}/{total}] b={budget} {dm.name:24s} n={nd:3d} s={s} "
                    f"centroid_rank={ranks.get('centroid')}",
                    flush=True,
                )
                atomic_dump({"done": False, "field": field, "rows": rows}, a.out)

    summary = {}
    for budget in budgets:
        br = [r for r in rows if r["budget"] == budget]
        if not br:
            continue
        summary[str(budget)] = {
            o: {
                "mean_rank": round(sum(r["ranks"][o] for r in br) / len(br), 3),
                "wins": sum(1 for r in br if r["ranks"][o] == 1),
            }
            for o in field
        }
    atomic_dump({"done": True, "field": field, "summary": summary, "rows": rows}, a.out)

    print("\n=== mean rank by budget (1=best of field) ===")
    for budget in budgets:
        sb = summary.get(str(budget))
        if not sb:
            continue
        ordered = sorted(field, key=lambda o: sb[o]["mean_rank"])
        print(
            f"  budget {budget}: "
            + "  ".join(f"{o}={sb[o]['mean_rank']}" for o in ordered)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
