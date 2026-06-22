"""E2 — Discovered optimizers, hardened (multi-budget, held-out).

Do the three 'discovered' optimizers beat the classic panel on HELD-OUT demos across
budgets, and where? Panel: NM, DE, CMA, ngCMA(pycma). Discovered:
  - centroid     : the LLM equal-blend (runs/simplex_warm_code/centroid.py)
  - surrogate    : the evolved 14-gene algo_dev genome (runs/v2.json best_genome)
  - unstructured : the best free-form Opus optimizer (runs/ablation_unstructured_code/)
Held-out = demos used by none of the selection sets. Crash-safe per-instance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
import algo_dev as ad  # noqa: E402
import numpy as np
from example_demos import DEMOS, disguise_demo  # noqa: E402
from rankcorr import run_opt  # noqa: E402  (humpday names + ngCMA)
from simplex_blend import compile_optimizer, select_demos  # noqa: E402

INF = float("inf")


def load_centroid():
    spec = importlib.util.spec_from_file_location(
        "centroid_opt", "runs/simplex_warm_code/centroid.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.optimize


def load_unstructured_best():
    d = json.load(open("runs/ablation_unstructured.json"))
    best = min(d["results"], key=lambda r: r["regret"])["label"]
    code = open(f"runs/ablation_unstructured_code/{best}.py").read()
    return compile_optimizer(code), best


def load_surrogate():
    g = json.load(open("runs/v2.json"))["best_genome"]
    return ad.make_candidate(g)


def run_discovered(opt, objective, n_dim, n_trials, seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        r = opt(objective, n_trials, n_dim)
        return float(r[0]) if r else INF
    except Exception:  # noqa: BLE001
        return INF


def atomic_dump(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=20)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="60,120,240,480")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/discovered_hardened.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    budgets = [int(b) for b in a.budgets.split(",")]
    if a.quick:
        seeds, budgets, a.demos = [0], [120], 3

    discovered = {"centroid": load_centroid(), "surrogate": load_surrogate()}
    try:
        uns, uns_label = load_unstructured_best()
        discovered["unstructured"] = uns
    except Exception as e:  # noqa: BLE001
        print(f"  (unstructured optimizer unavailable: {e})", flush=True)
    panel = ["NelderMead", "DifferentialEvolution", "CMAEvolutionStrategy", "ngCMA"]
    field = panel + list(discovered)

    # held-out: exclude every selection set used to pick the discovered optimizers
    used = {id(d) for d in select_demos(16, "spread")} | {id(d) for d in DEMOS[:15]}
    pool = sorted([d for d in DEMOS if id(d) not in used], key=lambda d: d.n_dim)
    n = min(a.demos, len(pool))
    idx = sorted({round(k * (len(pool) - 1) / max(n - 1, 1)) for k in range(n)})
    held = [pool[i] for i in idx]
    print(
        f"held-out demos ({len(held)}, dims {held[0].n_dim}-{held[-1].n_dim})\n",
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
                    f"[{c}/{total}] b={budget} {dm.name:22s} n={nd:3d} s={s} "
                    f"centroid_rank={ranks.get('centroid')}",
                    flush=True,
                )
                atomic_dump({"done": False, "field": field, "rows": rows}, a.out)

    # mean rank + win-rate per budget
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
