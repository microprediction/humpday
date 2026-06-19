"""
Block-Schur damping with KNOWN parameter-block structure (CMA-ES).

Many problems are "sets of parameters": one entity = a contiguous coordinate
group (a turbine's x,y; an atom's x,y,z; a centroid's x,y). Within an entity the
coordinates are genuinely coupled and worth estimating; across entities the
coupling is weaker and hard to estimate from few CMA samples. Block-Schur damping
keeps the within-block covariance and damps only the cross-block covariance by γ
(γ=1 full, γ=0 block-diagonal — within-block correlations survive, unlike the
blind diagonal damping in schur_cma which destroys them too).

This compares, per structured demo, CMA-ES under:
  - full        γ=1.00            (vanilla)
  - blind γ=0.5 / γ=0.0           (correlation damping toward FULL diagonal)
  - block γ=0.5 / γ=0.0           (block-Schur with the KNOWN block size)

Hypothesis: block-γ beats both full and blind, because it preserves the real
within-entity correlation while discarding the unreliable cross-entity part.

    python papers/dfo_recommender/block_schur.py --seeds 0,1,2 --trials 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from example_demos import DEMOS, disguise_demo  # noqa: E402
from schur_cma import cma_es  # noqa: E402

INF = float("inf")

# Known parameter-block sizes (one entity = contiguous coordinate group).
BLOCK_DEMOS = {
    "wind_farm": 2,             # 8 turbines x (x,y),  n=16  [high-dim]
    "circle_packing": 2,        # 6 circles  x (x,y),  n=12  [high-dim]
    "lennard_jones_cluster": 3, # 5 atoms    x (x,y,z),n=15  [high-dim]
    "sensor_localization": 2,   # 4 nodes    x (x,y),  n=8
    "kmeans_clustering": 2,     # 3 centroids x (x,y), n=6
}

VARIANTS = [
    ("full γ=1.00", dict(gamma=1.0)),
    ("blind γ=0.50", dict(gamma=0.5)),
    ("blind γ=0.00", dict(gamma=0.0)),
    ("block γ=0.50", dict(gamma=0.5, block=True)),
    ("block γ=0.00", dict(gamma=0.0, block=True)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--trials", type=int, default=150)
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    demos = [(d, BLOCK_DEMOS[d.name]) for d in DEMOS if d.name in BLOCK_DEMOS]
    print(
        f"Block-Schur (CMA-ES) on {len(demos)} structured demos x {len(seeds)} "
        f"seeds, {args.trials} trials.\n"
        f"demos: {', '.join(f'{d.name}(n={d.n_dim},blk={b})' for d, b in demos)}\n"
    )

    regret = {lab: [] for lab, _ in VARIANTS}
    per_demo = {}
    for i, (demo, bsize) in enumerate(demos):
        dvals = {lab: [] for lab, _ in VARIANTS}
        for s in seeds:
            inst = disguise_demo(demo, s)
            vals = {}
            for lab, kw in VARIANTS:
                kw2 = dict(kw)
                bs = kw2.pop("block", False)
                try:
                    v = cma_es(inst.objective, args.trials, inst.n_dim, seed=3000 + i * 7 + s,
                               block_size=(bsize if bs else None), **kw2)
                except Exception as e:  # noqa: BLE001
                    print(f"   ! {lab} on {inst.name}: {e}")
                    v = INF
                vals[lab] = v
            finite = [v for v in vals.values() if v < INF]
            mn, mx = (min(finite), max(finite)) if finite else (0.0, 1.0)
            for lab in vals:
                nr = 0.0 if mx <= mn or vals[lab] >= INF else (vals[lab] - mn) / (mx - mn)
                regret[lab].append(nr)
                dvals[lab].append(nr)
        per_demo[demo.name] = {lab: mean(dvals[lab]) for lab in dvals}

    print("=== per-demo normalised regret (lower=better; * = best in row) ===")
    labs = [lab for lab, _ in VARIANTS]
    print("  " + "demo".ljust(24) + "".join(l.rjust(14) for l in labs))
    for name, row in per_demo.items():
        best = min(row.values())
        cells = "".join((("*" if abs(row[l] - best) < 1e-9 else " ") + f"{row[l]:.3f}").rjust(14) for l in labs)
        print("  " + name.ljust(24) + cells)

    print("\n=== overall (mean normalised regret across structured demos) ===")
    for lab, r in sorted(((lab, mean(regret[lab])) for lab in labs), key=lambda t: t[1]):
        print(f"  {lab:14s} {r:.4f}  {'#' * int(r * 40)}")
    print(
        "\nWin condition: a `block` variant beats BOTH `full γ=1.00` and the best "
        "`blind` variant — i.e. keeping within-entity correlation while damping "
        "cross-entity correlation is what helps, not blind shrinkage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
