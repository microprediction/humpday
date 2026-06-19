"""
Seriated (HRP-style) block-Schur damping for CMA-ES, tested in HIGH dimensions.

The wind_farm failure showed assumed contiguous blocks can be exactly backwards
(it damped the cross-turbine wake coupling that IS the signal). Fix: discover the
blocks by clustering the adapted covariance (HRP seriation), then damp only the
genuinely-weak cross-cluster couplings.

Compares, on the high-dim demos (n>=10, the undersampled regime):
  - full      γ=1.00         vanilla CMA
  - blind     γ=0.00         shrink ALL correlations to diagonal
  - block     γ=0.50         assumed contiguous blocks (only where block size known)
  - seriated  γ=0.50 / 0.00  HRP-discovered blocks, damp cross-cluster by γ

    python papers/dfo_recommender/seriation_hidim.py --seeds 0,1,2 --trials 150
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

from block_schur import BLOCK_DEMOS  # noqa: E402
from example_demos import DEMOS, disguise_demo  # noqa: E402
from schur_cma import cma_es  # noqa: E402

INF = float("inf")
DIM_CUT = 10
SLOW = {"bowling"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--thresh", type=float, default=0.5, help="seriation cluster distance cut")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    demos = [d for d in DEMOS if d.n_dim >= DIM_CUT and d.name not in SLOW]
    print(
        f"Seriated block-Schur (CMA-ES), HIGH-dim only ({len(demos)} demos, n>={DIM_CUT}) "
        f"x {len(seeds)} seeds, {args.trials} trials, seriation cut={args.thresh}.\n"
        f"demos: {', '.join(f'{d.name}({d.n_dim})' for d in demos)}\n"
    )

    def variants_for(demo):
        v = [
            ("full γ=1.00", dict(gamma=1.0)),
            ("blind γ=0.00", dict(gamma=0.0)),
            ("seriated γ=.50", dict(gamma=0.5, seriate=True, dist_thresh=args.thresh)),
            ("seriated γ=.00", dict(gamma=0.0, seriate=True, dist_thresh=args.thresh)),
        ]
        if demo.name in BLOCK_DEMOS:
            v.insert(2, ("block γ=0.50", dict(gamma=0.5, block_size=BLOCK_DEMOS[demo.name])))
        return v

    all_labels = ["full γ=1.00", "blind γ=0.00", "block γ=0.50", "seriated γ=.50", "seriated γ=.00"]
    regret = {lab: [] for lab in all_labels}
    per_demo = {}
    for i, demo in enumerate(demos):
        vlist = variants_for(demo)
        dvals = {lab: [] for lab, _ in vlist}
        for s in seeds:
            inst = disguise_demo(demo, s)
            vals = {}
            for lab, kw in vlist:
                try:
                    vals[lab] = cma_es(inst.objective, args.trials, inst.n_dim, seed=4000 + i * 7 + s, **kw)
                except Exception as e:  # noqa: BLE001
                    print(f"   ! {lab} on {inst.name}: {e}")
                    vals[lab] = INF
            finite = [v for v in vals.values() if v < INF]
            mn, mx = (min(finite), max(finite)) if finite else (0.0, 1.0)
            for lab in vals:
                nr = 0.0 if mx <= mn or vals[lab] >= INF else (vals[lab] - mn) / (mx - mn)
                regret[lab].append(nr)
                dvals[lab].append(nr)
        per_demo[f"{demo.name}({demo.n_dim})"] = {lab: mean(dvals[lab]) for lab in dvals}

    labs = [l for l in all_labels if regret[l]]
    print("=== per-demo normalised regret (lower=better; * = best in row) ===")
    print("  " + "demo".ljust(22) + "".join(l.rjust(15) for l in labs))
    for name, row in per_demo.items():
        best = min(row.values())
        cells = "".join(
            (("*" if lab in row and abs(row[lab] - best) < 1e-9 else " ")
             + (f"{row[lab]:.3f}" if lab in row else "  -  ")).rjust(15)
            for lab in labs
        )
        print("  " + name.ljust(22) + cells)

    print("\n=== overall (mean normalised regret, high-dim) ===")
    for lab, r in sorted(((lab, mean(regret[lab])) for lab in labs), key=lambda t: t[1]):
        print(f"  {lab:15s} {r:.4f}  {'#' * int(r * 40)}")
    print(
        "\nWin condition: seriated beats full AND blind (discovered structure helps "
        "where assumed contiguous blocks did not) — especially on interaction "
        "problems like wind_farm where contiguous blocks were backwards."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
