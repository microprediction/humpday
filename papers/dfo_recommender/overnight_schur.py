"""
Overnight Schur/CMA-ES battery — the definitive, scaled test.

Sweeps damping variant x trial-budget x dimension, over all (fast) demos x many
seeds, bucketed by dimension. Free (pure numpy/scipy, no API). Checkpoints a JSON
after every demo so partial results survive a kill.

Key questions it settles:
  - Does any damping (blind diagonal / seriated HRP blocks) beat full-covariance
    CMA, and where — low-dim, high-dim, low-budget, high-budget?
  - Is the effect budget-dependent (damping should help most when the covariance
    is most undersampled: high dim AND low budget)?

    python papers/dfo_recommender/overnight_schur.py --seeds 0,1,2,3,4 \
        --budgets 60,120,240 --out runs/overnight_schur.json
"""

from __future__ import annotations

import argparse
import json
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

VARIANTS = [
    ("full γ=1.00", dict(gamma=1.0)),
    ("blind γ=0.25", dict(gamma=0.25)),
    ("blind γ=0.00", dict(gamma=0.0)),
    ("seriat γ=.50 t.5", dict(gamma=0.5, seriate=True, dist_thresh=0.5)),
    ("seriat γ=.00 t.5", dict(gamma=0.0, seriate=True, dist_thresh=0.5)),
    ("seriat γ=.50 t.35", dict(gamma=0.5, seriate=True, dist_thresh=0.35)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="60,120,240")
    ap.add_argument("--out", default="runs/overnight_schur.json")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    budgets = [int(b) for b in args.budgets.split(",")]
    demos = [d for d in DEMOS if d.name not in SLOW]
    ckpt = Path(args.out)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    labels = [lab for lab, _ in VARIANTS]
    # nested accumulators: regret[budget][bucket][label] -> list
    acc = {
        b: {bk: {lab: [] for lab in labels} for bk in ("all", "low", "high")}
        for b in budgets
    }
    done_demos = []

    print(
        f"Overnight Schur battery: {len(demos)} demos x {len(seeds)} seeds x "
        f"{len(budgets)} budgets x {len(VARIANTS)} variants.\n"
        f"checkpoint -> {ckpt}\n",
        flush=True,
    )

    def snapshot(done):
        out = {
            "done": done,
            "seeds": list(seeds),
            "budgets": budgets,
            "variants": labels,
            "demos_completed": done_demos,
            "tables": {},
        }
        for b in budgets:
            out["tables"][str(b)] = {}
            for bk in ("all", "low", "high"):
                tab = {
                    lab: (mean(acc[b][bk][lab]) if acc[b][bk][lab] else None)
                    for lab in labels
                }
                out["tables"][str(b)][bk] = tab
        tmp = Path(str(ckpt) + ".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(ckpt)

    for i, demo in enumerate(demos):
        bucket = "high" if demo.n_dim >= DIM_CUT else "low"
        for b in budgets:
            for s in seeds:
                inst = disguise_demo(demo, s)
                vals = {}
                for lab, kw in VARIANTS:
                    try:
                        vals[lab] = cma_es(
                            inst.objective, b, inst.n_dim, seed=9000 + i * 13 + s, **kw
                        )
                    except Exception:  # noqa: BLE001
                        vals[lab] = INF
                finite = [v for v in vals.values() if v < INF]
                mn, mx = (min(finite), max(finite)) if finite else (0.0, 1.0)
                for lab in labels:
                    nr = (
                        0.0
                        if mx <= mn or vals[lab] >= INF
                        else (vals[lab] - mn) / (mx - mn)
                    )
                    acc[b]["all"][lab].append(nr)
                    acc[b][bucket][lab].append(nr)
        done_demos.append(demo.name)
        print(f"  [{i + 1}/{len(demos)}] {demo.name}({demo.n_dim}) done", flush=True)
        snapshot(done=False)

    snapshot(done=True)
    print("\n=== FINAL: mean normalised regret by budget x bucket ===")
    for b in budgets:
        for bk in ("all", "low", "high"):
            tab = sorted(
                ((lab, mean(acc[b][bk][lab])) for lab in labels if acc[b][bk][lab]),
                key=lambda t: t[1],
            )
            print(f"\n--- budget={b}, {bk} ---")
            for lab, r in tab:
                print(f"  {lab:18s} {r:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
