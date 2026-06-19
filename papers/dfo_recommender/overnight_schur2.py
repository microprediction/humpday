"""
Overnight Schur battery — WAVE 2: reliability-adaptive γ* + budget extrapolation.

Wave 1 (overnight_schur.py) showed: fixed-γ damping ≈ neutral overall, but in
HIGH dimensions the damping benefit GROWS with budget (more adapted-but-overfit
covariance to tame), and seriated blocks win on structured problems.

Wave 2 tests the mechanism the writeup calls for:
  - `adaptive`  — reliability γ* (per-correlation noise-floor shrinkage, no fixed γ)
  - seriated at TIGHTER cuts (0.35, 0.20) — cut 0.5 gave trivial clusters
  - budget pushed to 480 — does damping keep pulling ahead in high-dim?

Question: does adaptive γ* match-or-beat the best fixed/seriated variant in EVERY
(budget, dim) cell — i.e. is it the self-tuning rule that wins everywhere?

    python papers/dfo_recommender/overnight_schur2.py --seeds 0,1,2,3,4 \
        --budgets 120,240,480 --out runs/overnight_schur2.json
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

from example_demos import DEMOS, disguise_demo  # noqa: E402
from schur_cma import cma_es  # noqa: E402

INF = float("inf")
DIM_CUT = 10
SLOW = {"bowling"}

VARIANTS = [
    ("full γ=1.00", dict(gamma=1.0)),
    ("blind γ=0.00", dict(gamma=0.0)),
    ("seriat t.35", dict(gamma=0.5, seriate=True, dist_thresh=0.35)),
    ("seriat t.20", dict(gamma=0.5, seriate=True, dist_thresh=0.20)),
    ("adaptive γ*", dict(adaptive=True)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="120,240,480")
    ap.add_argument("--out", default="runs/overnight_schur2.json")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    budgets = [int(b) for b in args.budgets.split(",")]
    demos = [d for d in DEMOS if d.name not in SLOW]
    ckpt = Path(args.out)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    labels = [lab for lab, _ in VARIANTS]
    acc = {b: {bk: {lab: [] for lab in labels} for bk in ("all", "low", "high")} for b in budgets}
    done_demos = []

    print(f"Schur WAVE 2: {len(demos)} demos x {len(seeds)} seeds x {len(budgets)} "
          f"budgets x {len(VARIANTS)} variants (incl. adaptive γ*).\ncheckpoint -> {ckpt}\n",
          flush=True)

    def snapshot(done):
        out = {"done": done, "seeds": list(seeds), "budgets": budgets, "variants": labels,
               "demos_completed": done_demos, "tables": {}}
        for b in budgets:
            out["tables"][str(b)] = {
                bk: {lab: (mean(acc[b][bk][lab]) if acc[b][bk][lab] else None) for lab in labels}
                for bk in ("all", "low", "high")}
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
                        vals[lab] = cma_es(inst.objective, b, inst.n_dim, seed=12000 + i * 13 + s, **kw)
                    except Exception:  # noqa: BLE001
                        vals[lab] = INF
                finite = [v for v in vals.values() if v < INF]
                mn, mx = (min(finite), max(finite)) if finite else (0.0, 1.0)
                for lab in labels:
                    nr = 0.0 if mx <= mn or vals[lab] >= INF else (vals[lab] - mn) / (mx - mn)
                    acc[b]["all"][lab].append(nr)
                    acc[b][bucket][lab].append(nr)
        done_demos.append(demo.name)
        print(f"  [{i + 1}/{len(demos)}] {demo.name}({demo.n_dim}) done", flush=True)
        snapshot(done=False)

    snapshot(done=True)
    print("\n=== FINAL: mean normalised regret by budget x bucket ===")
    for b in budgets:
        for bk in ("all", "low", "high"):
            tab = sorted(((lab, mean(acc[b][bk][lab])) for lab in labels if acc[b][bk][lab]),
                         key=lambda t: t[1])
            print(f"\n--- budget={b}, {bk} ---")
            for lab, r in tab:
                print(f"  {lab:14s} {r:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
