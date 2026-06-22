"""E3 / Phase 5 — runtime optimizer crossover via the ask/tell interface.

Question: do runtime-blended optimizers beat their individual components on the
disguised real-world suite? The new suggest_next/receive_update interface lets a
caller own the loop, so we can blend optimizers WITHOUT generating fused code:

  - single:    each component alone at the full budget (baseline)
  - portfolio: run k components, budget//k each, return the global best
  - interleave: round-robin suggestions from k components under ONE shared budget,
                global incumbent across all — the ask/tell-native blend

Compares pairs of complementary optimizers (a global explorer + a local refiner)
against their parts. Crash-safe per-(pair,demo,seed) checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
import numpy as np
from example_demos import DEMOS, disguise_demo  # noqa: E402

from humpday.optimizers.alloptimizers import (  # noqa: E402
    PURE_OPTIMIZERS,
    pure_optimize,
)

INF = float("inf")

# complementary pairs: a population/global method + a local/curvature method
PAIRS = [
    ("DifferentialEvolution", "NelderMead"),
    ("CMAEvolutionStrategy", "PatternSearch"),
    ("DifferentialEvolution", "PRIMA_BOBYQA"),
    ("ParticleSwarm", "PRIMA_NEWUOA"),
]


def _seed(s):
    random.seed(s)
    np.random.seed(s)


def run_single(name, objective, n_dim, n_trials, seed):
    _seed(seed)
    try:
        bv, _ = pure_optimize(objective, name, n_trials, n_dim)
        return float(bv)
    except Exception:  # noqa: BLE001
        return INF


def run_portfolio(names, objective, n_dim, n_trials, seed):
    k = len(names)
    each = max(1, n_trials // k)
    best = INF
    for i, nm in enumerate(names):
        _seed(seed + 17 * i)
        try:
            bv, _ = pure_optimize(objective, nm, each, n_dim)
            best = min(best, float(bv))
        except Exception:  # noqa: BLE001
            pass
    return best


def run_interleave(names, objective, n_dim, n_trials, seed):
    """Round-robin ask/tell across components under one shared budget."""
    _seed(seed)
    opts = [PURE_OPTIMIZERS[nm](objective, n_trials, n_dim) for nm in names]
    active = list(range(len(opts)))
    best = INF
    total = 0
    try:
        while active and total < n_trials:
            for i in list(active):
                if total >= n_trials:
                    break
                x = opts[i].suggest_next()
                if x is None:
                    active.remove(i)
                    continue
                y = float(objective([float(v) for v in x]))
                total += 1
                opts[i].receive_update(y)
                if y < best:
                    best = y
    finally:
        for o in opts:
            try:
                o.close()
            except Exception:  # noqa: BLE001
                pass
    return best


def atomic_dump(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=24)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/crossover.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    pairs = PAIRS
    if a.quick:
        seeds, pairs = [0], [("DifferentialEvolution", "NelderMead")]
        a.demos = 3

    ds = sorted(DEMOS, key=lambda d: d.n_dim)
    n = min(a.demos, len(ds))
    idx = sorted({round(k * (len(ds) - 1) / max(n - 1, 1)) for k in range(n)})
    demos = [ds[i] for i in idx]

    rows = []
    wins = {"single_best": 0, "portfolio": 0, "interleave": 0, "tie": 0}
    total = len(pairs) * len(demos) * len(seeds)
    c = 0
    for (na, nb) in pairs:
        for dm in demos:
            for s in seeds:
                c += 1
                inst = disguise_demo(dm, s)
                obj, nd = inst.objective, dm.n_dim
                va = run_single(na, obj, nd, a.trials, 800 + s)
                vb = run_single(nb, obj, nd, a.trials, 800 + s)
                vp = run_portfolio([na, nb], obj, nd, a.trials, 800 + s)
                vi = run_interleave([na, nb], obj, nd, a.trials, 800 + s)
                single_best = min(va, vb)
                cands = {"single_best": single_best, "portfolio": vp, "interleave": vi}
                w = min(cands, key=cands.get)
                # tie if interleave within 1e-9 of the best
                if abs(cands[w] - cands["single_best"]) <= 1e-9 and w != "single_best":
                    pass
                wins[w] += 1
                rows.append({"pair": f"{na}+{nb}", "demo": dm.name, "n": nd, "seed": s,
                             "a": va, "b": vb, "single_best": single_best,
                             "portfolio": vp, "interleave": vi, "winner": w})
                print(f"[{c}/{total}] {na[:4]}+{nb[:4]} {dm.name:20s} n={nd:3d} "
                      f"single={single_best:.4g} portf={vp:.4g} inter={vi:.4g} -> {w}", flush=True)
                atomic_dump({"done": False, "wins": wins, "rows": rows}, a.out)

    # how often does a blend beat the best single component?
    n_inst = len(rows)
    blend_beats = sum(1 for r in rows
                      if min(r["portfolio"], r["interleave"]) < r["single_best"] - 1e-9)
    interleave_beats = sum(1 for r in rows if r["interleave"] < r["single_best"] - 1e-9)
    summary = {"instances": n_inst, "winner_counts": wins,
               "blend_beats_best_single": blend_beats,
               "interleave_beats_best_single": interleave_beats,
               "blend_beat_rate": round(blend_beats / n_inst, 3) if n_inst else None}
    atomic_dump({"done": True, "summary": summary, "rows": rows}, a.out)
    print("\n=== crossover summary ===")
    print(f"  instances: {n_inst}")
    print(f"  a blend beats the best single component on {blend_beats}/{n_inst} "
          f"({summary['blend_beat_rate']})")
    print(f"  interleave specifically beats best single on {interleave_beats}/{n_inst}")
    print(f"  winner counts: {wins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
