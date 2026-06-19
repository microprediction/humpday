"""
HISTORICAL — this driver did its job. It evolved the surrogate-augmented template
and (with the follow-up ablation) showed the quadratic trust-region jump more than
halves regret, so the surrogate + its two genes are now native to
`algo_dev.make_candidate` / `_decode_genome` (GENOME_LEN is 14). The standard
`algo_dev --mode ga` now evolves the surrogate directly; this separate driver is
no longer needed and is kept only for provenance of runs/v2.json. Note: with the
genes folded in, `ad.GENOME_LEN` is already 14, so `V2_GENOME_LEN = GENOME_LEN+2`
below is now stale — do not re-run without revisiting that.

Evolve the surrogate-augmented template (`candidate_v2.make_candidate_v2`) with
the same (μ+λ) GA + crossover as `algo_dev`, but over a 14-gene genome:

    genes 0..11  base DE/ES knobs (algo_dev._decode_genome)
    gene  12     p_surrogate  — probability a generation fires a surrogate jump
    gene  13     r2_min scale — min model R² to trust the jump

This is the principled test of IDEAS.md §A.1: rather than a confounded single-
seed head-to-head, let evolution decide whether the local quadratic trust-region
jump lowers regret across the disguised suite, and at what intensity. Compare the
resulting fitness against the base-template GA (`algo_dev --mode ga`).

    python papers/dfo_recommender/evolve_v2.py --quick
    python papers/dfo_recommender/evolve_v2.py --generations 30 --mu 10 --lam 8 \
        --n-demos 15 --seeds 0,1 --trials 100 --out runs/v2.json
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

import random  # noqa: E402

import algo_dev as ad  # noqa: E402
from candidate_v2 import make_candidate_v2  # noqa: E402
from example_demos import DEMOS, disguise_demo  # noqa: E402

V2_GENOME_LEN = ad.GENOME_LEN + 2  # base 12 + p_surrogate + r2_min
INF = float("inf")

# warm start: the base default genome + a moderate, R²-gated surrogate.
DEFAULT_V2 = list(ad.DEFAULT_GENOME) + [0.4, 0.6]


def _candidate_best(candidate, demo, n_trials, run_seed):
    random.seed(run_seed)
    try:
        import numpy as np

        np.random.seed(run_seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        f_best, _ = candidate(demo.objective, n_trials, demo.n_dim)
        return float(f_best)
    except Exception:  # noqa: BLE001
        return INF


def v2_fitness(genome, base_demos, seeds, n_trials, panel=ad.PANEL):
    """Normalised regret of the v2 candidate vs the baseline panel — same metric
    as algo_dev.candidate_fitness, but builds the surrogate-augmented candidate."""
    candidate = make_candidate_v2(genome)
    scores = []
    for i, demo in enumerate(base_demos):
        for s in seeds:
            inst = disguise_demo(demo, s)
            seed = 5000 + 31 * i + s
            cand = _candidate_best(candidate, inst, n_trials, seed)
            vals = [cand] + [ad._panel_best(a, inst, n_trials, seed) for a in panel]
            finite = [v for v in vals if v < INF]
            if cand >= INF or not finite:
                scores.append(1.0)
                continue
            mn, mx = min(finite), max(finite)
            scores.append(0.0 if mx <= mn else (cand - mn) / (mx - mn))
    return mean(scores) if scores else 1.0


def _xover(a, b):
    child = []
    for j in range(V2_GENOME_LEN):
        if random.random() < 0.5:
            child.append(a[j] if random.random() < 0.5 else b[j])
        else:
            w = random.random()
            child.append(ad._clip01(w * a[j] + (1.0 - w) * b[j]))
    return child


def _mut(g, step, p_gene=0.5):
    return [
        ad._clip01(g[j] + random.gauss(0, step)) if random.random() < p_gene else g[j]
        for j in range(V2_GENOME_LEN)
    ]


def evolve_v2(generations, mu, lam, base, seeds, trials, p_crossover=0.7, n_warm=1,
              rng_seed=0, checkpoint_path=None):
    random.seed(rng_seed)
    pop = [list(DEFAULT_V2) for _ in range(min(n_warm, mu))]
    while len(pop) < mu:
        pop.append([random.random() for _ in range(V2_GENOME_LEN)])
    cache = {}

    def fit_of(g):
        key = tuple(round(x, 6) for x in g)
        if key not in cache:
            cache[key] = v2_fitness(g, base, seeds, trials)
        return cache[key]

    scored = sorted(((fit_of(g), g) for g in pop), key=lambda t: t[0])[:mu]
    best_fit, best_genome = scored[0]
    history = [best_fit]
    step = 0.2

    def checkpoint(gen, done=False):
        ad._write_checkpoint(checkpoint_path, {
            "template": "v2_surrogate", "generation": gen,
            "generations_planned": generations, "done": done, "mu": mu, "lam": lam,
            "seeds": list(seeds), "n_trials": trials, "panel": ad.PANEL,
            "demos": [d.name for d in base], "evals": len(cache),
            "best_fitness": best_fit, "best_genome": best_genome,
            "best_genome_decoded": {**ad._describe(best_genome[:ad.GENOME_LEN]),
                                    "p_surrogate": round(best_genome[12], 3),
                                    "r2_min_gene": round(best_genome[13], 3)},
            "population_fitness": [round(f, 4) for f, _ in scored], "history": history,
        })

    print(f"  gen 0: pop best regret = {best_fit:.4f}  "
          f"(pop {[round(f, 3) for f, _ in scored]})", flush=True)
    checkpoint(0)
    for gen in range(1, generations + 1):
        parents = [g for _, g in scored]
        offspring = []
        for _ in range(lam):
            if random.random() < p_crossover and len(parents) >= 2:
                pa, pb = random.sample(parents, 2)
                child = _xover(pa, pb)
            else:
                child = list(random.choice(parents))
            offspring.append(_mut(child, step))
        scored = sorted(((fit_of(g), g) for g in parents + offspring),
                        key=lambda t: t[0])[:mu]
        improved = scored[0][0] < best_fit - 1e-12
        best_fit, best_genome = scored[0]
        step = min(0.4, step * 1.15) if improved else max(0.05, step * 0.85)
        history.append(best_fit)
        print(f"  gen {gen}: best regret = {best_fit:.4f}  "
              f"(pop {[round(f, 3) for f, _ in scored]}, step {step:.2f}, "
              f"evals {len(cache)})", flush=True)
        checkpoint(gen, done=(gen == generations))
    return best_genome, best_fit, history


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--mu", type=int, default=10)
    ap.add_argument("--lam", type=int, default=8)
    ap.add_argument("--n-warm", type=int, default=2)
    ap.add_argument("--crossover", type=float, default=0.7)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--n-demos", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gens, mu, lam, trials = args.generations, args.mu, args.lam, args.trials
    seeds = tuple(int(s) for s in args.seeds.split(","))
    n_demos = args.n_demos
    if args.quick:
        gens, mu, lam, trials, n_demos, seeds = 3, 5, 4, 40, 4, (0, 1)
    ckpt = Path(args.out) if args.out else None
    base = DEMOS[:n_demos]
    print(f"Evolving SURROGATE template: ({mu}+{lam}) GA, {gens} gens, "
          f"fitness = regret vs {ad.PANEL}\n"
          f"across {len(base)} demos x {len(seeds)} seeds, {trials} trials each.\n"
          + (f"checkpointing to {ckpt}\n" if ckpt else ""))
    bg, bf, _ = evolve_v2(gens, mu, lam, base, seeds, trials,
                          p_crossover=args.crossover, n_warm=args.n_warm,
                          checkpoint_path=ckpt)
    print("\n=== Evolved surrogate optimizer ===")
    print(f"  normalised regret vs panel: {bf:.4f}")
    print(f"  base knobs: {ad._describe(bg[:ad.GENOME_LEN])}")
    print(f"  p_surrogate: {bg[12]:.3f}   r2_min_gene: {bg[13]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
