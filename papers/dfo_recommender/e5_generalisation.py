"""E5 — Does the evolved surrogate template GENERALISE? (train/test split)

Evolve a 14-gene genome on a TRAIN set of demos, then score the winner on a DISJOINT
TEST set (unseen demos AND unseen disguise seeds) against the panel. Closes the
EVOLVED_OPTIMIZER.md §7 gap (fitness was measured on the training demos). Compares the
evolved winner to the warm-start DEFAULT_GENOME on TEST — if evolution helps
out-of-sample, the discovery process generalises rather than overfitting.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
import algo_dev as ad  # noqa: E402
from example_demos import DEMOS, disguise_demo  # noqa: E402


def _xover(a, b):
    return [a[j] if random.random() < 0.5 else b[j] for j in range(ad.GENOME_LEN)]


def _mut(g, step):
    return [ad._clip01(g[j] + random.gauss(0, step)) for j in range(ad.GENOME_LEN)]


def evolve(train, seeds, trials, generations, mu, lam, n_warm):
    cache = {}

    def fit(g):
        key = tuple(round(v, 6) for v in g)
        if key not in cache:
            cache[key] = ad.candidate_fitness(g, train, seeds, trials)
        return cache[key]

    pop = [list(ad.DEFAULT_GENOME) for _ in range(min(n_warm, mu))]
    while len(pop) < mu:
        pop.append([random.random() for _ in range(ad.GENOME_LEN)])
    scored = sorted(((fit(g), g) for g in pop), key=lambda t: t[0])
    for gen in range(generations):
        step = 0.3 * (1 - gen / max(generations - 1, 1)) + 0.05
        kids = []
        for _ in range(lam):
            a, b = random.choice(scored)[1], random.choice(scored)[1]
            child = _mut(_xover(a, b), step) if random.random() < 0.7 else _mut(a, step)
            kids.append(child)
        scored = sorted(scored + [(fit(g), g) for g in kids], key=lambda t: t[0])[:mu]
        print(f"    gen {gen+1}/{generations}: train_best={scored[0][0]:.4f}", flush=True)
    return scored[0][1], scored[0][0]


def test_mean_rank(genome, test, seeds, trials):
    cand_opt = ad.make_candidate(genome)
    ranks = []
    for i, dm in enumerate(test):
        for s in seeds:
            inst = disguise_demo(dm, s)
            seed = 6000 + 31 * i + s
            cand = ad._candidate_best(cand_opt, inst, trials, seed)
            others = [ad._panel_best(p, inst, trials, seed) for p in ad.PANEL]
            ranks.append(1 + sum(1 for v in others if v < cand))
    return mean(ranks)


def atomic_dump(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-demos", type=int, default=18)
    ap.add_argument("--test-demos", type=int, default=18)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--mu", type=int, default=10)
    ap.add_argument("--lam", type=int, default=8)
    ap.add_argument("--n-warm", type=int, default=2)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/surrogate_generalisation.json")
    a = ap.parse_args()
    if a.quick:
        a.train_demos, a.test_demos, a.generations, a.mu, a.lam, a.trials = 4, 4, 3, 5, 4, 40

    # dim-spread, then split into disjoint train/test by alternating
    ds = sorted(DEMOS, key=lambda d: d.n_dim)
    take = min(a.train_demos + a.test_demos, len(ds))
    idx = sorted({round(k * (len(ds) - 1) / max(take - 1, 1)) for k in range(take)})
    chosen = [ds[i] for i in idx]
    train = chosen[0::2][: a.train_demos]
    test = chosen[1::2][: a.test_demos]
    train_seeds, test_seeds = (0, 1), (2, 3, 4)
    print(f"train: {len(train)} demos seeds {train_seeds} | test: {len(test)} demos "
          f"seeds {test_seeds} (disjoint)\n", flush=True)

    random.seed(0)
    print("  evolving on TRAIN...", flush=True)
    best, train_fit = evolve(train, train_seeds, a.trials, a.generations, a.mu, a.lam, a.n_warm)

    print("  scoring on TEST (held-out demos + unseen seeds)...", flush=True)
    evolved_rank = test_mean_rank(best, test, test_seeds, a.trials)
    default_rank = test_mean_rank(list(ad.DEFAULT_GENOME), test, test_seeds, a.trials)

    out = {"done": True, "train_demos": [d.name for d in train], "test_demos": [d.name for d in test],
           "train_seeds": list(train_seeds), "test_seeds": list(test_seeds),
           "best_genome": best, "train_fitness": train_fit,
           "evolved_test_mean_rank": round(evolved_rank, 3),
           "default_test_mean_rank": round(default_rank, 3),
           "generalises": evolved_rank <= default_rank + 1e-9}
    atomic_dump(out, a.out)
    print(f"\n=== generalisation ===\n  evolved winner  TEST mean rank: {evolved_rank:.3f}"
          f"\n  default genome  TEST mean rank: {default_rank:.3f}"
          f"\n  -> {'GENERALISES (evolution helps out-of-sample)' if out['generalises'] else 'OVERFIT (no out-of-sample gain)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
