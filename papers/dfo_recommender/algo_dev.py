"""
Goal-based optimizer development — evolve a black-box optimizer against the
disguised benchmark.

This is the capstone of the disguised-benchmark stack:

  example_applications/*  →  example_demos.disguised_demos()  →  fitness here

A *candidate optimizer* is produced from a small genome of behavioural knobs
(`make_candidate`). Its **fitness** is its mean rank against a fixed baseline
panel across many *disguised* instances (`candidate_fitness`) — disguised so the
candidate can't win by memorising optimum locations; it must genuinely search.
A simple (1 + λ) evolution strategy (`evolve`) mutates genomes to drive that
fitness down. Swap in a richer template or a real meta-optimizer later; the loop
and the memorisation-proof fitness are what matter.

    python papers/dfo_recommender/algo_dev.py --quick
    python papers/dfo_recommender/algo_dev.py --generations 12 --lam 6
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from example_demos import DEMOS, disguise_demo  # noqa: E402

from humpday.optimizers.alloptimizers import pure_optimize  # noqa: E402

INF = float("inf")
GENOME_LEN = 6

# Baseline panel the candidate is ranked against (kept small so a genome
# evaluation — which reruns the whole panel per instance — stays affordable).
PANEL = ["NelderMead", "DifferentialEvolution", "CMAEvolutionStrategy"]


def _clip01(x):
    return min(1.0, max(0.0, x))


def make_candidate(genome):
    """Build a concrete optimizer from a genome in [0,1]^6.

    The template is a real, general strategy: differential evolution with an
    occasional Gaussian local move around the incumbent and a one-shot restart
    of the worst half. The genome sets population size, F, crossover rate, the
    local-move probability, the local step, and when to restart."""
    g = [_clip01(v) for v in genome]
    pop = 4 + int(g[0] * 16)
    F = 0.3 + g[1] * 0.7
    CR = 0.1 + g[2] * 0.9
    p_local = g[3]
    sigma = 0.02 + g[4] * 0.28
    restart_at = 0.4 + g[5] * 0.5

    def optimize(objective, n_trials, n_dim, with_count=False):
        state = {"evals": 0, "best_f": INF, "best_x": None}

        def ev(x):
            x = [_clip01(xi) for xi in x]
            f = objective(x)
            state["evals"] += 1
            if f < state["best_f"]:
                state["best_f"], state["best_x"] = f, x[:]
            return f

        P = [[random.random() for _ in range(n_dim)] for _ in range(pop)]
        fit = [ev(p) for p in P]
        restarted = False
        while state["evals"] < n_trials:
            for i in range(pop):
                if state["evals"] >= n_trials:
                    break
                if state["best_x"] is not None and random.random() < p_local:
                    trial = [
                        state["best_x"][j] + random.gauss(0, sigma)
                        for j in range(n_dim)
                    ]
                else:
                    a, b, c = random.sample(range(pop), 3) if pop >= 3 else (0, 0, 0)
                    trial = [
                        P[a][j] + F * (P[b][j] - P[c][j])
                        if random.random() < CR
                        else P[i][j]
                        for j in range(n_dim)
                    ]
                ft = ev(trial)
                if ft < fit[i]:
                    P[i], fit[i] = trial, ft
            if not restarted and state["evals"] > restart_at * n_trials:
                order = sorted(range(pop), key=lambda k: fit[k])
                for k in order[pop // 2 :]:
                    if state["evals"] >= n_trials:
                        break
                    P[k] = [random.random() for _ in range(n_dim)]
                    fit[k] = ev(P[k])
                restarted = True
        if with_count:
            return state["best_f"], state["best_x"], state["evals"]
        return state["best_f"], state["best_x"]

    return optimize


def _panel_best(algo, demo, n_trials, run_seed):
    random.seed(run_seed)
    try:
        import numpy as np

        np.random.seed(run_seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        f_best, _ = pure_optimize(demo.objective, algo, n_trials, demo.n_dim)
        return float(f_best)
    except Exception:  # noqa: BLE001
        return INF


def _candidate_best(candidate, demo, n_trials, run_seed):
    random.seed(run_seed)
    try:
        f_best, _ = candidate(demo.objective, n_trials, demo.n_dim)
        return float(f_best)
    except Exception:  # noqa: BLE001
        return INF


def candidate_fitness(genome, base_demos, seeds, n_trials, panel=PANEL):
    """Mean rank (lower = better) of the candidate vs the baseline panel across
    the disguised instances of `base_demos`. This is the fitness to minimise."""
    candidate = make_candidate(genome)
    ranks = []
    for i, demo in enumerate(base_demos):
        for s in seeds:
            inst = disguise_demo(demo, s)
            seed = 5000 + 31 * i + s
            cand = _candidate_best(candidate, inst, n_trials, seed)
            others = [_panel_best(a, inst, n_trials, seed) for a in panel]
            # rank of candidate among [candidate] + panel (1 = best)
            rank = 1 + sum(1 for v in others if v < cand)
            ranks.append(rank)
    return mean(ranks) if ranks else INF


def evolve(generations, lam, base_demos, seeds, n_trials, rng_seed=0):
    """(1 + λ) evolution strategy over the genome. Returns (best_genome,
    best_fitness, history)."""
    random.seed(rng_seed)
    parent = [random.random() for _ in range(GENOME_LEN)]
    parent_fit = candidate_fitness(parent, base_demos, seeds, n_trials)
    history = [parent_fit]
    step = 0.25
    print(f"  gen 0: parent fitness (mean rank) = {parent_fit:.3f}", flush=True)
    for gen in range(1, generations + 1):
        best_child, best_child_fit = None, INF
        for _ in range(lam):
            child = [
                _clip01(parent[j] + random.gauss(0, step)) for j in range(GENOME_LEN)
            ]
            fit = candidate_fitness(child, base_demos, seeds, n_trials)
            if fit < best_child_fit:
                best_child, best_child_fit = child, fit
        if best_child_fit < parent_fit:
            parent, parent_fit = best_child, best_child_fit
            step = min(0.4, step * 1.15)  # success: explore a bit wider
        else:
            step = max(0.05, step * 0.8)  # stall: tighten
        history.append(parent_fit)
        print(
            f"  gen {gen}: best fitness = {parent_fit:.3f}  (child tried {best_child_fit:.3f}, step {step:.2f})",
            flush=True,
        )
    return parent, parent_fit, history


def _describe(genome):
    g = [_clip01(v) for v in genome]
    return {
        "pop": 4 + int(g[0] * 16),
        "F": round(0.3 + g[1] * 0.7, 3),
        "CR": round(0.1 + g[2] * 0.9, 3),
        "p_local": round(g[3], 3),
        "sigma": round(0.02 + g[4] * 0.28, 3),
        "restart_at": round(0.4 + g[5] * 0.5, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--lam", type=int, default=4)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument(
        "--n-demos", type=int, default=8, help="how many base demos to score on"
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    generations, lam, trials = args.generations, args.lam, args.trials
    seeds = tuple(int(s) for s in args.seeds.split(","))
    n_demos = args.n_demos
    if args.quick:
        generations, lam, trials, n_demos, seeds = 4, 3, 50, 5, (0, 1)

    base = DEMOS[:n_demos]
    print(
        f"Evolving an optimizer: (1+{lam}) ES, {generations} generations, fitness = mean rank\n"
        f"vs {PANEL} across {len(base)} demos x {len(seeds)} disguised seeds, {trials} trials each.\n"
    )
    best_genome, best_fit, _ = evolve(generations, lam, base, seeds, trials)
    print("\n=== Evolved optimizer ===")
    print(f"  fitness (mean rank vs panel of {len(PANEL)}): {best_fit:.3f}")
    print(f"  genome: {_describe(best_genome)}")
    print(
        "\nLower mean rank = the evolved optimizer beats more of the baseline panel\n"
        "on the (memorisation-proof) disguised instances. Widen --generations/--lam/\n"
        "--n-demos and enrich make_candidate to push further."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
