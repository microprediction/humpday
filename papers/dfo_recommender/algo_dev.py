"""
Goal-based optimizer development — evolve a black-box optimizer against the
disguised benchmark.

This is the capstone of the disguised-benchmark stack:

  example_applications/*  →  example_demos.disguised_demos()  →  fitness here

A *candidate optimizer* is produced from a 12-gene genome of behavioural knobs
(`make_candidate` — a DE/ES hybrid: rand/1 + current-to-best/1 moves, SHADE-style
F/CR self-adaptation, Gaussian + pattern-search local moves with a 1/5-rule step,
simulated-annealing acceptance, stagnation restarts). Its **fitness** is its
*normalised regret* vs a fixed baseline panel across many *disguised* instances
(`candidate_fitness`) — continuous (not integer rank) so the search has a
gradient, and disguised so the candidate can't win by memorising optimum
locations; it must genuinely search. A (1 + λ) evolution strategy (`evolve`),
warm-started from a sane genome, mutates to drive regret down. `candidate_mean_rank`
reports the interpretable rank. Swap in a richer template or a real meta-optimizer;
the loop and the memorisation-proof fitness are what matter.

    python papers/dfo_recommender/algo_dev.py --quick
    python papers/dfo_recommender/algo_dev.py --generations 12 --lam 6
"""

from __future__ import annotations

import argparse
import json
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
GENOME_LEN = 12

# Baseline panel the candidate is ranked against (kept small so a genome
# evaluation — which reruns the whole panel per instance — stays affordable).
PANEL = ["NelderMead", "DifferentialEvolution", "CMAEvolutionStrategy"]


def _clip01(x):
    return min(1.0, max(0.0, x))


def _decode_genome(genome):
    """Decode a [0,1]^12 genome into the behavioural knobs of the optimizer."""
    g = [_clip01(v) for v in genome] + [0.5] * (GENOME_LEN - len(genome))
    return {
        "pop": 4 + int(g[0] * 26),  # 4..30
        "F0": 0.3 + g[1] * 0.7,  # base differential weight
        "CR0": 0.05 + g[2] * 0.9,  # base crossover rate
        "ctb": g[3],  # P(current-to-best/1) vs rand/1
        "p_local": g[4] * 0.6,  # P(local move) vs DE move
        "sigma0": 0.01 + g[5] * 0.39,  # initial local step
        "adapt_sigma": g[6],  # 1/5-rule step adaptation strength
        "p_pattern": g[7],  # within a local move, P(pattern) vs Gaussian
        "adapt_fcr": g[8],  # SHADE-style F/CR self-adaptation strength
        "temp0": g[9] * 0.5,  # simulated-annealing acceptance temperature (0=greedy)
        "stagnate": 3 + int(g[10] * 12),  # generations of no gain before a restart
        "restart_frac": 0.2 + g[11] * 0.6,  # fraction reinitialised on restart
    }


def make_candidate(genome):
    """Build a concrete black-box optimizer from a genome in [0,1]^12.

    A unified DE/ES hybrid whose behaviour the genome selects and blends:
      - DE moves: rand/1 and current-to-best/1, mixed by `ctb`;
      - SHADE-style self-adaptation of F and CR from successful trials;
      - local moves around the incumbent: Gaussian or coordinate pattern search,
        with a per-member 1/5-success-rule step size;
      - simulated-annealing acceptance (greedy when temp0=0);
      - restart of the worst fraction on stagnation.
    Every move is budget-counted; the optimizer stops at exactly n_trials evals."""
    P = _decode_genome(genome)

    def optimize(objective, n_trials, n_dim, with_count=False):
        st = {"evals": 0, "best_f": INF, "best_x": None}

        def ev(x):
            x = [_clip01(xi) for xi in x]
            f = objective(x)
            st["evals"] += 1
            if f < st["best_f"]:
                st["best_f"], st["best_x"] = f, x[:]
            return f

        pop = min(P["pop"], max(4, n_trials // 2))
        pos = [[random.random() for _ in range(n_dim)] for _ in range(pop)]
        fit = [ev(p) for p in pos]
        sig = [P["sigma0"]] * pop
        muF, muCR = P["F0"], P["CR0"]
        spread = (max(fit) - min(fit)) or 1.0
        temp = P["temp0"] * spread
        gen = last_gain = 0

        while st["evals"] < n_trials:
            gen += 1
            sucF, sucCR = [], []
            gained = False
            for i in range(pop):
                if st["evals"] >= n_trials:
                    break
                local = random.random() < P["p_local"]
                if local:
                    base = st["best_x"]
                    if random.random() < P["p_pattern"]:
                        trial = base[:]
                        j = random.randrange(n_dim)
                        trial[j] += (1 if random.random() < 0.5 else -1) * sig[i]
                    else:
                        trial = [
                            base[k] + random.gauss(0, sig[i]) for k in range(n_dim)
                        ]
                    Fi = CRi = None
                else:
                    Fi = _clip01(random.gauss(muF, 0.1)) * 0.9 + 0.1
                    CRi = _clip01(random.gauss(muCR, 0.1))
                    if random.random() < P["ctb"]:
                        a, b = random.sample(range(pop), 2)
                        donor = [
                            pos[i][k]
                            + Fi * (st["best_x"][k] - pos[i][k])
                            + Fi * (pos[a][k] - pos[b][k])
                            for k in range(n_dim)
                        ]
                    else:
                        a, b, c = random.sample(range(pop), 3)
                        donor = [
                            pos[a][k] + Fi * (pos[b][k] - pos[c][k])
                            for k in range(n_dim)
                        ]
                    jr = random.randrange(n_dim)
                    trial = [
                        donor[k] if (random.random() < CRi or k == jr) else pos[i][k]
                        for k in range(n_dim)
                    ]
                ft = ev(trial)
                better = ft < fit[i]
                accept = better or (
                    temp > 0
                    and random.random() < math.exp(-(ft - fit[i]) / (temp + 1e-12))
                )
                if accept:
                    pos[i], fit[i] = trial, ft
                    if better and not local:
                        sucF.append(Fi)
                        sucCR.append(CRi)
                if local and P["adapt_sigma"] > 0:  # 1/5-rule on the local step
                    f = 1.0 + 0.2 * P["adapt_sigma"] * (1 if better else -1)
                    sig[i] = min(0.5, max(1e-3, sig[i] * f))
                gained = gained or better and ft <= st["best_f"] + 1e-12

            if P["adapt_fcr"] > 0 and sucF:  # SHADE memory (Lehmer mean of F)
                lr = 0.1 * P["adapt_fcr"]
                muF = (1 - lr) * muF + lr * (sum(x * x for x in sucF) / sum(sucF))
                muCR = (1 - lr) * muCR + lr * (sum(sucCR) / len(sucCR))
            temp *= 0.95  # cool
            if gained:
                last_gain = gen
            elif gen - last_gain >= P["stagnate"]:
                order = sorted(range(pop), key=lambda k: fit[k])
                for k in order[int(pop * (1 - P["restart_frac"])) :]:
                    if st["evals"] >= n_trials:
                        break
                    pos[k] = [random.random() for _ in range(n_dim)]
                    fit[k] = ev(pos[k])
                    sig[k] = P["sigma0"]
                last_gain = gen

        if with_count:
            return st["best_f"], st["best_x"], st["evals"]
        return st["best_f"], st["best_x"]

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


# A sane warm-start genome: moderate population, DE-focused, greedy acceptance
# (temp0=0), light local search + adaptation. Competitive out of the box, so the
# ES begins in a good region and refines rather than hunting from nonsense.
# order: pop,F0,CR0,ctb,p_local,sigma0,adapt_sigma,p_pattern,adapt_fcr,temp0,stagnate,restart_frac
DEFAULT_GENOME = [0.35, 0.45, 0.75, 0.30, 0.30, 0.25, 0.60, 0.30, 0.60, 0.0, 0.50, 0.40]


def candidate_fitness(genome, base_demos, seeds, n_trials, panel=PANEL):
    """Normalised regret (0 = candidate is best on every instance, 1 = worst) vs
    the baseline panel across the disguised instances — a CONTINUOUS fitness to
    minimise. Continuous (not integer rank) so the ES gets a gradient even when
    the candidate is close behind the panel rather than strictly winning."""
    candidate = make_candidate(genome)
    scores = []
    for i, demo in enumerate(base_demos):
        for s in seeds:
            inst = disguise_demo(demo, s)
            seed = 5000 + 31 * i + s
            cand = _candidate_best(candidate, inst, n_trials, seed)
            vals = [cand] + [_panel_best(a, inst, n_trials, seed) for a in panel]
            finite = [v for v in vals if v < INF]
            if cand >= INF or not finite:
                scores.append(1.0)
                continue
            mn, mx = min(finite), max(finite)
            scores.append(0.0 if mx <= mn else (cand - mn) / (mx - mn))
    return mean(scores) if scores else 1.0


def candidate_mean_rank(genome, base_demos, seeds, n_trials, panel=PANEL):
    """Reporting metric: mean rank of the candidate among {candidate} + panel
    (1 = best). Computed once at the end for interpretability."""
    candidate = make_candidate(genome)
    ranks = []
    for i, demo in enumerate(base_demos):
        for s in seeds:
            inst = disguise_demo(demo, s)
            seed = 5000 + 31 * i + s
            cand = _candidate_best(candidate, inst, n_trials, seed)
            others = [_panel_best(a, inst, n_trials, seed) for a in panel]
            ranks.append(1 + sum(1 for v in others if v < cand))
    return mean(ranks) if ranks else float(len(panel) + 1)


def _ck_payload(config, gen, parent, parent_fit, history, step, extra=None):
    payload = {
        "config": config or {},
        "panel": PANEL,
        "generation": gen,
        "best_fitness": parent_fit,
        "best_genome": parent,
        "genome_decoded": _describe(parent),
        "history": history,
        "step": step,
    }
    if extra:
        payload.update(extra)
    return payload


def _save_checkpoint(path, payload):
    """Atomic-ish JSON write (write to .tmp then rename)."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    Path(tmp).replace(path)


def evolve(
    generations,
    lam,
    base_demos,
    seeds,
    n_trials,
    rng_seed=0,
    checkpoint=None,
    resume=False,
    config=None,
):
    """(1 + λ) evolution strategy over the genome.

    When `checkpoint` is given, writes a JSON snapshot (best genome + decoded
    knobs + per-generation regret history) after every generation, so a long run
    on another machine is analysable and, with `resume=True`, restartable from
    where it left off. Returns (best_genome, best_fitness, history)."""
    random.seed(rng_seed)
    step = 0.25
    start_gen = 0
    if resume and checkpoint and Path(checkpoint).exists():
        ck = json.loads(Path(checkpoint).read_text())
        parent, parent_fit = ck["best_genome"], ck["best_fitness"]
        history, step = ck["history"], ck.get("step", step)
        start_gen = ck.get("generation", len(history) - 1)
        print(f"  resumed at gen {start_gen}: regret = {parent_fit:.4f}", flush=True)
    else:
        parent = list(DEFAULT_GENOME)  # warm start from a sane, competitive genome
        parent_fit = candidate_fitness(parent, base_demos, seeds, n_trials)
        history = [parent_fit]
        print(f"  gen 0: parent fitness (norm. regret) = {parent_fit:.4f}", flush=True)
        if checkpoint:
            _save_checkpoint(
                checkpoint, _ck_payload(config, 0, parent, parent_fit, history, step)
            )
    for gen in range(start_gen + 1, generations + 1):
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
            f"  gen {gen}: best regret = {parent_fit:.4f}  (best child {best_child_fit:.4f}, step {step:.2f})",
            flush=True,
        )
        if checkpoint:
            _save_checkpoint(
                checkpoint, _ck_payload(config, gen, parent, parent_fit, history, step)
            )
    return parent, parent_fit, history


def _describe(genome):
    d = _decode_genome(genome)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--lam", type=int, default=4)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument(
        "--n-demos", type=int, default=8, help="how many base demos to score on"
    )
    ap.add_argument(
        "--out",
        default=str(HERE / "algo_dev_run.json"),
        help="checkpoint JSON path written each generation ('' to disable)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume from the --out checkpoint if present",
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    generations, lam, trials = args.generations, args.lam, args.trials
    seeds = tuple(int(s) for s in args.seeds.split(","))
    n_demos = args.n_demos
    if args.quick:
        generations, lam, trials, n_demos, seeds = 4, 3, 50, 5, (0, 1)

    base = DEMOS[:n_demos]
    checkpoint = args.out or None
    config = {
        "generations": generations,
        "lam": lam,
        "seeds": list(seeds),
        "trials": trials,
        "n_demos": n_demos,
    }
    print(
        f"Evolving an optimizer: (1+{lam}) ES, {generations} generations,\n"
        f"fitness = normalised regret vs {PANEL}\n"
        f"across {len(base)} demos x {len(seeds)} disguised seeds, {trials} trials each."
    )
    if checkpoint:
        print(f"checkpoint: {checkpoint}{' (resuming)' if args.resume else ''}")
    print()
    best_genome, best_fit, history = evolve(
        generations,
        lam,
        base,
        seeds,
        trials,
        checkpoint=checkpoint,
        resume=args.resume,
        config=config,
    )
    rank = candidate_mean_rank(best_genome, base, seeds, trials)
    if checkpoint:  # final snapshot carries the (expensive) mean-rank readout
        _save_checkpoint(
            checkpoint,
            _ck_payload(
                config,
                generations,
                best_genome,
                best_fit,
                history,
                0.0,
                extra={"mean_rank": rank, "done": True},
            ),
        )
    print("\n=== Evolved optimizer ===")
    print(f"  normalised regret vs panel: {best_fit:.4f}  (0 = best on every instance)")
    print(
        f"  mean rank among {len(PANEL) + 1}: {rank:.3f}  (1 = beats the whole panel)"
    )
    print(f"  genome: {_describe(best_genome)}")
    print(
        "\nLower regret = the evolved optimizer is closer to (or beats) the best of\n"
        "the baseline panel on the memorisation-proof disguised instances. Widen\n"
        "--generations/--lam/--n-demos for a real study."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
