"""E1 — Benchmark-validity rank correlation (the SAD paper's spine).

Question: does the optimizer leaderboard on SYNTHETIC analytic functions predict the
leaderboard on DISGUISED REAL-WORLD objectives? We rank a fixed panel of optimizers
on each suite (per budget) and report Kendall-tau / Spearman between the two
leaderboards. Low correlation => synthetic benchmarks mis-rank optimizers for real
problems.

Panel: all humpday optimizers + ngCMA (production pycma via nevergrad).
Crash-safe: atomic per-instance checkpoints to --out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
import algo_dev as ad  # noqa: E402
import nevergrad as ng
import numpy as np
from example_demos import DEMOS, disguise_demo  # noqa: E402
from nevergrad.functions import ArtificialFunction
from scipy.stats import kendalltau, spearmanr

from humpday.optimizers.alloptimizers import (  # noqa: E402
    PURE_OPTIMIZERS,
    pure_optimize,
)

INF = float("inf")
SYNTH = ["sphere", "ellipsoid", "cigar", "rastrigin", "rosenbrock"]

# Per-call watchdog: a runaway optimizer that ignores its eval budget would spin a
# core forever and wedge the whole run (one wedged once for 3.5 days). SIGALRM (main
# thread, subprocess) interrupts the offending pure-Python call so we score it INF
# (worst) and move on. Set high enough that a genuinely slow-but-finite call (slow
# optimizers on high-dim demos at the largest budget have been seen near ~2 min) is
# never falsely killed, since a false INF would bias that optimizer's rank.
PER_CALL_TIMEOUT = int(os.environ.get("RANKCORR_CALL_TIMEOUT", "600"))


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def run_ngcma(objective, n_trials, n_dim, seed):
    param = ng.p.Array(shape=(n_dim,), lower=0.0, upper=1.0)
    param.random_state.seed(seed)
    opt = ng.optimizers.registry["CMA"](
        parametrization=param, budget=n_trials, num_workers=1
    )
    best = INF
    for _ in range(n_trials):
        c = opt.ask()
        loss = float(objective([float(v) for v in np.asarray(c.value)]))
        opt.tell(c, loss)
        best = min(best, loss)
    return best


def run_opt(name, objective, n_dim, n_trials, seed):
    random.seed(seed)
    np.random.seed(seed)
    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(PER_CALL_TIMEOUT)
    try:
        if name == "ngCMA":
            return run_ngcma(objective, n_trials, n_dim, seed)
        bv, _ = pure_optimize(objective, name, n_trials, n_dim)
        return float(bv)
    except _Timeout:
        print(
            f"    !! {name} exceeded {PER_CALL_TIMEOUT}s "
            f"(n_dim={n_dim}, budget={n_trials}) -> INF",
            flush=True,
        )
        return INF
    except Exception:  # noqa: BLE001
        return INF
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def make_synth(name, n, seed):
    af = ArtificialFunction(name=name, block_dimension=n, rotation=True)
    shift = np.random.default_rng(seed).uniform(-2.0, 2.0, size=n)

    def f(u):
        z = (np.asarray(u, dtype=float) - 0.5) * 10.0 - shift
        return float(af.function(z))

    return f


def atomic_dump(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def per_instance_ranks(vals_by_opt):
    """Return {opt: rank} (1=best) for one instance; INF handled (worst)."""
    opts = list(vals_by_opt)
    order = sorted(opts, key=lambda o: vals_by_opt[o])
    return {
        o: 1 + sum(1 for x in opts if vals_by_opt[x] < vals_by_opt[o] - 1e-12)
        for o in opts
    }


def leaderboard(rank_lists):
    """rank_lists: list of {opt: rank}; return {opt: mean_rank}."""
    opts = list(rank_lists[0])
    return {o: sum(rl[o] for rl in rank_lists) / len(rank_lists) for o in opts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--synth-dims", default="5,20,40")
    ap.add_argument("--real-demos", type=int, default=30)
    ap.add_argument("--budgets", default="60,120,240")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/rankcorr.json")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    synth_dims = [int(d) for d in a.synth_dims.split(",")]
    budgets = [int(b) for b in a.budgets.split(",")]
    funcs = SYNTH
    # BayesianOpt is EXCLUDED: GP fitting is O(N^3) and its acquisition search blows
    # up in high dimensions — it stalls the sweep on high-dim budget-240 instances.
    # (Rule: no Bayesian optimization in high-dimensional experiments.)
    panel = [o for o in sorted(PURE_OPTIMIZERS) if o != "BayesianOpt"] + ["ngCMA"]
    if a.quick:
        seeds, synth_dims, budgets, funcs = [0], [5], [40], ["sphere", "cigar"]
        panel = ["NelderMead", "DifferentialEvolution", "CMAEvolutionStrategy", "ngCMA"]

    # real-world instances: dim-spread subset
    ds = sorted(DEMOS, key=lambda d: d.n_dim)
    nreal = min(a.real_demos, len(ds))
    idx = sorted({round(k * (len(ds) - 1) / max(nreal - 1, 1)) for k in range(nreal)})
    real_demos = [ds[i] for i in idx]

    # build instance list: (suite, label, objective_factory, n_dim, seed)
    instances = []
    for d in synth_dims:
        for fn in funcs:
            for s in seeds:
                instances.append(
                    ("synthetic", f"{fn}-{d}d", make_synth(fn, d, s), d, s)
                )
    for dm in real_demos:
        for s in seeds:
            inst = disguise_demo(dm, s)
            instances.append(("real", dm.name, inst.objective, dm.n_dim, s))

    results = []  # {budget, suite, label, n_dim, seed, ranks:{opt:rank}}
    done = set()
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out))
            results = prev.get("results", [])
            done = {(r["budget"], r["suite"], r["label"], r["seed"]) for r in results}
        except Exception:  # noqa: BLE001
            pass

    total = len(budgets) * len(instances)
    c = 0
    for budget in budgets:
        for suite, label, obj, n_dim, seed in instances:
            c += 1
            key = (budget, suite, label, seed)
            if key in done:
                continue
            vals = {o: run_opt(o, obj, n_dim, budget, 7000 + seed) for o in panel}
            ranks = per_instance_ranks(vals)
            results.append(
                {
                    "budget": budget,
                    "suite": suite,
                    "label": label,
                    "n_dim": n_dim,
                    "seed": seed,
                    "ranks": ranks,
                }
            )
            print(
                f"[{c}/{total}] b={budget} {suite:9s} {label:22s} "
                f"best={min(r for r in ranks if ranks[r] == 1)}",
                flush=True,
            )
            atomic_dump({"done": False, "panel": panel, "results": results}, a.out)

    # synthesize leaderboards + correlation per budget
    summary = {}
    for budget in budgets:
        lbs = {}
        for suite in ("synthetic", "real"):
            rls = [
                r["ranks"]
                for r in results
                if r["budget"] == budget and r["suite"] == suite
            ]
            if rls:
                lbs[suite] = leaderboard(rls)
        if "synthetic" in lbs and "real" in lbs:
            opts = [o for o in panel if o in lbs["synthetic"] and o in lbs["real"]]
            sv = [lbs["synthetic"][o] for o in opts]
            rv = [lbs["real"][o] for o in opts]
            kt = kendalltau(sv, rv)
            sp = spearmanr(sv, rv)
            summary[str(budget)] = {
                "kendall_tau": round(float(kt.statistic), 4),
                "kendall_p": round(float(kt.pvalue), 4),
                "spearman": round(float(sp.statistic), 4),
                "synthetic_leaderboard": {
                    o: round(lbs["synthetic"][o], 3) for o in opts
                },
                "real_leaderboard": {o: round(lbs["real"][o], 3) for o in opts},
            }
    atomic_dump(
        {"done": True, "panel": panel, "summary": summary, "results": results}, a.out
    )

    print("\n=== rank correlation: synthetic vs real leaderboards ===")
    for b, s in summary.items():
        print(
            f"  budget {b}: Kendall-tau={s['kendall_tau']} (p={s['kendall_p']}) "
            f"Spearman={s['spearman']}"
        )
    print(
        "\nLow tau => synthetic benchmarks do NOT predict real-world optimizer ranking."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
