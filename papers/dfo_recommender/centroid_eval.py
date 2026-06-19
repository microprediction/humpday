"""Is the LLM-generated 'centroid' optimizer actually good? Honest head-to-head.

Strips away the simplex story: take the centroid optimizer (runs/simplex_warm_code/
centroid.py) and rank it, per instance, against strong baselines INCLUDING production
pycma — on demos it was NEVER selected on (held-out: the complement of the 16-demo
'spread' set used in the simplex run). Reports mean rank + win-rate + normalised
regret, multi-seed. Lower rank = better; rank 1 = best optimizer on that instance.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, random, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
import algo_dev as ad
from example_demos import DEMOS, disguise_demo
from simplex_blend import select_demos
import numpy as np
import nevergrad as ng

INF = float("inf")


def load_centroid():
    spec = importlib.util.spec_from_file_location("centroid_opt", "runs/simplex_warm_code/centroid.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.optimize


def run_ngcma(objective, n_trials, n_dim, seed):
    param = ng.p.Array(shape=(n_dim,), lower=0.0, upper=1.0)
    param.random_state.seed(seed)
    opt = ng.optimizers.registry["CMA"](parametrization=param, budget=n_trials, num_workers=1)
    best = INF
    for _ in range(n_trials):
        c = opt.ask()
        loss = float(objective([float(v) for v in np.asarray(c.value)]))
        opt.tell(c, loss)
        best = min(best, loss)
    return best


def run_one(name, opt_callable, inst, n_trials, seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        if name == "centroid":
            r = opt_callable(inst.objective, n_trials, inst.n_dim)
            return float(r[0]) if r else INF
        if name == "ngCMA(pycma)":
            return run_ngcma(inst.objective, n_trials, inst.n_dim, seed)
        return ad._panel_best(name, inst, n_trials, seed)  # humpday classics
    except Exception:  # noqa: BLE001
        return INF


def atomic_dump(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=16, help="held-out demos to test on")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--out", default="runs/centroid_eval.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    centroid = load_centroid()
    OPTS = ["centroid", "NelderMead", "DifferentialEvolution",
            "CMAEvolutionStrategy", "ngCMA(pycma)"]
    callbacks = {"centroid": centroid}

    # held-out = demos NOT in the 16-demo 'spread' set the simplex run used
    used = {id(d) for d in select_demos(16, "spread")}
    pool = [d for d in DEMOS if id(d) not in used]
    pool.sort(key=lambda d: d.n_dim)
    n = min(args.demos, len(pool))
    idx = sorted({round(k * (len(pool) - 1) / max(n - 1, 1)) for k in range(n)})
    held = [pool[i] for i in idx]
    print(f"held-out demos ({len(held)}, dims {held[0].n_dim}-{held[-1].n_dim}): "
          f"{', '.join(d.name for d in held)}\n", flush=True)

    ranks = {o: [] for o in OPTS}
    wins = {o: 0 for o in OPTS}
    regret_centroid = []
    rows = []
    total = len(held) * len(seeds)
    c = 0
    for demo in held:
        for s in seeds:
            c += 1
            inst = disguise_demo(demo, s)
            seed = 9000 + s
            vals = {o: run_one(o, callbacks.get(o), inst, args.trials, seed) for o in OPTS}
            finite = [v for v in vals.values() if v < INF]
            mn, mx = (min(finite), max(finite)) if finite else (0.0, 1.0)
            # per-instance rank (1 = best); normalised regret for centroid
            order = sorted(OPTS, key=lambda o: vals[o])
            best_v = vals[order[0]]
            for o in OPTS:
                ranks[o].append(1 + sum(1 for x in OPTS if vals[x] < vals[o] - 1e-12))
            if vals["centroid"] <= best_v + 1e-12:
                wins["centroid"] += 1
            reg = 0.0 if mx <= mn else (vals["centroid"] - mn) / (mx - mn)
            regret_centroid.append(reg)
            rows.append({"demo": demo.name, "n": demo.n_dim, "seed": s,
                         **{o: (None if vals[o] >= INF else round(vals[o], 5)) for o in OPTS}})
            print(f"[{c}/{total}] {demo.name:24s} n={demo.n_dim:3d} s={s} "
                  f"centroid_rank={ranks['centroid'][-1]} winner={order[0]}", flush=True)
            atomic_dump({"done": False, "opts": OPTS, "rows": rows}, args.out)

    summary = {o: {"mean_rank": round(sum(ranks[o]) / len(ranks[o]), 3),
                   "wins": wins.get(o, 0)} for o in OPTS}
    summary["centroid"]["mean_regret_vs_field"] = round(sum(regret_centroid) / len(regret_centroid), 4)
    atomic_dump({"done": True, "opts": OPTS, "n_instances": total,
                 "summary": summary, "rows": rows}, args.out)

    print("\n=== mean rank (1=best of 5) over held-out instances ===")
    for o in sorted(OPTS, key=lambda o: summary[o]["mean_rank"]):
        w = f"  outright-best on {summary[o]['wins']}/{total}" if o == "centroid" else ""
        print(f"  {summary[o]['mean_rank']:.3f}  {o}{w}")
    print(f"\ncentroid mean normalised regret vs field: {summary['centroid']['mean_regret_vs_field']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
