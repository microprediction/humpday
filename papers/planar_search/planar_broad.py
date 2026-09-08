"""Broad line-vs-plane test across the full HumpDay catalog:
all 51 analytic objectives (at d=2,5,10) and all physics simulators
(at their native dimension). Same budget-matched protocol and the same
four strategies as planar_experiment.py, which this imports.
Usage: python planar_broad.py [--budget 90] [--seeds 16]
"""
import argparse, json, os, sys, importlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "example_applications"))
from planar_experiment import run  # same strategies, same protocol

from humpday.objectives.allobjectives import OBJECTIVES

def analytic_tasks():
    for f in OBJECTIVES:
        for d in (2, 5, 10):
            yield (f.__name__, "analytic", d, f)

def sim_tasks():
    appdir = os.path.join(HERE, "..", "..", "example_applications")
    for demo in sorted(os.listdir(appdir)):
        p = os.path.join(appdir, demo, "problem.py")
        if not os.path.isfile(p) or demo.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{demo}.problem")
        except Exception:
            continue
        obj = None
        for name in ("objective", "simulate_throw_objective"):
            if hasattr(mod, name):
                obj = getattr(mod, name); break
        nd = getattr(mod, "N_DIM", None)
        if obj is None or nd is None:
            continue
        # keep only LIGHT simulators: skip if an eval is slow
        try:
            import time as _t
            x0 = list(np.random.rand(int(nd)))
            t0 = _t.perf_counter()
            for _ in range(3):
                obj(x0)
            ms = (_t.perf_counter() - t0) / 3 * 1000.0
        except Exception:
            continue
        if ms > 2.0:  # heavier than ~2 ms/eval: skip (chess, protein, art, ...)
            continue
        yield (demo, "sim", int(nd), obj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=90)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--source", choices=["analytic","sim","all"], default="all")
    ap.add_argument("--max_dim", type=int, default=999)
    ap.add_argument("--out", default="planar_broad_results.json")
    args = ap.parse_args()
    strategies = ["line", "planar", "planar_informed", "random"]
    rows = []
    tasks = []
    if args.source in ("analytic","all"): tasks += list(analytic_tasks())
    if args.source in ("sim","all"): tasks += list(sim_tasks())
    tasks = [t for t in tasks if t[2] <= args.max_dim]
    outpath = os.path.join(HERE, args.out)
    for name, src, d, obj in tasks:
        res = {s: [] for s in strategies}
        try:
            for seed in range(args.seeds):
                for s in strategies:
                    res[s].append(run(obj, d, s, args.budget, seed))
        except Exception:
            continue
        pw = sum(1 for i in range(args.seeds) if res["planar"][i] < res["line"][i])
        iw = sum(1 for i in range(args.seeds) if res["planar_informed"][i] < res["line"][i])
        rows.append(dict(objective=name, source=src, d=d,
                         planar_wins=pw, informed_wins=iw, seeds=args.seeds))
        print(f"[{len(rows):3d}] {name:22s} {src:8s} d={d:<3} "
              f"planar {pw}/{args.seeds}  informed {iw}/{args.seeds}", flush=True)
        json.dump(dict(budget=args.budget, seeds=args.seeds, rows=rows),
                  open(outpath, "w"))  # incremental: survives a kill
    # aggregate by source x dimension bucket
    def rate(sel, key):
        w = sum(r[key] for r in rows if sel(r)); t = sum(r["seeds"] for r in rows if sel(r))
        return (w, t, w / t if t else float("nan"))
    print(f"tasks scored: {len(rows)} (budget {args.budget}, seeds {args.seeds})")
    for src in ("analytic", "sim", None):
        for lab, lo, hi in (("low d<=3", 0, 3), ("high d>=4", 4, 999)):
            sel = lambda r, src=src, lo=lo, hi=hi: (src is None or r["source"] == src) and lo <= r["d"] <= hi
            pw = rate(sel, "planar_wins"); iw = rate(sel, "informed_wins")
            tag = src or "ALL"
            print(f"  {tag:8s} {lab:9s}: planar {pw[2]:.3f} ({pw[0]}/{pw[1]})  "
                  f"informed {iw[2]:.3f} ({iw[0]}/{iw[1]})")
    json.dump(dict(budget=args.budget, seeds=args.seeds, rows=rows),
              open(outpath, "w"), indent=2)
    print(f"wrote {args.out} ({len(rows)} tasks)")

if __name__ == "__main__":
    main()
