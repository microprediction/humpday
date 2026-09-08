"""Line vs plane: does an off-line (planar) third placement help?

A budget-matched, apples-to-apples test of the paper's central claim.
Each step spends two evaluations from the incumbent along a shared
random direction; the only difference between the strategies is where
the second of the two goes.

  LINE   -- the third point stays on the line: extrapolate past the
            trial when it improved, reflect back when it did not.
  PLANAR -- the third point is the equilateral apex over the base
            (incumbent, trial) in a random plane containing the line.
  RANDOM -- two fresh cube points per step (a control floor).

Everything else is identical across strategies for a given
(objective, dimension, seed): start point, direction stream, adaptive
step size. We report the median best value over seeds and, scale-free,
the fraction of seeds on which PLANAR beats LINE.

Self-contained: numpy + humpday classic objectives only.
Usage: python planar_experiment.py [--budget 120] [--seeds 24]
"""
import argparse
import json
import numpy as np

from humpday.objectives.classic import (
    rastrigin_on_cube, ackley_on_cube, griewank_on_cube, schwefel_on_cube,
    drop_wave_on_cube, salomon_on_cube, styblinski_tang_on_cube,
    rosenbrock_on_cube, zakharov_on_cube, rotated_hyper_ellipsoid_on_cube,
)

OBJECTIVES = {
    "rastrigin": rastrigin_on_cube, "ackley": ackley_on_cube,
    "griewank": griewank_on_cube, "schwefel": schwefel_on_cube,
    "drop_wave": drop_wave_on_cube, "salomon": salomon_on_cube,
    "styblinski_tang": styblinski_tang_on_cube, "rosenbrock": rosenbrock_on_cube,
    "zakharov": zakharov_on_cube, "hyper_ellipsoid": rotated_hyper_ellipsoid_on_cube,
}
# multimodal (correlated-structure home turf) vs unimodal/smooth
MULTIMODAL = {"rastrigin", "ackley", "griewank", "schwefel", "drop_wave", "salomon"}


def clip(x):
    return np.clip(x, 0.0, 1.0)


def orthonormal(v, rng):
    g = rng.standard_normal(len(v))
    g = g - (g @ v) * v
    n = np.linalg.norm(g)
    while n < 1e-9:
        g = rng.standard_normal(len(v))
        g = g - (g @ v) * v
        n = np.linalg.norm(g)
    return g / n


def run(obj, d, strategy, budget, seed):
    rng = np.random.default_rng(seed)
    f = lambda u: float(obj(list(u)))
    x = clip(rng.random(d)); fx = f(x)
    best_x, best_f = x.copy(), fx
    a = 0.3
    used = 1
    while used + 2 <= budget:
        v = rng.standard_normal(d); v = v / np.linalg.norm(v)
        x1 = clip(best_x + a * v); f1 = f(x1); used += 1
        if strategy == "random":
            x2 = clip(rng.random(d))
        elif strategy == "line":
            # extrapolate past the trial if it helped, else reflect back
            x2 = clip(best_x + (2.0 * a) * v) if f1 < best_f \
                else clip(best_x - a * v)
        else:  # planar: equilateral apex over the base in a random plane
            vp = orthonormal(v, rng)
            x2 = clip(0.5 * (best_x + x1) + (np.sqrt(3) / 2.0) * a * vp)
        f2 = f(x2); used += 1
        # new incumbent
        cand = [(fx, best_x)] if False else []
        for xx, ff in ((x1, f1), (x2, f2)):
            if ff < best_f:
                best_f, best_x = ff, xx.copy()
        # adaptive step: grow on progress, shrink on stall
        improved = min(f1, f2) < fx
        fx = best_f
        a = min(0.5, a * 1.3) if improved else max(1e-3, a * 0.7)
    return best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    args = ap.parse_args()
    strategies = ["line", "planar", "random"]
    rows = []
    planar_wins = {"multimodal": [0, 0], "unimodal": [0, 0]}  # [wins, total]
    for name, obj in OBJECTIVES.items():
        for d in args.dims:
            res = {s: [] for s in strategies}
            for seed in range(args.seeds):
                for s in strategies:
                    res[s].append(run(obj, d, s, args.budget, seed))
            med = {s: float(np.median(res[s])) for s in strategies}
            # paired seed wins: planar strictly below line
            wins = sum(1 for i in range(args.seeds)
                       if res["planar"][i] < res["line"][i])
            grp = "multimodal" if name in MULTIMODAL else "unimodal"
            planar_wins[grp][0] += wins
            planar_wins[grp][1] += args.seeds
            rows.append(dict(objective=name, d=d, group=grp,
                             line=med["line"], planar=med["planar"],
                             random=med["random"], planar_wins=wins,
                             seeds=args.seeds))
            print(f"{name:16s} d={d:<2} [{grp:10s}]  line={med['line']:.4g}  "
                  f"planar={med['planar']:.4g}  planar_wins={wins}/{args.seeds}")
    print("\n=== planar-beats-line seed-win rate ===")
    for grp, (w, t) in planar_wins.items():
        print(f"  {grp:10s}: {w}/{t} = {w/t:.3f}")
    out = dict(budget=args.budget, seeds=args.seeds, dims=args.dims,
               rows=rows, planar_win_rate=planar_wins)
    with open("planar_experiment_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote planar_experiment_results.json")


if __name__ == "__main__":
    main()
