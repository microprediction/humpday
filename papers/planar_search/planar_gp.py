"""Does a fitted GP aim the off-line probe better than fixed geometry?

Same budget-matched protocol as planar_experiment.py, but the planar
strategy now chooses its one off-line point by a GP acquisition rather
than a fixed equilateral apex. Everything the GP uses (history) is free;
each step still spends two objective evaluations.

  line      -- third point on the line (extrapolate / reflect).
  planar_gp -- third point = the in-plane candidate maximizing expected
               improvement under a GP fit to recent history; candidates
               span several transverse directions and heights.
  planar    -- fixed equilateral apex in a random plane (the reference).

Minimal RBF GP (numpy); history capped for speed. Usage:
  python planar_gp.py --budget 90 --seeds 16 [--objectives rastrigin ...]
"""
import argparse, json
import numpy as np
from humpday.objectives.allobjectives import OBJECTIVES

OBJ = {f.__name__.replace("_on_cube", ""): f for f in OBJECTIVES}


def clip(x):
    return np.clip(x, 0.0, 1.0)


def orthonormal(v, rng):
    g = rng.standard_normal(len(v)); g = g - (g @ v) * v
    n = np.linalg.norm(g)
    while n < 1e-9:
        g = rng.standard_normal(len(v)); g = g - (g @ v) * v; n = np.linalg.norm(g)
    return g / n


class GP:
    """Zero-mean RBF GP on standardized targets; length scale = median
    pairwise distance of the retained points."""
    def __init__(self, X, y, cap=40, noise=1e-6):
        if len(X) > cap:
            X, y = X[-cap:], y[-cap:]
        self.X = np.asarray(X); self.y = np.asarray(y, float)
        self.mu_y = self.y.mean(); self.sd_y = self.y.std() or 1.0
        yz = (self.y - self.mu_y) / self.sd_y
        n = len(self.X)
        D = np.sqrt(((self.X[:, None] - self.X[None]) ** 2).sum(-1))
        ell = np.median(D[D > 0]) if np.any(D > 0) else 0.2
        self.ell = max(ell, 1e-3)
        K = np.exp(-0.5 * (D / self.ell) ** 2) + noise * np.eye(n)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yz))

    def ei(self, Xs, best_z):
        Xs = np.atleast_2d(Xs)
        d = np.sqrt(((Xs[:, None] - self.X[None]) ** 2).sum(-1))
        Ks = np.exp(-0.5 * (d / self.ell) ** 2)
        mu = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        s2 = np.clip(1.0 - (v ** 2).sum(0), 1e-12, None)
        s = np.sqrt(s2)
        z = (best_z - mu) / s  # minimization: improvement below best
        from math import erf, pi
        Phi = 0.5 * (1 + np.vectorize(lambda t: erf(t / np.sqrt(2)))(z))
        phi = np.exp(-0.5 * z * z) / np.sqrt(2 * pi)
        return (best_z - mu) * Phi + s * phi


def run(obj, d, strategy, budget, seed):
    rng = np.random.default_rng(seed)
    f = lambda u: float(obj(list(u)))
    x = clip(rng.random(d)); fx = f(x)
    X, Y = [x.copy()], [fx]
    best_x, best_f = x.copy(), fx
    a = 0.3; used = 1
    while used + 2 <= budget:
        v = rng.standard_normal(d); v = v / np.linalg.norm(v)
        x1 = clip(best_x + a * v); f1 = f(x1); used += 1
        X.append(x1.copy()); Y.append(f1)
        if strategy == "line":
            x2 = clip(best_x + 2 * a * v) if f1 < best_f else clip(best_x - a * v)
        elif strategy == "planar":
            vp = orthonormal(v, rng)
            x2 = clip(0.5 * (best_x + x1) + (np.sqrt(3) / 2) * a * vp)
        else:  # GP-guided placement; line_gp restricts to the line
            base = 0.5 * (best_x + x1)
            online = [clip(best_x + t * a * v)
                      for t in (-1.5, -1.0, -0.5, 1.5, 2.0, 2.5, 3.0)]
            cands = list(online)
            if strategy == "planar_gp":  # add off-line (in-plane) candidates
                for _ in range(6):
                    vp = orthonormal(v, rng)
                    for h in (0.5, 1.0, 1.5):
                        cands.append(clip(base + h * (np.sqrt(3) / 2) * a * vp))
            cands = np.array(cands)
            gp = GP(X, Y)
            best_z = (best_f - gp.mu_y) / gp.sd_y
            x2 = cands[int(np.argmax(gp.ei(cands, best_z)))]
        f2 = f(x2); used += 1
        X.append(x2.copy()); Y.append(f2)
        for xx, ff in ((x1, f1), (x2, f2)):
            if ff < best_f:
                best_f, best_x = ff, xx.copy()
        improved = min(f1, f2) < fx; fx = best_f
        a = min(0.5, a * 1.3) if improved else max(1e-3, a * 0.7)
    return best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=90)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--objectives", nargs="+",
                    default=["rastrigin", "ackley", "griewank", "schwefel",
                             "drop_wave", "rosenbrock", "zakharov", "salomon",
                             "styblinski_tang", "rotated_hyper_ellipsoid"])
    args = ap.parse_args()
    strat = ["line", "line_gp", "planar_gp"]
    rows = []
    for name in args.objectives:
        obj = OBJ[name]
        for d in args.dims:
            res = {s: [run(obj, d, s, args.budget, sd) for sd in range(args.seeds)]
                   for s in strat}
            vs_line = sum(1 for i in range(args.seeds) if res["planar_gp"][i] < res["line"][i])
            vs_gp = sum(1 for i in range(args.seeds) if res["planar_gp"][i] < res["line_gp"][i])
            rows.append(dict(objective=name, d=d, gp_vs_line=vs_line,
                             gp_vs_linegp=vs_gp, seeds=args.seeds))
            print(f"{name:20s} d={d:<2} planar_gp>line {vs_line}/{args.seeds}  "
                  f"planar_gp>line_gp {vs_gp}/{args.seeds}", flush=True)
    for key, lab in (("gp_vs_line", "gp>line"), ("gp_vs_linegp", "gp>line_gp")):
        for lo, hi, dl in ((0, 3, "low"), (4, 999, "high")):
            w = sum(r[key] for r in rows if lo <= r["d"] <= hi)
            t = sum(r["seeds"] for r in rows if lo <= r["d"] <= hi)
            print(f"  {lab:10s} {dl:4s}: {w}/{t} = {w/t:.3f}")
    json.dump(dict(rows=rows), open("planar_gp_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
