"""
Schur-damped Bayesian optimization — does damping the GP kernel keep BayesOpt
from blowing up in higher dimensions?

Motivation (IDEAS.md §E): the GP posterior covariance IS a Schur complement,
    S = K** - K*X (K_XX + σ²I)^{-1} K_X*,
and in higher dimensions K_XX becomes ill-conditioned (near-duplicate points,
long length-scales), the inverse explodes, the posterior variance goes haywire,
and the acquisition misbehaves — BayesOpt "blows up". A γ-damping of (K_XX+σ²I)
toward its diagonal (the SAME `schur_damp` dial used for CMA-ES) restores
diagonal dominance => the matrix stays well-conditioned, and the model shrinks
toward its prior where data is too sparse to trust (most of the space, in high
dim). γ=1 -> full GP (vanilla), γ=0 -> independent observations (max damping).

Hypothesis: vanilla (γ=1) degrades / destabilises as n_dim grows; a damped
(γ<1) GP degrades gracefully. Tested by bucketing the disguised demos by
dimension and also counting numerical-instability events (Cholesky failures /
non-finite posteriors) per γ.

    python papers/dfo_recommender/schur_bayes.py --demos 12 --seeds 0,1 --trials 80
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from example_demos import DEMOS, disguise_demo  # noqa: E402
from schur_cma import schur_damp, schur_damp_blocks  # reuse the damping dials  # noqa: E402

INF = float("inf")


def _rbf(X1, X2, ls):
    d2 = (
        np.sum(X1 * X1, 1)[:, None]
        + np.sum(X2 * X2, 1)[None, :]
        - 2.0 * X1 @ X2.T
    )
    return np.exp(-0.5 * np.clip(d2, 0, None) / (ls * ls))


class _Stats:
    def __init__(self):
        self.unstable = 0  # Cholesky failures / non-finite posteriors


def gp_lcb(Xtr, ytr, Xq, ls, noise, gamma, kappa, stats):
    """Damped-GP Lower Confidence Bound at query points Xq (for minimisation)."""
    K = _rbf(Xtr, Xtr, ls) + noise * np.eye(len(Xtr))
    K = schur_damp(K, gamma)  # <-- the Schur dial on the conditioning matrix
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        stats.unstable += 1
        L = np.linalg.cholesky(K + 1e-3 * np.eye(len(K)))
    ymean = ytr.mean()
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr - ymean))
    Ks = _rbf(Xtr, Xq, ls)
    mu = ymean + Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = 1.0 - np.sum(v * v, 0)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(var)):
        stats.unstable += 1
        mu = np.nan_to_num(mu, nan=ymean)
        var = np.nan_to_num(var, nan=0.0)
    std = np.sqrt(np.clip(var, 0, None))
    return mu - kappa * std


def bayes_opt(objective, n_trials, n_dim, gamma=1.0, seed=0, kappa=2.0):
    rng = np.random.default_rng(seed)
    stats = _Stats()
    n_init = min(max(2 * n_dim, 5), max(5, n_trials // 2))
    X = rng.random((n_init, n_dim))
    y = np.array([float(objective(x.tolist())) for x in X])
    evals = n_init
    noise = 1e-6
    pool_size = min(40 * n_dim + 50, 1200)
    while evals < n_trials:
        # median-heuristic length scale on current data (robust across dims)
        if len(X) > 1:
            dd = np.sqrt(np.clip(
                np.sum(X * X, 1)[:, None] + np.sum(X * X, 1)[None, :] - 2 * X @ X.T, 0, None))
            ls = max(np.median(dd[dd > 0]) if np.any(dd > 0) else 0.3, 0.05)
        else:
            ls = 0.3
        cand = rng.random((pool_size, n_dim))
        acq = gp_lcb(X, y, cand, ls, noise, gamma, kappa, stats)
        x_next = cand[int(np.argmin(acq))]
        f = float(objective(x_next.tolist()))
        X = np.vstack([X, x_next])
        y = np.append(y, f)
        evals += 1
    return float(y.min()), stats.unstable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", type=int, default=12)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--gammas", default="0.0,0.5,0.75,0.9,1.0")
    ap.add_argument("--skip", default="bowling")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    base = [d for d in DEMOS if d.name not in skip][: args.demos]
    seeds = tuple(int(s) for s in args.seeds.split(","))
    gammas = [float(g) for g in args.gammas.split(",")]
    variants = [(f"γ={g:.2f}", g) for g in gammas]

    DIM_CUT = 10
    buckets = {b: {lab: [] for lab, _ in variants} for b in ("all", "low(<10)", "high(>=10)")}
    unstable = {lab: 0 for lab, _ in variants}
    print(
        f"Schur-damped BayesOpt γ-sweep: {len(base)} demos x {len(seeds)} seeds, "
        f"{args.trials} trials. high-dim cut n>={DIM_CUT}.\n"
    )
    for i, demo in enumerate(base):
        b = "high(>=10)" if demo.n_dim >= DIM_CUT else "low(<10)"
        for s in seeds:
            inst = disguise_demo(demo, s)
            vals = {}
            for lab, g in variants:
                try:
                    v, unst = bayes_opt(inst.objective, args.trials, inst.n_dim, gamma=g, seed=2000 + i * 7 + s)
                    unstable[lab] += unst
                except Exception as e:  # noqa: BLE001
                    print(f"    ! {lab} on {inst.name}: {e}")
                    v = INF
                vals[lab] = v
            finite = [v for v in vals.values() if v < INF]
            mn, mx = (min(finite), max(finite)) if finite else (0.0, 1.0)
            for lab in vals:
                nr = 0.0 if mx <= mn or vals[lab] >= INF else (vals[lab] - mn) / (mx - mn)
                buckets["all"][lab].append(nr)
                buckets[b][lab].append(nr)

    n_high = sum(1 for d in base if d.n_dim >= DIM_CUT)
    print(f"(high-dim: {n_high} demos, low-dim: {len(base) - n_high} demos)\n")
    for bname in ("all", "low(<10)", "high(>=10)"):
        reg = buckets[bname]
        table = sorted(((lab, mean(reg[lab]) if reg[lab] else INF) for lab, _ in variants), key=lambda t: t[1])
        print(f"=== {bname}: BayesOpt γ-sweep (normalised regret; lower = better) ===")
        for lab, r in table:
            print(f"  {lab:10s} {r:.4f}  {'#' * int(r * 40)}")
        print()
    print("=== numerical-instability events (Cholesky failures / non-finite posteriors) ===")
    for lab, _ in variants:
        print(f"  {lab:10s} {unstable[lab]}")
    print(
        "\nRead: if vanilla γ=1.00 shows more instability AND worse high-dim "
        "regret while a damped γ<1 stays stable, damping extends BayesOpt to "
        "higher dimensions — the hypothesis holds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
