"""
Schur-damped CMA-ES — does a covariance-damping dial γ help low-budget CMA-ES?

Motivation (IDEAS.md §E, schur.microprediction.org): Schur damping is a dial
γ∈[0,1] interpolating between a diagonal/block covariance (γ=0, robust when the
covariance is undersampled) and the full covariance (γ=1), with a reliability
γ* set by how undersampled the estimate is. CMA-ES adapts a full covariance C
from only λ samples per generation — undersampled exactly when λ is small vs the
dimension n, which is the low-budget DFO regime. So CMA-ES is the prime
candidate.

This is a compact standalone CMA-ES (Hansen, rank-1 + rank-μ) with one extra
step: after each C update, `C <- schur_damp(C, γ)` before sampling.

THE DAMPING (honest scope): `schur_damp` here is the **correlation-damping form**
of the dial — it preserves the variances (diag C) and scales the *correlations*
by γ, so γ=0 → diagonal (sep-CMA-like), γ=1 → full C, exactly the endpoints Schur
damping interpolates. It is NOT yet the faithful recursive Schur-complement block
construction (cluster the coordinates, augment each block with the Schur
complement of the others) — that is the follow-up *if* an interior γ or the
reliability γ* shows signal here. This script answers the cheap prior question:
"is there any interior γ that beats both the diagonal and full endpoints?"

    python papers/dfo_recommender/schur_cma.py            # sweep γ on disguised demos
    python papers/dfo_recommender/schur_cma.py --demos 8 --seeds 0,1,2 --trials 150
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

INF = float("inf")


def schur_damp(C, gamma):
    """Correlation-damping form of the Schur dial: keep variances, scale
    correlations by γ. γ=1 -> full C, γ=0 -> diag(C). Returns an SPD matrix."""
    d = np.sqrt(np.clip(np.diag(C), 1e-30, None))
    R = C / np.outer(d, d)  # correlation matrix
    R = (1.0 - gamma) * np.eye(len(C)) + gamma * R  # damp correlations
    Cg = R * np.outer(d, d)
    # tiny PSD safeguard
    w = np.linalg.eigvalsh(Cg)
    if w.min() < 1e-12:
        Cg = Cg + (1e-12 - w.min()) * np.eye(len(C))
    return Cg


def schur_damp_blocks(C, gamma, block_size):
    """Block-Schur damping: keep the WITHIN-block covariance intact (the
    structure we trust — e.g. one turbine's x,y, one atom's x,y,z) and damp only
    the CROSS-block covariance by γ. γ=1 -> full C, γ=0 -> block-diagonal (NOT
    fully diagonal — within-block correlations survive). `block_size` groups
    contiguous coordinates into entities."""
    n = len(C)
    blk = np.arange(n) // block_size
    mask = np.where(blk[:, None] == blk[None, :], 1.0, gamma)
    Cg = C * mask  # diagonal is within-block, so untouched
    w = np.linalg.eigvalsh(Cg)
    if w.min() < 1e-12:
        Cg = Cg + (1e-12 - w.min()) * np.eye(n)
    return Cg


def schur_damp_seriated(C, gamma, dist_thresh=0.5):
    """HRP-style block discovery: cluster the correlation matrix (seriation /
    quasi-diagonalisation) into data-driven blocks, then damp CROSS-cluster
    covariance by γ. Unlike fixed contiguous blocks, genuinely coupled
    coordinates (e.g. interacting turbines) land in the SAME cluster and are not
    damped — fixing the 'wrong blocks' failure of assumed contiguous structure."""
    n = len(C)
    if n < 3:
        return schur_damp(C, gamma)
    d = np.sqrt(np.clip(np.diag(C), 1e-30, None))
    R = np.clip(C / np.outer(d, d), -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - R), 0.0, None))  # HRP correlation distance
    np.fill_diagonal(dist, 0.0)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=dist_thresh, criterion="distance")  # data-driven #clusters
    mask = np.where(labels[:, None] == labels[None, :], 1.0, gamma)
    Cg = C * mask
    w = np.linalg.eigvalsh(Cg)
    if w.min() < 1e-12:
        Cg = Cg + (1e-12 - w.min()) * np.eye(n)
    return Cg


def schur_damp_adaptive(C, lam, c=1.0):
    """Reliability-adaptive damping (no fixed γ). Each correlation ρ_ij is kept
    in proportion to how much it exceeds the sampling-noise floor ~1/λ: a
    per-entry Wiener/James-Stein weight s = ρ²/(ρ² + c/λ). Correlations below the
    noise floor (spurious, undersampled) are shrunk toward 0; genuinely strong
    ones survive. Automatically damps more in high dim (small λ → high floor) and
    keeps real structure — unifying the fixed-γ / seriated results into one
    self-tuning rule."""
    n = len(C)
    d = np.sqrt(np.clip(np.diag(C), 1e-30, None))
    R = np.clip(C / np.outer(d, d), -1.0, 1.0)
    noise = c / max(lam, 1)
    s = R * R / (R * R + noise)  # reliability weight in [0,1)
    np.fill_diagonal(s, 1.0)
    Cg = (R * s) * np.outer(d, d)
    w = np.linalg.eigvalsh(Cg)
    if w.min() < 1e-12:
        Cg = Cg + (1e-12 - w.min()) * np.eye(n)
    return Cg


def cma_es(objective, n_trials, n_dim, gamma=1.0, reliability=False, seed=0,
           block_size=None, seriate=False, dist_thresh=0.5, adaptive=False):
    """Minimal CMA-ES on [0,1]^n with a Schur-damping dial. Returns best value."""
    rng = np.random.default_rng(seed)
    n = n_dim
    lam = 4 + int(3 * math.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w**2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    # reliability γ*: more effective samples per dimension -> trust correlations
    if reliability:
        gamma = float(np.clip(mueff / (mueff + n), 0.0, 1.0))

    mean_x = 0.3 + 0.4 * rng.random(n)
    sigma = 0.2
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    evals = 0
    best = INF

    def ev(x):
        nonlocal evals, best
        xc = np.clip(x, 0, 1)
        f = float(objective(xc.tolist()))
        evals += 1
        if f < best:
            best = f
        return f

    gen = 0
    while evals < n_trials:
        gen += 1
        if adaptive:
            Cd = schur_damp_adaptive(C, lam)
        elif seriate:
            Cd = schur_damp_seriated(C, gamma, dist_thresh)
        elif block_size:
            Cd = schur_damp_blocks(C, gamma, block_size)
        else:
            Cd = schur_damp(C, gamma)
        try:
            L = np.linalg.cholesky(Cd)
        except np.linalg.LinAlgError:
            L = np.eye(n)
        pop = []
        for _ in range(lam):
            if evals >= n_trials:
                break
            z = rng.standard_normal(n)
            y = L @ z
            x = mean_x + sigma * y
            f = ev(x)
            pop.append((f, x, y))
        if len(pop) < mu:
            break
        pop.sort(key=lambda t: t[0])
        old_mean = mean_x.copy()
        mean_x = sum(w[i] * pop[i][1] for i in range(mu))
        yw = (mean_x - old_mean) / sigma
        # use damped C for invsqrt (consistent with what we sampled from)
        D, B = np.linalg.eigh(Cd)
        invsqrtC = B @ np.diag(1.0 / np.sqrt(np.clip(D, 1e-30, None))) @ B.T
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ yw)
        hsig = 1.0 if np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (n + 1)) * chiN else 0.0
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
        artmp = np.array([(pop[i][1] - old_mean) / sigma for i in range(mu)])
        C = (
            (1 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            + cmu * (artmp.T * w) @ artmp
        )
        C = np.triu(C) + np.triu(C, 1).T
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, 1.0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", type=int, default=6)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--gammas", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--skip", default="", help="comma-sep demo names to exclude (slow ones)")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    base = [d for d in DEMOS if d.name not in skip][: args.demos]
    seeds = tuple(int(s) for s in args.seeds.split(","))
    gammas = [float(g) for g in args.gammas.split(",")]
    variants = [(f"γ={g:.2f}", dict(gamma=g)) for g in gammas]
    variants.append(("γ*=reliab", dict(reliability=True)))

    print(
        f"Schur-damped CMA-ES γ-sweep: {len(base)} demos x {len(seeds)} seeds, "
        f"{args.trials} trials.\n"
        f"demos (n_dim): {', '.join(f'{d.name}({d.n_dim})' for d in base)}\n"
    )

    # per-instance min-max normalised regret within the variant set, bucketed
    # by dimension (the undersampling regime where damping should matter).
    DIM_CUT = 10  # n_dim >= DIM_CUT is the "high-dim / undersampled" bucket
    buckets = {"all": {label: [] for label, _ in variants},
               "low(<10)": {label: [] for label, _ in variants},
               "high(>=10)": {label: [] for label, _ in variants}}
    for i, demo in enumerate(base):
        b = "high(>=10)" if demo.n_dim >= DIM_CUT else "low(<10)"
        for s in seeds:
            inst = disguise_demo(demo, s)
            vals = {}
            for label, kw in variants:
                try:
                    vals[label] = cma_es(inst.objective, args.trials, inst.n_dim, seed=1000 + i * 7 + s, **kw)
                except Exception as e:  # noqa: BLE001
                    print(f"    ! {label} on {inst.name}: {e}")
                    vals[label] = INF
            finite = [v for v in vals.values() if v < INF]
            mn, mx = (min(finite), max(finite)) if finite else (0.0, 1.0)
            for label in vals:
                v = vals[label]
                nr = 0.0 if mx <= mn or v >= INF else (v - mn) / (mx - mn)
                buckets["all"][label].append(nr)
                buckets[b][label].append(nr)

    n_high = sum(1 for d in base if d.n_dim >= DIM_CUT)
    print(f"(bucket sizes: high-dim n>={DIM_CUT}: {n_high} demos, low-dim: {len(base) - n_high} demos)\n")
    for bname in ("all", "low(<10)", "high(>=10)"):
        reg = buckets[bname]
        table = sorted(((label, mean(reg[label]) if reg[label] else INF) for label, _ in variants), key=lambda t: t[1])
        print(f"=== {bname}: γ-sweep (within-instance normalised regret; lower = better) ===")
        for label, r in table:
            print(f"  {label:12s} {r:.4f}  {'#' * int(r * 40)}")
        print()
    print(
        "Read: if damping (interior γ < 1) helps in the HIGH-dim bucket but not "
        "low-dim, that matches the undersampling theory — Schur damping is a "
        "high-dimensional fix, exactly like the Bayes-in-high-dim hypothesis."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
