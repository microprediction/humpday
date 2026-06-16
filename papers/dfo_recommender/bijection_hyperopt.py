"""
Amortized hyper-optimization of the cube->simplex bijection.

The cube->simplex map `phi_theta` reshapes the landscape an optimizer
actually sees. Tuning theta is learning a *preconditioner* for simplex-
valued problems: the ideal theta puts the optimum near the cube centroid
(where optimizers sample densely) and makes its basin locally isotropic
in cube coordinates -- i.e. whitening the optimum's basin, expressed in
Aitchison (log-ratio) geometry. Cf. Snoek et al. 2014, "Input Warping
for Bayesian Optimization", which learns Beta-CDF input warps for the
same reason in the box-domain setting.

This is the AMORTIZED version (the one worth doing): we tune a single,
dimension-agnostic theta over a *family* of training problems, then ship
it as a reusable preconditioner. A held-out TEST family measures whether
the learned theta generalizes -- the whole point of amortizing. The
per-problem inline version is deliberately not built: it burns the
optimization budget learning geometry it could have spent optimizing,
and degenerates by overfitting to where the (unknown) optimum sits.

Two families (--family):
  * bowls     -- synthetic anisotropic quadratic bowls in log-ratio coords
                 with off-center optima and known optimum value 0.
  * portfolio -- randomized non-convex portfolio problems (factor-model
                 covariances, varied risk aversion / holding cost) in the
                 mould of example_applications/portfolio_frontier. The
                 SHIPPED portfolio_frontier instance is held out and its
                 specific improvement reported -- "does the learned
                 preconditioner help the actual demo?"

Run:
    python papers/dfo_recommender/bijection_hyperopt.py --family portfolio
    python papers/dfo_recommender/bijection_hyperopt.py --family bowls
    python papers/dfo_recommender/bijection_hyperopt.py --quick

Outer loop optimizes theta with humpday itself (meta-humpday). Inner
score = median best value a fixed optimizer finds under phi_theta, with
COMMON RANDOM NUMBERS across theta (same seeds reused) so the objective
difference reflects theta, not seed luck. Because portfolio objectives
are negative, the outer metric is a sign-agnostic *normalized
improvement* over the (s,gamma)=(1,1) baseline, scaled per problem by the
objective's spread over random simplex points.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from humpday.optimizers.alloptimizers import pure_optimize  # noqa: E402

RESULTS_PATH = HERE / "bijection_hyperopt_results.json"

# theta = (s, gamma), both dimension-agnostic so one learned theta applies
# to any simplex size. s is a global log-scale (centroid <-> corner reach);
# gamma is a probit-tail warp. (s, gamma) = (1, 1) is the plain logistic-
# normal map -- the de-dep'd default now living in thurstone_transform.py.
THETA_BOX = {"s": (0.25, 4.0), "gamma": (0.5, 2.0)}
THETA0 = (1.0, 1.0)
# A deliberately corner-reaching lift, used only to locate approximate
# optima for the interpretable centroid proxy (not for scoring).
PROBE_THETA = (3.0, 1.0)

_CLIP = 1e-10


def _unpack_theta(u_outer: list[float]) -> tuple[float, float]:
    """Map an outer-loop cube point [0,1]^2 to (s, gamma) in THETA_BOX."""
    (s_lo, s_hi), (g_lo, g_hi) = THETA_BOX["s"], THETA_BOX["gamma"]
    s = s_lo + u_outer[0] * (s_hi - s_lo)
    g = g_lo + u_outer[1] * (g_hi - g_lo)
    return s, g


def phi(u, theta: tuple[float, float]) -> np.ndarray:
    """Parametrized cube->simplex bijection. u in (0,1)^n -> p in simplex^{n+1}.

    z = probit(u);  z' = sign(z)|z|^gamma;  p = softmax(s * [0, z'])."""
    s, gamma = theta
    z = norm.ppf(np.clip(np.asarray(u, dtype=float), _CLIP, 1 - _CLIP))
    zp = np.sign(z) * np.abs(z) ** gamma
    logits = np.concatenate([[0.0], s * zp])
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def phi_inv(p, theta: tuple[float, float]) -> np.ndarray:
    """Inverse map, simplex^{n+1} -> cube (0,1)^n. Used for the centroid check."""
    s, gamma = theta
    p = np.maximum(np.asarray(p, dtype=float), _CLIP)
    ell = np.log(p[1:] / p[0])  # log-ratios vs reference component 0
    zp = ell / s
    z = np.sign(zp) * np.abs(zp) ** (1.0 / gamma)
    return norm.cdf(z)


# --------------------------------------------------------------------------
# A Problem is a cost defined on the simplex (one higher dimension than the
# cube we optimise over), plus a per-problem objective `scale` for fair
# cross-problem averaging and an optional known/approx optimum `p_star`.
# --------------------------------------------------------------------------


@dataclass
class Problem:
    name: str
    n_dim: int  # cube/manifold dimension; simplex has n_dim+1 components
    objective: Callable[[np.ndarray], float]  # simplex point -> cost
    scale: float = 1.0  # objective spread, for normalized improvement
    p_star: np.ndarray | None = field(default=None)  # optimum, if known


def _objective_scale(objective: Callable, n_components: int, seed: int) -> float:
    """Spread of the objective over random simplex points -- a positive
    per-problem scale so improvements are comparable across problems."""
    rng = np.random.default_rng(seed)
    vals = [objective(rng.dirichlet(np.ones(n_components))) for _ in range(128)]
    return max(float(np.std(vals)), _CLIP)


# ---- bowls family ---------------------------------------------------------


def _bowl_objective(p_star: np.ndarray, A: np.ndarray) -> Callable:
    ell_star = np.log(np.maximum(p_star, _CLIP)[1:] / max(p_star[0], _CLIP))

    def f(p) -> float:
        p = np.maximum(np.asarray(p, dtype=float), _CLIP)
        d = np.log(p[1:] / p[0]) - ell_star
        return float(d @ A @ d)

    return f


def _make_bowls(n_dims: tuple[int, ...], seeds: range, cond: float) -> list[Problem]:
    fam: list[Problem] = []
    for n in n_dims:
        for sd in seeds:
            rng = np.random.default_rng(10_000 * n + sd)
            p_star = rng.dirichlet(np.ones(n + 1))
            Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
            eig = np.exp(rng.uniform(-np.log(cond) / 2, np.log(cond) / 2, size=n))
            A = (Q * eig) @ Q.T
            obj = _bowl_objective(p_star, A)
            scale = _objective_scale(obj, n + 1, seed=999 * n + sd)
            fam.append(Problem(f"bowl_n{n}_s{sd}", n, obj, scale, p_star))
    return fam


# ---- portfolio family -----------------------------------------------------


def _portfolio_objective(mu, cov, gamma, kappa, tau) -> Callable:
    mu = np.asarray(mu)
    cov = np.asarray(cov)

    def f(w) -> float:
        w = np.asarray(w, dtype=float)
        ret = float(mu @ w)
        var = float(w @ cov @ w)
        cost = kappa * float(np.sum(1.0 - np.exp(-w / tau)))
        return -ret + 0.5 * gamma * var + cost

    return f


def _make_portfolios(n_assets: tuple[int, ...], seeds: range) -> list[Problem]:
    """Randomized non-convex portfolios: factor-model covariances and varied
    risk aversion / holding cost, in the mould of portfolio_frontier."""
    fam: list[Problem] = []
    for n in n_assets:
        for sd in seeds:
            rng = np.random.default_rng(20_000 * n + sd)
            mu = rng.uniform(0.02, 0.15, n)
            B = rng.normal(0.0, 0.18, (n, 2))  # 2 latent factors
            idio = rng.uniform(0.04, 0.20, n) ** 2
            cov = B @ B.T + np.diag(idio)
            gamma = rng.uniform(2.0, 6.0)
            kappa = rng.uniform(0.002, 0.010)
            obj = _portfolio_objective(mu, cov, gamma, kappa, tau=0.02)
            scale = _objective_scale(obj, n, seed=777 * n + sd)
            fam.append(Problem(f"pf_n{n}_s{sd}", n - 1, obj, scale))
    return fam


def _shipped_portfolio() -> Problem:
    from example_applications.portfolio_frontier import problem as PF

    obj = PF.simplex_objective  # simplex weights -> cost
    scale = _objective_scale(obj, len(PF.ASSETS), seed=4242)
    return Problem("portfolio_frontier(shipped)", PF.N_DIM, obj, scale)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _inner_best(
    prob: Problem,
    theta: tuple[float, float],
    algo: str,
    n_trials: int,
    seeds: list[int],
) -> float:
    """Median best objective `algo` finds on `prob` under phi_theta, over seeds.

    Common random numbers: the caller passes the SAME `seeds` for every theta."""

    def lifted(u_cube) -> float:
        return prob.objective(phi(u_cube, theta))

    bests = []
    for sd in seeds:
        np.random.seed(sd)
        try:
            f_best, _ = pure_optimize(lifted, algo, n_trials, prob.n_dim)
            bests.append(float(f_best))
        except Exception as e:  # noqa: BLE001
            print(f"    ! {prob.name} {algo} seed {sd}: {e}", flush=True)
            bests.append(float("inf"))
    valid = [b for b in bests if np.isfinite(b)]
    return median(valid) if valid else float("inf")


def _improvement(
    theta: tuple[float, float],
    family: list[Problem],
    baseline: dict[str, float],
    algo: str,
    n_trials: int,
    seeds: list[int],
) -> float:
    """Mean over the family of (baseline_best - theta_best) / scale.

    Positive = theta beats the (1,1) baseline. Lower objective is better, so
    a positive numerator means theta found a lower cost. Per-problem scaling
    keeps any one problem from dominating; sign-agnostic, unlike a log-ratio."""
    gains = []
    for prob in family:
        m_theta = _inner_best(prob, theta, algo, n_trials, seeds)
        m_base = baseline[prob.name]
        gains.append((m_base - m_theta) / prob.scale)
    return float(np.mean(gains))


def _approx_optimum(prob: Problem, algo: str = "DifferentialEvolution") -> np.ndarray:
    """Approximate optimum on the simplex via a corner-reaching multistart.
    Used only for the interpretable centroid proxy, not for scoring."""
    best_f, best_u = float("inf"), None
    for sd in range(3):
        np.random.seed(sd)

        def lifted(u_cube) -> float:
            return prob.objective(phi(u_cube, PROBE_THETA))

        f, u = pure_optimize(lifted, algo, 400, prob.n_dim)
        if f < best_f:
            best_f, best_u = f, u
    return phi(best_u, PROBE_THETA)


def _centroid_distance(family: list[Problem], theta: tuple[float, float]) -> float:
    """Mean ||phi^{-1}(p*) - centroid|| over problems with a known optimum.
    Lower = the bijection places optima nearer where optimizers sample
    densely (interpretable signal). Skips problems lacking p_star."""
    ds = [
        float(np.linalg.norm(phi_inv(prob.p_star, theta) - 0.5))
        for prob in family
        if prob.p_star is not None
    ]
    return float(np.mean(ds)) if ds else float("nan")


# --------------------------------------------------------------------------


def _build_families(family: str, quick: bool):
    """Return (train, test, headline) where headline is the shipped instance
    to report separately (or None)."""
    if family == "bowls":
        dims = (3, 5) if quick else (3, 5, 8)
        train = _make_bowls(dims, range(0, 4), cond=100.0)
        test = _make_bowls(dims, range(100, 104), cond=100.0)
        return train, test, None
    if family == "portfolio":
        assets = (6, 8) if quick else (6, 8, 10)
        train = _make_portfolios(assets, range(0, 4))
        test = _make_portfolios(assets, range(100, 102))
        headline = _shipped_portfolio()
        return train, test + [headline], headline
    raise ValueError(f"unknown family {family!r}")


def run(
    family: str = "portfolio",
    inner_algo: str = "PRIMA_BOBYQA",
    outer_algo: str = "DifferentialEvolution",
    n_trials: int = 60,
    outer_trials: int = 40,
    n_seeds: int = 3,
    quick: bool = False,
) -> dict:
    seeds = list(range(n_seeds))
    train, test, headline = _build_families(family, quick)
    print(
        f"Family={family}   Train: {len(train)} problems   "
        f"Test: {len(test)} problems (held-out)"
    )

    # Baselines at theta0=(1,1), with the SAME seeds reused everywhere.
    base_train = {
        p.name: _inner_best(p, THETA0, inner_algo, n_trials, seeds) for p in train
    }
    base_test = {
        p.name: _inner_best(p, THETA0, inner_algo, n_trials, seeds) for p in test
    }

    # Outer loop: maximize normalized improvement on the TRAIN family by
    # minimizing its negative with humpday (meta-humpday).
    evals: list[dict] = []

    def J(u_outer) -> float:
        theta = _unpack_theta(u_outer)
        gain = _improvement(theta, train, base_train, inner_algo, n_trials, seeds)
        evals.append({"theta": list(theta), "train_improvement": gain})
        print(
            f"  theta=({theta[0]:.3f},{theta[1]:.3f})  train improvement={gain:+.4f}",
            flush=True,
        )
        return -gain

    print(f"\nOuter optimize ({outer_algo}, {outer_trials} trials) over theta...")
    np.random.seed(0)
    _, u_best = pure_optimize(J, outer_algo, outer_trials, 2)
    theta_star = _unpack_theta(list(u_best))

    train_gain = _improvement(
        theta_star, train, base_train, inner_algo, n_trials, seeds
    )
    test_gain = _improvement(theta_star, test, base_test, inner_algo, n_trials, seeds)

    out = {
        "config": {
            "family": family,
            "inner_algo": inner_algo,
            "outer_algo": outer_algo,
            "n_trials": n_trials,
            "outer_trials": outer_trials,
            "n_seeds": n_seeds,
        },
        "theta_star": {"s": theta_star[0], "gamma": theta_star[1]},
        "train_improvement": train_gain,
        "test_improvement": test_gain,
        "outer_trace": evals,
    }

    print("\n=== Amortized bijection hyper-optimization ===")
    print(f"  family          : {family}")
    print(f"  learned theta*  : s={theta_star[0]:.3f}, gamma={theta_star[1]:.3f}")
    print(f"  train improvement vs (1,1): {train_gain:+.4f}  (higher = better)")
    print(f"  TEST  improvement vs (1,1): {test_gain:+.4f}  (held-out payoff)")

    # Headline: the shipped portfolio_frontier, in its own (absolute) units.
    if headline is not None:
        m0 = base_test[headline.name]
        m_star = _inner_best(headline, theta_star, inner_algo, n_trials, seeds)
        out["shipped"] = {
            "name": headline.name,
            "baseline_best": m0,
            "learned_best": m_star,
            "absolute_gain": m0 - m_star,
        }
        print(
            f"  SHIPPED {headline.name}: best {m0:.5f} -> {m_star:.5f}  "
            f"(absolute gain {m0 - m_star:+.5f}, lower objective is better)"
        )

    # Interpretable proxy: does theta* pull optima toward the cube centre?
    # bowls have known optima; for portfolios approximate them via multistart.
    proxy_set = test
    if family == "portfolio":
        for prob in proxy_set:
            prob.p_star = _approx_optimum(prob)
    cd0 = _centroid_distance(proxy_set, THETA0)
    cds = _centroid_distance(proxy_set, theta_star)
    out["centroid_distance"] = {"baseline_theta0": cd0, "learned_theta": cds}
    print(
        f"  centroid dist (test): {cd0:.3f} -> {cds:.3f}  "
        f"(lower = optima nearer cube centre)"
    )

    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS_PATH.relative_to(REPO_ROOT)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=("portfolio", "bowls"), default="portfolio")
    ap.add_argument("--inner-algo", default="PRIMA_BOBYQA")
    ap.add_argument("--outer-algo", default="DifferentialEvolution")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--outer-trials", type=int, default=40)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    args = ap.parse_args()

    if args.quick:
        run(
            family=args.family,
            inner_algo=args.inner_algo,
            outer_algo=args.outer_algo,
            n_trials=25,
            outer_trials=12,
            n_seeds=2,
            quick=True,
        )
    else:
        run(
            family=args.family,
            inner_algo=args.inner_algo,
            outer_algo=args.outer_algo,
            n_trials=args.n_trials,
            outer_trials=args.outer_trials,
            n_seeds=args.n_seeds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
