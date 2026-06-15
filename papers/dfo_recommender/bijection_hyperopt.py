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
it as a reusable preconditioner. The held-out TEST family measures
whether the learned theta generalizes -- the whole point of amortizing.
The per-problem inline version is deliberately not built: it burns the
optimization budget learning geometry it could have spent optimizing,
and degenerates by overfitting to where the (unknown) optimum sits.

Run:
    python papers/dfo_recommender/bijection_hyperopt.py            # default
    python papers/dfo_recommender/bijection_hyperopt.py --quick    # fast smoke

Outer loop optimizes theta with humpday itself (meta-humpday). Inner
score = median best value a fixed optimizer finds under phi_theta, with
COMMON RANDOM NUMBERS across theta (same seeds reused) so J(theta)
differences reflect theta, not seed luck.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median

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

_CLIP = 1e-10


def _unpack_theta(u_outer: list[float]) -> tuple[float, float]:
    """Map an outer-loop cube point [0,1]^2 to (s, gamma) in THETA_BOX."""
    (s_lo, s_hi), (g_lo, g_hi) = THETA_BOX["s"], THETA_BOX["gamma"]
    s = s_lo + u_outer[0] * (s_hi - s_lo)
    g = g_lo + u_outer[1] * (g_hi - g_lo)
    return s, g


def phi(u: list[float], theta: tuple[float, float]) -> np.ndarray:
    """Parametrized cube->simplex bijection. u in (0,1)^n -> p in simplex^{n+1}.

    z = probit(u);  z' = sign(z)|z|^gamma;  p = softmax(s * [0, z'])."""
    s, gamma = theta
    z = norm.ppf(np.clip(np.asarray(u, dtype=float), _CLIP, 1 - _CLIP))
    zp = np.sign(z) * np.abs(z) ** gamma
    logits = np.concatenate([[0.0], s * zp])
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def phi_inv(p: np.ndarray, theta: tuple[float, float]) -> np.ndarray:
    """Inverse map, simplex^{n+1} -> cube (0,1)^n. Used for the centroid check."""
    s, gamma = theta
    p = np.maximum(np.asarray(p, dtype=float), _CLIP)
    ell = np.log(p[1:] / p[0])  # log-ratios vs reference component 0
    zp = ell / s
    z = np.sign(zp) * np.abs(zp) ** (1.0 / gamma)
    return norm.cdf(z)


# --------------------------------------------------------------------------
# Training / test family: quadratic bowls in log-ratio (alr) coordinates with
# random off-center optima and random anisotropic conditioning -- exactly the
# regime a preconditioner should help. Known optimum value 0.
# --------------------------------------------------------------------------


@dataclass
class SimplexProblem:
    name: str
    n_dim: int  # cube/manifold dimension (simplex has n_dim+1 components)
    p_star: np.ndarray  # optimum on the simplex
    A: np.ndarray  # SPD conditioning in alr coordinates

    def objective(self, p: np.ndarray) -> float:
        p = np.maximum(np.asarray(p, dtype=float), _CLIP)
        ell = np.log(p[1:] / p[0])
        ell_star = np.log(self.p_star[1:] / self.p_star[0])
        d = ell - ell_star
        return float(d @ self.A @ d)


def _make_family(
    n_dims: tuple[int, ...], seeds: range, cond: float
) -> list[SimplexProblem]:
    fam: list[SimplexProblem] = []
    for n in n_dims:
        for sd in seeds:
            rng = np.random.default_rng(10_000 * n + sd)
            # off-center optimum on the simplex
            p_star = rng.dirichlet(np.ones(n + 1))
            # anisotropic SPD bowl: eigenvalues spanning `cond` orders
            Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
            eig = np.exp(rng.uniform(-np.log(cond) / 2, np.log(cond) / 2, size=n))
            A = (Q * eig) @ Q.T
            fam.append(SimplexProblem(f"n{n}_s{sd}", n, p_star, A))
    return fam


def _inner_score(
    prob: SimplexProblem,
    theta: tuple[float, float],
    algo: str,
    n_trials: int,
    seeds: list[int],
) -> float:
    """Median best objective `algo` finds on `prob` under phi_theta, over seeds.

    Common random numbers: the caller passes the SAME `seeds` for every theta."""

    def lifted(u_cube: list[float]) -> float:
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


def _outer_objective(
    theta: tuple[float, float],
    family: list[SimplexProblem],
    baseline: dict[str, float],
    algo: str,
    n_trials: int,
    seeds: list[int],
) -> float:
    """Mean over the family of log(score_theta / score_baseline_theta).

    Negative = theta beats the (s,gamma)=(1,1) logistic-normal baseline.
    Per-problem log-ratio normalization keeps any single problem's scale
    from dominating the average."""
    ratios = []
    for prob in family:
        s = _inner_score(prob, theta, algo, n_trials, seeds)
        b = baseline[prob.name]
        ratios.append(np.log(max(s, _CLIP) / max(b, _CLIP)))
    return float(np.mean(ratios))


def _centroid_distance(
    family: list[SimplexProblem], theta: tuple[float, float]
) -> float:
    """Mean ||phi^{-1}(p*) - centroid|| over the family. Lower = the bijection
    places optima nearer where optimizers sample densely (interpretable signal)."""
    ds = []
    for prob in family:
        u = phi_inv(prob.p_star, theta)
        ds.append(float(np.linalg.norm(u - 0.5)))
    return float(np.mean(ds))


def run(
    inner_algo: str = "PRIMA_BOBYQA",
    outer_algo: str = "DifferentialEvolution",
    n_trials: int = 60,
    outer_trials: int = 40,
    n_seeds: int = 3,
    n_dims: tuple[int, ...] = (3, 5, 8),
    cond: float = 100.0,
) -> dict:
    seeds = list(range(n_seeds))
    train = _make_family(n_dims, range(0, 4), cond)
    test = _make_family(n_dims, range(100, 104), cond)
    print(f"Train family: {len(train)} problems   Test family: {len(test)} problems")

    # Baseline scores at theta0=(1,1), with the SAME seeds reused everywhere.
    base_train = {
        p.name: _inner_score(p, THETA0, inner_algo, n_trials, seeds) for p in train
    }
    base_test = {
        p.name: _inner_score(p, THETA0, inner_algo, n_trials, seeds) for p in test
    }

    # Outer loop: optimize theta with humpday on the TRAIN family only.
    evals: list[dict] = []

    def J(u_outer: list[float]) -> float:
        theta = _unpack_theta(u_outer)
        val = _outer_objective(theta, train, base_train, inner_algo, n_trials, seeds)
        evals.append({"theta": list(theta), "train_log_improvement": val})
        print(
            f"  theta=({theta[0]:.3f},{theta[1]:.3f})  train logΔ={val:+.4f}",
            flush=True,
        )
        return val

    print(f"\nOuter optimize ({outer_algo}, {outer_trials} trials) over theta...")
    np.random.seed(0)
    _, u_best = pure_optimize(J, outer_algo, outer_trials, 2)
    theta_star = _unpack_theta(list(u_best))

    # Amortization payoff: evaluate the learned theta on the held-out TEST family.
    test_improvement = _outer_objective(
        theta_star, test, base_test, inner_algo, n_trials, seeds
    )
    train_improvement = _outer_objective(
        theta_star, train, base_train, inner_algo, n_trials, seeds
    )

    out = {
        "config": {
            "inner_algo": inner_algo,
            "outer_algo": outer_algo,
            "n_trials": n_trials,
            "outer_trials": outer_trials,
            "n_seeds": n_seeds,
            "n_dims": list(n_dims),
            "cond": cond,
        },
        "theta_star": {"s": theta_star[0], "gamma": theta_star[1]},
        "train_log_improvement": train_improvement,
        "test_log_improvement": test_improvement,
        "centroid_distance": {
            "baseline_theta0_test": _centroid_distance(test, THETA0),
            "learned_theta_test": _centroid_distance(test, theta_star),
        },
        "outer_trace": evals,
    }

    print("\n=== Amortized bijection hyper-optimization ===")
    print(f"  learned theta*  : s={theta_star[0]:.3f}, gamma={theta_star[1]:.3f}")
    print(f"  train  logΔ vs (1,1): {train_improvement:+.4f}  (negative = better)")
    print(f"  TEST   logΔ vs (1,1): {test_improvement:+.4f}  (held-out payoff)")
    cd = out["centroid_distance"]
    print(
        f"  centroid dist (test): {cd['baseline_theta0_test']:.3f} -> "
        f"{cd['learned_theta_test']:.3f}  (lower = optima nearer cube centre)"
    )
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS_PATH.relative_to(REPO_ROOT)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inner-algo", default="PRIMA_BOBYQA")
    ap.add_argument("--outer-algo", default="DifferentialEvolution")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--outer-trials", type=int, default=40)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    args = ap.parse_args()

    if args.quick:
        run(
            inner_algo=args.inner_algo,
            outer_algo=args.outer_algo,
            n_trials=25,
            outer_trials=12,
            n_seeds=2,
            n_dims=(3, 5),
        )
    else:
        run(
            inner_algo=args.inner_algo,
            outer_algo=args.outer_algo,
            n_trials=args.n_trials,
            outer_trials=args.outer_trials,
            n_seeds=args.n_seeds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
