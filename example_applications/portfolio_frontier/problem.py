"""
Portfolio Frontier — non-convex allocation on the unit simplex.

Allocate capital across eight assets (long-only weights summing to 1) to
minimise a mean-variance utility *plus a per-holding fixed cost*. The
weights are a composition — a point on the 7-simplex — so, like
`cocktail_blend`, we lift a plain `[0,1]^7` cube point onto the simplex
via `humpday.transforms.cubetosimplex`.

What it stresses (and how it differs from `cocktail_blend`):
  - Unit-simplex geometry again, but the landscape is **non-convex**.
    Classic Markowitz (mean - gamma*variance) is convex and unimodal; we
    add a smoothed cardinality penalty — a fixed cost per asset actually
    held — which is concave near zero and therefore creates competing
    basins over *which subset* of assets to own. That makes this a demo
    that genuinely discriminates between optimisers, not just a smooth
    bowl every method nails.

Objective to MINIMISE:
    -mu . w  +  (gamma/2) w^T Sigma w  +  kappa * sum_i (1 - exp(-w_i / tau))

The first two terms are negative mean-variance utility (annualised); the
third charges ~kappa per asset with a non-trivial weight, smoothed so the
objective stays continuous. Raising gamma traces the efficient frontier;
raising kappa pushes toward more concentrated portfolios.
"""

from __future__ import annotations

import math

from humpday.transforms.cubetosimplex import cube_to_simplex

ASSETS = (
    "tech_a",
    "tech_b",
    "bond",
    "gold",
    "energy",
    "utility",
    "em_equity",
    "reit",
)

# Annualised expected returns (decimal).
MU = (0.14, 0.12, 0.03, 0.05, 0.09, 0.06, 0.13, 0.08)

# Annualised volatilities (standard deviation).
VOL = (0.28, 0.26, 0.05, 0.15, 0.24, 0.12, 0.30, 0.18)

# Correlation matrix (symmetric, unit diagonal). Sector structure: the two
# tech names move together and with EM; bonds and gold are diversifiers.
CORR = (
    (1.00, 0.82, -0.10, -0.15, 0.35, 0.25, 0.55, 0.40),
    (0.82, 1.00, -0.08, -0.12, 0.32, 0.22, 0.52, 0.38),
    (-0.10, -0.08, 1.00, 0.20, -0.05, 0.30, -0.10, 0.10),
    (-0.15, -0.12, 0.20, 1.00, 0.10, 0.05, 0.00, 0.05),
    (0.35, 0.32, -0.05, 0.10, 1.00, 0.20, 0.45, 0.30),
    (0.25, 0.22, 0.30, 0.05, 0.20, 1.00, 0.20, 0.35),
    (0.55, 0.52, -0.10, 0.00, 0.45, 0.20, 1.00, 0.45),
    (0.40, 0.38, 0.10, 0.05, 0.30, 0.35, 0.45, 1.00),
)

# Covariance Sigma_ij = corr_ij * vol_i * vol_j.
COV = tuple(
    tuple(CORR[i][j] * VOL[i] * VOL[j] for j in range(len(ASSETS)))
    for i in range(len(ASSETS))
)

GAMMA = 3.0  # risk aversion (tuned so the optimum is a diversified interior blend)
KAPPA = 0.006  # fixed cost charged per asset actually held
TAU = 0.02  # weight at which a holding counts as "fully active"

N_DIM = len(ASSETS) - 1  # = 7


def _portfolio_return(w):
    return sum(MU[i] * w[i] for i in range(len(ASSETS)))


def _portfolio_variance(w):
    return sum(
        w[i] * COV[i][j] * w[j] for i in range(len(ASSETS)) for j in range(len(ASSETS))
    )


def _holding_cost(w):
    """Smoothed cardinality penalty: ~KAPPA per asset with a real position."""
    return KAPPA * sum(1.0 - math.exp(-w[i] / TAU) for i in range(len(ASSETS)))


def simplex_objective(w):
    """Cost of a portfolio given weights `w` already on the simplex
    (long-only, summing to 1): negative mean-variance utility plus the
    per-holding cost. Kept separate from the cube->simplex lift so the
    geometry (the bijection) can be varied independently of the finance
    — see papers/dfo_recommender/bijection_hyperopt.py."""
    ret = _portfolio_return(w)
    var = _portfolio_variance(w)
    return -ret + 0.5 * GAMMA * var + _holding_cost(w)


def objective(u):
    """HumpDay-style objective: `u ∈ [0,1]^7` -> negative utility + costs.

    Lifts the cube point onto the 8-asset simplex of long-only weights,
    then evaluates mean-variance utility plus the per-holding cost."""
    return simplex_objective(cube_to_simplex(u))


def decode(u):
    """Convenience: weights + return/vol/Sharpe + holdings for a `[0,1]^7` point."""
    w = cube_to_simplex(u)
    ret = _portfolio_return(w)
    var = _portfolio_variance(w)
    vol = math.sqrt(max(var, 0.0))
    return {
        "weights": {name: w[i] for i, name in enumerate(ASSETS)},
        "expected_return": ret,
        "volatility": vol,
        "sharpe": ret / vol if vol > 0 else float("inf"),
        "n_holdings": sum(1 for wi in w if wi > 0.01),
        "objective": objective(u),
    }
