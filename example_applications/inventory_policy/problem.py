"""
(s, S) inventory control under stochastic demand — a classic
operations-research policy-tuning problem.

We run a single-product periodic-review inventory system for T=40
periods. The control is an (s, S) policy: at the end of each period,
if on-hand inventory has dropped to or below the reorder point `s`,
place an order that brings it back up to the order-up-to level `S`.
Ordering costs a fixed `K=50` plus `c=2` per unit. Each period we also
pay a holding cost `h=1.0` per unit left on the shelf and a shortage
penalty `p=8.0` per unit of unmet demand.

The HumpDay objective takes a 2-D point in [0,1]^2 and maps it to
  s ∈ [0, 60],  S ∈ [0, 120]
(with S clamped up to s whenever S < s, so the policy is always valid),
then returns the total cost of ONE simulated rollout.

Variables (N_DIM=2):
  u[0] → s, the reorder point.
  u[1] → S, the order-up-to level.

Pathology: STOCHASTIC objective. Demand is a fresh
`max(0, round(gauss(20, 7)))` draw every period, so repeated
evaluations of the same (s, S) return different costs — the true
landscape is a smooth expected-cost bowl buried under sampling noise.
Like cart_pole_policy, an optimiser that over-trusts a single lucky
evaluation gets punished; the recommender harness seeds the objective
so runs are reproducible.
"""

from __future__ import annotations

import random

N_DIM = 2

T = 40  # periods per rollout

# Demand distribution (Gaussian, truncated at zero, integer units).
DEMAND_MEAN = 20.0
DEMAND_STD = 7.0

# Cost parameters.
K = 50.0  # fixed ordering cost
C = 2.0  # per-unit ordering cost
H = 1.0  # per-unit holding cost
P = 8.0  # per-unit shortage penalty

# Bounds for the decision variables.
S_REORDER_MAX = 60.0  # s ∈ [0, 60]
S_ORDER_UP_MAX = 120.0  # S ∈ [0, 120]


def _scale(u):
    """[0,1]^2 → (s, S) with S clamped up to s so the policy is valid."""
    s = S_REORDER_MAX * u[0]
    big_s = S_ORDER_UP_MAX * u[1]
    if big_s < s:
        big_s = s
    return s, big_s


def _rollout(s, big_s, rng):
    """Simulate T periods of the (s, S) policy. Returns total cost."""
    inventory = big_s
    total_cost = 0.0
    for _ in range(T):
        # Reorder up to S if at or below the reorder point.
        if inventory <= s:
            q = big_s - inventory
            total_cost += K + C * q
            inventory += q
        # Realise demand and fill what we can.
        d = max(0, round(rng.gauss(DEMAND_MEAN, DEMAND_STD)))
        sales = min(inventory, d)
        shortage = d - sales
        inventory -= sales
        # Holding and shortage costs for the period.
        total_cost += H * max(inventory, 0.0) + P * shortage
    return total_cost


def objective(u, seed_offset=None):
    """HumpDay objective: total cost of one stochastic rollout of the
    (s, S) policy. Lower is better.

    The demand stream is genuinely random, so this objective is NOISY.
    `seed_offset` lets a test harness fix the rollout's seed; the
    optimiser should call it positionally with just `u`."""
    s, big_s = _scale(u)
    rng = random.Random(seed_offset)
    return _rollout(s, big_s, rng)


def decode(u, seed_offset=None):
    """Convenience: return the policy `(s, S)` for a `[0,1]^2` point,
    plus the (noisy) total cost from one rollout."""
    s, big_s = _scale(u)
    rng = random.Random(seed_offset)
    cost = _rollout(s, big_s, rng)
    return {
        "s": s,
        "S": big_s,
        "cost": cost,
        "valid": big_s >= s,
    }
