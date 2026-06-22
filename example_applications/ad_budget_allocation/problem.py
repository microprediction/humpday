"""
Marketing budget allocation across channels with diminishing returns.

A fixed budget is split across eight advertising channels. Each channel converts spend
into customers through a saturating response c_i(s) = cap_i * (1 - exp(-eff_i * s)):
early dollars are productive, later ones saturate. The best split equalises marginal
returns across channels (a water-filling allocation) subject to the budget.

The HumpDay objective takes an 8-D point in [0,1]^8, normalises it to the budget, and
returns the negative total conversions (to minimise).
"""

from __future__ import annotations

import math

N_CH = 8
N_DIM = N_CH

CAP = (5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0)  # max conversions per channel
EFF = (0.3, 0.5, 0.4, 0.8, 0.6, 1.0, 0.7, 1.2)  # responsiveness
BUDGET = 10.0


def decode(u):
    s = sum(max(0.0, v) for v in u) or 1.0
    return [BUDGET * max(0.0, v) / s for v in u]


def objective(u):
    spend = decode(u)
    conversions = sum(
        CAP[i] * (1.0 - math.exp(-EFF[i] * spend[i])) for i in range(N_CH)
    )
    return -conversions
