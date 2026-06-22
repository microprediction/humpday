"""
Revenue management: pricing across segments under shared capacity.

A firm sets prices for eight customer segments. Each segment's demand falls linearly
with its price, q_i = a_i - b_i * p_i (floored at zero), and the segments draw on a
shared, limited capacity. Priced too low, demand exceeds capacity and is penalised;
priced too high, demand and revenue collapse. The optimum is the capacity-constrained
analogue of segment-by-segment monopoly pricing, raising prices on the segments that
consume scarce capacity most.

The HumpDay objective takes an 8-D point in [0,1]^8 (prices, mapped to 0..p_max per
segment) and returns negative revenue plus an over-capacity penalty.
"""

from __future__ import annotations

N_SEG = 8
N_DIM = N_SEG
A = (100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0)  # demand intercepts
B = (8.0, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0)  # price sensitivities
P_MAX = 15.0
CAPACITY = 220.0
OVERAGE_PENALTY = 3.0


def decode(u):
    return [P_MAX * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    p = decode(u)
    q = [max(0.0, A[i] - B[i] * p[i]) for i in range(N_SEG)]
    revenue = sum(p[i] * q[i] for i in range(N_SEG))
    overage = max(0.0, sum(q) - CAPACITY)
    return -(revenue) + OVERAGE_PENALTY * overage * overage
