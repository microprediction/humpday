"""
Reservoir release scheduling for hydropower and downstream demand.

Over twelve months a reservoir receives seasonal inflow and we choose the monthly
release. Releasing water generates hydropower, but the power per unit release rises with
the head, so the same release is worth more when the reservoir is full. Downstream demand
must be met (shortfalls are penalised) and storage must stay within its bounds (overflow
spills and underflow is penalised). The optimum holds water to keep the head high while
still covering demand and respecting the bounds.

The HumpDay objective takes a 12-D point in [0,1]^12 (monthly releases, mapped to
0..R_max) and returns negative hydropower plus demand-shortfall and storage penalties.
"""

from __future__ import annotations

T = 12
N_DIM = T
INFLOW = (30, 28, 35, 55, 80, 70, 50, 40, 38, 42, 48, 40)
DEMAND = (40, 40, 45, 50, 55, 60, 60, 55, 48, 45, 42, 40)
S_MIN, S_MAX, S0 = 20.0, 200.0, 120.0
R_MAX = 90.0
SHORT_PENALTY = 3.0
BOUND_PENALTY = 1.0


def decode(u):
    return [R_MAX * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    r = decode(u)
    S = S0
    energy = 0.0
    penalty = 0.0
    for t in range(T):
        head = S / S_MAX
        energy += r[t] * head
        penalty += SHORT_PENALTY * max(0.0, DEMAND[t] - r[t]) ** 2
        S = S + INFLOW[t] - r[t]
        if S < S_MIN:
            penalty += BOUND_PENALTY * (S_MIN - S) ** 2
            S = S_MIN
        elif S > S_MAX:
            penalty += BOUND_PENALTY * (S - S_MAX) ** 2
            S = S_MAX
    return -energy + penalty
