"""
Discrete delta-hedging of a European call option.

A trader sells a call and hedges it over twelve steps by holding a fraction of a share
between rebalances. As the underlying moves along a path, the hedge portfolio is marked
to market; at expiry it should reproduce the option payoff. We choose the hedge ratio
at each step to minimise the squared replication error plus a transaction cost on every
rebalance. The optimal schedule tracks the option's delta along the path, trading off
tracking accuracy against churn.

The HumpDay objective takes a 12-D point in [0,1]^12 (per-step hedge ratios) and returns
squared replication error plus transaction cost.
"""
from __future__ import annotations

import math

STEPS = 12
N_DIM = STEPS
S0 = 100.0
STRIKE = 100.0
TXN = 0.01   # proportional transaction cost

# Deterministic underlying path (drift plus a smooth oscillation).
PATH = [S0 * math.exp(0.02 * t / STEPS + 0.15 * math.sin(3.0 * t / STEPS))
        for t in range(STEPS + 1)]


def decode(u):
    return [min(1.0, max(0.0, v)) for v in u]


def objective(u):
    h = decode(u)
    cash = 0.0
    prev = 0.0
    for t in range(STEPS):
        dh = h[t] - prev
        cash -= dh * PATH[t]             # rebalance the share holding
        cash -= TXN * abs(dh) * PATH[t]  # transaction cost
        prev = h[t]
    cash += prev * PATH[STEPS]           # liquidate at expiry
    payoff = max(0.0, PATH[STEPS] - STRIKE)
    return (cash - payoff) ** 2
