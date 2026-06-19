"""
Index tracking: replicate a benchmark with a small basket.

A fund wants to track a benchmark's return series using eight tradeable assets, holding
non-negative weights that sum to one. The assets follow a deterministic factor model, and
the benchmark is a target series the eight assets can only approximate, so perfect
replication is impossible and the job is to minimise tracking error: the mean squared gap
between the portfolio's return and the benchmark's over the history. It is a regression on
the simplex.

The HumpDay objective takes an 8-D point in [0,1]^8, normalises it to portfolio weights,
and returns the mean squared tracking error (scaled).
"""
from __future__ import annotations

import math

N_ASSETS = 8
N_DIM = N_ASSETS
T = 20


def _ret(t, i):
    return 0.010 * math.sin(0.5 * t + i) + 0.008 * math.cos(0.3 * t * (i + 1))


R = [[_ret(t, i) for i in range(N_ASSETS)] for t in range(T)]
BMK = [0.012 * math.sin(0.5 * t) + 0.004 * math.cos(0.7 * t) for t in range(T)]


def decode(u):
    s = sum(max(0.0, v) for v in u) or 1.0
    return [max(0.0, v) / s for v in u]


def objective(u):
    w = decode(u)
    te = 0.0
    for t in range(T):
        port = sum(w[i] * R[t][i] for i in range(N_ASSETS))
        te += (port - BMK[t]) ** 2
    return 1000.0 * te / T
