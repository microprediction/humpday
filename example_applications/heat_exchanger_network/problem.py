"""
Heat-exchanger network sizing (effectiveness-NTU).

A cold process stream is heated in series by six exchangers, each against a hot stream
held at a fixed temperature. We choose each exchanger's area. The effectiveness of a
unit is eps = 1 - exp(-U*A / (m*cp)), so larger area buys more heat transfer with
diminishing returns; the outlet temperature builds up stage by stage toward the cold
stream's target. Area costs money, and the target carries a shortfall penalty, so the
good design puts area where the temperature driving force is largest rather than
spreading it evenly.

The HumpDay objective takes a 6-D point in [0,1]^6 (areas, mapped to 0..A_max) and
returns total area cost plus a penalty for missing the outlet-temperature target.
"""
from __future__ import annotations

import math

N_STAGES = 6
N_DIM = N_STAGES
T_IN = 20.0                # cold-stream inlet (C)
T_TARGET = 140.0           # desired outlet (C)
HOT = (90.0, 120.0, 150.0, 180.0, 210.0, 240.0)   # hot-side temperatures per stage
MCP = 8.0                  # cold stream heat-capacity rate
U = 0.5                    # overall heat-transfer coefficient
A_MAX = 30.0
AREA_COST = 0.4
SHORTFALL_PENALTY = 5.0


def decode(u):
    return [A_MAX * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    areas = decode(u)
    t = T_IN
    for i in range(N_STAGES):
        eps = 1.0 - math.exp(-U * areas[i] / MCP)
        t = t + eps * (HOT[i] - t)          # approach the hot temperature
    cost = AREA_COST * sum(areas)
    shortfall = max(0.0, T_TARGET - t)
    return cost + SHORTFALL_PENALTY * shortfall
