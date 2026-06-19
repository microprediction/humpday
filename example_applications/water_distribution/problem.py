"""
Water-distribution pipe sizing.

A branched water network carries known demand flows through eight pipes from a pumped
source. We choose each pipe's diameter. Head loss along a pipe follows a
Hazen-Williams form, h ~ L * q^1.85 / d^4.87, so a wider pipe loses less head (less
pumping energy) but costs more material (cost ~ L * d^2). The optimum trades pipe
capital against lifetime pumping cost, widening the high-flow trunk pipes more than the
low-flow branches.

The HumpDay objective takes an 8-D point in [0,1]^8 (diameters, mapped to 0.1..1.0 m)
and returns material cost plus pumping cost.
"""
from __future__ import annotations

N_PIPES = 8
N_DIM = N_PIPES
FLOW = (0.9, 0.6, 0.5, 0.35, 0.3, 0.2, 0.15, 0.1)   # m^3/s carried by each pipe
LENGTH = (500, 400, 350, 300, 250, 200, 180, 150)   # m
D_MIN, D_MAX = 0.10, 1.00
MATERIAL_COST = 800.0   # per (m of length * m^2 of cross-section)
PUMP_COST = 2.0         # per unit of head loss


def decode(u):
    return [D_MIN + (D_MAX - D_MIN) * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    d = decode(u)
    material = 0.0
    pumping = 0.0
    for i in range(N_PIPES):
        material += MATERIAL_COST * LENGTH[i] * d[i] ** 2
        pumping += PUMP_COST * LENGTH[i] * FLOW[i] ** 1.85 / d[i] ** 4.87
    return (material + pumping) / 1e5
