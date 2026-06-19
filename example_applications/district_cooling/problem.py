"""
Chiller-plant load allocation (district cooling).

A cooling plant has five chillers of differing size that together must meet a fixed
cooling demand. We choose how to split the load across them. Each chiller's coefficient
of performance varies with its part-load ratio, peaking near a sweet spot and falling
off when it is run nearly idle or flat out, so the electrical energy to deliver a given
cooling load depends strongly on how the load is shared. The good allocation keeps the
running chillers near their efficient part-load point rather than dumping everything on
the largest unit.

The HumpDay objective takes a 5-D point in [0,1]^5, normalises it so the chillers
exactly meet demand, and returns total electrical energy (cooling delivered divided by
each chiller's COP).
"""
from __future__ import annotations

N_CHILLERS = 5
N_DIM = N_CHILLERS
CAPACITY = (400.0, 350.0, 300.0, 250.0, 200.0)   # cooling capacity per chiller (kW)
COP_PEAK = (6.5, 6.0, 5.8, 5.5, 5.2)             # best-case COP
PLR_SWEET = (0.7, 0.65, 0.6, 0.6, 0.55)          # part-load ratio of peak efficiency
DEMAND = 900.0                                    # total cooling load to meet (kW)


def decode(u):
    s = sum(max(0.0, v) for v in u) or 1.0
    return [DEMAND * max(0.0, v) / s for v in u]   # loads summing to DEMAND


def objective(u):
    loads = decode(u)
    energy = 0.0
    for i in range(N_CHILLERS):
        load = min(loads[i], CAPACITY[i])
        plr = load / CAPACITY[i]
        # COP peaks at PLR_SWEET, degrades quadratically away from it; small floor
        cop = max(1.0, COP_PEAK[i] * (1.0 - 1.2 * (plr - PLR_SWEET[i]) ** 2))
        energy += load / cop
        # unmet load (capacity overflow) penalised at a poor effective COP
        energy += max(0.0, loads[i] - CAPACITY[i]) / 1.0
    return energy
