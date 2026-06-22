"""
Microgrid dispatch: meet a day's load from solar, a diesel genset, and a battery.

Over twelve daytime periods a remote microgrid serves a varying electrical load. Solar
power is free (a midday bell), a diesel generator can be dispatched at a fuel cost, and a
battery can charge (storing surplus solar) or discharge (covering the evening load),
bounded by its state of charge and cycled at a wear cost. The optimum stores midday solar
and discharges it into the evening so the genset runs as little as possible.

The HumpDay objective takes a 24-D point in [0,1]^24 (twelve diesel levels, then twelve
battery powers) and returns fuel cost plus battery wear plus penalties for unmet load and
state-of-charge violations.
"""

from __future__ import annotations

import math

T = 12
N_DIM = 2 * T
LOAD = (40, 38, 42, 55, 70, 80, 85, 82, 78, 70, 58, 46)
SOLAR = tuple(max(0.0, 95.0 * math.sin(math.pi * t / (T - 1))) for t in range(T))
DIESEL_MAX = 60.0
BATT_MAX = 30.0
BATT_CAP = 120.0
SOC_INIT = 60.0
FUEL_COST = 1.0
WEAR_COST = 0.05
SHORTFALL_PENALTY = 5.0
SOC_PENALTY = 2.0


def decode(u):
    diesel = [DIESEL_MAX * min(1.0, max(0.0, u[t])) for t in range(T)]
    batt = [(min(1.0, max(0.0, u[T + t])) - 0.5) * 2 * BATT_MAX for t in range(T)]
    return diesel, batt


def objective(u):
    diesel, batt = decode(u)
    soc = SOC_INIT
    cost = 0.0
    for t in range(T):
        supply = SOLAR[t] + diesel[t] + batt[t]  # batt>0 discharges, <0 charges
        shortfall = max(0.0, LOAD[t] - supply)
        cost += FUEL_COST * diesel[t] + WEAR_COST * abs(batt[t])
        cost += SHORTFALL_PENALTY * shortfall * shortfall
        soc -= batt[t]
        if soc < 0.0:
            cost += SOC_PENALTY * soc * soc
            soc = 0.0
        elif soc > BATT_CAP:
            cost += SOC_PENALTY * (soc - BATT_CAP) ** 2
            soc = BATT_CAP
    return cost
