"""
NYC building HVAC setpoint scheduling: a demand-response problem.

Over 24 hours we choose an hourly indoor temperature setpoint for a commercial
building. The envelope leaks heat toward the (time-varying) outdoor temperature, so
each hour the HVAC must spend energy to pull the drifted indoor temperature back to
the chosen setpoint; energy costs the hour's electricity price. Letting the building
coast warmer during the expensive afternoon peak (within a comfort band) saves money,
exactly the demand-response trade a smart controller exploits.

The HumpDay objective takes a 24-D point in [0,1]^24 (hourly setpoints, mapped to
18..28 C) and returns price-weighted conditioning energy plus a comfort penalty for
setpoints outside [21, 25] C.
"""
from __future__ import annotations

import math

HOURS = 24
N_DIM = HOURS

# Normalised electricity price: cheap overnight, sharp evening peak.
PRICE = (0.30, 0.25, 0.22, 0.20, 0.22, 0.30, 0.45, 0.60, 0.70, 0.65, 0.60, 0.58,
         0.55, 0.55, 0.60, 0.70, 0.85, 1.00, 0.95, 0.80, 0.65, 0.50, 0.40, 0.35)
# Outdoor temperature over a summer day (peaks mid-afternoon).
OUT = tuple(26.0 + 6.0 * math.sin((h - 9) / 24.0 * 2 * math.pi) for h in range(HOURS))
T_LO, T_HI = 21.0, 25.0   # comfort band
ENVELOPE = 0.25           # fraction of the indoor-outdoor gap lost per hour
COP = 0.4                 # energy per degree C of active conditioning


def decode(u):
    return [18.0 + 10.0 * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    sp = decode(u)
    temp = 24.0
    cost = 0.0
    comfort = 0.0
    for h in range(HOURS):
        temp = temp + ENVELOPE * (OUT[h] - temp)   # passive drift toward outdoor
        cost += COP * abs(temp - sp[h]) * PRICE[h]  # condition back to setpoint
        temp = sp[h]
        if sp[h] < T_LO:
            comfort += (T_LO - sp[h]) ** 2
        elif sp[h] > T_HI:
            comfort += (sp[h] - T_HI) ** 2
    return cost + 2.0 * comfort
