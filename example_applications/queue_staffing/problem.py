"""
Call-centre staffing across shifts (queueing-delay model).

Six shifts of a day have different call arrival rates. For each shift we choose a
staffing level; the expected caller wait follows a queueing-delay model that grows
without bound as utilisation approaches one. Staffing costs money and waits above an SLA
target are penalised, so the good plan staffs heavily in the busy shifts and lightly in
the quiet ones rather than uniformly.

The HumpDay objective takes a 6-D point in [0,1]^6 (staffing per shift, mapped to a
range of agents) and returns staffing cost plus SLA-wait penalty.
"""
from __future__ import annotations

N_SHIFTS = 6
N_DIM = N_SHIFTS
ARRIVAL = (20.0, 35.0, 55.0, 70.0, 45.0, 25.0)   # calls per period
MU = 8.0          # calls an agent can serve per period
C_MIN, C_MAX = 1.0, 14.0
STAFF_COST = 1.0
SLA_PENALTY = 40.0
TARGET_WAIT = 0.1


def decode(u):
    return [C_MIN + (C_MAX - C_MIN) * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    agents = decode(u)
    cost = 0.0
    for t in range(N_SHIFTS):
        capacity = agents[t] * MU
        rho = ARRIVAL[t] / capacity if capacity > 0 else 2.0
        if rho >= 0.999:
            wait = 100.0 * rho     # overloaded: grows with utilisation (gives a
                                   # gradient back toward feasibility, no flat plateau)
        else:
            wait = rho / (capacity * (1.0 - rho))
        cost += STAFF_COST * agents[t] + SLA_PENALTY * max(0.0, wait - TARGET_WAIT)
    return cost
