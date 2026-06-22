"""
Vaccine allocation across population groups (multi-group SIR).

A fixed supply of doses must be split across eight population groups that differ in
size and contact rate. Vaccinating a group removes susceptibles before an epidemic
runs; the groups mix through a shared force of infection weighted by contact rate, so
where the doses go changes how large the outbreak gets. The good allocations protect
the high-contact, high-population groups that drive transmission.

The HumpDay objective takes an 8-D point in [0,1]^8, normalises it to a fixed dose
budget, simulates the multi-group SIR, and returns the total number infected.
"""

from __future__ import annotations

N_GROUPS = 8
N_DIM = N_GROUPS

POP = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)  # relative group sizes
CONTACT = (1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5)  # contacts per unit time
VAX_BUDGET = 2.5  # total doses, in population units
GAMMA = 0.3  # recovery rate
STEPS = 80
DT = 1.0
TOTAL_POP = sum(POP)


def decode(u):
    s = sum(max(0.0, v) for v in u) or 1.0
    return [VAX_BUDGET * max(0.0, v) / s for v in u]


def objective(u):
    doses = decode(u)
    S = [max(0.0, POP[i] - doses[i]) for i in range(N_GROUPS)]
    # outbreak seeded in the two highest-contact groups, so protecting the right
    # spreaders changes how far it propagates
    I = [0.01 * POP[i] if i < 2 else 0.0 for i in range(N_GROUPS)]
    total_infected = 0.0
    for _ in range(STEPS):
        force = sum(CONTACT[j] * I[j] for j in range(N_GROUPS)) / TOTAL_POP
        for i in range(N_GROUPS):
            new = min(S[i], CONTACT[i] * S[i] * force * DT)
            S[i] -= new
            I[i] += new - GAMMA * I[i] * DT
            total_infected += new
    return total_infected
