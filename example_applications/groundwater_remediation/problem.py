"""
Groundwater remediation: pump-and-treat well-rate selection.

A contaminant plume spreads from a source through an aquifer. Six extraction wells can
pump at chosen rates; pumping at a well lowers the contaminant concentration at nearby
monitoring points, with the effect falling off with distance. Pumping costs energy, and
concentrations above a regulatory limit at any monitoring point are penalised, so the
optimum pumps just enough at the wells best placed to capture the plume rather than
running every well flat out.

The HumpDay objective takes a 6-D point in [0,1]^6 (well rates, mapped to 0..Q_max) and
returns pumping cost plus the over-limit concentration penalty.
"""
from __future__ import annotations

import math

N_WELLS = 6
N_DIM = N_WELLS
Q_MAX = 10.0
SRC_STRENGTH = 8.0
LIMIT = 1.0
PUMP_COST = 0.3
PENALTY = 20.0
CAPTURE = 3.0
SOURCE = (2.0, 5.0)
WELLS = ((4.0, 5.0), (6.0, 4.0), (6.0, 6.0), (8.0, 5.0), (5.0, 3.0), (5.0, 7.0))
MONITORS = ((7.0, 5.0), (9.0, 5.0), (8.0, 3.0), (8.0, 7.0), (10.0, 5.0))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def decode(u):
    return [Q_MAX * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    q = decode(u)
    cost = PUMP_COST * sum(q)
    for m in MONITORS:
        conc = SRC_STRENGTH / (1.0 + _dist(m, SOURCE))
        for i, w in enumerate(WELLS):
            conc -= q[i] * CAPTURE / (1.0 + _dist(w, m) ** 2)
        conc = max(0.0, conc)
        cost += PENALTY * max(0.0, conc - LIMIT) ** 2
    return cost
