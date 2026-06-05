"""
Circle-packing objective: pack N equal circles into a unit square.

A pure-Python "maximise the minimum" geometry problem. N equal circles live in
the unit square; the HumpDay objective takes a 2N-D point in [0,1]^(2N) — every
circle centre — and grows all circles to the largest shared radius that fits
without overlap or spilling out of the square. It returns the negative radius
(normalised so 100 ≈ the best known packing). Minimising it tightens the pack.

Every layout is valid (there are no infeasible points), but the objective is
**non-smooth**: the achievable radius is the minimum over all pairwise gaps and
all wall clearances, so it has sharp ridges where the binding constraint
switches. The optimum for 6 circles is two offset columns of three (r ≈ 0.1875
of the side).

Mirrors the browser demo docs/applications/packing.html.
"""

from __future__ import annotations

import math

N_CIRCLES = 6
N_DIM = N_CIRCLES * 2
R_REF = 0.1875  # best-known radius for 6 circles, for the 0..100 score


def _decode(u):
    return [(u[2 * i], u[2 * i + 1]) for i in range(N_CIRCLES)]


def packing_radius(centres):
    """Largest shared radius: min of wall clearances and half-gaps."""
    r = math.inf
    for cx, cy in centres:
        r = min(r, cx, 1 - cx, cy, 1 - cy)
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            d = math.hypot(centres[i][0] - centres[j][0], centres[i][1] - centres[j][1])
            r = min(r, d / 2)
    return r


def score(u):
    """Percentage of the best-known packing radius (higher = tighter)."""
    return 100.0 * packing_radius(_decode(u)) / R_REF


def objective(u):
    """HumpDay objective: negative packing score (minimise)."""
    return -score(u)
