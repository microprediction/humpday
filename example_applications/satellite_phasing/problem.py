"""
Satellite constellation phasing for ground coverage.

Six satellites orbit a ring; we choose each one's phase. Ground targets sit at fixed
longitudes with population weights, and a target's coverage cost is the squared angular
distance to its nearest satellite. Bunching satellites leaves wide gaps over the
unserved longitudes, so the optimum spreads them to cover the weighted targets, leaning
toward the heavily populated ones.

The HumpDay objective takes a 6-D point in [0,1]^6 (phases, mapped to [0, 2 pi)) and
returns the weighted sum of squared nearest-satellite angular distances.
"""

from __future__ import annotations

import math

N_SATS = 6
N_DIM = N_SATS
TWO_PI = 2 * math.pi
# (longitude in radians, population weight)
TARGETS = (
    (0.3, 3.0),
    (0.9, 1.0),
    (1.6, 2.5),
    (2.4, 1.5),
    (3.0, 2.0),
    (3.8, 1.0),
    (4.5, 2.5),
    (5.2, 1.5),
    (5.9, 1.0),
    (0.0, 2.0),
)


def decode(u):
    return [TWO_PI * min(1.0, max(0.0, v)) for v in u]


def _ang_dist(a, b):
    d = abs(a - b) % TWO_PI
    return min(d, TWO_PI - d)


def objective(u):
    phases = decode(u)
    total = 0.0
    for lon, weight in TARGETS:
        nearest = min(_ang_dist(lon, p) for p in phases)
        total += weight * nearest * nearest
    return total
