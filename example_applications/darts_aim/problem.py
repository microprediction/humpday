"""
Where to Aim on a Dartboard — expected-score maximisation under throw noise.

You choose an aim point (x, y) in millimetres on a standard dartboard.
Because nobody throws perfectly, the dart actually lands at
(x + dx, y + dy) where (dx, dy) is isotropic Gaussian throw error with
standard deviation SIGMA = 35 mm (an amateur). The payoff is the
expected dartboard score over that scatter, and the HumpDay objective
MINIMISES the negative expected score.

Variables (n_dim = 2): the unit point u ∈ [0,1]^2 maps to an aim point
    x = (u0 - 0.5) * 340,  y = (u1 - 0.5) * 340   (mm)
so the aim may roam over the whole 340 mm board and a little beyond.

Scoring (radii in mm from centre, r = hypot(x, y)):
    r <= 6.35   -> 50   (inner bull)
    r <= 15.9   -> 25   (outer bull)
    r > 170.0   -> 0    (miss, off the scoring area)
    otherwise   -> S * M  where S is the sector base number from the
                   angle and M is the ring multiplier from the radius:
                       double  162.0 < r <= 170.0 -> M = 2
                       treble   99.0 < r <= 107.0 -> M = 3
                       single   otherwise          -> M = 1

The expected score is computed DETERMINISTICALLY: a fixed bank of ~600
standard-normal offset pairs is precomputed once at import (fixed-seed
LCG + Box–Muller) and reused on every call, so `objective` is a fixed,
repeatable function of the aim.

Pathology: a deterministic but strongly MULTIMODAL landscape. The board
is a pinwheel that alternates big and small numbers (20 sits next to 1
and 5), so the raw score surface is jagged. Convolving it with a 35 mm
Gaussian smooths the spikes but leaves several competing local maxima —
near treble-20 at the top, near the 19/16 cluster at the lower-left, and
near the centre. Hill-climbers happily settle on the wrong hump.

The famous result (Tibshirani, Price & Taylor, 2011 — "A Statistician
Plays Darts"): a perfect player aims at the treble-20, but as throw
noise grows the optimal aim slides off treble-20, down and to the
lower-left of centre, eventually landing just below the bullseye.
"""

from __future__ import annotations

import math

# Throw-noise standard deviation (mm). 35 mm ≈ an amateur.
SIGMA = 35.0

# Board geometry (mm).
BOARD_RADIUS = 170.0
INNER_BULL_R = 6.35
OUTER_BULL_R = 15.9
TREBLE_INNER_R = 99.0
TREBLE_OUTER_R = 107.0
DOUBLE_INNER_R = 162.0
DOUBLE_OUTER_R = 170.0

# u -> aim coordinate half-extent (board diameter, mm).
SPAN = 340.0

# Sector base numbers, clockwise starting from the 20 at the top.
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Number of fixed throw-noise offsets averaged per evaluation.
N_OFFSETS = 600


def _gaussian_offsets(n, seed=1234567):
    """Deterministic standard-normal (gx, gy) pairs via LCG + Box–Muller."""
    # Numerical Recipes LCG constants.
    a, c, m = 1664525, 1013904223, 2**32
    state = seed
    offsets = []
    while len(offsets) < n:
        state = (a * state + c) % m
        u1 = (state + 1.0) / (m + 1.0)
        state = (a * state + c) % m
        u2 = (state + 1.0) / (m + 1.0)
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        offsets.append((radius * math.cos(angle), radius * math.sin(angle)))
    return offsets


# Precompute once at import so the objective is fully deterministic.
_OFFSETS = _gaussian_offsets(N_OFFSETS)


def score(x, y):
    """Standard dartboard score of a single landing point (x, y) in mm."""
    r = math.hypot(x, y)
    if r <= INNER_BULL_R:
        return 50
    if r <= OUTER_BULL_R:
        return 25
    if r > BOARD_RADIUS:
        return 0

    # Sector base number. Standard math angle (0 = +x axis, CCW); the 20
    # wedge is centred at the top (+y, 90 deg) and each wedge spans 18 deg.
    theta = math.atan2(y, x)
    idx = int(math.floor(((90.0 - math.degrees(theta) + 9.0) % 360.0) / 18.0))
    s = SECTORS[idx]

    # Ring multiplier from the radius.
    if DOUBLE_INNER_R < r <= DOUBLE_OUTER_R:
        m = 2
    elif TREBLE_INNER_R < r <= TREBLE_OUTER_R:
        m = 3
    else:
        m = 1

    return s * m


def expected_score(x, y):
    """Mean score over the fixed bank of throw-noise offsets."""
    total = 0
    for gx, gy in _OFFSETS:
        total += score(x + SIGMA * gx, y + SIGMA * gy)
    return total / len(_OFFSETS)


def _unit_to_aim(u):
    """Map a point in [0,1]^2 to an aim coordinate (x, y) in mm."""
    return (u[0] - 0.5) * SPAN, (u[1] - 0.5) * SPAN


def objective(u):
    """HumpDay-style objective: input is `u ∈ [0,1]^2`, output is the
    negative expected score (so MINIMISING maximises expected points)."""
    x, y = _unit_to_aim(u)
    return -expected_score(x, y)


def decode(u):
    """Convenience: return the aim point `(x, y)` in mm for a `[0,1]^2`
    point, plus the expected score there."""
    x, y = _unit_to_aim(u)
    return {
        "x": x,
        "y": y,
        "expected_score": expected_score(x, y),
    }


N_DIM = 2
