"""
Brachistochrone objective: shape a ramp for the fastest slide.

The classic calculus-of-variations problem. A marble starts at rest at the
top-left and slides under gravity along a piecewise-linear ramp to the
bottom-right; the HumpDay objective takes an 8-D point in [0,1]^8 — the heights
of eight control points — and returns the **descent time**. Minimising it finds
the fastest curve, which approaches the cycloid (steep drop early to build speed,
then a shallow run-out) — NOT the straight line.

This pure-Python port scores the ramp by energy conservation: speed at any point
is sqrt(2 g h) for the height dropped, and each segment's time is its length
divided by its mean speed. (The browser demo rolls a Matter.js marble instead;
the cycloid optimum is the same. Control heights are bounded at or below the
start, since a marble at rest can't climb above where it began.)

Mirrors the browser demo docs/applications/brachistochrone.html.
"""

from __future__ import annotations

import math

# Screen-style coordinates: y increases DOWNWARD, so larger y = lower = faster.
START = (60.0, 60.0)
END = (740.0, 380.0)
CONTROL_XS = (135.0, 210.0, 285.0, 360.0, 435.0, 510.0, 585.0, 660.0)
Y_MIN = 60.0  # can't go above the start height
Y_MAX = 430.0  # just above the floor
N_DIM = len(CONTROL_XS)
G = 1.0  # gravity scale (only sets the time unit, not the optimum)


def _decode(u):
    return [Y_MIN + (Y_MAX - Y_MIN) * v for v in u]


def descent_time(u):
    """Time for the marble to slide along the ramp (energy conservation)."""
    ys = [START[1]] + _decode(u) + [END[1]]
    xs = [START[0]] + list(CONTROL_XS) + [END[0]]

    def speed(y):
        return math.sqrt(2 * G * max(0.0, y - START[1]))

    t = 0.0
    for i in range(len(xs) - 1):
        length = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        v_mean = (speed(ys[i]) + speed(ys[i + 1])) / 2
        t += length / max(v_mean, 1e-6)
    return t


def objective(u):
    """HumpDay objective: descent time, in arbitrary units (minimise)."""
    return descent_time(u)
