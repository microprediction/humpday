"""
Mini-golf objective: sink the putt (REDUCED-ORDER model).

NOTE: the browser demo (docs/applications/mini-golf.html) rolls the ball on a
sloped green with the Matter.js engine. This is a deliberately SIMPLIFIED
roll-to-the-hole model with the SAME 3-D parameterisation and the SAME hole — a
faithful optimisation *problem*, not a bit-identical sim.

Take a 3-D point in [0,1]^3 — aim angle, power, and a slope-compensation trim —
roll the ball (it decelerates under friction and curves as it crosses the sloped
green), and score by how close it finishes to the hole. Too little power stops
short; too much overshoots; the slope must be read and compensated.
"""

from __future__ import annotations

import math

N_DIM = 3
HOLE_DIST = 300.0  # straight-line distance to the hole (px)
HOLE_BEARING = 8.0  # bearing to the hole from the tee (deg)
HOLE_R = 14.0
MAX_RANGE = 520.0  # roll distance at full power


def decode(u):
    return [5 + 65 * u[0], 15 + 115 * u[1], -2.0 + 4.0 * u[2]]


def _final_pos(params):
    aim_deg, power, slope_trim = params
    rng = MAX_RANGE * (power / 130.0)
    # the slope curves the path; the trim compensates for it
    curve_deg = 6.0 - slope_trim * 3.0
    eff_bearing = aim_deg + curve_deg
    fx = rng * math.sin(math.radians(eff_bearing))
    fy = rng * math.cos(math.radians(eff_bearing))
    hx = HOLE_DIST * math.sin(math.radians(HOLE_BEARING))
    hy = HOLE_DIST * math.cos(math.radians(HOLE_BEARING))
    return math.hypot(fx - hx, fy - hy)


def finish_distance(u):
    """Distance from the hole where the ball finishes (px)."""
    return _final_pos(decode(u))


def objective(u):
    """HumpDay objective: negative putt score (sink = 100; minimise)."""
    d = finish_distance(u)
    if d < HOLE_R:
        return -100.0
    return -max(0.0, 100 * math.exp(-(((d - HOLE_R) / 90.0) ** 2)))
