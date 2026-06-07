"""
Pool objective: pot the object ball (REDUCED-ORDER model).

NOTE: the browser demo (docs/applications/pool.html) simulates the full table —
cue ball, object balls, cushions and pockets — with the Matter.js engine.
Reproducing multi-ball collisions in pure Python would mean porting a physics
engine, so this is a deliberately SIMPLIFIED single-cut model with the SAME 3-D
parameterisation — a faithful optimisation *problem*, not a bit-identical sim.

Take a 3-D point in [0,1]^3 — cue aim angle, power, and side-spin (english) —
strike the cue ball at a fixed object ball; the cut angle and english set the
object ball's direction, and the objective scores how well that direction lines
up with the nearest pocket (with enough power to reach it). The "ghost-ball" cut
angle that pots the ball is a narrow target — a small, precise optimum.
"""

from __future__ import annotations

import math

N_DIM = 3
# geometry (px): cue ball, object ball, and the pocket we aim the object at.
CUE = (200.0, 250.0)
OBJ = (430.0, 210.0)
POCKET = (760.0, 60.0)


def decode(u):
    return [-10 + 20 * u[0], 20 + 50 * u[1], -0.8 + 1.6 * u[2]]


def _alignment(params):
    aim_deg, power, english = params
    # cue travel direction (base aim is straight at the object ball)
    base = math.atan2(OBJ[1] - CUE[1], OBJ[0] - CUE[0])
    # the cut angle (and a little english) steer the object ball's departure
    obj_dir = base + math.radians(aim_deg) * 2.0 + 0.12 * english
    want = math.atan2(POCKET[1] - OBJ[1], POCKET[0] - OBJ[0])
    ang_err = abs((obj_dir - want + math.pi) % (2 * math.pi) - math.pi)
    reach = min(1.0, power / 45.0)  # enough pace to reach the pocket
    return ang_err, reach


def objective(u):
    """HumpDay objective: negative pot score (minimise)."""
    ang_err, reach = _alignment(decode(u))
    return -(100 * math.exp(-((ang_err / 0.07) ** 2)) * reach)
