"""
Curling objective: stop the stone on the button (REDUCED-ORDER model).

NOTE: the browser demo (docs/applications/curling.html) uses the Matter.js
rigid-body engine (sliding friction + collisions). This is a deliberately
SIMPLIFIED slide-with-curl model sharing the SAME 4-D parameterisation and the
SAME house geometry — a faithful optimisation *problem*, not a bit-identical sim.

Take a 4-D point in [0,1]^4 — aim angle, throwing weight (speed), curl (handle),
and a fine-aim trim — slide the stone down the sheet (it decelerates under
friction and drifts sideways with the curl), and score by how close it stops to
the button at the centre of the house (scoring rings at 25/50/75 px). Too little
weight stops short; too much sails through the back.
"""

from __future__ import annotations

import math

N_DIM = 4
HOUSE_R_12 = 75.0  # outer ring
BUTTON_DIST = 520.0  # distance from the hack to the button (px)
MAX_TRAVEL = 760.0  # travel at full weight (px)


def decode(u):
    return [-5 + 10 * u[0], 8 + 7 * u[1], -1.0 + 2.0 * u[2], u[3]]


def _final_offset(params):
    aim_deg, weight, curl, trim = params
    travel = MAX_TRAVEL * (weight / 15.0) ** 2  # weight sets distance
    along = travel - BUTTON_DIST  # +past / -short of button
    lateral = math.tan(math.radians(aim_deg)) * travel  # aim sweeps sideways
    lateral += curl * 0.04 * travel  # curl drifts the stone
    lateral += (trim - 0.5) * 40  # fine trim
    return along, lateral


def stop_distance(u):
    """Distance from the button where the stone stops (px)."""
    along, lateral = _final_offset(decode(u))
    return math.hypot(along, lateral)


def objective(u):
    """HumpDay objective: negative house score (minimise)."""
    return -max(0.0, 100 * (1 - stop_distance(u) / HOUSE_R_12))
