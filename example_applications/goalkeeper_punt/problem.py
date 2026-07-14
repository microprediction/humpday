"""
Goalkeeper punt objective: two steps, one punt, hit the overhanging wire.

A pure-Python 2-D (side-view) ball-flight simulator. The HumpDay objective
takes a 4-D point in [0,1]^4 — two run-up step lengths, loft and leg power —
walks the keeper forward, punts the ball from the hands, and integrates its
flight at 120 Hz under gravity and linear drag. Strung across the pitch,
30 m downfield and 13 m up, hangs a TV spider-cam wire. You win if the ball
hits it: a wire strike scores 100+, anything else is shaped by how close the
trajectory came.

The steps are not decoration. Longer strides add run-up speed to the punt,
but the ball must be released before the edge of the penalty area — overstep
and it's a foul (handling outside the box). Stride rhythm matters too: the
cleanest contact comes from an accelerating second step, so the two step
dimensions couple with each other and with power. Loft and power trade off
along two distinct hit manifolds (clip the wire on the way up, or drop onto
it on the way down), which makes the landscape multimodal.

Mirrors the browser demo docs/applications/punt-the-wire.html.
"""

from __future__ import annotations

import math

N_DIM = 4

DT = 1 / 120
MAX_T = 6.0
G = 9.81
DRAG = 0.06  # per-second linear drag

# Keeper + penalty box geometry (metres, keeper's start = x 0).
RELEASE_HEIGHT = 0.9  # drop-punt contact height
BOX_ROOM = 3.5  # room to the 18-yard line from the start spot
PLANT_LUNGE = 0.6  # forward lunge during the strike itself

# Run-up model.
STEP_MIN, STEP_MAX = 0.2, 1.6
RUN_SPEED_PER_M = 1.7  # run-up speed gained per metre of stride
RUN_CARRY = 0.9  # fraction of run-up speed carried into the ball
RHYTHM_BEST_GAP = 0.25  # ideal: second step this much longer than first
RHYTHM_WIDTH = 0.5
QUALITY_FLOOR = 0.75  # contact quality at worst rhythm

# The overhanging wire (a spider-cam cable crossing the pitch).
WIRE_X = 30.0  # downfield from the keeper's start
WIRE_Z = 13.0  # height above the turf
HIT_RADIUS = 0.7  # matches the drawn wire + ball sizes in the browser demo


def decode(u):
    return [
        STEP_MIN + (STEP_MAX - STEP_MIN) * u[0],  # step 1 (m)
        STEP_MIN + (STEP_MAX - STEP_MIN) * u[1],  # step 2 (m)
        15 + 55 * u[2],  # loft (deg)
        16 + 12 * u[3],  # leg power (m/s)
    ]


def run_punt(params):
    """Simulate one punt; return (score, result)."""
    step1, step2, loft_deg, power = params

    # ---- the two steps ---------------------------------------------------
    advance = step1 + step2 + PLANT_LUNGE
    overstep = advance - BOX_ROOM
    if overstep > 0:
        # Released the ball beyond the 18-yard line: free kick, no punt.
        return max(0.0, 8 - 20 * overstep), "FOUL - outside the box"

    v_run = RUN_SPEED_PER_M * (step1 + step2)
    rhythm = math.exp(-(((step2 - step1 - RHYTHM_BEST_GAP) / RHYTHM_WIDTH) ** 2))
    quality = QUALITY_FLOOR + (1 - QUALITY_FLOOR) * rhythm
    v0 = (power + RUN_CARRY * v_run) * quality

    # ---- the flight ------------------------------------------------------
    loft = loft_deg * math.pi / 180
    x, z = advance, RELEASE_HEIGHT
    vx, vz = v0 * math.cos(loft), v0 * math.sin(loft)
    d_min = math.hypot(WIRE_X - x, WIRE_Z - z)
    speed_at_closest = math.hypot(vx, vz)
    t = 0.0
    while t < MAX_T:
        vx *= 1 - DRAG * DT
        vz *= 1 - DRAG * DT
        vz -= G * DT
        x += vx * DT
        z += vz * DT
        d = math.hypot(WIRE_X - x, WIRE_Z - z)
        if d < d_min:
            d_min = d
            speed_at_closest = math.hypot(vx, vz)
        if d_min <= HIT_RADIUS or z < 0:
            break
        t += DT

    # ---- scoring ---------------------------------------------------------
    if d_min <= HIT_RADIUS:
        score = 100 + min(10.0, 0.35 * speed_at_closest)
        return score, "WIRE!"
    score = 90 * math.exp(-d_min / 2.0) + min(2.0, 0.4 * v_run)
    return score, f"miss by {d_min:.1f} m"


def evaluate_punt(u):
    return run_punt(decode(u))


def objective(u):
    """HumpDay objective: negative punt score (minimise)."""
    return -run_punt(decode(u))[0]
