"""
Slingshot objective: knock down the most blocks (REDUCED-ORDER model).

NOTE: the browser demo (docs/applications/slingshot.html) topples stacked blocks
with the Matter.js rigid-body engine. Reproducing toppling stacks in pure Python
would mean porting a physics engine, so this is a deliberately SIMPLIFIED
ballistic-fly-through model with the SAME 3-D parameterisation — a faithful
optimisation *problem*, not a bit-identical sim.

Take a 3-D point in [0,1]^3 — launch angle, power, and a release-height trim —
fire a projectile on a ballistic arc; score the number of stacked blocks whose
zone the trajectory passes through. The blocks sit in two groups across a gap, so
a flat hard shot rakes the near stack while a lofted shot drops onto the far one
— two basins, a mildly multimodal landscape.
"""

from __future__ import annotations

import math

N_DIM = 3
G = 900.0  # px/s^2
LAUNCH = (100.0, 250.0)  # launch point (x, y), y up
# block centres (x, y) — two stacks across a gap (12 + 8 blocks)
# two narrow, tall stacks so a well-placed arc rakes several blocks in one pass
BLOCKS = [(410 + 26 * (i % 2), 50 + 26 * (i // 2)) for i in range(12)] + [
    (660 + 26 * (i % 2), 50 + 26 * (i // 2)) for i in range(8)
]
BLOCK_R = 24.0


def decode(u):
    return [5 + 70 * u[0], 25 + 105 * u[1], -1.2 + 2.4 * u[2]]


def _hits(params):
    angle_deg, power, height_trim = params
    a = math.radians(angle_deg)
    v = power * 4.0
    x0, y0 = LAUNCH[0], LAUNCH[1] + height_trim * 30
    vx, vy = v * math.cos(a), v * math.sin(a)
    if vx < 1e-6:
        return 0
    # sample the arc finely; a block is knocked if the path passes within BLOCK_R
    # of it (so a steep descent rakes a whole stack).
    knocked = [False] * len(BLOCKS)
    dt = 0.01
    t = 0.0
    while t < 4.0:
        x = x0 + vx * t
        y = y0 + vy * t - 0.5 * G * t * t
        t += dt
        if x > 820 or y < -40:
            break
        for k, (bx, by) in enumerate(BLOCKS):
            if not knocked[k] and (x - bx) ** 2 + (y - by) ** 2 < BLOCK_R * BLOCK_R:
                knocked[k] = True
    return sum(knocked)


def blocks_hit(u):
    return _hits(decode(u))


def objective(u):
    """HumpDay objective: negative number of blocks hit (minimise)."""
    return -float(_hits(decode(u)))
