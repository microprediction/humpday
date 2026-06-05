"""
Walking-creature objective: evolve a gait that walks as far as possible.

A pure-Python stylised kinematic walker. A two-legged body is driven by leg
oscillators: each foot follows a sine wave, sweeping and lifting. While a foot is
planted (stance) it grips the ground, so its backward sweep carries the body
forward; while both feet are airborne the body falls and can faceplant. The
HumpDay objective takes a 6-D point in [0,1]^6, stretches it into the gait
parameters (frequency, stride, the PHASE OFFSET between the legs, lift timing,
lift duty, forward lean), simulates an 8-second walk, and returns the
**negative distance** travelled (with a faceplant penalty).

Nobody tells the optimiser that legs should alternate; it reliably rediscovers
it — the best gaits emerge with the two legs almost exactly out of phase. The
landscape is forgiving (even Random Search shuffles forward), but PRIMA_NEWUOA
can get trapped near its start and never learn to walk.

Mirrors the browser demo docs/applications/creature.html.
"""

from __future__ import annotations

import math

N_DIM = 6
T = 8.0
DT = 0.02
STEPS = int(round(T / DT))
LEG_LEN = 1.0
MAX_DROP = 0.55


def _decode(u):
    return {
        "w": 3 + u[0] * 9,
        "A": 0.15 + u[1] * 1.05,
        "dphi": u[2] * 2 * math.pi,
        "zeta": u[3] * 2 * math.pi,
        "duty": 0.2 + u[4] * 0.6,
        "lean": (u[5] - 0.5) * 0.8,
    }


def _leg_phase(g, i, t):
    return g["w"] * t + (g["dphi"] if i else 0.0)


def _lifted(g, i, t):
    thr = 1 - 2 * g["duty"]
    return math.sin(_leg_phase(g, i, t) + g["zeta"]) > thr


def _foot_vx(g, i, t):
    return g["A"] * g["w"] * math.cos(_leg_phase(g, i, t))


def simulate(u):
    """Return (distance, fell) for a gait in [0,1]^6."""
    g = _decode(u)
    x, y, vy = 0.0, LEG_LEN, 0.0
    fell = False
    for s in range(STEPS):
        t = s * DT
        stance = [i for i in (0, 1) if not _lifted(g, i, t)]
        if stance:
            v = sum(-_foot_vx(g, i, t) for i in stance) / len(stance)
            x += v * DT
            y += (LEG_LEN - y) * min(1.0, 8 * DT)
            vy = 0.0
        else:
            vy -= 6.0 * DT
            y += vy * DT
            if y < LEG_LEN - MAX_DROP:
                fell = True
        if fell:
            break
    return x, fell


def objective(u):
    """HumpDay objective: negative distance walked, with faceplant penalty."""
    dist, fell = simulate(u)
    d = dist
    if fell:
        d = min(d, d * 0.3) - 1.0
    return -d
