"""
Trebuchet objective: hit the target 60 m away (REDUCED-ORDER model).

NOTE: the browser demo (docs/applications/trebuchet.html) simulates the full
articulated trebuchet with the Matter.js rigid-body engine. Reproducing that in
pure Python would mean porting a physics engine, so this is a deliberately
SIMPLIFIED energy/ballistics model with the SAME 4-D parameterisation and the
SAME target — a faithful optimisation *problem*, not a bit-identical sim.

The HumpDay objective takes a 4-D point in [0,1]^4 — counterweight mass, arm
ratio, sling ratio, and release angle — estimates the launch speed (heavier
counterweight and well-matched arm/sling ratios transfer energy more
efficiently) and launch angle, computes the projectile range, and returns the
negative score (100 when it lands on the target 60 m away, falling off with
miss distance). The arm/sling efficiency peaks at an interior optimum, so the
landscape rewards tuning, not just "max counterweight".
"""

from __future__ import annotations

import math

N_DIM = 4
G = 9.81
TARGET_M = 60.0


def decode(u):
    return [
        50 + 450 * u[0],  # counterweight mass (kg)
        0.15 + 0.30 * u[1],  # short/long arm ratio
        0.4 + 1.1 * u[2],  # sling length / long arm
        40 + 160 * u[3],  # release angle (deg)
    ]


def _launch(params):
    cw, arm_ratio, sling_ratio, release_deg = params
    # efficiency peaks at a well-matched arm and sling ratio
    eff = math.exp(-(((arm_ratio - 0.28) / 0.12) ** 2)) * math.exp(
        -(((sling_ratio - 0.95) / 0.40) ** 2)
    )
    v = 1.087 * math.sqrt(cw) * eff  # launch speed (m/s); 60 m reachable at peak
    launch_deg = 20 + (release_deg - 40) / 160 * 60  # release -> 20..80 deg
    return v, math.radians(launch_deg)


def throw_range(u):
    """Projectile range in metres for a design in [0,1]^4."""
    v, ang = _launch(decode(u))
    return v * v * math.sin(2 * ang) / G


def objective(u):
    """HumpDay objective: negative hit-the-target score (minimise)."""
    miss = abs(throw_range(u) - TARGET_M)
    return -max(0.0, 100 * (1 - miss / 30.0))
