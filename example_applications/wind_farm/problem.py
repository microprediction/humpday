"""
Wind-farm layout objective: place turbines to maximise expected power.

A pure-Python reduced-order layout problem. Eight turbines sit in a rectangular
field; the HumpDay objective takes a 16-D point in [0,1]^16 (each turbine's x, y)
and computes the **expected power** over a 12-sector wind rose using the Jensen
wake model — turbines in another's wake see slower wind, and power scales with
the cube of wind speed. The objective returns the negative score (power minus a
spacing penalty plus a small spread-to-the-boundary bonus).

This is the canonical early-stage wind-farm problem: the optimiser must spread
turbines out of each other's wakes, respecting a minimum spacing, while favouring
the prevailing wind directions. The wake coupling makes it a genuinely
interacting, non-separable objective.

Mirrors the browser demo docs/applications/wind-farm.html.
"""

from __future__ import annotations

import math

N_TURBINES = 8
N_DIM = N_TURBINES * 2
FIELD_X0, FIELD_X1 = 60.0, 740.0
FIELD_Y0, FIELD_Y1 = 60.0, 390.0
ROTOR_R = 14.0
AXIAL_INDUCTION = 0.42
WAKE_DECAY = 0.08
MIN_SPACING = ROTOR_R * 2 * 3  # 3-rotor-diameter minimum spacing
BOUNDARY_BONUS_PTS = 2.5

# (compass-from direction, weight); angle = (compass + 90) in radians.
_ROSE = [
    (270, 0.22),
    (300, 0.14),
    (330, 0.10),
    (0, 0.04),
    (30, 0.03),
    (60, 0.03),
    (90, 0.03),
    (120, 0.04),
    (150, 0.05),
    (180, 0.06),
    (210, 0.10),
    (240, 0.16),
]
WIND_ROSE = [(((c + 90) * math.pi / 180), w) for c, w in _ROSE]


def decode(u):
    return [
        (
            FIELD_X0 + u[2 * i] * (FIELD_X1 - FIELD_X0),
            FIELD_Y0 + u[2 * i + 1] * (FIELD_Y1 - FIELD_Y0),
        )
        for i in range(N_TURBINES)
    ]


def _wind_speeds(positions, wind_angle):
    cos, sin = math.cos(-wind_angle), math.sin(-wind_angle)
    rot = [(p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos) for p in positions]
    v = []
    for i in range(len(rot)):
        def_sq = 0.0
        for j in range(len(rot)):
            if i == j:
                continue
            dx = rot[i][0] - rot[j][0]
            dy = rot[i][1] - rot[j][1]
            if dx <= 0:  # j not upstream of i
                continue
            wake_r = ROTOR_R + WAKE_DECAY * dx
            if abs(dy) > wake_r:  # i outside j's wake
                continue
            deficit = (2 * AXIAL_INDUCTION) / (1 + WAKE_DECAY * dx / ROTOR_R) ** 2
            def_sq += deficit * deficit
        v.append(max(0.0, 1 - math.sqrt(def_sq)))
    return v


def _expected_power(positions):
    total = 0.0
    for angle, weight in WIND_ROSE:
        v = _wind_speeds(positions, angle)
        total += weight * sum(vi**3 for vi in v)
    return total


def _spacing_penalty(positions):
    penalty = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = math.hypot(
                positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]
            )
            if d < MIN_SPACING:
                viol = (MIN_SPACING - d) / MIN_SPACING
                penalty += viol * viol
    return penalty


def _boundary_bonus(positions):
    cxf, cyf = (FIELD_X0 + FIELD_X1) / 2, (FIELD_Y0 + FIELD_Y1) / 2
    half_w, half_h = (FIELD_X1 - FIELD_X0) / 2, (FIELD_Y1 - FIELD_Y0) / 2
    total = sum(max(abs(x - cxf) / half_w, abs(y - cyf) / half_h) for x, y in positions)
    return total / len(positions)


def evaluate_layout(u):
    """Return (score, power_fraction, spacing_penalty) for a layout in [0,1]^16."""
    positions = decode(u)
    power_fraction = _expected_power(positions) / N_TURBINES
    penalty = _spacing_penalty(positions)
    boundary = _boundary_bonus(positions)
    score = 100 * power_fraction + BOUNDARY_BONUS_PTS * boundary - 40 * penalty
    return score, power_fraction, penalty


def objective(u):
    """HumpDay objective: negative layout score (minimise)."""
    return -evaluate_layout(u)[0]
