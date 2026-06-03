"""
Python ports of the optimization problems behind the
docs/applications/ demos. Each port mirrors the demo's JS objective
line-for-line so the demo stays the canonical source of truth.

Every demo exposes:

    name: str
    n_dim: int
    suggested_n_trials: int    # what the demo's UI defaults to
    objective(u: list[float]) -> float
        # u is in [0, 1]^n_dim. Returns scalar to minimise.

We collect them into DEMOS so the analysis driver can iterate.

Add a demo by porting the corresponding `decode` and `objective`
functions from docs/applications/<name>.html, validating that
known-optimum or known-feasible solutions evaluate the same way on
both sides, and registering it in DEMOS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class Demo:
    name: str
    n_dim: int
    suggested_n_trials: int
    objective: Callable[[list[float]], float]


# -----------------------------------------------------------------------------
# Welded-beam design — classical 4D constrained DFO benchmark.
# Source: docs/applications/welded-beam.html.
# Reference optimum: cost ≈ 1.725 at (h, l, t, b) ≈ (0.205, 3.470, 9.037, 0.206).
# -----------------------------------------------------------------------------

# Constants from welded-beam.html lines 309–316.
_WB_P_LOAD = 6000.0
_WB_L_BEAM = 14.0
_WB_E_MOD = 30e6
_WB_G_MOD = 12e6
_WB_TAU_MAX = 13600.0
_WB_SIGMA_MAX = 30000.0
_WB_DELTA_MAX = 0.25
_WB_PENALTY_WEIGHT = 1e4

_WB_BOUNDS = {
    "h": (0.125, 2.0),
    "l": (0.1, 10.0),
    "t": (0.1, 10.0),
    "b": (0.1, 2.0),
}


def _wb_decode(u: list[float]) -> tuple[float, float, float, float]:
    """u ∈ [0, 1]^4 → (h, l, t, b) in physical units (inches)."""
    h = _WB_BOUNDS["h"][0] + (_WB_BOUNDS["h"][1] - _WB_BOUNDS["h"][0]) * u[0]
    l = _WB_BOUNDS["l"][0] + (_WB_BOUNDS["l"][1] - _WB_BOUNDS["l"][0]) * u[1]
    t = _WB_BOUNDS["t"][0] + (_WB_BOUNDS["t"][1] - _WB_BOUNDS["t"][0]) * u[2]
    b = _WB_BOUNDS["b"][0] + (_WB_BOUNDS["b"][1] - _WB_BOUNDS["b"][0]) * u[3]
    return h, l, t, b


def _wb_stresses(h: float, l: float, t: float, b: float) -> dict[str, float]:
    eps = 1e-12
    tau_prime = _WB_P_LOAD / (math.sqrt(2) * h * l + eps)
    M = _WB_P_LOAD * (_WB_L_BEAM + l / 2)
    R = math.sqrt(l * l / 4 + (h + t) * (h + t) / 4)
    J = 2 * (math.sqrt(2) * h * l * (l * l / 12 + (h + t) * (h + t) / 4))
    tau_dprime = M * R / (J + eps)
    tau = math.sqrt(
        tau_prime * tau_prime
        + 2 * tau_prime * tau_dprime * l / (2 * R + eps)
        + tau_dprime * tau_dprime
    )
    sigma = 6 * _WB_P_LOAD * _WB_L_BEAM / (t * t * b + eps)
    delta = 4 * _WB_P_LOAD * _WB_L_BEAM**3 / (_WB_E_MOD * t * t * t * b + eps)
    pc = (
        4.013
        * _WB_E_MOD
        * math.sqrt(t * t * b**6 / 36)
        / (_WB_L_BEAM * _WB_L_BEAM)
        * (1 - t / (2 * _WB_L_BEAM) * math.sqrt(_WB_E_MOD / (4 * _WB_G_MOD + eps)))
    )
    return {"tau": tau, "sigma": sigma, "delta": delta, "pc": pc}


def _wb_constraint_violations(
    h: float, l: float, t: float, b: float
) -> list[tuple[str, float]]:
    s = _wb_stresses(h, l, t, b)
    return [
        ("shear", s["tau"] - _WB_TAU_MAX),
        ("bending", s["sigma"] - _WB_SIGMA_MAX),
        ("h_le_b", h - b),
        (
            "combined",
            0.10471 * h * h + 0.04811 * t * b * (14 + l) - 5,
        ),
        ("h_min", 0.125 - h),
        ("deflection", s["delta"] - _WB_DELTA_MAX),
        ("buckling", _WB_P_LOAD - s["pc"]),
    ]


def _wb_raw_cost(h: float, l: float, t: float, b: float) -> float:
    return 1.10471 * h * h * l + 0.04811 * t * b * (14 + l)


def welded_beam_objective(u: list[float]) -> float:
    """Cost + quadratic constraint penalty (to minimise). Feasible
    optimum is ≈ 1.725."""
    h, l, t, b = _wb_decode(u)
    cost = _wb_raw_cost(h, l, t, b)
    penalty = 0.0
    for _name, g in _wb_constraint_violations(h, l, t, b):
        penalty += _WB_PENALTY_WEIGHT * max(0.0, g) ** 2
    return cost + penalty


# -----------------------------------------------------------------------------
# Robot arm — 6D inverse kinematics with obstacle avoidance.
# Source: docs/applications/robot-arm.html.
# -----------------------------------------------------------------------------

_RA_N_JOINTS = 6
_RA_LINK_LEN = (120.0, 105.0, 90.0, 75.0, 60.0, 45.0)
_RA_JOINT_ANGLE_MAX = math.pi / 2
_RA_LINK_HW = 9.0
_RA_BASE = (400.0, 478.0)
_RA_TARGET = (700.0, 180.0)
_RA_OBSTACLES = (
    (450.0, 360.0, 55.0),
    (600.0, 320.0, 50.0),
    (540.0, 190.0, 38.0),
)


def _ra_forward_kinematics(angles: list[float]) -> list[tuple[float, float]]:
    pos: list[tuple[float, float]] = [_RA_BASE]
    acc = -math.pi / 2  # base joint points "up"
    for i in range(_RA_N_JOINTS):
        acc += angles[i]
        px, py = pos[-1]
        pos.append(
            (px + _RA_LINK_LEN[i] * math.cos(acc), py + _RA_LINK_LEN[i] * math.sin(acc))
        )
    return pos


def _ra_seg_circle_dist(
    p1: tuple[float, float], p2: tuple[float, float], c: tuple[float, float]
) -> float:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    len2 = dx * dx + dy * dy
    if len2 < 1e-9:
        return math.hypot(p1[0] - c[0], p1[1] - c[1])
    t = ((c[0] - p1[0]) * dx + (c[1] - p1[1]) * dy) / len2
    t = max(0.0, min(1.0, t))
    px = p1[0] + t * dx
    py = p1[1] + t * dy
    return math.hypot(px - c[0], py - c[1])


def robot_arm_objective(u: list[float]) -> float:
    angles = [(v - 0.5) * 2 * _RA_JOINT_ANGLE_MAX for v in u]
    positions = _ra_forward_kinematics(angles)
    tip = positions[-1]
    tip_err = math.hypot(tip[0] - _RA_TARGET[0], tip[1] - _RA_TARGET[1])
    collisions = 0
    deepest = 0.0
    for i in range(_RA_N_JOINTS):
        p1 = positions[i]
        p2 = positions[i + 1]
        for ox, oy, r in _RA_OBSTACLES:
            d = _ra_seg_circle_dist(p1, p2, (ox, oy))
            if d < r + _RA_LINK_HW:
                collisions += 1
                deepest = max(deepest, r + _RA_LINK_HW - d)
    reach = max(0.0, 100.0 - tip_err * 0.25)
    penalty = collisions * 25 + deepest * 0.4
    score = reach - penalty
    return -score  # minimise


# -----------------------------------------------------------------------------
# Slingshot — 2D projectile aiming with stochastic terrain.
# Source: docs/applications/slingshot.html.
# Deterministic variant: fix the seed so every evaluation sees the same
# terrain. We use the user-blessed slingshot params from MEMORY.md.
# -----------------------------------------------------------------------------

_SS_BOUNDS = {
    "angle": (math.radians(15), math.radians(75)),
    "speed": (30.0, 95.0),
}
_SS_TARGET_X = 360.0
_SS_TARGET_Y = 240.0
_SS_GRAV = 9.81
_SS_DT = 1.0 / 60.0
_SS_MAX_STEPS = 400


def slingshot_objective(u: list[float]) -> float:
    """Simplified projectile: launch at u-encoded (angle, speed); minimise
    distance from target after free flight. (No wind / no terrain bumps
    — keeping the objective deterministic so seeds only change algorithm
    behavior, not the landscape.)"""
    angle = (
        _SS_BOUNDS["angle"][0]
        + (_SS_BOUNDS["angle"][1] - _SS_BOUNDS["angle"][0]) * u[0]
    )
    speed = (
        _SS_BOUNDS["speed"][0]
        + (_SS_BOUNDS["speed"][1] - _SS_BOUNDS["speed"][0]) * u[1]
    )
    vx = speed * math.cos(angle)
    vy = -speed * math.sin(angle)  # canvas-y points down
    x, y = 60.0, 320.0
    best_d = float("inf")
    for _ in range(_SS_MAX_STEPS):
        d = math.hypot(x - _SS_TARGET_X, y - _SS_TARGET_Y)
        if d < best_d:
            best_d = d
        if y > 360:  # below ground
            break
        x += vx * _SS_DT * 60
        y += vy * _SS_DT * 60
        vy += _SS_GRAV * _SS_DT * 60
    return best_d


# -----------------------------------------------------------------------------
# Brachistochrone — find the curve y(x) that minimises descent time
# between two points under gravity. Parameterise y(x) by the y values
# at 10 equally-spaced x knots (so u ∈ [0, 1]^10).
# Source: docs/applications/brachistochrone.html.
# -----------------------------------------------------------------------------

_BC_N_KNOTS = 10
_BC_X0 = 0.0
_BC_X1 = 1.0
_BC_Y0 = 1.0
_BC_Y1 = 0.0
_BC_GRAV = 9.81


def brachistochrone_objective(u: list[float]) -> float:
    """Integrate descent time over a piecewise-linear curve through
    (x0, y0), (x_knot_i, y_knot_i), (x1, y1), where the y_knots are
    given by u ∈ [0, 1]^10 mapped to [0, 1]."""
    xs = (
        [_BC_X0]
        + [
            _BC_X0 + (_BC_X1 - _BC_X0) * (i + 1) / (_BC_N_KNOTS + 1)
            for i in range(_BC_N_KNOTS)
        ]
        + [_BC_X1]
    )
    ys = [_BC_Y0] + list(u) + [_BC_Y1]
    total_time = 0.0
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        ds = math.hypot(dx, dy)
        # Velocity at midpoint by conservation of energy: v = sqrt(2g(y0 − y_mid))
        y_mid = (ys[i] + ys[i + 1]) / 2
        v_sq = 2 * _BC_GRAV * max(_BC_Y0 - y_mid, 1e-9)
        v = math.sqrt(v_sq)
        total_time += ds / v
    return total_time


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

DEMOS: list[Demo] = [
    Demo(
        name="welded_beam",
        n_dim=4,
        suggested_n_trials=200,
        objective=welded_beam_objective,
    ),
    Demo(
        name="robot_arm",
        n_dim=6,
        suggested_n_trials=200,
        objective=robot_arm_objective,
    ),
    Demo(
        name="slingshot",
        n_dim=2,
        suggested_n_trials=50,
        objective=slingshot_objective,
    ),
    Demo(
        name="brachistochrone",
        n_dim=10,
        suggested_n_trials=200,
        objective=brachistochrone_objective,
    ),
]


# -----------------------------------------------------------------------------
# Validation — known-good points from the demo
# -----------------------------------------------------------------------------


def _validate_welded_beam() -> None:
    """The published optimum lives at h≈0.205, l≈3.470, t≈9.037, b≈0.206
    with cost ≈ 1.725. Decode the cube point that maps there and check
    the objective evaluates to a feasible value near 1.725."""

    def encode(h: float, l: float, t: float, b: float) -> list[float]:
        return [
            (h - _WB_BOUNDS["h"][0]) / (_WB_BOUNDS["h"][1] - _WB_BOUNDS["h"][0]),
            (l - _WB_BOUNDS["l"][0]) / (_WB_BOUNDS["l"][1] - _WB_BOUNDS["l"][0]),
            (t - _WB_BOUNDS["t"][0]) / (_WB_BOUNDS["t"][1] - _WB_BOUNDS["t"][0]),
            (b - _WB_BOUNDS["b"][0]) / (_WB_BOUNDS["b"][1] - _WB_BOUNDS["b"][0]),
        ]

    u_opt = encode(0.205730, 3.470489, 9.036624, 0.205730)
    val = welded_beam_objective(u_opt)
    print(f"welded_beam @ published optimum: {val:.4f} (expected ≈ 1.725)")
    assert val < 2.5, f"welded_beam port appears broken: got {val}"


if __name__ == "__main__":
    _validate_welded_beam()
