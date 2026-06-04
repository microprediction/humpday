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
# Battery dispatch — 24-D charge/discharge schedule against a day-ahead
# price curve. State-of-charge bounds couple consecutive hours, so the
# landscape is smooth but constrained.
# Source: docs/applications/battery-dispatch.html.
# -----------------------------------------------------------------------------

_BD_HOURS = 24
_BD_P_MAX = 25.0  # MW
_BD_E_MAX = 100.0  # MWh
_BD_SOC_LO = 10.0
_BD_SOC_HI = 90.0
_BD_SOC_INIT = 50.0
_BD_ETA_C = 0.92
_BD_ETA_D = 0.92
_BD_PRICE = (
    30,
    25,
    22,
    20,
    22,
    35,
    55,
    75,
    90,
    100,
    90,
    75,
    55,
    40,
    38,
    45,
    80,
    150,
    240,
    220,
    160,
    100,
    60,
    40,
)


def battery_dispatch_objective(u: list[float]) -> float:
    schedule = [(u[h] - 0.5) * 2 * _BD_P_MAX for h in range(_BD_HOURS)]
    SoC = _BD_SOC_INIT
    revenue = 0.0
    for h in range(_BD_HOURS):
        p_actual = schedule[h]
        if p_actual < 0:
            max_charge = (_BD_SOC_HI - SoC) / _BD_ETA_C
            if -p_actual > max_charge:
                p_actual = -max_charge
        elif p_actual > 0:
            max_discharge = (SoC - _BD_SOC_LO) * _BD_ETA_D
            if p_actual > max_discharge:
                p_actual = max_discharge
        revenue += p_actual * _BD_PRICE[h]
        if p_actual < 0:
            SoC += -p_actual * _BD_ETA_C
        else:
            SoC -= p_actual / _BD_ETA_D
    return -revenue  # maximise revenue ↔ minimise -revenue


# -----------------------------------------------------------------------------
# Reactor T-profile — 10-D temperature profile for an A→B→C series
# reactor with Arrhenius kinetics. Maximise yield of intermediate B.
# Source: docs/applications/reactor-tprofile.html.
# -----------------------------------------------------------------------------

_RX_N_ZONES = 10
_RX_TAU_PER_ZONE = 0.1
_RX_T_REF = 400.0
_RX_B1 = 6.0
_RX_B2 = 22.0
_RX_T_MIN = 300.0
_RX_T_MAX = 480.0
_RX_N_PROFILE_SAMPLES = 80


def _rx_arr(T: float, B: float) -> float:
    return math.exp(B * (1 - _RX_T_REF / T))


def _rx_advance(CA: float, CB: float, T: float, dtau: float) -> tuple[float, float]:
    k1 = _rx_arr(T, _RX_B1)
    k2 = _rx_arr(T, _RX_B2)
    e1 = math.exp(-k1 * dtau)
    e2 = math.exp(-k2 * dtau)
    CAn = CA * e1
    if abs(k2 - k1) < 1e-9:
        CBn = (CA * k1 * dtau + CB) * e1
    else:
        CBn = (k1 * CA / (k2 - k1)) * (e1 - e2) + CB * e2
    return CAn, CBn


def reactor_tprofile_objective(u: list[float]) -> float:
    T = [_RX_T_MIN + ui * (_RX_T_MAX - _RX_T_MIN) for ui in u]
    sub_per_zone = max(1, _RX_N_PROFILE_SAMPLES // _RX_N_ZONES)
    dsub = _RX_TAU_PER_ZONE / sub_per_zone
    CA, CB = 1.0, 0.0
    for zi in range(_RX_N_ZONES):
        for _ in range(sub_per_zone):
            CA, CB = _rx_advance(CA, CB, T[zi], dsub)
    return -CB  # maximise yield of B


# -----------------------------------------------------------------------------
# Wind farm — 16-D placement of 8 turbines in a rectangular field. Jensen
# wake model, 12-bin wind rose. Maximise expected farm power.
# Source: docs/applications/wind-farm.html.
# -----------------------------------------------------------------------------

_WF_FIELD_X0, _WF_FIELD_X1 = 60.0, 740.0
_WF_FIELD_Y0, _WF_FIELD_Y1 = 60.0, 390.0
_WF_N_TURBINES = 8
_WF_ROTOR_R = 14.0
_WF_ROTOR_D = _WF_ROTOR_R * 2
_WF_AXIAL_INDUCTION = 0.42
_WF_WAKE_DECAY = 0.08
_WF_MIN_SPACING = _WF_ROTOR_D * 3
_WF_BOUNDARY_BONUS_PTS = 2.5

_WF_WIND_ROSE = (
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
)
_WF_WIND_ROSE_RAD = tuple(
    ((compass + 90) * math.pi / 180.0, weight) for compass, weight in _WF_WIND_ROSE
)


def _wf_wind_speeds(
    positions: list[tuple[float, float]], wind_angle: float
) -> list[float]:
    cos = math.cos(-wind_angle)
    sin = math.sin(-wind_angle)
    rot = [(p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos) for p in positions]
    v: list[float] = []
    for i, (xi, yi) in enumerate(rot):
        def_sq = 0.0
        for j, (xj, yj) in enumerate(rot):
            if i == j:
                continue
            dx = xi - xj  # downwind distance from j to i
            dy = yi - yj
            if dx <= 0:
                continue
            wake_r = _WF_ROTOR_R + _WF_WAKE_DECAY * dx
            if abs(dy) > wake_r:
                continue
            deficit = (2 * _WF_AXIAL_INDUCTION) / (
                1 + _WF_WAKE_DECAY * dx / _WF_ROTOR_R
            ) ** 2
            def_sq += deficit * deficit
        v.append(max(0.0, 1 - math.sqrt(def_sq)))
    return v


def wind_farm_objective(u: list[float]) -> float:
    positions = [
        (
            _WF_FIELD_X0 + u[2 * i] * (_WF_FIELD_X1 - _WF_FIELD_X0),
            _WF_FIELD_Y0 + u[2 * i + 1] * (_WF_FIELD_Y1 - _WF_FIELD_Y0),
        )
        for i in range(_WF_N_TURBINES)
    ]
    expected_power = 0.0
    for angle, weight in _WF_WIND_ROSE_RAD:
        speeds = _wf_wind_speeds(positions, angle)
        p = sum(v**3 for v in speeds)
        expected_power += weight * p
    power_fraction = expected_power / _WF_N_TURBINES

    # Spacing penalty (squared violation per pair).
    penalty = 0.0
    for i in range(_WF_N_TURBINES):
        for j in range(i + 1, _WF_N_TURBINES):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.hypot(dx, dy)
            if d < _WF_MIN_SPACING:
                violation = (_WF_MIN_SPACING - d) / _WF_MIN_SPACING
                penalty += violation * violation

    # Boundary bonus (Chebyshev distance from centre, averaged).
    cxf = (_WF_FIELD_X0 + _WF_FIELD_X1) / 2
    cyf = (_WF_FIELD_Y0 + _WF_FIELD_Y1) / 2
    halfW = (_WF_FIELD_X1 - _WF_FIELD_X0) / 2
    halfH = (_WF_FIELD_Y1 - _WF_FIELD_Y0) / 2
    boundary = (
        sum(max(abs(x - cxf) / halfW, abs(y - cyf) / halfH) for x, y in positions)
        / _WF_N_TURBINES
    )

    score = 100 * power_fraction + _WF_BOUNDARY_BONUS_PTS * boundary - 40 * penalty
    return -score


# -----------------------------------------------------------------------------
# CartPole-v1 direct policy search — 5-dim linear policy weights trained by
# black-box optimization. Direct port of docs/applications/cart-pole.html,
# which is itself a port of OpenAI Gym's classic CartPole-v1 physics.
# Non-smooth objective: returns are integer step counts (1..MAX_STEPS).
# -----------------------------------------------------------------------------

_CP_N_EPISODES = 8
_CP_MAX_STEPS = 500
_CP_WEIGHT_RANGE = 5.0
_CP_GRAVITY = 9.8
_CP_MASS_CART = 1.0
_CP_MASS_POLE = 0.1
_CP_TOTAL_MASS = _CP_MASS_CART + _CP_MASS_POLE
_CP_LENGTH = 0.5  # half pole length (m)
_CP_POLEMASS_LENGTH = _CP_MASS_POLE * _CP_LENGTH
_CP_FORCE_MAG = 10.0
_CP_TAU = 0.02  # seconds per step
_CP_X_THRESHOLD = 2.4
_CP_THETA_THRESHOLD = 12.0 * math.pi / 180.0


def _cp_mulberry32(seed: int) -> Callable[[], float]:
    """Reproduce the Mulberry32 PRNG used in the JS cart-pole demo. The
    `& 0xFFFFFFFF` masks emulate JS's `| 0` int32 truncation so the same
    seed yields the same sequence as the in-browser run."""
    s = (seed & 0xFFFFFFFF) or 1

    def next_value() -> float:
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = (s ^ (s >> 15)) * (1 | s) & 0xFFFFFFFF
        t = ((t + ((t ^ (t >> 7)) * (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return next_value


def _cp_step(
    state: tuple[float, float, float, float], action: int
) -> tuple[float, float, float, float]:
    x, x_dot, theta, theta_dot = state
    force = _CP_FORCE_MAG if action == 1 else -_CP_FORCE_MAG
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    temp = (
        force + _CP_POLEMASS_LENGTH * theta_dot * theta_dot * sin_t
    ) / _CP_TOTAL_MASS
    theta_acc = (_CP_GRAVITY * sin_t - cos_t * temp) / (
        _CP_LENGTH * (4.0 / 3.0 - _CP_MASS_POLE * cos_t * cos_t / _CP_TOTAL_MASS)
    )
    x_acc = temp - _CP_POLEMASS_LENGTH * theta_acc * cos_t / _CP_TOTAL_MASS
    return (
        x + _CP_TAU * x_dot,
        x_dot + _CP_TAU * x_acc,
        theta + _CP_TAU * theta_dot,
        theta_dot + _CP_TAU * theta_acc,
    )


def _cp_rollout(policy: tuple[float, float, float, float, float], seed: int) -> int:
    """Run one CartPole episode under the given linear policy and return
    the number of steps survived (1..MAX_STEPS)."""
    rng = _cp_mulberry32(seed)
    state = (
        rng() * 0.1 - 0.05,
        rng() * 0.1 - 0.05,
        rng() * 0.1 - 0.05,
        rng() * 0.1 - 0.05,
    )
    w1, w2, w3, w4, bias = policy
    for t in range(_CP_MAX_STEPS):
        x, x_dot, theta, theta_dot = state
        score = w1 * x + w2 * x_dot + w3 * theta + w4 * theta_dot + bias
        action = 1 if score > 0 else 0
        state = _cp_step(state, action)
        if abs(state[0]) > _CP_X_THRESHOLD or abs(state[2]) > _CP_THETA_THRESHOLD:
            return t + 1
    return _CP_MAX_STEPS


def cart_pole_objective(u: list[float]) -> float:
    """Minimise −(mean return) across N_EPISODES rollouts of a linear
    policy whose 5 weights are decoded from u ∈ [0,1]^5 via the affine
    map w = 10u − 5. Best achievable value is −500 (perfectly balanced
    across all eight rollouts of MAX_STEPS = 500)."""
    policy = tuple(_CP_WEIGHT_RANGE * (2.0 * ui - 1.0) for ui in u)
    total = 0
    for i in range(_CP_N_EPISODES):
        total += _cp_rollout(policy, i)
    return -total / _CP_N_EPISODES


# -----------------------------------------------------------------------------
# MLP-on-XOR — 17-dim weight-space optimisation of a 2-4-1 tanh/sigmoid net
# trained on the four XOR examples. Pure stdlib math, no torch/numpy dep.
# Famously non-convex: many local minima where the net learns 3 of 4 patterns.
# -----------------------------------------------------------------------------

_MLP_X: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
)
_MLP_Y: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0)
_MLP_N_HIDDEN = 4
_MLP_WEIGHT_SCALE = 3.0  # u∈[0,1] → weight in [-3, 3]


def _mlp_sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def mlp_xor_objective(u: list[float]) -> float:
    """MSE of a 2-4-1 MLP (tanh hidden, sigmoid output) on the XOR truth table,
    with the 17 weights drawn from u∈[0,1]^17 via the affine map w = 6u − 3.

    Decode order: W1 (8 = 2 inputs × 4 hidden), b1 (4), W2 (4), b2 (1)."""
    w = [_MLP_WEIGHT_SCALE * (2 * ui - 1) for ui in u]
    # W1[i][j] is the weight from input i to hidden unit j.
    W1 = [w[4 * i : 4 * (i + 1)] for i in range(2)]
    b1 = w[8:12]
    W2 = w[12:16]
    b2 = w[16]

    loss = 0.0
    for (x0, x1), y in zip(_MLP_X, _MLP_Y):
        h = [
            math.tanh(W1[0][j] * x0 + W1[1][j] * x1 + b1[j])
            for j in range(_MLP_N_HIDDEN)
        ]
        z = sum(W2[j] * h[j] for j in range(_MLP_N_HIDDEN)) + b2
        yhat = _mlp_sigmoid(z)
        loss += (yhat - y) ** 2
    return loss / len(_MLP_X)


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
    Demo(
        name="battery_dispatch",
        n_dim=24,
        suggested_n_trials=200,
        objective=battery_dispatch_objective,
    ),
    Demo(
        name="reactor_tprofile",
        n_dim=10,
        suggested_n_trials=200,
        objective=reactor_tprofile_objective,
    ),
    Demo(
        name="wind_farm",
        n_dim=16,
        suggested_n_trials=200,
        objective=wind_farm_objective,
    ),
    Demo(
        name="mlp_xor",
        n_dim=17,
        suggested_n_trials=200,
        objective=mlp_xor_objective,
    ),
    Demo(
        name="cart_pole",
        n_dim=5,
        suggested_n_trials=200,
        objective=cart_pole_objective,
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
