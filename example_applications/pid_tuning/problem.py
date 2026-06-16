"""
PID Controller Tuning — second-order plant step-response, control engineering.

Tune the three gains of a textbook PID controller so that an underdamped
second-order plant tracks a unit-step reference as tightly as possible.

The plant is the classic mass–spring–damper / RLC transfer function

    y'' + 2·ζ·ωn·y' + ωn²·y = ωn²·u

with damping ratio ζ = 0.3 and natural frequency ωn = 1.0 (lightly damped,
so it overshoots and rings if controlled badly). The controller closes the
loop on the unit-step reference r = 1 via

    e    = r − y
    u    = Kp·e + Ki·I + Kd·ėdot      (I = ∫e dt, ėdot = de/dt)

We integrate with explicit Euler (dt = 0.01) over a horizon T = 30 (~3000
steps) and score the integral of squared error, ISE = Σ e²·dt.

Variables (N_DIM = 3), mapped linearly from [0,1]:
    Kp ∈ [0, 30]    proportional gain
    Ki ∈ [0, 15]    integral gain
    Kd ∈ [0, 15]    derivative gain

Pathology: TWO bad behaviours at once.
  1. An ill-conditioned valley. Kp and Kd trade off against each other (more
     derivative damping lets you crank up proportional gain) and Ki couples
     in to kill steady-state error, so the good region is a curved, skewed
     ravine rather than an axis-aligned bowl.
  2. An instability cliff. Push the gains too high or out of balance and the
     explicit-Euler loop diverges; the response is clamped and scored with a
     flat PENALTY = 1e3. That penalty plateau borders the good basin, so an
     optimiser must descend into the stable ravine without stepping off the
     edge. There is no clean closed-form optimum; a competent tuning reaches
     ISE on the order of ~1–3.
"""

from __future__ import annotations

import math

# Gain bounds, mapped linearly from the unit cube.
LOWER = (0.0, 0.0, 0.0)
UPPER = (30.0, 15.0, 15.0)

# Plant parameters: underdamped second-order system.
ZETA = 0.3  # damping ratio
WN = 1.0  # natural frequency (rad/s)

# Closed-loop unit-step reference.
R = 1.0

# Euler integration grid.
DT = 0.01
T_HORIZON = 30.0
N_STEPS = int(round(T_HORIZON / DT))

# Anything beyond this magnitude is treated as a diverging (unstable) loop.
DIVERGENCE = 1e3

# Flat cost returned for an unstable / diverging tuning.
PENALTY = 1e3


def _scale_unit_to_gains(u):
    """Map a point in [0,1]^3 to controller gains (Kp, Ki, Kd)."""
    return tuple(LOWER[i] + (UPPER[i] - LOWER[i]) * u[i] for i in range(3))


def _simulate(kp, ki, kd):
    """Run the closed-loop step response and return (ISE, stable).

    Explicit Euler on (y, ydot). If the state ever exceeds DIVERGENCE the
    loop is unstable: we stop and report (PENALTY, False) rather than letting
    the integration overflow.
    """
    y = 0.0
    ydot = 0.0
    integral = 0.0
    e_prev = R - y  # error at t = 0

    ise = 0.0
    for _ in range(N_STEPS):
        e = R - y
        integral += e * DT
        edot = (e - e_prev) / DT
        e_prev = e

        # PID control signal.
        u_ctrl = kp * e + ki * integral + kd * edot

        # Plant acceleration: y'' = ωn²·u − 2·ζ·ωn·y' − ωn²·y.
        yddot = WN * WN * u_ctrl - 2.0 * ZETA * WN * ydot - WN * WN * y

        # Euler step.
        y += ydot * DT
        ydot += yddot * DT

        if not math.isfinite(y) or abs(y) > DIVERGENCE:
            return PENALTY, False

        ise += e * e * DT

    return ise, True


def objective(u):
    """HumpDay-style objective: input is `u ∈ [0,1]^3`, output is the integral
    of squared tracking error (ISE) to MINIMISE. Diverging gains return a flat
    PENALTY = 1e3 instead of overflowing."""
    kp, ki, kd = _scale_unit_to_gains(u)
    ise, _ = _simulate(kp, ki, kd)
    return ise


def decode(u):
    """Convenience: return the gains `(Kp, Ki, Kd)` for a `[0,1]^3` point,
    plus the achieved ISE and a stability flag."""
    kp, ki, kd = _scale_unit_to_gains(u)
    ise, stable = _simulate(kp, ki, kd)
    return {
        "Kp": kp,
        "Ki": ki,
        "Kd": kd,
        "ISE": ise,
        "stable": stable,
    }


N_DIM = 3
