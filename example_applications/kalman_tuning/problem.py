"""
Kalman Filter Noise Tuning — 1-D constant-velocity target tracking.

A 2-D optimisation problem from state estimation. A target drifts along a
line under a near-constant-velocity model; we observe only its noisy
position. A 2-state Kalman filter (position, velocity) reconstructs the
trajectory, but its accuracy hinges on two covariance hyper-parameters:
the process-noise scale `q` and the measurement-noise variance `r`. The
objective tunes `(q, r)` to minimise the position tracking RMSE against
the (hidden) true trajectory.

Variables (N_DIM=2), both log-scaled from the unit interval:
  u0 -> q in [1e-5, 1e-1]   (continuous-white-noise-acceleration scale)
  u1 -> r in [1e-2, 1e1]    (measurement-noise variance)

Pathology: ILL-CONDITIONED. Filter performance is governed almost
entirely by the RATIO q/r, which sets the steady-state Kalman gain.
The good region is therefore a long, thin DIAGONAL VALLEY in
(log q, log r) space rather than a round basin — only the offset along
the valley matters, the position across it barely does. Log-scaling
both axes is essential; on a linear scale the valley collapses against
the origin and most samples land in the flat, useless interior.

A well-tuned filter drives the position RMSE well below the raw
measurement-noise std (~2.0), into roughly 1.0–1.5.
"""

from __future__ import annotations

import math

# Trajectory and observation model constants.
N_STEPS = 80  # number of time steps
DT = 1.0  # time increment per step

# True-process and measurement noise used to GENERATE the data once.
TRUE_PROCESS_ACCEL_STD = 0.05  # std of the velocity random-walk increments
MEAS_NOISE_STD = 2.0  # std of the additive position-measurement noise

# Log-scaled bounds for the two tuning variables.
Q_LO, Q_HI = 1e-5, 1e-1
R_LO, R_HI = 1e-2, 1e1


class _LCG:
    """Tiny deterministic linear-congruential generator (Numerical Recipes
    constants) plus Box–Muller, so the dataset is fixed without `random`."""

    def __init__(self, seed):
        self._state = seed & 0xFFFFFFFF
        self._spare = None

    def _next_uniform(self):
        # Returns a float in the open interval (0, 1).
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        # Avoid exactly 0 so log() in Box–Muller is safe.
        return (self._state + 0.5) / 4294967296.0

    def gauss(self):
        """Standard-normal sample via Box–Muller (caches the spare)."""
        if self._spare is not None:
            z = self._spare
            self._spare = None
            return z
        u1 = self._next_uniform()
        u2 = self._next_uniform()
        radius = math.sqrt(-2.0 * math.log(u1))
        self._spare = radius * math.sin(2.0 * math.pi * u2)
        return radius * math.cos(2.0 * math.pi * u2)


def _build_dataset():
    """Generate the fixed true trajectory and noisy position measurements.

    Returns (true_pos, true_vel, measurements) as parallel lists of length
    N_STEPS. Built ONCE at import with a fixed seed so `objective` is
    fully deterministic.
    """
    rng = _LCG(seed=20260616)

    pos = 0.0
    vel = 1.0  # initial velocity
    true_pos = []
    true_vel = []
    measurements = []
    for _ in range(N_STEPS):
        # Velocity does a slow random walk; position integrates velocity.
        vel += TRUE_PROCESS_ACCEL_STD * rng.gauss()
        pos += vel * DT
        true_pos.append(pos)
        true_vel.append(vel)
        measurements.append(pos + MEAS_NOISE_STD * rng.gauss())
    return true_pos, true_vel, measurements


# Fixed data, computed once at import.
TRUE_POS, TRUE_VEL, MEASUREMENTS = _build_dataset()


def _scale_unit_to_params(u):
    """Map a point in [0,1]^2 to (q, r), both log-scaled."""
    u0 = min(1.0, max(0.0, u[0]))
    u1 = min(1.0, max(0.0, u[1]))
    q = Q_LO * (Q_HI / Q_LO) ** u0
    r = R_LO * (R_HI / R_LO) ** u1
    return q, r


def _tracking_rmse(q, r):
    """Run the standard predict/update Kalman recursion over MEASUREMENTS
    and return the position RMSE of the filtered estimate vs TRUE_POS.

    2-state filter, x = [position, velocity]:
        F = [[1, dt], [0, 1]]          (constant-velocity transition)
        H = [1, 0]                     (position-only observation)
        Q = q * [[dt^4/4, dt^3/2],     (continuous-white-noise-accel model)
                 [dt^3/2, dt^2  ]]
        R = r                          (scalar measurement variance)
    """
    dt = DT
    # Process-noise covariance (continuous white-noise acceleration model).
    q00 = q * dt**4 / 4.0
    q01 = q * dt**3 / 2.0
    q11 = q * dt**2

    x0, x1 = MEASUREMENTS[0], 0.0
    p00, p01, p10, p11 = 1e3, 0.0, 0.0, 1e3

    sq_err = 0.0
    for k, z in enumerate(MEASUREMENTS):
        # Predict.
        xp0 = x0 + dt * x1
        xp1 = x1
        a00 = p00 + dt * p10
        a01 = p01 + dt * p11
        a10 = p10
        a11 = p11
        pp00 = a00 + dt * a01 + q00
        pp01 = a01 + q01
        pp10 = a10 + dt * a11 + q01
        pp11 = a11 + q11

        # Update.
        s = pp00 + r
        k0 = pp00 / s
        k1 = pp10 / s
        y = z - xp0
        x0 = xp0 + k0 * y
        x1 = xp1 + k1 * y
        p00 = (1.0 - k0) * pp00
        p01 = (1.0 - k0) * pp01
        p10 = pp10 - k1 * pp00
        p11 = pp11 - k1 * pp01

        sq_err += (x0 - TRUE_POS[k]) ** 2

    return math.sqrt(sq_err / N_STEPS)


def objective(u):
    """HumpDay-style objective: input is `u in [0,1]^2`, output is the
    filtered-position tracking RMSE (lower is better)."""
    q, r = _scale_unit_to_params(u)
    return _tracking_rmse(q, r)


def decode(u):
    """Convenience: return the tuned `(q, r)` for a `[0,1]^2` point plus
    the resulting tracking RMSE."""
    q, r = _scale_unit_to_params(u)
    rmse = _tracking_rmse(q, r)
    return {"q": q, "r": r, "rmse": rmse}


N_DIM = 2
