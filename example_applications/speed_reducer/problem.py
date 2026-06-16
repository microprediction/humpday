"""
Speed Reducer (Golinski gearbox) — a constrained, mixed-integer classic.

Minimise the weight of a single-stage gearbox over seven design variables
subject to eleven nonlinear inequality constraints drawn from real mechanical
physics (gear-tooth bending and surface stress, shaft deflections and shaft
stresses, plus geometric limits). One variable — the number of pinion teeth —
is an integer.

This is one of the most-cited constrained engineering benchmarks. Its known
global optimum lets HumpDay *verify* a result, not just rank methods:

    x* = (3.5, 0.7, 17, 7.3, 7.7153190, 3.350282, 5.286654),  weight ≈ 2994.4712

What it stresses (and how it differs from welded_beam):
  - Eleven simultaneously-relevant nonlinear constraints (vs welded_beam's seven),
    several active at the optimum — a narrow feasible corner.
  - A mixed-integer variable (pinion teeth), like tuned_mass_damper.
  - A nonconvex (signomial / generalized-geometric-program) objective, so
    metaheuristics cannot *guarantee* the optimum — the known value is the check.

Variables (scaled from [0,1]^7):
  x1 face width b           [2.6, 3.6]
  x2 tooth module m         [0.7, 0.8]
  x3 number of pinion teeth [17, 28]  (integer)
  x4 shaft-1 length l1      [7.3, 8.3]
  x5 shaft-2 length l2      [7.3, 8.3]
  x6 shaft-1 diameter d1    [2.9, 3.9]
  x7 shaft-2 diameter d2    [5.0, 5.5]

Refs: Golinski (1970); standard CEC/engineering constrained-optimisation suites
(e.g. arXiv 2505.03512 reproduces weight 2994.471921 at the optimum above).
"""

from __future__ import annotations

import math

N_DIM = 7

LOWER = (2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0)
UPPER = (3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5)

# Penalty per unit of (squared) constraint violation. The constraints are
# normalised (g_i ~ O(1)), and cutting a corner only saves ~150 in weight, so
# the penalty must be large enough that even a small violation outweighs that —
# otherwise optimisers sit just outside the feasible region.
PENALTY_WEIGHT = 1e7

# The optimum lies ON several active constraints, so any near-optimal design
# carries a tiny (sub-0.1%) violation under a finite penalty. Treat that as
# feasible for reporting — it's engineering round-off, not a real violation.
FEASIBLE_TOL = 1e-3


def _scale(u):
    """Map [0,1]^7 to physical variables; x3 (pinion teeth) is an integer."""
    x = [LOWER[i] + (UPPER[i] - LOWER[i]) * u[i] for i in range(7)]
    x[2] = float(round(x[2]))  # integer number of teeth
    return x


def _weight(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    return (
        0.7854 * x1 * x2 * x2 * (3.3333 * x3 * x3 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6 * x6 + x7 * x7)
        + 7.4777 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    )


def _constraints(x):
    """Return the 11 g_i(x); g_i <= 0 is feasible (positive = violation)."""
    x1, x2, x3, x4, x5, x6, x7 = x
    return [
        27.0 / (x1 * x2 * x2 * x3) - 1.0,
        397.5 / (x1 * x2 * x2 * x3 * x3) - 1.0,
        1.93 * x4**3 / (x2 * x3 * x6**4) - 1.0,
        1.93 * x5**3 / (x2 * x3 * x7**4) - 1.0,
        math.sqrt((745.0 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110.0 * x6**3) - 1.0,
        math.sqrt((745.0 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85.0 * x7**3) - 1.0,
        x2 * x3 / 40.0 - 1.0,
        5.0 * x2 / x1 - 1.0,
        x1 / (12.0 * x2) - 1.0,
        (1.5 * x6 + 1.9) / x4 - 1.0,
        (1.1 * x7 + 1.9) / x5 - 1.0,
    ]


def objective(u):
    """HumpDay-style objective: `u ∈ [0,1]^7` -> gearbox weight plus a quadratic
    penalty for each violated constraint."""
    x = _scale(u)
    base = _weight(x)
    penalty = sum(PENALTY_WEIGHT * max(0.0, g) ** 2 for g in _constraints(x))
    return base + penalty


def decode(u):
    """Convenience: physical design + weight + feasibility for a `[0,1]^7` point."""
    x = _scale(u)
    g = _constraints(x)
    max_violation = max((max(0.0, gi) for gi in g), default=0.0)
    return {
        "vars": {
            "b": x[0],
            "m": x[1],
            "teeth": int(x[2]),
            "l1": x[3],
            "l2": x[4],
            "d1": x[5],
            "d2": x[6],
        },
        "weight": _weight(x),
        "max_violation": max_violation,
        "feasible": max_violation <= FEASIBLE_TOL,
    }
