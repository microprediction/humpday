"""
Welded Beam Design — Ragsdell & Phillips (1976) formulation.

A 4-D constrained optimisation problem from structural engineering.
Find the cheapest welded beam (variables h, l, t, b) that survives
a 6000-lb point load under seven physical constraints.

The objective for HumpDay maps the unit hypercube [0,1]^4 to the
physical bounds, evaluates the cost, and adds a quadratic penalty
for each violated constraint.

Reported global minimum cost ≈ 1.725 at (h, l, t, b) ≈
(0.2057, 3.4705, 9.0366, 0.2057). Many published metaheuristics
report 1.7248 — 1.725; constraint-aware methods on HumpDay should
find designs in [1.73, 2.0] within ~300 evaluations.
"""

from __future__ import annotations

import math

# Physical bounds for each variable in inches.
LOWER = (0.125, 0.1, 0.1, 0.1)
UPPER = (2.0, 10.0, 10.0, 2.0)

# Loading and material constants.
P = 6000.0  # lb, applied load
L = 14.0  # in, beam length
E = 30e6  # psi, Young's modulus
G = 12e6  # psi, shear modulus
TAU_MAX = 13600.0  # psi, allowable shear stress
SIGMA_MAX = 30000.0  # psi, allowable bending stress
DELTA_MAX = 0.25  # in, allowable end deflection

# Penalty weight for each unit of constraint violation. Large enough that
# infeasible designs always dominate any sub-optimal feasible one.
PENALTY_WEIGHT = 1e4


def _scale_unit_to_physical(u):
    """Map a point in [0,1]^4 to (h, l, t, b) in physical units."""
    return tuple(LOWER[i] + (UPPER[i] - LOWER[i]) * u[i] for i in range(4))


def _cost(h, l, t, b):
    """Fabrication + labour + material cost of a (feasible or not) design."""
    return 1.10471 * h * h * l + 0.04811 * t * b * (14.0 + l)


def _stresses(h, l, t, b):
    """Compute (shear τ, bending σ, deflection δ, buckling Pc).

    All formulas from Ragsdell & Phillips (1976). Guarded against
    divide-by-zero at the bounds of the unit cube.
    """
    eps = 1e-12

    # Primary and secondary shear.
    tau_prime = P / (math.sqrt(2.0) * h * l + eps)

    # Moment about weld centroid.
    M = P * (L + l / 2.0)
    R = math.sqrt(l * l / 4.0 + (h + t) * (h + t) / 4.0)
    # Polar moment of inertia of the weld group.
    J = 2.0 * (math.sqrt(2.0) * h * l * (l * l / 12.0 + (h + t) * (h + t) / 4.0))

    tau_double_prime = M * R / (J + eps)

    # Total shear stress (combine primary and secondary via law of cosines
    # using cos(angle) = l / (2R)).
    tau = math.sqrt(
        tau_prime * tau_prime
        + 2.0 * tau_prime * tau_double_prime * l / (2.0 * R + eps)
        + tau_double_prime * tau_double_prime
    )

    # Bending stress and tip deflection.
    sigma = 6.0 * P * L / (t * t * b + eps)
    delta = 4.0 * P * L * L * L / (E * t * t * t * b + eps)

    # Buckling load (Euler-like).
    pc = (
        4.013
        * E
        * math.sqrt((t * t * b * b * b * b * b * b) / 36.0)
        / (L * L)
        * (1.0 - t / (2.0 * L) * math.sqrt(E / (4.0 * G + eps)))
    )

    return tau, sigma, delta, pc


def _constraint_violations(h, l, t, b):
    """Return list of g_i(x) values; positive values are violations."""
    tau, sigma, delta, pc = _stresses(h, l, t, b)
    return [
        tau - TAU_MAX,  # g1 shear
        sigma - SIGMA_MAX,  # g2 bending
        h - b,  # g3 thickness ordering
        0.10471 * h * h + 0.04811 * t * b * (14.0 + l) - 5.0,  # g4 combined
        0.125 - h,  # g5 minimum weld thickness
        delta - DELTA_MAX,  # g6 deflection
        P - pc,  # g7 buckling
    ]


def objective(u):
    """HumpDay-style objective: input is `u ∈ [0,1]^4`, output is cost
    (with quadratic penalty for any constraint violation)."""
    h, l, t, b = _scale_unit_to_physical(u)
    base = _cost(h, l, t, b)
    violations = _constraint_violations(h, l, t, b)
    penalty = sum(PENALTY_WEIGHT * max(0.0, g) ** 2 for g in violations)
    return base + penalty


def decode(u):
    """Convenience: return the physical design `(h, l, t, b)` for a
    `[0,1]^4` point, plus the (cost, max_violation) pair."""
    h, l, t, b = _scale_unit_to_physical(u)
    cost = _cost(h, l, t, b)
    violations = _constraint_violations(h, l, t, b)
    max_violation = max((max(0.0, g) for g in violations), default=0.0)
    return {
        "h": h,
        "l": l,
        "t": t,
        "b": b,
        "cost": cost,
        "max_violation": max_violation,
        "feasible": max_violation <= 1e-6,
    }


N_DIM = 4
