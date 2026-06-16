"""
Cantilever Beam Design — classic single-constraint structural benchmark.

A 5-section hollow square cantilever beam carries a tip load. Each of
the five segments has a height variable x1..x5; the goal is to minimise
the total weight while keeping the tip deflection within an allowable
bound. The deflection limit collapses to a single nonlinear inequality.

Variables (section heights), each in physical [0.01, 100], mapped
linearly from the unit interval; N_DIM = 5.

Weight (proportional, to MINIMISE):

    f = 0.0624 * (x1 + x2 + x3 + x4 + x5)

Single deflection constraint (feasible when g <= 0):

    g = 61/x1**3 + 37/x2**3 + 19/x3**3 + 7/x4**3 + 1/x5**3 - 1

The HumpDay objective folds infeasibility into the cost via a quadratic
penalty, since the API takes no explicit constraints:

    objective = f + PENALTY_WEIGHT * max(0, g)**2

Pathology: a single smooth-but-curved nonlinear constraint carves the
feasible region, and the global optimum sits exactly ON that boundary
(the constraint is active there). An optimiser cannot simply drive every
variable up or down; it must ride the curved boundary to trade weight
against deflection. Penalty-based handling turns the boundary into a
steep valley wall.

Known global optimum: weight ≈ 1.33996 at
x ≈ (6.016, 5.309, 4.494, 3.502, 2.153).
"""

from __future__ import annotations

import math

# Physical bounds for each section height. Mapped linearly to [0, 1]^5.
LOWER = 0.01
UPPER = 100.0

# Coefficients on each 1/x**3 term in the deflection constraint.
DEFLECTION_COEFFS = (61.0, 37.0, 19.0, 7.0, 1.0)

# Weight coefficient (per unit of summed section height).
WEIGHT_COEFF = 0.0624

# Penalty weight per unit of (squared) constraint violation. The optimum
# weight is ~1.34 and g is O(1) near the boundary, so a weight of 1e4
# makes even a small violation dominate any feasible weight saving.
PENALTY_WEIGHT = 1e4

N_DIM = 5


def _scale_unit_to_physical(u):
    """Map a point in [0,1]^5 to section heights (x1..x5) in physical units."""
    return tuple(LOWER + (UPPER - LOWER) * u[i] for i in range(N_DIM))


def _weight(x):
    """Proportional weight of the beam (the quantity to minimise)."""
    return WEIGHT_COEFF * sum(x)


def _constraint(x):
    """Deflection constraint g(x); feasible when g <= 0."""
    eps = 1e-12
    return sum(c / (xi**3 + eps) for c, xi in zip(DEFLECTION_COEFFS, x)) - 1.0


def objective(u):
    """HumpDay-style objective: input is `u in [0,1]^5`, output is weight
    (with a quadratic penalty for any deflection-constraint violation)."""
    x = _scale_unit_to_physical(u)
    base = _weight(x)
    g = _constraint(x)
    penalty = PENALTY_WEIGHT * max(0.0, g) ** 2
    return base + penalty


def decode(u):
    """Convenience: return the physical design for a `[0,1]^5` point, plus
    weight, the constraint value, and a feasibility flag (with tolerance)."""
    x = _scale_unit_to_physical(u)
    weight = _weight(x)
    g = _constraint(x)
    return {
        "x": list(x),
        "weight": weight,
        "g": g,
        "violation": max(0.0, g),
        "feasible": g <= 1e-4,
    }
