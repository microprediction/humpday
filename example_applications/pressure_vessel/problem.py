"""
Pressure Vessel Design — classic constrained, mixed-integer benchmark.

A 4-D constrained optimisation problem from mechanical engineering. Find
the cheapest cylindrical pressure vessel (a cylinder capped by two
hemispherical heads) that holds a fixed volume under four physical
constraints.

Variables (physical):
  Ts — shell thickness, an INTEGER multiple of 0.0625 in in [0.0625, 6.1875]
  Th — head thickness,  an INTEGER multiple of 0.0625 in in [0.0625, 6.1875]
  R  — inner radius, in [10, 200]
  L  — length of the cylindrical section, in [10, 200]

The objective for HumpDay maps the unit hypercube [0,1]^4 to the physical
bounds, evaluates the fabrication cost, and adds a quadratic penalty for
each violated constraint.

Landscape pathology: the two thickness variables are quantised (rounded
to the nearest 1/16 in), so the cost surface is piecewise-flat with
discontinuous steps along Ts and Th — the gradient is zero almost
everywhere and undefined at the steps. That defeats smooth local methods
and trust-region surrogates, which assume continuity. The volume
constraint g3 carves a thin curved feasible sliver out of the box, and
the optimum sits hard against the g1/g3 boundary, so penalty-based
search must thread a narrow corridor without slipping infeasible.

Reported global minimum cost ≈ 6059.714 at (Ts, Th, R, L) ≈
(0.8125, 0.4375, 42.0984, 176.6366).
"""

from __future__ import annotations

import math

# Thickness quantisation: thicknesses are integer multiples of this gauge.
GAUGE = 0.0625
# Integer multiplier k in [1, 99] => thickness in [0.0625, 6.1875].
N_GAUGE = 98  # k = round(1 + 98 * u) spans 1..99

# Continuous bounds for the radius and length, in inches.
R_LO, R_HI = 10.0, 200.0
L_LO, L_HI = 10.0, 200.0

# Penalty weight for each unit of squared constraint violation. The
# optimum sits hard against the g1/g3 boundary, so population methods are
# tempted to camp a hair past it (violations of ~0.005 on g1). At 1e6
# that buys them ~25 cost units — too cheap, so they settle infeasible.
# Raising it to 1e7 makes the just-infeasible ledge cost ~250 and pushes
# the search back onto the feasible side without turning the boundary into
# a cliff that the smoother methods (CMA) can no longer approach.
PENALTY_WEIGHT = 1e7


def _scale_unit_to_physical(u):
    """Map a point in [0,1]^4 to (Ts, Th, R, L) in physical units.

    Ts and Th are snapped to the nearest 1/16 in gauge, mimicking the
    discrete plate thicknesses available in fabrication.
    """
    ts = GAUGE * round(1.0 + N_GAUGE * u[0])
    th = GAUGE * round(1.0 + N_GAUGE * u[1])
    r = R_LO + (R_HI - R_LO) * u[2]
    length = L_LO + (L_HI - L_LO) * u[3]
    return ts, th, r, length


def _cost(ts, th, r, length):
    """Fabrication cost: material + welding + forming."""
    return (
        0.6224 * ts * r * length
        + 1.7781 * th * r * r
        + 3.1661 * ts * ts * length
        + 19.84 * ts * ts * r
    )


def _constraint_violations(ts, th, r, length):
    """Return list of g_i(x) values; g_i <= 0 is feasible."""
    return [
        -ts + 0.0193 * r,  # g1 shell thickness vs pressure
        -th + 0.00954 * r,  # g2 head thickness vs pressure
        -math.pi * r * r * length
        - (4.0 / 3.0) * math.pi * r**3
        + 1296000.0,  # g3 minimum volume
        length - 240.0,  # g4 maximum length
    ]


def objective(u):
    """HumpDay-style objective: input is `u ∈ [0,1]^4`, output is cost
    (with quadratic penalty for any constraint violation)."""
    ts, th, r, length = _scale_unit_to_physical(u)
    base = _cost(ts, th, r, length)
    violations = _constraint_violations(ts, th, r, length)
    penalty = sum(PENALTY_WEIGHT * max(0.0, g) ** 2 for g in violations)
    return base + penalty


def decode(u):
    """Convenience: return the physical design `(Ts, Th, R, L)` for a
    `[0,1]^4` point, plus cost and feasibility."""
    ts, th, r, length = _scale_unit_to_physical(u)
    cost = _cost(ts, th, r, length)
    violations = _constraint_violations(ts, th, r, length)
    max_violation = max((max(0.0, g) for g in violations), default=0.0)
    return {
        "Ts": ts,
        "Th": th,
        "R": r,
        "L": length,
        "cost": cost,
        "max_violation": max_violation,
        "feasible": max_violation <= 1e-4,
    }


N_DIM = 4
