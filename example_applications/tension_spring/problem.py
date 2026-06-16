"""
Tension/Compression Spring Design — Arora (1989) / Belegundu (1982).

A 3-D constrained optimisation problem from mechanical engineering.
Find the lightest helical coil spring (variables d, D, N) that satisfies
four physical inequality constraints on deflection, shear stress, surge
frequency, and outer-diameter feasibility.

The objective for HumpDay maps the unit hypercube [0,1]^3 to the
physical bounds, evaluates the spring weight, and adds a quadratic
penalty for each violated constraint.

Variables:
  d = wire diameter      in [0.05, 2.0]
  D = mean coil diameter in [0.25, 1.3]
  N = number of active coils in [2.0, 15.0]

Pathology:
  The minimised weight is tiny (≈ 0.0127), while a typical constraint
  violation is O(1). Folding constraints into the objective therefore
  requires a *very* large penalty weight, producing steep penalty walls
  around a narrow feasible corridor. The denominators in g1/g2/g3 also
  blow up as d → 0, so the landscape has near-singular ridges that punish
  naive scaling. This rewards trust-region and population methods and
  trips up unguided local search.

Reported global minimum weight ≈ 0.012665 at
(d, D, N) ≈ (0.05169, 0.35672, 11.289).
"""

from __future__ import annotations

# Physical bounds for each variable.
LOWER = (0.05, 0.25, 2.0)
UPPER = (2.0, 1.3, 15.0)

# Penalty weight for each unit of (squared) constraint violation. The weight
# at the optimum is ~0.0127 while violations are O(1), so the penalty must be
# large enough that any infeasible design loses to the feasible optimum.
# 1e3..1e5 all work; 1e4 keeps a clean margin without overflowing far-out
# infeasible probes.
PENALTY_WEIGHT = 1e4


def _scale_unit_to_physical(u):
    """Map a point in [0,1]^3 to (d, D, N) in physical units."""
    return tuple(LOWER[i] + (UPPER[i] - LOWER[i]) * u[i] for i in range(3))


def _weight(d, D, N):
    """Weight (objective) of the coil spring."""
    return (N + 2.0) * D * d * d


def _constraint_violations(d, D, N):
    """Return list of g_i(x) values; positive values are violations.

    Denominators are guarded with a tiny eps against divide-by-zero at the
    bounds of the unit cube.
    """
    eps = 1e-12

    g1 = 1.0 - (D**3 * N) / (71785.0 * d**4 + eps)
    g2 = (
        (4.0 * D * D - d * D) / (12566.0 * (D * d**3 - d**4) + eps)
        + 1.0 / (5108.0 * d * d + eps)
        - 1.0
    )
    g3 = 1.0 - 140.45 * d / (D * D * N + eps)
    g4 = (D + d) / 1.5 - 1.0
    return [g1, g2, g3, g4]


def objective(u):
    """HumpDay-style objective: input is `u ∈ [0,1]^3`, output is weight
    (with quadratic penalty for any constraint violation)."""
    d, D, N = _scale_unit_to_physical(u)
    base = _weight(d, D, N)
    violations = _constraint_violations(d, D, N)
    penalty = sum(PENALTY_WEIGHT * max(0.0, g) ** 2 for g in violations)
    return base + penalty


def decode(u):
    """Convenience: return the physical design `(d, D, N)` for a
    `[0,1]^3` point, plus the weight and feasibility."""
    d, D, N = _scale_unit_to_physical(u)
    weight = _weight(d, D, N)
    violations = _constraint_violations(d, D, N)
    max_violation = max((max(0.0, g) for g in violations), default=0.0)
    return {
        "d": d,
        "D": D,
        "N": N,
        "weight": weight,
        "max_violation": max_violation,
        "feasible": max_violation <= 1e-4,
    }


N_DIM = 3
