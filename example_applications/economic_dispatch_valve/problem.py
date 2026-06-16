"""
Economic Dispatch with Valve-Point Loading — a 3-unit power-systems benchmark.

Schedule three thermal generators so that their combined output meets a
fixed system demand of D = 850 MW at minimum fuel cost. Each unit's fuel
cost is a smooth quadratic in its power output P plus a rectified-sine
"valve-point" ripple:

    C_i(P) = a_i + b_i·P + c_i·P²  +  |e_i · sin(f_i · (Pmin_i − P))|

The valve-point term models the sharp increase in heat rate each time a
new steam-admission valve opens. It turns each unit's otherwise convex
cost curve into a rippled, non-differentiable surface, and the sum of
three such curves is highly multimodal.

Variables (N_DIM = 3): P1, P2, P3, each mapped linearly from [0, 1] to
the unit's [Pmin, Pmax]. HumpDay's API has no explicit constraints, so the
demand-balance equality P1 + P2 + P3 = D is folded into the objective with
a large additive penalty (PENALTY = 1000 $/MW per MW of imbalance).

Pathology: the rectified-sine valve ripples create many local minima and
non-smooth kinks on top of an equality constraint — the classic test that
breaks gradient/trust-region methods and rewards global search.

Reference: the best-known cost for this canonical instance is
≈ 8234.07 $/h, near P ≈ (300.3, 400.0, 149.7) MW.
"""

from __future__ import annotations

import math

# Total system demand (MW) the three units must jointly supply.
DEMAND = 850.0

# Penalty weight ($/MW) on the demand-balance equality violation. Large
# enough that any meaningful imbalance dominates the fuel-cost savings.
PENALTY = 1000.0

# Per-unit data: (a, b, c, e, f, Pmin, Pmax).
#   a, b, c : quadratic fuel-cost coefficients
#   e, f    : valve-point ripple amplitude and frequency
#   Pmin/Pmax : output bounds (MW)
UNITS = (
    (561.0, 7.92, 0.001562, 300.0, 0.0315, 100.0, 600.0),  # U1
    (310.0, 7.85, 0.00194, 200.0, 0.042, 100.0, 400.0),  # U2
    (78.0, 7.97, 0.00482, 150.0, 0.063, 50.0, 200.0),  # U3
)


def _scale_unit_to_power(u):
    """Map a point in [0,1]^3 to (P1, P2, P3) in MW within each unit's bounds."""
    powers = []
    for i, (_a, _b, _c, _e, _f, pmin, pmax) in enumerate(UNITS):
        powers.append(pmin + (pmax - pmin) * u[i])
    return tuple(powers)


def _unit_cost(p, unit):
    """Fuel cost ($/h) of one unit at output p (MW), including valve ripple."""
    a, b, c, e, f, pmin, _pmax = unit
    return a + b * p + c * p * p + abs(e * math.sin(f * (pmin - p)))


def _fuel_cost(powers):
    """Total fuel cost ($/h) summed over all units (no penalty)."""
    return sum(_unit_cost(p, unit) for p, unit in zip(powers, UNITS))


def objective(u):
    """HumpDay-style objective: input `u ∈ [0,1]^3`, output total cost ($/h)
    with a penalty for any deviation from the demand-balance equality."""
    powers = _scale_unit_to_power(u)
    fuel = _fuel_cost(powers)
    mismatch = sum(powers) - DEMAND
    return fuel + PENALTY * abs(mismatch)


def decode(u):
    """Convenience: return the per-unit powers (MW), total fuel cost ($/h),
    and demand mismatch (MW) for a `[0,1]^3` point."""
    powers = _scale_unit_to_power(u)
    fuel = _fuel_cost(powers)
    mismatch = sum(powers) - DEMAND
    return {
        "P1": powers[0],
        "P2": powers[1],
        "P3": powers[2],
        "fuel_cost": fuel,
        "demand_mismatch": mismatch,
    }


N_DIM = 3
