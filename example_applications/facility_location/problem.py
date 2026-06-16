"""
Continuous p-median / Weber facility location.

Place `p = 3` facilities in the unit square to serve a fixed set of
~30 demand points. The service cost of a demand point is the Euclidean
distance to its *nearest* facility; the total cost is the sum over all
demand points. We minimise that total.

Variables: the 2D coordinates of the three facilities, so
`N_DIM = 2 * p = 6`. HumpDay's `u ∈ [0,1]^6` maps directly to three
`(x, y)` positions in the unit square.

The demand points are generated ONCE at import as three jittered blobs
around three fixed cluster centres, using a small deterministic LCG so
the landscape is reproducible without any external dependency.

Pathology: the per-point `min` over facilities makes the objective
NON-SMOOTH — it has kinks wherever the nearest-facility assignment
flips — and strongly MULTIMODAL. Which facility serves which cluster is
a combinatorial choice (3! permutations of facilities among the blobs),
and the assignment boundaries add further local minima. There is no
simple closed-form optimum; we report the best value found by the run.
"""

from __future__ import annotations

import math

# Number of facilities to place, and resulting search dimension.
P = 3
N_DIM = 2 * P

# Three blob centres in the unit square.
CLUSTER_CENTRES = ((0.2, 0.25), (0.75, 0.3), (0.5, 0.8))

# Demand points per cluster and jitter spread.
POINTS_PER_CLUSTER = 10
JITTER = 0.08

# LCG parameters (glibc-style) and fixed seed for reproducibility.
_LCG_A = 1103515245
_LCG_C = 12345
_LCG_M = 2**31
_LCG_SEED = 20240611


def _build_demand_points():
    """Generate the fixed demand points once, deterministically.

    Uses a self-contained linear congruential generator seeded at module
    import so the point cloud is identical on every run and platform.
    """
    state = _LCG_SEED

    def _next_unit():
        nonlocal state
        state = (_LCG_A * state + _LCG_C) % _LCG_M
        return state / _LCG_M

    points = []
    for cx, cy in CLUSTER_CENTRES:
        for _ in range(POINTS_PER_CLUSTER):
            # Centred jitter in [-JITTER, JITTER], clamped to the unit square.
            x = cx + (_next_unit() - 0.5) * 2.0 * JITTER
            y = cy + (_next_unit() - 0.5) * 2.0 * JITTER
            x = min(1.0, max(0.0, x))
            y = min(1.0, max(0.0, y))
            points.append((x, y))
    return points


# Generated ONCE at import — these are the fixed demand points.
DEMAND_POINTS = _build_demand_points()


def _facilities_from_unit(u):
    """Map a point in [0,1]^6 to three (x, y) facility positions."""
    return [(u[2 * i], u[2 * i + 1]) for i in range(P)]


def _total_service_cost(facilities):
    """Sum over demand points of the distance to the nearest facility."""
    total = 0.0
    for px, py in DEMAND_POINTS:
        best = math.inf
        for fx, fy in facilities:
            d = math.hypot(px - fx, py - fy)
            if d < best:
                best = d
        total += best
    return total


def objective(u):
    """HumpDay-style objective: input is `u ∈ [0,1]^6`, output is the
    total nearest-facility service cost to MINIMISE."""
    return _total_service_cost(_facilities_from_unit(u))


def decode(u):
    """Convenience: return the three facility positions and the total
    service cost for a `[0,1]^6` point."""
    facilities = _facilities_from_unit(u)
    return {
        "facilities": facilities,
        "total_cost": _total_service_cost(facilities),
    }
