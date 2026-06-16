"""
Sensor-Network Localisation from noisy ranges — robotics / signal processing.

We have 3 ANCHOR nodes at known, fixed positions in the unit square and
4 UNKNOWN nodes whose 2-D positions we must recover. For every
unknown–anchor pair and every unknown–unknown pair we are given a noisy
measured distance (the true layout is generated once at import; small
Gaussian noise, std ≈ 0.02, is added to each true pairwise distance).

The objective maps the unit hypercube [0,1]^8 to four candidate node
positions in the unit square and returns the sum of squared range
residuals, `Σ (estimated_distance − measured_distance)²`, over all
measured pairs. The global minimum recovers the true layout and bottoms
out near the noise floor.

Variables (N_DIM = 2 × 4 = 8): the (x, y) of each unknown node, each
component mapped from [0,1] to [0,1] (the unit square).

Pathology — strongly MULTIMODAL with flip / fold ambiguities. Distance
constraints are invariant to reflections of a node across the line(s)
joining its constraining neighbours, so a partially-pinned node can fold
to a mirror position with (almost) the same residual. With only three
anchors the network is not rigidly pinned, producing many spurious local
minima; local and trust-region methods routinely settle into a flipped /
folded configuration with a residual well above the noise floor, while
good global searches recover the true layout.
"""

from __future__ import annotations

import math

# Three anchors at known fixed positions in the unit square.
ANCHORS = ((0.1, 0.1), (0.9, 0.15), (0.5, 0.9))

N_UNKNOWN = 4
N_DIM = 2 * N_UNKNOWN

NOISE_STD = 0.02


def _lcg(seed):
    """Numerical-Recipes LCG yielding floats in [0,1)."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 0x100000000


def _gaussian(rng, std):
    """One N(0, std) draw via Box–Muller from a uniform generator."""
    u1 = next(rng)
    u2 = next(rng)
    # Guard the log against u1 == 0.
    u1 = max(u1, 1e-12)
    return std * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build():
    """Generate the true unknown-node layout and the noisy measured
    distances ONCE, deterministically, at import time.

    Returns (true_unknowns, measurements) where `measurements` is a list
    of (kind, i, j, measured_distance):
      - ("ua", u_index, anchor_index, d): unknown–anchor pair
      - ("uu", u_index, u_index, d):      unknown–unknown pair
    """
    rng = _lcg(20240611)

    # True positions of the unknown nodes, drawn in the interior so they
    # sit comfortably inside the unit square.
    true_unknowns = []
    for _ in range(N_UNKNOWN):
        x = 0.15 + 0.7 * next(rng)
        y = 0.15 + 0.7 * next(rng)
        true_unknowns.append((x, y))

    measurements = []
    # Every unknown–anchor pair.
    for ui, p in enumerate(true_unknowns):
        for ai, a in enumerate(ANCHORS):
            d = _dist(p, a) + _gaussian(rng, NOISE_STD)
            measurements.append(("ua", ui, ai, d))
    # Every unknown–unknown pair.
    for i in range(N_UNKNOWN):
        for j in range(i + 1, N_UNKNOWN):
            d = _dist(true_unknowns[i], true_unknowns[j]) + _gaussian(rng, NOISE_STD)
            measurements.append(("uu", i, j, d))

    return true_unknowns, measurements


TRUE_UNKNOWNS, MEASUREMENTS = _build()


def _positions(u):
    """Map u ∈ [0,1]^8 to a list of four (x, y) candidate positions."""
    return [(u[2 * k], u[2 * k + 1]) for k in range(N_UNKNOWN)]


def _residual(positions):
    """Sum of squared range residuals over all measured pairs."""
    total = 0.0
    for kind, i, j, d_meas in MEASUREMENTS:
        if kind == "ua":
            d_est = _dist(positions[i], ANCHORS[j])
        else:  # "uu"
            d_est = _dist(positions[i], positions[j])
        diff = d_est - d_meas
        total += diff * diff
    return total


def objective(u):
    """HumpDay-style objective: input `u ∈ [0,1]^8`, output the total
    squared range residual against the noisy measured distances."""
    return _residual(_positions(u))


def true_u():
    """The unit-cube point corresponding to the true layout (the global
    optimum, up to the noise floor)."""
    u = []
    for x, y in TRUE_UNKNOWNS:
        u.append(x)
        u.append(y)
    return u


def decode(u):
    """Return the four estimated node positions and the total residual."""
    positions = _positions(u)
    return {
        "positions": positions,
        "residual": _residual(positions),
    }
