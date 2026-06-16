"""
k-Means Clustering as Continuous Global Optimisation — multimodal.

A fixed 2-D dataset of 75 points lives in the unit square, drawn as three
well-separated Gaussian-ish blobs (≈25 points each). The task is to place
k = 3 cluster centroids so as to minimise the within-cluster sum of
squares (WCSS): for every data point we take the SQUARED Euclidean
distance to its nearest centroid and sum those over the dataset.

The decision vector is the 3 centroids' (x, y) coordinates, so
N_DIM = 2·k = 6. HumpDay hands us `u ∈ [0,1]^6`; each pair maps directly
onto the unit square in which the data live.

Pathology: this is the classic reason k-means needs random restarts. The
objective is strongly MULTIMODAL — there are many local minima, e.g. two
centroids splitting a single blob while the third lone centroid is forced
to cover the remaining two blobs. It is also NON-SMOOTH: the nearest-
centroid assignment changes discontinuously across the perpendicular
bisectors between centroids, so the WCSS surface is piecewise-quadratic
with creases. The global optimum places exactly one centroid in each
blob, giving a low WCSS; optimisers trapped in a "2-1" split report a
markedly higher WCSS. That spread is the whole point of the demo.
"""

from __future__ import annotations

import math

K = 3  # number of centroids
N_DIM = 2 * K  # decision vector is (x, y) per centroid

# Three asymmetric ground-truth blobs. Two are close together on the left and
# one is a tight, dense blob on the right. The uneven geometry creates deep
# competing local minima: the "natural" assignment is one centroid per blob,
# but spending two centroids on the diffuse left pair (and only one on the
# right) is a strong attractor — the classic k-means restart trap. The result
# is a genuinely multimodal WCSS surface.
_BLOB_SPECS = (
    # (centre_x, centre_y, jitter, n_points)
    (0.28, 0.35, 0.10, 30),  # diffuse left-lower
    (0.34, 0.68, 0.10, 30),  # diffuse left-upper (overlaps the one above)
    (0.82, 0.50, 0.04, 15),  # tight, dense, well-separated right blob
)


def _lcg(seed):
    """Tiny deterministic linear congruential generator yielding floats in [0,1).

    Numerical Recipes constants (modulus 2^32). Returns a zero-arg sampler.
    """
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 4294967296.0

    return nxt


def _build_dataset():
    """Generate the fixed blob dataset once, deterministically, at import."""
    rng = _lcg(20240611)
    points = []
    for cx, cy, jitter, n_points in _BLOB_SPECS:
        for _ in range(n_points):
            # Box-Muller jitter from two uniforms -> approx Gaussian offset.
            u1 = rng()
            u2 = rng()
            r = math.sqrt(-2.0 * math.log(u1 + 1e-12))
            gx = r * math.cos(2.0 * math.pi * u2)
            gy = r * math.sin(2.0 * math.pi * u2)
            x = min(1.0, max(0.0, cx + jitter * gx))
            y = min(1.0, max(0.0, cy + jitter * gy))
            points.append((x, y))
    return points


# Fixed dataset, built deterministically at import time.
DATA = _build_dataset()


def _centroids_from_unit(u):
    """Map `u ∈ [0,1]^6` to a list of K (x, y) centroids in the unit square."""
    return [(u[2 * i], u[2 * i + 1]) for i in range(K)]


def _wcss(centroids):
    """Within-cluster sum of squared distances to the nearest centroid."""
    total = 0.0
    for px, py in DATA:
        best = float("inf")
        for cx, cy in centroids:
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d < best:
                best = d
        total += best
    return total


def objective(u):
    """HumpDay-style objective: input `u ∈ [0,1]^6`, output WCSS to MINIMISE."""
    return _wcss(_centroids_from_unit(u))


def decode(u):
    """Convenience: return the K centroids and the within-cluster SSE for a
    `[0,1]^6` point."""
    centroids = _centroids_from_unit(u)
    return {
        "centroids": centroids,
        "sse": _wcss(centroids),
    }
