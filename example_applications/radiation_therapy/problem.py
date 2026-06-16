"""
Radiation Therapy — beam-weight optimisation on the unit simplex.

A medical-physics allocation problem. Five radiation beams irradiate a
patient from five fixed angles. We choose their relative intensities —
the beam *weights* — which form a *composition* (non-negative, summing
to 1): a point on the 4-simplex, not a box point. That makes this the
canonical case for the cube->simplex bijection, exactly like
`cocktail_blend`: HumpDay sees an ordinary `[0,1]^4` objective whose
first act is to lift the cube point onto the simplex via
`humpday.transforms.cubetosimplex`. With four free cube coordinates the
bijection returns the five beam weights.

Geometry (fixed, built deterministically at import): 10 TUMOR voxels we
want to bring to the prescription dose Dp = 1.0, and 12 ORGAN-AT-RISK
(OAR) voxels we want to keep below the tolerance Dmax = 0.5. A
dose-influence matrix D[beam][voxel] says how much dose each beam
deposits in each voxel; each beam favours a different cluster of voxels.
The dose at a voxel is the weighted sum over beams, scaled by TOTAL so
doses land at O(1).

Objective (minimise): tumor uniformity + OAR sparing =
    sum_{tumor}  (dose - Dp)**2  +  ALPHA * sum_{OAR} max(0, dose - Dmax)**2

What it stresses:
  - Unit-simplex geometry (intensities summing to 1) via the bijection —
    an allocation constraint the box-bounded demos never exercise.
  - COMPETING objectives. The beams that best cover the tumor uniformly
    also spill dose into the OAR, so cranking tumor coverage drives the
    OAR penalty up. The optimum is a genuine interior trade-off — a
    balance between tumor coverage and OAR sparing, never a vertex.
"""

from __future__ import annotations

import math

from humpday.transforms.cubetosimplex import cube_to_simplex

# Prescription / planning constants.
N_BEAMS = 5
N_TUMOR = 10
N_OAR = 12
DP = 1.0  # prescription dose at every tumor voxel
DMAX = 0.5  # dose tolerance at every OAR voxel
ALPHA = 2.0  # relative weight of OAR sparing vs tumor uniformity
TOTAL = 1.5  # overall fluence scale, chosen so the optimum tumor dose sits near Dp

# The simplex of beam weights lives one dimension above the cube we
# optimise over: four cube coordinates -> five beam weights.
N_DIM = N_BEAMS - 1  # = 4


def _lcg(seed):
    """Deterministic uniform(0,1) generator (numerical-recipes LCG)."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 0x100000000


def _build_influence():
    """Dose-influence matrix D[beam][voxel] over tumor then OAR voxels.

    Built deterministically at import. Each beam favours a different
    cluster of voxels, so no single beam can cover the whole tumor
    without also dosing some OAR voxels — the source of the trade-off.
    """
    rng = _lcg(seed=20240611)
    n_vox = N_TUMOR + N_OAR
    matrix = []
    for b in range(N_BEAMS):
        # Each beam aims at a different band of voxels; influence decays
        # with distance from that band, plus a small positive floor.
        focus = (b + 0.5) * n_vox / N_BEAMS
        spread = n_vox / 3.0
        row = []
        for v in range(n_vox):
            falloff = math.exp(-((v - focus) ** 2) / (2.0 * spread * spread))
            jitter = 0.5 + next(rng)  # in [0.5, 1.5), keeps it positive
            row.append(0.05 + falloff * jitter)
        matrix.append(row)
    return matrix


# D[beam][voxel]: fixed at import.
D = _build_influence()


def _doses(weights):
    """Dose at each voxel = TOTAL * sum_b weight_b * D[b][voxel]."""
    n_vox = N_TUMOR + N_OAR
    return [
        TOTAL * sum(weights[b] * D[b][v] for b in range(N_BEAMS)) for v in range(n_vox)
    ]


def objective(u):
    """HumpDay-style objective: `u in [0,1]^4` -> plan score (minimise).

    Lifts the cube point onto the 5-component simplex of beam weights,
    then scores tumor uniformity plus OAR sparing."""
    weights = cube_to_simplex(u)  # 5 non-negative weights summing to 1
    dose = _doses(weights)
    tumor_term = sum((dose[v] - DP) ** 2 for v in range(N_TUMOR))
    oar_term = sum(
        max(0.0, dose[v] - DMAX) ** 2 for v in range(N_TUMOR, N_TUMOR + N_OAR)
    )
    return tumor_term + ALPHA * oar_term


def decode(u):
    """Convenience: beam weights and dose statistics for a `[0,1]^4` point."""
    weights = cube_to_simplex(u)
    dose = _doses(weights)
    tumor_dose = dose[:N_TUMOR]
    oar_dose = dose[N_TUMOR:]
    return {
        "beam_weights": list(weights),
        "tumor_mean": sum(tumor_dose) / N_TUMOR,
        "tumor_min": min(tumor_dose),
        "tumor_max": max(tumor_dose),
        "oar_max": max(oar_dose),
        "score": objective(u),
    }
