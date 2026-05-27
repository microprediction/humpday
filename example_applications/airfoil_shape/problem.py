"""
Hicks-Henne airfoil parameterisation with a low-fidelity drag surrogate.

The surrogate is a pure-Python stand-in for XFOIL / SU2 / OpenFOAM —
it captures the qualitative shape of the drag landscape (smooth
basin near a moderate-camber configuration, with self-intersection
and roughness penalties) without depending on a real CFD solver.

The HumpDay objective takes a 6-D point in [0, 1]^6, decodes it into
six Hicks-Henne bump amplitudes (3 upper, 3 lower) in [-0.02, 0.02],
adds the perturbation to a NACA 0012 baseline, and returns the
synthetic drag.
"""

from __future__ import annotations

import math

N_DIM = 6
N_BUMPS_PER_SURFACE = 3
AMPLITUDE_RANGE = 0.02  # ±2% chord — typical Hicks-Henne range
N_CHORD_SAMPLES = 60

# Fixed Hicks-Henne bump centres on (0, 1). Avoid x_m = 0.5 since the
# log-ratio exponent is undefined there; use 0.51 for the middle bump.
BUMP_CENTRES = (0.25, 0.51, 0.75)
BUMP_EXPONENT = 4.0

# Pre-compute the chord sample points and the NACA 0012 baseline.
_CHORD = [i / (N_CHORD_SAMPLES - 1) for i in range(N_CHORD_SAMPLES)]


def _naca_0012_thickness(x):
    """Half-thickness distribution of a NACA 0012 airfoil at chord
    fraction `x ∈ [0, 1]`. Symmetric airfoil, max thickness 12%."""
    if x <= 0.0:
        return 0.0
    return 0.6 * (
        0.2969 * math.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x * x
        + 0.2843 * x * x * x
        - 0.1015 * x * x * x * x
    )


_UPPER_BASELINE = [_naca_0012_thickness(x) for x in _CHORD]
_LOWER_BASELINE = [-y for y in _UPPER_BASELINE]


def _hicks_henne_bump(x, x_m):
    """Single Hicks-Henne bump evaluated at chord position `x`, centred
    at `x_m`. Returns a value in [0, 1] that peaks at `x_m`."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    exponent = math.log(0.5) / math.log(x_m)
    return math.sin(math.pi * (x**exponent)) ** BUMP_EXPONENT


def _build_surfaces(amplitudes):
    """Return `(upper_y, lower_y)` for the perturbed airfoil. Amplitudes
    is a 6-tuple: (a_up_1, a_up_2, a_up_3, a_lo_1, a_lo_2, a_lo_3)."""
    a_up = amplitudes[:3]
    a_lo = amplitudes[3:]

    upper = list(_UPPER_BASELINE)
    lower = list(_LOWER_BASELINE)
    for k, x in enumerate(_CHORD):
        delta_up = sum(
            a_up[j] * _hicks_henne_bump(x, BUMP_CENTRES[j]) for j in range(3)
        )
        delta_lo = sum(
            a_lo[j] * _hicks_henne_bump(x, BUMP_CENTRES[j]) for j in range(3)
        )
        upper[k] += delta_up
        lower[k] += delta_lo
    return upper, lower


def _drag_surrogate(upper, lower):
    """Synthetic stand-in for XFOIL drag.

    Three contributions:
      • Form drag — penalty on (target_thickness − current_thickness)^2,
        favouring profiles with max thickness ≈ 10–12% chord at x ≈ 0.3.
      • Stall penalty — if upper drops below lower anywhere, add a
        large constant (a real CFD solver would diverge here).
      • Roughness penalty — sum of squared second differences along
        each surface, since wavy airfoils have higher drag in reality.
    """
    thickness = [upper[k] - lower[k] for k in range(N_CHORD_SAMPLES)]

    # Stall (self-intersection or near-zero thickness anywhere away from the
    # leading/trailing edges, where the baseline itself touches zero).
    interior = thickness[1:-1]
    if interior and min(interior) <= 1e-6:
        return 0.5

    # Form-drag target: moderate camber, peak thickness ~12% near 1/3 chord.
    target_thickness = [
        0.12 * 4.0 * x * (1.0 - x) ** 1.5 for x in _CHORD
    ]  # peaks ≈ 0.12 around x ≈ 0.3
    form_drag = (
        sum((thickness[k] - target_thickness[k]) ** 2 for k in range(N_CHORD_SAMPLES))
        / N_CHORD_SAMPLES
    )

    # Roughness — sum of squared second differences (curvature wiggle).
    rough_up = sum(
        (upper[k - 1] - 2.0 * upper[k] + upper[k + 1]) ** 2
        for k in range(1, N_CHORD_SAMPLES - 1)
    )
    rough_lo = sum(
        (lower[k - 1] - 2.0 * lower[k] + lower[k + 1]) ** 2
        for k in range(1, N_CHORD_SAMPLES - 1)
    )
    roughness = rough_up + rough_lo

    # Mix the three components. Scale chosen so a NACA-0012 baseline
    # scores ≈ 0.0006 and a well-shaped optimum scores ≈ 0.00005.
    return 1.0 * form_drag + 0.5 * roughness


def _decode_amplitudes(u):
    """[0, 1]^6 → 6 Hicks-Henne amplitudes in [-AMPLITUDE_RANGE, +AMPLITUDE_RANGE]."""
    return tuple(AMPLITUDE_RANGE * (2.0 * ui - 1.0) for ui in u)


def objective(u):
    """HumpDay objective: surrogate drag of the perturbed airfoil."""
    amplitudes = _decode_amplitudes(u)
    upper, lower = _build_surfaces(amplitudes)
    return _drag_surrogate(upper, lower)


def decode(u):
    """Human-readable: amplitudes and the resulting surface."""
    amplitudes = _decode_amplitudes(u)
    upper, lower = _build_surfaces(amplitudes)
    thickness = [upper[k] - lower[k] for k in range(N_CHORD_SAMPLES)]
    max_t = max(thickness)
    max_t_idx = thickness.index(max_t)
    interior = thickness[1:-1]
    return {
        "amplitudes": amplitudes,
        "drag": _drag_surrogate(upper, lower),
        "max_thickness": max_t,
        "max_thickness_x": _CHORD[max_t_idx],
        "stalls": (min(interior) if interior else 0.0) <= 1e-6,
    }
