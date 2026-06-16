"""
Michaelis–Menten enzyme-kinetics curve fit — a 2-D ill-conditioned
least-squares problem.

The reaction velocity of a single-substrate enzyme follows

    v(S) = Vmax · S / (Km + S)

where `Vmax` is the saturating velocity and `Km` is the substrate
concentration at half-maximal velocity. We measure `v` at a fixed grid
of substrate concentrations and fit `(Vmax, Km)` by minimising the sum
of squared residuals.

Variables (N_DIM=2), mapped from the unit square [0,1]^2:
  - `Vmax`  linear from [0,1] to [0.5, 5.0]
  - `Km`    LOG-scaled from [0,1] to [0.05, 5.0], i.e.
            Km = 0.05 · (5.0/0.05) ** u1 — because Km spans two decades
            and the curve responds to it multiplicatively.

Pathology: a mildly ILL-CONDITIONED fit. `Vmax` and `Km` are
correlated — at high `S` the curve saturates at `Vmax` regardless of
`Km`, and at low `S` only the ratio `Vmax/Km` (the initial slope) is
constrained. The data therefore pin down a curved, stretched diagonal
trough in parameter space rather than a round bowl, so optimisers that
assume isotropic curvature crawl along the valley floor.

True parameters: Vmax = 2.0, Km = 0.8. Synthetic data is generated once
at import from these values plus small Gaussian noise (sigma ≈ 0.03), so
the achievable minimum sits at the noise floor: SSE ≈ n·sigma² and
RMS ≈ 0.03 near the true `(Vmax, Km)`.
"""

from __future__ import annotations

import math

# Substrate concentrations at which velocity is measured.
S_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

# Ground-truth parameters used to synthesise the data.
TRUE_VMAX = 2.0
TRUE_KM = 0.8

# Measurement-noise standard deviation.
NOISE_SIGMA = 0.03

# Parameter bounds.
VMAX_LO, VMAX_HI = 0.5, 5.0
KM_LO, KM_HI = 0.05, 5.0


def _model(vmax, km, s):
    """Michaelis–Menten reaction velocity at substrate concentration `s`."""
    return vmax * s / (km + s)


class _LCG:
    """Tiny deterministic linear-congruential generator (numpy-free).

    Parameters are the Numerical Recipes constants; this only needs to
    be reproducible, not cryptographically sound.
    """

    def __init__(self, seed):
        self._state = seed & 0xFFFFFFFF

    def random(self):
        """Return a float in [0, 1)."""
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state / 0x100000000


def _gaussian_noise(n, sigma, seed):
    """`n` Gaussian samples ~ N(0, sigma) via Box–Muller, deterministic."""
    rng = _LCG(seed)
    out = []
    while len(out) < n:
        u1 = rng.random()
        u2 = rng.random()
        # Guard the log against u1 == 0.
        r = math.sqrt(-2.0 * math.log(u1 + 1e-12))
        out.append(sigma * r * math.cos(2.0 * math.pi * u2))
        if len(out) < n:
            out.append(sigma * r * math.sin(2.0 * math.pi * u2))
    return out[:n]


# Build the noisy observed velocities ONCE at import time.
_NOISE = _gaussian_noise(len(S_GRID), NOISE_SIGMA, seed=20260616)
V_OBS = [_model(TRUE_VMAX, TRUE_KM, s) + e for s, e in zip(S_GRID, _NOISE)]


def _decode_params(u):
    """Map a point in [0,1]^2 to physical parameters `(Vmax, Km)`."""
    vmax = VMAX_LO + (VMAX_HI - VMAX_LO) * u[0]
    km = KM_LO * (KM_HI / KM_LO) ** u[1]
    return vmax, km


def objective(u):
    """HumpDay-style objective: input `u ∈ [0,1]^2`, output is the sum of
    squared residuals between the Michaelis–Menten model and the data."""
    vmax, km = _decode_params(u)
    sse = 0.0
    for s, v in zip(S_GRID, V_OBS):
        r = _model(vmax, km, s) - v
        sse += r * r
    return sse


def decode(u):
    """Convenience: return the fitted `(Vmax, Km)` for a `[0,1]^2` point,
    plus the sum of squared residuals and the RMS residual."""
    vmax, km = _decode_params(u)
    sse = objective(u)
    rms = math.sqrt(sse / len(S_GRID))
    return {
        "Vmax": vmax,
        "Km": km,
        "sse": sse,
        "rms": rms,
    }


N_DIM = 2
