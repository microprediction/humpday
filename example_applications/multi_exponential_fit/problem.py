"""
Multi-Exponential Fit — the genuinely ill-conditioned valley.

Recover a sum of two decaying exponentials from noisy data:

    y(t) = A1 * exp(-k1 t) + A2 * exp(-k2 t)

This is the canonical "sloppy", ill-posed inverse problem of spectroscopy,
fluorescence lifetime imaging, and pharmacokinetics. The objective (sum of
squared residuals over the decay curve) has a long, **curving, degenerate
valley**: when the two rates k1, k2 are close, the two exponentials become
nearly indistinguishable and a whole ridge of (A, k) combinations fit the
data almost equally well. The Jacobian's condition number diverges as
k1 -> k2, so the normal equations go singular — the textbook ill-conditioned
landscape that gradient-ish / trust-region methods crawl through and that
the rest of the demo suite does not exercise.

What it stresses:
  - Ill-conditioning (a curving Rosenbrock-like valley), not multimodality.
  - A label-swap symmetry: (A1,k1,A2,k2) and (A2,k2,A1,k1) are the same fit,
    so there are two equivalent global optima.

Decision variables (mapped from [0,1]^4): two amplitudes and two rates. Rates
are mapped on a log scale (they span an order of magnitude). The global
minimum sits at the noise floor, at the true parameters (up to the swap).

Refs: Sethna (Cornell), "Fitting Exponentials"; Transtrum et al. on sloppy
models; Nature Sci. Rep. 2022 (s41598-022-08638-7).
"""

from __future__ import annotations

import math

N_DIM = 4

# True parameters used to generate the synthetic data. The two rates are
# deliberately within a factor of ~2 so the problem is meaningfully
# ill-conditioned (close rates -> near-degenerate valley) yet still solvable.
TRUE_A = (1.0, 0.6)
TRUE_K = (1.0, 2.2)

# Physical ranges for the search. Amplitudes are linear; rates are log-scaled.
A_LO, A_HI = 0.0, 2.0
K_LO, K_HI = 0.2, 6.0

# Sampling grid and a small, fixed (reproducible) noise realisation.
N_SAMPLES = 60
T_MAX = 5.0
NOISE_SIGMA = 0.01


def _model(t, a1, k1, a2, k2):
    return a1 * math.exp(-k1 * t) + a2 * math.exp(-k2 * t)


def _gauss_noise(n, sigma, seed=12345):
    """Deterministic Gaussian noise (Box–Muller on a tiny LCG) so the demo's
    optimum is reproducible without importing numpy/random."""
    out = []
    state = seed
    for _ in range((n + 1) // 2):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        u1 = (state + 1) / 0x80000000
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        u2 = (state + 1) / 0x80000000
        r = sigma * math.sqrt(-2.0 * math.log(u1))
        out.append(r * math.cos(2 * math.pi * u2))
        out.append(r * math.sin(2 * math.pi * u2))
    return out[:n]


_TIMES = [T_MAX * i / (N_SAMPLES - 1) for i in range(N_SAMPLES)]
_NOISE = _gauss_noise(N_SAMPLES, NOISE_SIGMA)
_DATA = [
    _model(t, TRUE_A[0], TRUE_K[0], TRUE_A[1], TRUE_K[1]) + e
    for t, e in zip(_TIMES, _NOISE)
]


def _decode_params(u):
    """Map a [0,1]^4 cube point to (A1, k1, A2, k2)."""
    a1 = A_LO + (A_HI - A_LO) * u[0]
    a2 = A_LO + (A_HI - A_LO) * u[1]
    # log-scaled rates
    k1 = K_LO * (K_HI / K_LO) ** u[2]
    k2 = K_LO * (K_HI / K_LO) ** u[3]
    return a1, k1, a2, k2


def objective(u):
    """HumpDay-style objective: `u ∈ [0,1]^4` -> sum of squared residuals
    between the two-exponential model and the noisy data."""
    a1, k1, a2, k2 = _decode_params(u)
    sse = 0.0
    for t, y in zip(_TIMES, _DATA):
        r = _model(t, a1, k1, a2, k2) - y
        sse += r * r
    return sse


def decode(u):
    """Convenience: fitted parameters + RMS residual for a `[0,1]^4` point."""
    a1, k1, a2, k2 = _decode_params(u)
    sse = objective(u)
    return {
        "A1": a1,
        "k1": k1,
        "A2": a2,
        "k2": k2,
        "rms_residual": math.sqrt(sse / N_SAMPLES),
        "sse": sse,
    }
