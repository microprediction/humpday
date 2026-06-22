"""
FIR low-pass filter design.

Choose the eleven taps of a finite-impulse-response filter so its frequency response
approximates an ideal low-pass: gain near one below the passband edge and near zero above
the stopband edge. The error is summed over a frequency grid, comparing the realised
magnitude response to the target in the passband and stopband (the transition band is
free). It is a smooth least-squares filter-design problem.

The HumpDay objective takes an 11-D point in [0,1]^11 (taps, mapped to [-0.5, 0.5]) and
returns the summed passband and stopband response error.
"""

from __future__ import annotations

import math

N_TAPS = 11
N_DIM = N_TAPS
WC = 0.25 * math.pi  # passband edge
WS = 0.40 * math.pi  # stopband edge
NGRID = 40
GRID = [math.pi * i / NGRID for i in range(NGRID + 1)]


def decode(u):
    return [min(1.0, max(0.0, v)) - 0.5 for v in u]


def objective(u):
    h = decode(u)
    err = 0.0
    for w in GRID:
        hr = sum(h[k] * math.cos(w * k) for k in range(N_TAPS))
        hi = -sum(h[k] * math.sin(w * k) for k in range(N_TAPS))
        mag = math.hypot(hr, hi)
        if w <= WC:
            err += (mag - 1.0) ** 2
        elif w >= WS:
            err += mag**2
    return err
