"""
Antenna-array objective: place elements to maximise forward gain.

A pure-Python reduced-order array-factor model. Seven isotropic radiators sit on
a line; the HumpDay objective takes a 7-D point in [0,1]^7, stretches it into the
element positions (in wavelengths, over a span of 4 lambda), and returns the
**negative forward directivity in dBi**. Minimising it maximises the gain of the
forward beam.

The lesson is optimiser-beats-intuition: a human spaces the elements evenly,
which scores about 8 dBi; the optimiser finds an irregular spacing that
concentrates more energy forward and beats the uniform array by a couple of
decibels. The landscape is wiggly (interference lobes), so it rewards a real
search.

Mirrors the browser demo docs/applications/antenna.html.
"""

from __future__ import annotations

import math

N_DIM = 7
SPAN = 4.0  # element positions live in [0, SPAN] wavelengths


def _scale(u):
    return [SPAN * v for v in u]


def _gain(theta, xs):
    """Array-factor power gain in direction theta (broadside-referenced)."""
    re = im = 0.0
    c = math.cos(theta) - 1
    for x in xs:
        ph = 2 * math.pi * x * c
        re += math.cos(ph)
        im += math.sin(ph)
    return re * re + im * im


def _avg_gain(xs, m=240):
    a = 0.0
    for i in range(m):
        th = (i + 0.5) / m * math.pi
        a += _gain(th, xs) * math.sin(th)
    return a * (math.pi / m) / 2


def directivity_dbi(xs):
    d = _gain(0.0, xs) / max(_avg_gain(xs), 1e-9)
    return 10 * math.log10(d)


def _uniform_xs():
    return [SPAN * i / (N_DIM - 1) for i in range(N_DIM)]


UNIFORM_DBI = directivity_dbi(_uniform_xs())


def objective(u):
    """HumpDay objective: negative forward gain in dBi (minimise)."""
    return -directivity_dbi(_scale(u))
