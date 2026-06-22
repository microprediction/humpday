"""
Solar-farm row-tilt layout with inter-row shading.

Eight rows of panels run across a field at fixed spacing. We choose each row's tilt
angle. A row captures most energy near a sweet-spot tilt, but a steeper row casts a
longer shadow on the row behind it, costing that row output. So front rows face a
trade-off (tilt down to avoid shading the next row, at a small capture loss) while the
back row, shading no one, wants the capture-optimal tilt.

The HumpDay objective takes an 8-D point in [0,1]^8 (row tilts, mapped to 0..1.2 rad) and
returns the negative of total captured energy minus shading losses.
"""

from __future__ import annotations

import math

N_ROWS = 8
N_DIM = N_ROWS
TILT_OPT = 0.6
SPACING = 0.45
SHADE = 1.5


def decode(u):
    return [1.2 * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    tilt = decode(u)
    energy = 0.0
    for i in range(N_ROWS):
        energy += math.cos(tilt[i] - TILT_OPT)  # capture, peaks at TILT_OPT
        if i < N_ROWS - 1:  # shades the row behind it
            shadow = math.sin(tilt[i])
            energy -= SHADE * max(0.0, shadow - SPACING)
    return -energy
