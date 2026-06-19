"""
Cellular base-station power control.

Six base stations along a corridor serve users spread between them. We set each
station's transmit power. A user is served by the nearest station; its
signal-to-interference-plus-noise ratio (SINR) is the served power over noise plus the
sum of the other stations' powers, all attenuated by distance squared. Raising a
station's power helps its own users but interferes with everyone else's, so coverage is
a non-separable tug of war, and blanket maximum power is not optimal. Power also costs
energy.

The HumpDay objective takes a 6-D point in [0,1]^6 (transmit powers, mapped to
0..P_max) and returns the negative of (covered users minus an energy penalty).
"""
from __future__ import annotations

N_TOWERS = 6
N_DIM = N_TOWERS
TOWERS = tuple(2.0 + 16.0 * i / (N_TOWERS - 1) for i in range(N_TOWERS))  # positions
USERS = tuple(0.5 * j for j in range(40))                                 # positions 0..19.5
P_MAX = 10.0
NOISE = 0.05
SINR_THRESHOLD = 1.0
ENERGY_PENALTY = 0.15


def decode(u):
    return [P_MAX * min(1.0, max(0.0, v)) for v in u]


def _gain(a, b):
    d2 = (a - b) ** 2
    return 1.0 / (1.0 + d2)


def objective(u):
    power = decode(u)
    covered = 0.0
    for x in USERS:
        serv = min(range(N_TOWERS), key=lambda t: abs(TOWERS[t] - x))
        signal = power[serv] * _gain(x, TOWERS[serv])
        interference = sum(power[t] * _gain(x, TOWERS[t]) for t in range(N_TOWERS) if t != serv)
        sinr = signal / (NOISE + interference)
        covered += 1.0 if sinr >= SINR_THRESHOLD else sinr / SINR_THRESHOLD
    return -(covered - ENERGY_PENALTY * sum(power))
