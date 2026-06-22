"""
Traffic-signal timing on a coordinated arterial.

Four signalised intersections along a main road share a common cycle. For each we
choose the green split (fraction of the cycle given to the main street versus the side
street) and an offset (when its cycle starts relative to the others). Webster's formula
gives the average delay per approach as a function of the split and the degree of
saturation; coordinating the offsets to the travel time between intersections lets a
platoon ride a green wave, cutting main-street delay. Splitting too far toward the main
street starves the side streets, and vice versa.

The HumpDay objective takes an 8-D point in [0,1]^8 (split and offset per intersection)
and returns total vehicle-delay (scaled).
"""

from __future__ import annotations

N_INT = 4
N_DIM = N_INT * 2
CYCLE = 90.0
MAIN = (1000.0, 1100.0, 950.0, 1050.0)  # main-approach flow (veh/h)
SIDE = (400.0, 350.0, 500.0, 300.0)  # side-approach flow
SAT = 1800.0  # saturation flow (veh/h of green)
TRAVEL = (30.0, 35.0, 28.0)  # seconds between consecutive intersections


def decode(u):
    g = [0.3 + 0.4 * min(1.0, max(0.0, u[2 * i])) for i in range(N_INT)]
    off = [CYCLE * min(1.0, max(0.0, u[2 * i + 1])) for i in range(N_INT)]
    return g, off


def _webster(flow, green_frac):
    cap = SAT * green_frac
    x = min(0.95, flow / cap) if cap > 0 else 0.95
    red = 1.0 - green_frac
    return CYCLE * red * red / (2.0 * (1.0 - x) + 1e-6)


def objective(u):
    g, off = decode(u)
    delay = 0.0
    for i in range(N_INT):
        delay += MAIN[i] * _webster(MAIN[i], g[i])
        delay += SIDE[i] * _webster(SIDE[i], 1.0 - g[i])
    # progression: offset mismatch versus travel time adds main-street delay
    for i in range(N_INT - 1):
        ideal = TRAVEL[i] % CYCLE
        mism = abs(((off[i + 1] - off[i]) % CYCLE) - ideal)
        mism = min(mism, CYCLE - mism)
        delay += MAIN[i + 1] * 0.5 * mism
    return delay / 1000.0
