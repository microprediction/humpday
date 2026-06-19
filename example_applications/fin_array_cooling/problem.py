"""
Heat-sink fin-array design.

A heat sink has eight cooling fins in a row; we choose each fin's height. A taller fin has
more surface area but its tip runs less effectively (fin efficiency falls as
tanh(m h)/(m h)), and taller fins together choke the airflow shared across the array, so
every fin cools a little less. Fins nearer the front of the array see more airflow than
those behind. The optimum makes the front fins taller and the rear ones shorter rather
than sizing them all the same.

The HumpDay objective takes an 8-D point in [0,1]^8 (fin heights, mapped to 0..H_max) and
returns the negative total heat dissipated.
"""
from __future__ import annotations

import math

N_FINS = 8
N_DIM = N_FINS
H_MAX = 40.0
M = 0.05          # fin parameter (sets where efficiency rolls off)
DRAG = 0.004      # airflow choking per unit total fin height
AIRFLOW = (1.0, 0.92, 0.85, 0.78, 0.72, 0.66, 0.60, 0.55)


def decode(u):
    return [H_MAX * min(1.0, max(0.0, v)) for v in u]


def objective(u):
    h = decode(u)
    airflow_factor = 1.0 / (1.0 + DRAG * sum(h))
    q = 0.0
    for i in range(N_FINS):
        mh = M * h[i]
        eff = math.tanh(mh) / mh if mh > 1e-6 else 1.0
        q += AIRFLOW[i] * h[i] * eff
    return -(airflow_factor * q)
