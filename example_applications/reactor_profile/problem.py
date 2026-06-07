"""
Reactor temperature-profile objective: maximise the intermediate-product yield.

A pure-Python plug-flow reactor with two first-order reactions in series,
A -> B -> C, each with Arrhenius temperature dependence. B is the valuable
product; push the temperature too high and it runs straight through to worthless
C. The HumpDay objective takes a 10-D point in [0,1]^10 — the temperature in each
of ten zones along the reactor (300-480 K) — integrates the kinetics, and returns
the negative **final yield of B**.

Because the second reaction is far more temperature-sensitive than the first, the
optimum is a *profile*, not a single temperature: run hot early to convert A
quickly, then cool to stop B decomposing. The best isothermal temperature is
beaten by a well-shaped schedule — the calculus-of-variations flavour of optimal
reactor control.

Mirrors the browser demo docs/applications/reactor-tprofile.html.
"""

from __future__ import annotations

import math

N_ZONES = 10
N_DIM = N_ZONES
TAU_PER_ZONE = 0.1  # total residence time tau = 1
T_REF = 400.0
B1 = 6.0
B2 = 22.0  # B->C is much more temperature-sensitive than A->B
T_MIN, T_MAX = 300.0, 480.0
N_PROFILE_SAMPLES = 80


def _arr(T, B):
    return math.exp(B * (1 - T_REF / T))


def _advance(ca, cb, T, dtau):
    k1, k2 = _arr(T, B1), _arr(T, B2)
    e1, e2 = math.exp(-k1 * dtau), math.exp(-k2 * dtau)
    ca_n = ca * e1
    if abs(k2 - k1) < 1e-9:
        cb_n = (ca * k1 * dtau + cb) * e1
    else:
        cb_n = (k1 * ca / (k2 - k1)) * (e1 - e2) + cb * e2
    return ca_n, cb_n


def decode(u):
    return [T_MIN + u[i] * (T_MAX - T_MIN) for i in range(N_ZONES)]


def simulate(t_profile):
    """Return (CA_final, CB_final, CC_final) for a 10-zone temperature profile."""
    ca, cb = 1.0, 0.0
    sub_per_zone = N_PROFILE_SAMPLES // N_ZONES
    dsub = TAU_PER_ZONE / sub_per_zone
    for zi in range(N_ZONES):
        for _ in range(sub_per_zone):
            ca, cb = _advance(ca, cb, t_profile[zi], dsub)
    return ca, cb, 1 - ca - cb


def yield_b(u):
    """Final yield of intermediate product B (0..1; higher = better)."""
    return simulate(decode(u))[1]


def objective(u):
    """HumpDay objective: negative final yield of B (minimise)."""
    return -simulate(decode(u))[1]
