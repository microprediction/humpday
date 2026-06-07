"""
Tuned-mass-damper objective: cut a tower's earthquake sway with one damper.

A 6-storey shear building is shaken by a synthetic earthquake. A tuned mass
damper — a heavy block on a spring attached to one floor — can soak up the sway
if its mass, tuning and damping are right and it sits on a good floor. The
HumpDay objective takes a 4-D point in [0,1]^4 and decodes it into the damper's
mass (% of building mass), frequency tuning, damping ratio, and the **integer
floor** it attaches to, then runs the full earthquake via Newmark-beta time
integration and returns the negative **percentage reduction in peak roof sway**.

This is a MIXED-INTEGER, noisy-looking dynamics problem: three continuous knobs
plus one discrete choice (which floor), evaluated by a 1600-step structural
simulation. The ground motion is narrowband near the building's first mode, so
the bare tower resonates and the tuning genuinely matters. Uses numpy for the
modal eigenvalues and the per-step linear solves.

Mirrors the browser demo docs/applications/tuned-mass-damper.html.
"""

from __future__ import annotations

import numpy as np

N_DIM = 4
N = 6  # storeys
FLOOR_M = 1.0
STORY_K = 700.0
STRUCT_ZETA = 0.015  # 1.5% Rayleigh structural damping
DT = 0.02
L = 1600
BETA, GAMMA = 0.25, 0.5


def _structural_k():
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] += STORY_K
        if i + 1 < N:
            K[i, i] += STORY_K
            K[i, i + 1] -= STORY_K
            K[i + 1, i] -= STORY_K
    return K


MSTRUCT = np.eye(N) * FLOOR_M
KSTRUCT = _structural_k()
_w = np.sort(np.sqrt(np.maximum(np.linalg.eigvalsh(KSTRUCT / FLOOR_M), 0)))
OMEGA1, OMEGA2 = _w[0], _w[1]
_A1 = 2 * STRUCT_ZETA / (OMEGA1 + OMEGA2)
_A0 = 2 * STRUCT_ZETA * OMEGA1 * OMEGA2 / (OMEGA1 + OMEGA2)
CSTRUCT = _A0 * MSTRUCT + _A1 * KSTRUCT


def _ground_motion():
    """Seeded Kanai-Tajimi white noise with a Jennings envelope (deterministic)."""
    seed = 20260604

    def rng():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return seed / 0x7FFFFFFF * 2 - 1

    wg, zg = OMEGA1, 0.40
    xf = vf = 0.0
    ag = np.empty(L)
    t_rise, t_decay = 2.5, 14.0
    for i in range(L):
        t = i * DT
        white = rng()
        af = white - 2 * zg * wg * vf - wg * wg * xf
        vf += af * DT
        xf += vf * DT
        if t < t_rise:
            env = (t / t_rise) ** 2
        elif t < t_decay:
            env = 1.0
        else:
            env = np.exp(-0.18 * (t - t_decay))
        ag[i] = env * (2 * zg * wg * vf + wg * wg * xf)
    ag *= 0.6 / np.max(np.abs(ag))  # normalise PGA
    return ag


AG = _ground_motion()


def _newmark(M, C, K, roof_idx):
    """Average-acceleration Newmark-beta; returns peak |displacement| at roof_idx."""
    a0 = 1 / (BETA * DT * DT)
    a1 = GAMMA / (BETA * DT)
    a2 = 1 / (BETA * DT)
    a3 = 1 / (2 * BETA) - 1
    a4 = GAMMA / BETA - 1
    a5 = DT * (GAMMA / (2 * BETA) - 1)
    keff = K + a1 * C + a0 * M
    keff_inv = np.linalg.inv(keff)
    mr = M.sum(axis=1)  # M . 1 influence load
    n = M.shape[0]
    u = np.zeros(n)
    v = np.zeros(n)
    acc = np.full(n, -AG[0])
    peak = 0.0
    for s in range(1, L):
        t1 = a0 * u + a2 * v + a3 * acc
        t2 = a1 * u + a4 * v + a5 * acc
        peff = -mr * AG[s] + M @ t1 + C @ t2
        u_new = keff_inv @ peff
        acc_new = a0 * (u_new - u) - a2 * v - a3 * acc
        v = v + DT * ((1 - GAMMA) * acc + GAMMA * acc_new)
        u, acc = u_new, acc_new
        peak = max(peak, abs(u[roof_idx]))
    return peak


_D0 = _newmark(MSTRUCT, CSTRUCT, KSTRUCT, N - 1)  # bare-building peak roof sway


def _decode(u):
    return {
        "mu": 0.003 + 0.037 * u[0],
        "tuning": 0.4 + 1.2 * u[1],
        "zeta": 0.40 * u[2],
        "floor": 1 + round(u[3] * (N - 1)),
    }


def run_damper(u):
    """Return % reduction in peak roof sway for a damper design in [0,1]^4."""
    d = _decode(u)
    n = N + 1
    mt = d["mu"] * (N * FLOOR_M)
    wt = d["tuning"] * OMEGA1
    kt = mt * wt * wt
    ct = 2 * d["zeta"] * mt * wt
    p = d["floor"] - 1
    M = np.zeros((n, n))
    K = np.zeros((n, n))
    C = np.zeros((n, n))
    M[:N, :N] = MSTRUCT
    K[:N, :N] = KSTRUCT
    C[:N, :N] = CSTRUCT
    M[N, N] = mt
    K[p, p] += kt
    K[p, N] -= kt
    K[N, p] -= kt
    K[N, N] += kt
    C[p, p] += ct
    C[p, N] -= ct
    C[N, p] -= ct
    C[N, N] += ct
    peak = _newmark(M, C, K, N - 1)
    return (_D0 - peak) / _D0 * 100.0, d


def objective(u):
    """HumpDay objective: negative % reduction in peak roof sway (minimise)."""
    return -run_damper(u)[0]
