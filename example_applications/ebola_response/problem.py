"""
Ebola-response objective: time a public-health intervention to minimise harm.

A pure-Python SEIR epidemic model (Susceptible -> Exposed -> Infectious ->
Removed). The outbreak runs for 180 days, split into 8 equal control windows.
The HumpDay objective takes an 8-D point in [0,1]^8 — the control intensity in
each window — where higher control suppresses transmission (beta) but costs
economic effort. The score is the **percentage of harm avoided** versus doing
nothing, where harm = deaths + control cost; the objective returns its negative.

This is a multi-objective control-timing problem in disguise: spend too little
and the epidemic burns through the population; spend too much, too early, and
you pay for control the outbreak didn't need yet. The optimiser learns the
epidemiologist's playbook — do little at first, hit the growth phase hard, then
stand down.

Mirrors the browser demo docs/applications/ebola.html.
"""

from __future__ import annotations

N_DIM = 8
T = 180  # days
SIGMA = 1 / 9  # E -> I rate
GAMMA = 1 / 10  # I -> R rate
R0 = 2.2
BETA0 = R0 * GAMMA
CFR = 0.6  # case fatality ratio
K = 8  # control windows
WIN = T / K
DEATH_W = 1.0
ECON_W = 0.18


def _run(u):
    """Integrate the SEIR model under a windowed control policy."""
    S, E, I, R = 1 - 2e-4, 1e-4, 1e-4, 0.0
    econ = 0.0
    for day in range(T):
        k = min(K - 1, int(day / WIN))
        ctrl = max(0.0, min(1.0, u[k]))
        beta = BETA0 * (1 - 0.92 * ctrl)
        n_e = beta * S * I
        n_i = SIGMA * E
        n_r = GAMMA * I
        S -= n_e
        E += n_e - n_i
        I += n_i - n_r
        R += n_r
        econ += ctrl
    return {"death_frac": CFR * R, "econ_frac": econ / T}


def _harm(o):
    return DEATH_W * o["death_frac"] + ECON_W * o["econ_frac"]


_J_NOACT = _harm(_run([0.0] * K))


def policy_score(u):
    """Percentage of harm avoided versus doing nothing (higher = better)."""
    return 100.0 * (1 - _harm(_run([max(0.0, min(1.0, v)) for v in u[:K]])) / _J_NOACT)


def objective(u):
    """HumpDay objective: negative % harm avoided (minimise)."""
    return -policy_score(u)
