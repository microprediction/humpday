"""
Plinko-funnel objective: steer a noisy cascade onto an off-centre target.

A pure-Python Galton board. Balls drop through 14 rows of pegs, bouncing left or
right; on a flat board they pile up in the middle, so almost none reach an
off-centre bin. Each region of the board has a tunable LEAN that biases the
left/right coin flip. The HumpDay objective takes a 7-D point in [0,1]^7,
stretches it into a 7-point lean profile in [-1,1]^7, drops 400 random balls,
and returns the **negative percentage** landing in the target bin. Minimising it
funnels the most balls onto the target.

The objective is INHERENTLY noisy — every ball is a fresh coin-flip cascade — so
this is a stochastic-optimisation example: the optimiser is steering a random
process, not a deterministic one. With no lean only ~3% reach the off-centre
bin; a good funnel gets more than half of them there.

Mirrors the browser demo docs/applications/plinko.html.
"""

from __future__ import annotations

import random

N_DIM = 7
N_BINS = 15
N_ROWS = 14
TARGET = 11
N_CTRL = 7
N_BALLS = 400


def _scale(u):
    return [2.0 * ui - 1.0 for ui in u]


def _lean_at(col, ctrl):
    t = col / (N_BINS - 1) * (N_CTRL - 1)
    i = int(t)
    f = t - i
    if i >= N_CTRL - 1:
        return ctrl[N_CTRL - 1]
    return ctrl[i] * (1 - f) + ctrl[i + 1] * f


def _drop_ball(ctrl, rng):
    col = (N_BINS - 1) / 2.0
    for _ in range(N_ROWS):
        lean = max(-1.0, min(1.0, _lean_at(col, ctrl)))
        p = 0.5 + 0.45 * lean
        col += 0.5 if rng.random() < p else -0.5
        col = max(0.0, min(N_BINS - 1, col))
    return int(round(col))


def percent_in_target(ctrl, seed):
    rng = random.Random(seed)
    hit = sum(1 for _ in range(N_BALLS) if _drop_ball(ctrl, rng) == TARGET)
    return 100.0 * hit / N_BALLS


def objective(u, seed_offset=0):
    """HumpDay objective: negative % of balls in the target bin (minimise)."""
    ctrl = _scale(u)
    return -percent_in_target(ctrl, seed_offset + 1)


def evaluate_profile(u, n_boards=12, seed_offset=700_000):
    """Held-out evaluation: average % in target over fresh ball cascades."""
    ctrl = _scale(u)
    vals = [percent_in_target(ctrl, seed_offset + i * 101 + 1) for i in range(n_boards)]
    vals.sort()
    n = len(vals)
    return {"mean": sum(vals) / n, "median": vals[n // 2], "min": vals[0], "max": vals[-1]}
