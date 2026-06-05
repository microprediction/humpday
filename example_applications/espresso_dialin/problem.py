"""
Espresso dial-in objective: find the god shot in as few pulls as possible.

A pure-Python response surface for an espresso shot. The HumpDay objective takes
a 4-D point in [0,1]^4 and stretches it into the four dials — grind, dose,
temperature, time. A good shot needs the **extraction yield** near 20% AND the
**brew ratio** near 1:2; those two conditions are coupled, so the sweet spot is
small. Each pull is **noisy** (real shots vary), and the objective returns the
negative shot score.

The point of this example is **sample efficiency**: with a realistic budget of
only a dozen-or-so pulls, interpolation / trust-region methods (PRIMA_BOBYQA)
and Bayesian optimisation find the sweet spot while population methods, which
need many generations to adapt, are still wandering. This is the
expensive-evaluation regime derivative-free optimisation is really for.

Mirrors the browser demo docs/applications/espresso.html.
"""

from __future__ import annotations

import math
import random

N_DIM = 4
NOISE_SD = 5.0


def _scale(u):
    """[0,1]^4 -> (grind, dose g, temp C, time s)."""
    return {
        "grind": u[0],
        "dose": 14 + 8 * u[1],
        "temp": 88 + 8 * u[2],
        "time": 20 + 18 * u[3],
    }


def eval_shot(u):
    """Noise-free shot evaluation. Returns (extraction_yield, ratio, score)."""
    p = _scale(u)
    g_n = p["grind"]
    d_n = (p["dose"] - 14) / 8
    t_temp = (p["temp"] - 88) / 8
    t_n = (p["time"] - 20) / 18
    ey = 10 + 8.5 * g_n + 3.5 * t_temp + 5.5 * t_n - 4.5 * d_n + 2.5 * g_n * t_n
    vol = (0.9 + 1.7 * t_n) * (1 - 0.35 * g_n)
    ratio = vol / (0.65 + 0.7 * d_n)
    ey_bonus = math.exp(-(((ey - 20) / 3.2) ** 2))
    r_bonus = math.exp(-(((ratio - 1.75) / 0.42) ** 2))
    return ey, ratio, 100.0 * ey_bonus * r_bonus


def pull(u, seed=None):
    """One noisy pull of the shot (what the optimiser actually tastes)."""
    rng = random.Random(seed) if seed is not None else random
    # sum of 4 uniforms ~ approx normal, matching the browser demo
    noise = (rng.random() + rng.random() + rng.random() + rng.random() - 2) * 1.4142
    return max(0.0, min(100.0, eval_shot(u)[2] + noise * NOISE_SD))


_counter = [0]


def objective(u, seed_offset=0):
    """HumpDay objective: negative noisy shot score (minimise)."""
    _counter[0] += 1
    return -pull(u, seed=seed_offset + _counter[0] if seed_offset else None)


def evaluate_recipe(u):
    """Noise-free 'true' quality of the chosen recipe (for the test column)."""
    ey, ratio, score = eval_shot(u)
    return {"score": score, "extraction": ey, "ratio": ratio}
