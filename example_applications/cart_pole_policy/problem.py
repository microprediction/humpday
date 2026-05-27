"""
CartPole-v1-equivalent objective for direct policy search.

A pure-Python re-implementation of the OpenAI Gym CartPole dynamics
(see https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
so this example has no external runtime dependencies.

The optimisation objective for HumpDay takes a 5-D point in [0,1]^5,
stretches it into linear-policy parameters in [-5, 5]^5, runs
`N_EPISODES` rollouts of up to `MAX_STEPS` steps each with different
random seeds, and returns the **negative mean return** — so minimising
the HumpDay objective maximises survival.
"""

from __future__ import annotations

import math
import random

N_DIM = 5
N_EPISODES = 8  # rollouts per evaluation
MAX_STEPS = 500
WEIGHT_RANGE = 5.0  # [0,1]^5 → [-5, 5]^5

# CartPole physical constants (matching the OpenAI Gym defaults).
GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
LENGTH = 0.5  # half pole length
POLEMASS_LENGTH = MASS_POLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds per simulation step

# Termination thresholds (cart escapes the track, or pole falls past 12°).
X_THRESHOLD = 2.4
THETA_THRESHOLD = 12 * 2 * math.pi / 360


def _scale(u):
    """[0,1]^5 → [-WEIGHT_RANGE, WEIGHT_RANGE]^5."""
    return [WEIGHT_RANGE * (2.0 * ui - 1.0) for ui in u]


def _step(state, action):
    """Single CartPole simulation step (semi-implicit Euler)."""
    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS
    theta_acc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASS_POLE * costheta * costheta / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * costheta / TOTAL_MASS

    x = x + TAU * x_dot
    x_dot = x_dot + TAU * x_acc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * theta_acc
    return (x, x_dot, theta, theta_dot)


def _rollout(policy_params, seed):
    """Run one CartPole episode under a linear policy. Returns the
    number of steps survived (max MAX_STEPS)."""
    rng = random.Random(seed)
    # Initial state: small random perturbation in [-0.05, 0.05] per dim.
    state = tuple(rng.uniform(-0.05, 0.05) for _ in range(4))
    w1, w2, w3, w4, bias = policy_params

    for step in range(MAX_STEPS):
        x, x_dot, theta, theta_dot = state
        # Sign of a linear combination decides the action.
        score = w1 * x + w2 * x_dot + w3 * theta + w4 * theta_dot + bias
        action = 1 if score > 0 else 0

        state = _step(state, action)
        x, _, theta, _ = state
        if abs(x) > X_THRESHOLD or abs(theta) > THETA_THRESHOLD:
            return step + 1

    return MAX_STEPS


def objective(u, seed_offset=0):
    """HumpDay objective: negative mean return across N_EPISODES
    rollouts. Lower (more negative) = better policy.

    `seed_offset` lets the test harness re-evaluate the same policy
    on a different seed cohort; the optimiser shouldn't pass it."""
    params = _scale(u)
    returns = [_rollout(params, seed=seed_offset + i) for i in range(N_EPISODES)]
    mean_return = sum(returns) / len(returns)
    return -mean_return


def evaluate_policy(u, n_episodes=20, seed_offset=10_000):
    """Independent test-set evaluation. Returns (mean_return, median_return,
    min_return, max_return). Seeds disjoint from training cohort."""
    params = _scale(u)
    returns = [_rollout(params, seed=seed_offset + i) for i in range(n_episodes)]
    returns.sort()
    n = len(returns)
    return {
        "mean": sum(returns) / n,
        "median": returns[n // 2],
        "min": returns[0],
        "max": returns[-1],
        "returns": returns,
    }
