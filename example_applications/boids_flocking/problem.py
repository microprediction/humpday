"""
Boids-flocking objective: get a swarm through a chicane to a goal.

A vectorised (numpy) Reynolds boids simulation. Forty boids start on the left and
must reach a goal on the right, threading a two-obstacle chicane. Each boid obeys
five local rules — separation, alignment, cohesion, goal-seeking, and
obstacle-avoidance — and the HumpDay objective takes a 5-D point in [0,1]^5 (the
five rule weights), simulates the flock, and returns the **negative percentage of
boids that reach the goal** (crashed boids don't count). Different start jitters
make it a mildly noisy objective.

The lesson is emergent behaviour from a scalar reward: nobody tells the swarm how
to flock; the optimiser just balances "get to the goal" against "don't crash"
and a coherent navigating flock falls out. (Note: this port updates all boids
simultaneously each step, where the browser demo updates them sequentially — the
emergent character is the same.)

Mirrors the browser demo docs/applications/boids.html.
"""

from __future__ import annotations

import numpy as np

N_DIM = 5
WX, WY = 100.0, 60.0
T = 320
NB = 40
PERC, SEP_R = 12.0, 5.0
MAXV = 1.0
GOAL = np.array([90.0, 30.0])
GOAL_R = 7.0
OBST = [(40.0, 22.0, 7.0), (58.0, 40.0, 7.0)]  # offset chicane (x, y, r)


def _decode(u):
    return {
        "sep": 3 * u[0],
        "align": 2.2 * u[1],
        "coh": 2.2 * u[2],
        "goal": 2.5 * u[3],
        "obs": 7 * u[4],
    }


def simulate(u, seed):
    """Return the fraction of boids (0..1) that reach the goal."""
    w = _decode(u)
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    pos = np.column_stack([6 + rng.rand(NB) * 8, 18 + rng.rand(NB) * 24])
    vel = (rng.rand(NB, 2) - 0.5) * 0.2
    done = np.zeros(NB, dtype=int)  # 0 active, 1 reached, 2 crashed
    reached = 0

    for _ in range(T):
        active = done == 0
        if not active.any():
            break
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        d2 = dx * dx + dy * dy
        act = active[None, :]
        neigh = (d2 < PERC * PERC) & act
        np.fill_diagonal(neigh, False)
        n = neigh.sum(axis=1)
        has = n > 0
        nn = np.where(has, n, 1)

        # alignment + cohesion (mean neighbour velocity / position)
        avx = (neigh @ vel[:, 0]) / nn
        avy = (neigh @ vel[:, 1]) / nn
        cvx = (neigh @ pos[:, 0]) / nn
        cvy = (neigh @ pos[:, 1]) / nn
        # separation (close neighbours only)
        close = (d2 < SEP_R * SEP_R) & neigh
        d = np.sqrt(d2) + 1e-6
        sx = (close * (dx / d)).sum(axis=1)
        sy = (close * (dy / d)).sum(axis=1)

        fx = w["sep"] * sx + np.where(
            has,
            w["align"] * (avx - vel[:, 0]) + w["coh"] * (cvx - pos[:, 0]) * 0.05,
            0.0,
        )
        fy = w["sep"] * sy + np.where(
            has,
            w["align"] * (avy - vel[:, 1]) + w["coh"] * (cvy - pos[:, 1]) * 0.05,
            0.0,
        )

        # goal seeking
        gdx, gdy = GOAL[0] - pos[:, 0], GOAL[1] - pos[:, 1]
        gd = np.hypot(gdx, gdy) + 1e-6
        fx += w["goal"] * gdx / gd
        fy += w["goal"] * gdy / gd

        # obstacle avoidance
        for ox, oy, r in OBST:
            odx, ody = pos[:, 0] - ox, pos[:, 1] - oy
            od = np.hypot(odx, ody) + 1e-6
            m = r + 6
            push = np.where(od < m, w["obs"] * (m - od) / m, 0.0)
            fx += odx / od * push
            fy += ody / od * push

        vel[:, 0] += fx * 0.1
        vel[:, 1] += fy * 0.1
        sp = np.hypot(vel[:, 0], vel[:, 1])
        scale = np.where(sp > MAXV, MAXV / np.where(sp > 0, sp, 1), 1.0)
        vel *= scale[:, None]
        pos += vel

        # bounds bounce
        oob = (pos[:, 0] < 0) | (pos[:, 0] > WX) | (pos[:, 1] < 0) | (pos[:, 1] > WY)
        pos[:, 0] = np.clip(pos[:, 0], 0, WX)
        pos[:, 1] = np.clip(pos[:, 1], 0, WY)
        vel[oob] *= -0.4

        # collisions + goal (only for active boids)
        for ox, oy, r in OBST:
            hit = active & (np.hypot(pos[:, 0] - ox, pos[:, 1] - oy) < r)
            done[hit] = 2
        active = done == 0
        at_goal = active & (np.hypot(pos[:, 0] - GOAL[0], pos[:, 1] - GOAL[1]) < GOAL_R)
        reached += int(at_goal.sum())
        done[at_goal] = 1

    return reached / NB


def objective(u, seed_offset=0):
    """HumpDay objective: negative mean % reaching the goal over 2 starts."""
    seeds = (seed_offset + 1, seed_offset + 2)
    return -100.0 * sum(simulate(u, s) for s in seeds) / len(seeds)


def evaluate_weights(u, n_seeds=8, seed_offset=500_000):
    """Held-out mean % reaching the goal over fresh start jitters."""
    vals = [100.0 * simulate(u, seed_offset + i * 17 + 1) for i in range(n_seeds)]
    vals.sort()
    return {"mean": sum(vals) / len(vals), "min": vals[0], "max": vals[-1]}
