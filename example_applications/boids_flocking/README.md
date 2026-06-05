# Boids Flocking — Emergent Navigation

Get a swarm of 40 boids through a chicane to a goal by tuning **five rule
weights** — separation, alignment, cohesion, goal-seeking, obstacle-avoidance.
A vectorised (numpy) Reynolds boids simulation runs the flock; HumpDay's
`[0,1]^5` cube sets the weights, and the objective returns the negative
**percentage of boids that reach the goal** (crashed ones don't count).

## What this stresses

- **Emergent behaviour from a scalar reward.** Nobody tells the swarm how to
  flock; the optimiser just balances "reach the goal" against "don't crash" and a
  coherent navigating flock falls out.
- **A broad, forgiving basin.** Many weightings work, so even Random Search does
  well — the structured methods win on *consistency* across start jitters, not
  feasibility. `NelderMead`, by contrast, can collapse to a non-flocking corner.
- **Mild noise.** Random start positions make each run a slightly different
  problem, so `run.py` reports a held-out cohort alongside the training seeds.

(Implementation note: this port updates all boids simultaneously each step where
the browser demo updates them sequentially; the emergent character is the same.)

## Running

```bash
python -m example_applications.boids_flocking.run
```

Expect most optimisers to get ~90–100% of the swarm to the goal, with
Nelder-Mead the outlier. Mirrors the browser demo
[`docs/applications/boids.html`](../../docs/applications/boids.html).
