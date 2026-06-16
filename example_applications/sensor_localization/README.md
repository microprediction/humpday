# Sensor-Network Localisation

Recover the 2-D positions of 4 unknown sensor nodes from noisy pairwise
range measurements, given 3 anchor nodes at known fixed positions in the
unit square. A canonical hard global-optimisation problem from robotics
and signal processing.

The true layout and a set of noisy measured distances (every
unknown–anchor and unknown–unknown pair, Gaussian noise with std ≈ 0.02)
are generated **once** at import via a fixed-seed LCG + Box–Muller, so
the objective is deterministic across runs.

## What this stresses

- **Flip / fold multimodality.** Distance constraints are invariant to
  reflecting a node across the line joining the neighbours that constrain
  it. A partially-pinned node can fold to a mirror position with almost
  the same residual, so the landscape is littered with deep spurious
  local minima.

- **Under-determined geometry.** Three anchors do not rigidly pin a
  4-node network, so several distinct layouts fit the measurements nearly
  as well as the truth. Local and trust-region methods routinely settle
  into a flipped configuration; the spread between those and the global
  recovery is the point of the demo.

## Formulation

- Variables: `N_DIM = 2 × 4 = 8`, the `(x, y)` of each unknown node,
  each component mapped from `[0,1]` to the unit square.
- Objective (minimise): `Σ (estimated_distance − measured_distance)²`
  over all measured pairs, with anchor positions held fixed.
- Global optimum: the true layout, residual ≈ the noise floor.

## Running

```bash
python -m example_applications.sensor_localization.run
```

Output is a small table of optimiser → best residual → estimated node
positions. Good global searches approach the noise floor; trapped
optimisers report a visibly larger residual.
