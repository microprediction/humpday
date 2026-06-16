# Kalman Filter Noise Tuning

A 2-D state-estimation benchmark. A target drifts along a line under a
near-constant-velocity model; we observe only its noisy position. A
2-state Kalman filter (position, velocity) reconstructs the trajectory,
but its accuracy depends entirely on two covariance hyper-parameters:
the process-noise scale `q` and the measurement-noise variance `r`. The
task is to tune `(q, r)` to minimise the position tracking RMSE against
the (hidden) true trajectory.

The fixed true trajectory and noisy measurements are generated **once at
import** with a deterministic LCG + Box–Muller, so the objective is fully
reproducible.

## What this stresses

- **Ill-conditioning.** Filter performance is governed almost entirely by
  the **ratio** `q / r`, which sets the steady-state Kalman gain. The good
  region is a long, thin **diagonal valley** in `(log q, log r)` space, not
  a round basin: move along the valley and almost nothing changes, step
  across it and the RMSE spikes. Optimisers that don't adapt their search
  geometry to the elongated, rotated basin waste evaluations.

- **Log-scaled magnitudes.** `q ∈ [1e-5, 1e-1]` and `r ∈ [1e-2, 1e1]` span
  several orders of magnitude, so both unit-cube axes are mapped
  logarithmically. On a linear scale the useful valley collapses against
  the origin and most random samples land in the flat, useless interior.

## Variables

| Variable | Symbol | Range | Mapping |
|---|---|---|---|
| Process-noise scale     | `q` | 1e-5 – 1e-1 | log from `u0` |
| Measurement variance    | `r` | 1e-2 – 1e1  | log from `u1` |

## Filter

State `x = [position, velocity]`, `dt = 1`, over `N = 80` steps:

    F = [[1, dt], [0, 1]]                 transition (constant velocity)
    H = [1, 0]                            position-only observation
    Q = q · [[dt⁴/4, dt³/2],              continuous white-noise accel.
             [dt³/2, dt²  ]]
    R = r                                 scalar measurement variance

The standard predict/update recursion runs over the measurements; the
objective is the RMSE between the filtered position estimate and the true
position. See `problem.py` for the exact recursion.

## Running

```bash
python -m example_applications.kalman_tuning.run
```

Output is a small comparison table of optimiser → best RMSE → tuned
`(q, r)`. The raw measurement-noise std is ≈ 2.0; a well-tuned filter
should drive the position RMSE down to roughly 1.0–1.5.
