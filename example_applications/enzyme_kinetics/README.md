# Enzyme Kinetics Fit

A 2-D nonlinear least-squares benchmark. Fit the Michaelis–Menten
reaction-velocity model

    v(S) = Vmax · S / (Km + S)

to a fixed grid of substrate concentrations by recovering `(Vmax, Km)`.

The synthetic data is generated once at import from the true parameters
`Vmax = 2.0`, `Km = 0.8` plus small Gaussian noise (`sigma ≈ 0.03`), so
the best attainable fit sits at the noise floor: good optimisers recover
`Vmax ≈ 2.0`, `Km ≈ 0.8` with `RMS ≈ 0.03`.

## What this stresses

- **Ill-conditioning / correlated parameters.** At high `S` the curve
  saturates at `Vmax` regardless of `Km`; at low `S` only the initial
  slope `Vmax/Km` is constrained. The least-squares surface is therefore
  a stretched, curved diagonal trough rather than a round bowl.
  Optimisers that assume isotropic curvature inch along the valley.

- **Log-scaled parameter.** `Km` spans two decades, so it is mapped
  logarithmically from `[0, 1]`; the conditioning of the search depends
  on that reparameterisation.

## Variables

| Variable | Range | Mapping from `u ∈ [0,1]` |
|---|---|---|
| `Vmax` | 0.5 – 5.0 | linear |
| `Km`   | 0.05 – 5.0 | log: `Km = 0.05·(5.0/0.05)**u` |

Substrate grid: `S = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]`.

Objective: sum of squared residuals between model and data over the grid.

## Running

```bash
python -m example_applications.enzyme_kinetics.run
```

Output is a small comparison table of optimiser → best SSE → RMS → fitted
`(Vmax, Km)`. See `problem.py` for the exact equations.
