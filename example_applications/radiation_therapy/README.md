# Radiation Therapy — beam-weight optimisation on the unit simplex

Choose the relative intensities (**weights**) of five radiation beams so
that a tumor receives a uniform prescription dose while nearby organs at
risk (OAR) stay below tolerance. The weights are a *composition* —
non-negative and summing to 1 — so the decision variable is a point on
the 4-simplex, not a box point.

## Why it's here

Like `cocktail_blend`, the decision variable is a composition, so we use
the **cube→simplex bijection** (`humpday.transforms.cubetosimplex`): the
objective HumpDay sees is an ordinary `[0,1]^4` function whose first act
lifts the cube point onto the five beam weights. Simplex problems become
box problems for free.

What makes this one different is **competing objectives**. A fixed
dose-influence matrix (built deterministically at import) means each beam
deposits dose into both tumor and OAR voxels. The beams that best cover
the tumor uniformly also spill dose into the OAR, so pushing tumor
coverage drives the sparing penalty up. The optimum is a genuine interior
trade-off, never a single-beam vertex.

## Formulation

- 5 beam weights `w` on the simplex (`sum w = 1`), via `cube_to_simplex`.
- 10 tumor voxels (target dose `Dp = 1.0`), 12 OAR voxels
  (tolerance `Dmax = 0.5`).
- Dose at voxel `v`: `TOTAL * sum_b w_b * D[b][v]`.
- Objective (minimise):

      sum_tumor (dose - Dp)^2  +  ALPHA * sum_OAR max(0, dose - Dmax)^2

  with `ALPHA = 2.0`.

## Running

```bash
python -m example_applications.radiation_therapy.run
```

Output is a small comparison table of optimiser → best score → tumor
dose statistics (mean/min/max) and OAR max dose. Tumor doses cluster near
`Dp = 1.0`; OAR dose is held lower where the trade-off allows.
