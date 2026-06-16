# Facility Location (continuous p-median)

Place `p = 3` facilities in the unit square to serve ~30 fixed demand
points. Each demand point is served by its **nearest** facility, and the
total cost is the sum of those nearest-facility distances. We minimise
the total. The search variable is the three facilities' `(x, y)`
coordinates, so `N_DIM = 6`.

The demand points are three jittered blobs around three fixed cluster
centres, generated once at import by a self-contained deterministic LCG
(no external dependencies, identical on every platform).

## What this stresses

- **Non-smoothness.** The per-point `min` over facilities introduces
  kinks: the objective is piecewise-smooth, with a derivative
  discontinuity everywhere the nearest-facility assignment flips. This
  punishes optimisers that implicitly assume a smooth landscape.

- **Multimodality.** Which facility serves which cluster is a
  combinatorial choice — there are `3! = 6` ways to assign three
  facilities to three blobs, each a distinct local basin, plus the
  assignment boundaries between them. Pure local search tends to get
  stuck on a sub-optimal permutation; population methods that sample the
  whole box (DifferentialEvolution, ParticleSwarm, CMA-ES) usually do
  better here.

There is no simple closed-form optimum, so the run reports the best
value it finds. A good solution parks one facility near each cluster
centre.

## Running

```bash
python -m example_applications.facility_location.run
```

Output is a small comparison table of optimiser → best cost → the three
facility positions found.
