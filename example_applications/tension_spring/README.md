# Tension/Compression Spring Design

A canonical constrained mechanical-engineering benchmark. Find the
lightest 3-D helical coil spring `(d, D, N)` that satisfies four physical
inequality constraints on deflection, shear stress, surge frequency, and
outer-diameter feasibility.

The world-class reported minimum weight is ≈ **0.012665** (Arora, 1989;
Belegundu, 1982; verified by many subsequent metaheuristic papers) at
`(d, D, N) ≈ (0.05169, 0.35672, 11.289)`. Feasible designs under ≈ 0.013
are excellent; large values indicate the optimiser is still wandering in
the penalised infeasible interior.

## What this stresses

- **Tiny objective, huge penalty.** The minimised weight is ≈ 0.0127,
  yet a typical constraint violation is O(1). Because HumpDay's API has
  no explicit constraints, infeasibility is folded in via a large
  additive penalty (`PENALTY_WEIGHT = 1e4`). The result is a landscape
  with a steep wall around a narrow feasible corridor.

- **Near-singular ridges.** The denominators in `g1`, `g2`, `g3` scale
  with powers of the wire diameter `d`, so as `d → 0` the constraint
  surfaces blow up. Optimisers that don't adapt their effective scale
  per coordinate burn evaluations near these ridges.

- **Narrow feasible region.** Four constraints carve a thin slice out of
  the bounding box, so most random samples are infeasible.

## Mechanical formulation

| Variable | Symbol | Range |
|---|---|---|
| Wire diameter      | `d` | 0.05 – 2.0 |
| Mean coil diameter | `D` | 0.25 – 1.3 |
| Active coils       | `N` | 2.0  – 15.0 |

Weight (objective):

    f(d, D, N) = (N + 2)·D·d²

Constraints (all `g_i(x) ≤ 0`):

    g1: 1 − (D³·N) / (71785·d⁴)                                  ≤ 0
    g2: (4D² − dD) / (12566·(D·d³ − d⁴)) + 1/(5108·d²) − 1       ≤ 0
    g3: 1 − 140.45·d / (D²·N)                                    ≤ 0
    g4: (D + d)/1.5 − 1                                          ≤ 0

See `problem.py` for the exact equations (denominators are guarded with a
tiny eps against divide-by-zero at the bounds of the unit cube).

## Running

```bash
python -m example_applications.tension_spring.run
```

Output is a small comparison table of optimiser → best weight → best
design. A handful of HumpDay's optimisers should consistently discover
feasible designs near 0.0127; the rest will reveal which families are
ill-suited to tightly constrained problems.
