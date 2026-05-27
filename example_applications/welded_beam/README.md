# Welded Beam Design

A canonical constrained structural engineering benchmark. Find the
cheapest 4-D weld geometry `(h, l, t, b)` that survives seven physical
inequality constraints under a 6000-lb load.

The world-class reported minimum cost is ≈ **1.725** (Ragsdell &
Phillips, 1976; verified by many subsequent metaheuristic papers).
Anything under 2.0 is a very good design; values above 5 typically
indicate the optimiser is still wandering in the infeasible interior.

## What this stresses

- **Disparate dimension scales.** Weld thickness `h ∈ [0.125, 2.0]`,
  weld length `l ∈ [0.1, 10.0]`, beam thickness/width `t, b ∈ [0.1, 2.0]`.
  When HumpDay maps these to `[0, 1]^4`, every internal step the
  optimiser takes is a 100×-mismatched move in physical space. Optimisers
  that don't adapt their effective scale per coordinate quickly waste
  evaluations.

- **Narrow feasible region.** Seven constraints (shear, bending,
  deflection, buckling, dimension-feasibility) carve a tiny corridor
  out of the bounding box. Most random samples are infeasible, so
  unguided search burns its budget on penalised designs.

- **Penalty cliffs.** Because HumpDay's API doesn't accept explicit
  constraints, infeasibility is folded into the objective via a large
  additive penalty. The resulting landscape is piecewise smooth with
  steep walls — exactly the kind of geometry that breaks gradient-based
  heuristics but rewards trust-region and population methods.

## Mechanical formulation

| Variable | Symbol | Range (in) |
|---|---|---|
| Weld thickness  | `h` | 0.125 – 2.0 |
| Weld length     | `l` | 0.1   – 10.0 |
| Beam thickness  | `t` | 0.1   – 2.0 |
| Beam width      | `b` | 0.1   – 2.0 |

Constants: load `P = 6000 lb`, length `L = 14 in`, `E = 30e6 psi`,
`G = 12e6 psi`, allowable shear `τ_max = 13600 psi`, allowable bending
`σ_max = 30000 psi`, allowable deflection `δ_max = 0.25 in`.

Cost (fabrication + labour + material):

    f(h, l, t, b) = 1.10471·h²·l + 0.04811·t·b·(14 + l)

Constraints (all `g_i(x) ≤ 0`):

    g1: τ(h,l,t,b)   − 13600  ≤ 0       (shear stress)
    g2: σ(t,b)       − 30000  ≤ 0       (bending stress)
    g3: h − b                 ≤ 0       (weld vs beam thickness feasibility)
    g4: 0.10471·h² + 0.04811·t·b·(14 + l) − 5  ≤ 0   (combined cost bound)
    g5: 0.125 − h             ≤ 0       (minimum weld thickness)
    g6: δ(t,b)       − 0.25   ≤ 0       (deflection)
    g7: P − Pc(t,b)           ≤ 0       (buckling)

where the state variables (shear, bending, deflection, buckling load)
follow the classical Ragsdell–Phillips formulation. See `problem.py`
for the exact equations.

## Running

```bash
python -m example_applications.welded_beam.run
```

Output is a small comparison table of optimiser → best cost → best
geometry. A handful of HumpDay's 22 optimisers should consistently
discover designs below 2.0; the rest will reveal which families are
ill-suited to constrained problems.
