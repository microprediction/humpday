# Gear Train Design — a discrete, staircase landscape

Choose four gears' tooth counts so the compound train hits a target ratio. The
classic Sandgren (1990) formulation:

```
minimise  ( 1/6.931 − (z1·z2)/(z3·z4) )²,   each z_i ∈ {12, …, 60}
```

```bash
python -m example_applications.gear_ratios.run
```

## Why it's here

A landscape **type** nothing else in the suite exercises: **discreteness /
plateaus**. The objective depends only on the *rounded* integer tooth counts, so
it is piecewise-constant — flat over every cell of the integer lattice, with **no
gradient to follow**. Methods that assume smoothness (Powell, trust-region) stall
on a plateau; only methods that hop around the lattice (Differential Evolution,
Particle Swarm, annealing) keep finding better near-ties. And there are *many*
near-ties — distinct tooth combinations giving nearly the same ratio — so the
true optimum is a needle.

Best-known optimum: `z = (19, 16, 43, 49)` → ratio `304/2107 ≈ 0.1442809`,
`f ≈ 2.7e-12` (the swaps `z1↔z2`, `z3↔z4` are equivalent). In practice optimisers
land on assorted near-ties around `1e-9`–`1e-6`; reaching the `1e-12` needle is
genuinely hard precisely because the lattice gives no descent direction.

## References

- Sandgren, E. (1990), *Nonlinear integer and discrete programming in mechanical
  design optimization*, J. Mech. Design — a standard mixed-integer DFO benchmark.
