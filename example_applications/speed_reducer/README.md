# Speed Reducer (Golinski gearbox) — constrained, mixed-integer, *verifiable*

Minimise the weight of a single-stage gearbox over **seven** design variables
subject to **eleven** nonlinear inequality constraints from real mechanical
physics — gear-tooth bending and surface stress, shaft deflections and stresses,
plus geometric limits. One variable (the number of pinion teeth) is an integer.

```bash
python -m example_applications.speed_reducer.run
```

## Why it's here

It's one of the most-cited constrained engineering benchmarks, and unlike most
demos it has a **known global optimum**, so HumpDay can *verify* a result, not
just rank methods:

```
x* = (b, m, teeth, l1, l2, d1, d2) = (3.5, 0.7, 17, 7.3, 7.7153, 3.3503, 5.2867)
weight ≈ 2994.471
```

How it differs from `welded_beam`:
- **Eleven** simultaneously-relevant constraints (vs seven), several *active* at
  the optimum — a narrow feasible corner that sits right on the constraint
  boundary.
- A **mixed-integer** variable (pinion teeth), like `tuned_mass_damper`.
- A **nonconvex** (signomial / generalized-geometric-program) objective, so
  metaheuristics cannot *guarantee* the optimum — the known value is the check.

In practice Powell and the local/trust-region methods get closest (≈2994–3005);
population and Bayesian methods spread higher (3005–3300, occasionally landing on
18 teeth instead of 17). Because the optimum lies *on* active constraints, any
near-optimal design carries a sub-0.1% violation under a finite penalty — treated
as feasible for reporting (`FEASIBLE_TOL`), which is engineering round-off, not a
real violation.

## References

- Golinski, J. (1970), original speed-reducer formulation.
- Standard CEC / engineering constrained-optimisation suites; e.g. arXiv 2505.03512
  reproduces weight 2994.471921 at the optimum above.
