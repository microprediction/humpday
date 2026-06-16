# Economic Dispatch with Valve-Point Loading

A canonical, famously multimodal power-systems benchmark. Schedule three
thermal generators so their combined output meets a fixed demand of
**D = 850 MW** at minimum fuel cost.

Each unit's fuel cost is a smooth quadratic plus a rectified-sine
**valve-point** ripple:

    C_i(P) = a_i + b_i·P + c_i·P²  +  |e_i · sin(f_i · (Pmin_i − P))|

The `|sin(...)|` term models the surge in heat rate each time a new
steam-admission valve opens. It studs each unit's cost curve with
non-differentiable kinks, and the sum of three rippled curves is riddled
with local minima.

The best-known cost for this classic instance is ≈ **8234.07 $/h**, near
`P ≈ (300.3, 400.0, 149.7) MW`. Good global optimisers should approach
that value with a near-zero demand mismatch.

## What this stresses

- **Multimodality.** The rectified-sine ripples create dozens of local
  minima. Gradient and trust-region methods latch onto the nearest dimple;
  population and annealing methods fare better.
- **Non-smoothness.** `|sin(...)|` is non-differentiable at every kink, so
  derivative-based heuristics get noisy or stuck.
- **Equality constraint.** HumpDay's API takes no explicit constraints, so
  the demand-balance equality `P1 + P2 + P3 = 850` is folded into the
  objective via an additive penalty (`PENALTY = 1000 $/MW`).

## Unit data

| Unit | a | b | c | e | f | Pmin | Pmax |
|---|---|---|---|---|---|---|---|
| U1 | 561 | 7.92 | 0.001562 | 300 | 0.0315 | 100 | 600 |
| U2 | 310 | 7.85 | 0.00194  | 200 | 0.042  | 100 | 400 |
| U3 | 78  | 7.97 | 0.00482  | 150 | 0.063  | 50  | 200 |

Variables (`N_DIM = 3`): `P1, P2, P3`, each mapped linearly from `[0, 1]`
to its `[Pmin, Pmax]`. See `problem.py` for the exact equations.

## Running

```bash
python -m example_applications.economic_dispatch_valve.run
```

Output is a small comparison table of optimiser → best fuel cost →
demand mismatch → dispatch. A few of HumpDay's optimisers should
consistently land near 8234 $/h with a tiny mismatch.
