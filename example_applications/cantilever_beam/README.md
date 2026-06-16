# Cantilever Beam Design

A canonical single-constraint structural benchmark. A hollow square
cantilever beam is divided into five segments, each with a height
variable `x1..x5`. Minimise the beam's weight subject to a single tip
deflection constraint.

The reported global minimum weight is ≈ **1.33996** at
`x ≈ (6.016, 5.309, 4.494, 3.502, 2.153)`.

## What this stresses

- **An active nonlinear constraint.** The feasible region is bounded by
  one smooth-but-curved surface
  `g = 61/x1³ + 37/x2³ + 19/x3³ + 7/x4³ + 1/x5³ − 1 ≤ 0`,
  and the optimum lies exactly on it. Driving any height down lowers
  weight but pushes deflection up, so an optimiser has to *ride the
  boundary* rather than retreat into the interior.

- **Penalty cliffs.** HumpDay's API takes no explicit constraints, so
  the violation is folded into the cost via a large quadratic penalty.
  The landscape is a smooth weight bowl with a steep wall along the
  constraint — easy to fall off, hard to hug.

## Formulation

| Variable | Range |
|---|---|
| Section heights `x1..x5` | 0.01 – 100 |

Weight (proportional, to minimise):

    f(x) = 0.0624 · (x1 + x2 + x3 + x4 + x5)

Constraint (`g ≤ 0` feasible):

    g(x) = 61/x1³ + 37/x2³ + 19/x3³ + 7/x4³ + 1/x5³ − 1

HumpDay objective: `f + PENALTY_WEIGHT · max(0, g)²` with
`PENALTY_WEIGHT = 1e4`. See `problem.py` for the exact equations.

## Running

```bash
python -m example_applications.cantilever_beam.run
```

Output is a small comparison table of optimiser → best weight → design.
Constraint-aware methods should consistently reach feasible designs near
the reference weight of ≈ 1.34.
