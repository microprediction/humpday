# Pressure Vessel Design

A classic constrained, mixed-integer engineering benchmark: minimise the
fabrication cost of a cylindrical pressure vessel (a cylinder closed by
two hemispherical heads) that must hold a fixed volume.

Four variables: shell thickness `Ts` and head thickness `Th` (both
quantised to integer multiples of 1/16 in), inner radius `R`, and
cylindrical length `L`. Four inequality constraints (pressure-based
thickness lower bounds, a minimum-volume requirement, and a maximum
length). Constraints are folded into the objective as a quadratic penalty.

## Why it's here (pathology)

The two thickness variables are rounded to the nearest 1/16 in, so the
cost landscape is piecewise-flat with discontinuous steps: gradients are
zero almost everywhere and undefined at the steps, which breaks smooth
local methods and trust-region surrogates. The minimum-volume constraint
carves a thin curved feasible region, and the optimum sits hard against
that boundary, so penalty-based search must thread a narrow corridor
without slipping infeasible.

## Run

    python -m example_applications.pressure_vessel.run

## Known optimum

Cost ≈ **6059.714** at `(Ts, Th, R, L) ≈ (0.8125, 0.4375, 42.0984, 176.6366)`.

## Reference

Kannan, B. K. & Kramer, S. N. (1994), "An augmented Lagrange multiplier
based method for mixed integer discrete continuous optimization and its
applications to mechanical design," *Journal of Mechanical Design*
116(2), 405–411. Widely reused as a metaheuristic benchmark (e.g.
Coello Coello 2000).
