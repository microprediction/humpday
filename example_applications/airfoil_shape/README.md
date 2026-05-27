# Aerodynamic Shape Optimization (Low-Fidelity Surrogate)

Optimise the 6-D Hicks-Henne bump coefficients that perturb a NACA 0012
baseline airfoil, minimising a **low-fidelity drag surrogate** that
mimics the qualitative landscape of XFOIL output at a fixed lift
coefficient.

A genuine engineering deployment would replace the surrogate with a
real flow solver (XFOIL, SU2, OpenFOAM). The interface here is the same:
the optimiser sees `objective(u: list[float]) -> float`, and HumpDay
doesn't care what's inside.

## What this stresses

- **Severely restricted evaluation budget.** We cap `n_trials=50`.
  In a real CFD setting each call would take minutes; here we just
  budget the optimiser accordingly. This is the regime where Bayesian
  / surrogate-based methods are supposed to dominate evolutionary
  ones — there isn't enough budget for evolutionary methods to fill
  their initial population, let alone breed.

- **Implicit constraints.** The Hicks-Henne perturbations must not
  produce a self-intersecting airfoil (upper surface dropping below
  the lower). The surrogate penalises this implicitly: an intersecting
  airfoil produces "stall" — a large drag jump.

- **Multimodal but with a clear basin.** Real airfoil optimisation
  has multiple aerodynamic regimes (laminar / turbulent transition,
  shock formation). The surrogate models a single basin with a
  ridge of moderate-drag local optima, so algorithms that prefer
  exploration over exploitation reach the deep minimum.

## Hicks-Henne parameterisation

The airfoil surface is the NACA 0012 baseline plus a linear
combination of `N_BUMPS = 3` sine-like bumps on each of the upper and
lower surfaces (6 coefficients total).

A single Hicks-Henne bump at peak location `x_m ∈ (0, 1)`:

    b(x) = sin( π x^(log 0.5 / log x_m) )^t

with bump centres fixed at the quarter, half, three-quarter chord
positions and exponent `t = 4`.

The 6 HumpDay parameters are the bump amplitudes (decoded from
`[0,1]` to `[-0.02, 0.02]` — small, since airfoil perturbations are
~2% chord at most).

## The drag surrogate

A purely synthetic stand-in. It evaluates:

1. The thickness distribution along the chord.
2. The maximum-thickness location (the "max-thickness point").
3. A "self-intersection penalty" if upper < lower anywhere.
4. A curvature-roughness penalty (high-frequency wobble = high drag).

The closed-form value approximates the qualitative behaviour of XFOIL
at Re=1e6, α=2°: drag has a smooth minimum near a 5% camber profile
with the bulk of thickness in the front half of the chord.

## Running

```bash
python -m example_applications.airfoil_shape.run
```

Watch how the Bayesian optimiser (and to a lesser extent CMA-ES)
beats the population-based algorithms when there are only 50 evaluations
to spend.

## Real-world hookup

To use a real flow solver, replace `problem._drag_surrogate` with a
call to XFOIL (e.g. via `subprocess` and the `pyxfoil` wrapper) or
your CFD pipeline of choice. Make sure to wrap each call in a try/except
that returns a large penalty value on solver failure — both XFOIL
and panel methods can fail to converge for adversarial geometries the
optimiser explores.
