# Objective functions

Two lists. Import both from `humpday.objectives`.

```python
from humpday.objectives import SURFACES, physics_objectives, morphed_surfaces
```

| | what it is | dimension |
|---|---|---|
| `SURFACES` | classic formulaic surfaces — sphere, Rastrigin, Ackley, Griewank and friends, each mapped onto the unit cube | any; you choose |
| `physics_objectives()` | the engineering demos under `example_applications` — brachistochrone, truss, cantilever, cart-pole, antenna array | fixed; the problem owns it |
| `morphed_surfaces()` | `SURFACES` with a fresh random shift, rotation, scale, noise and modal frequency each call | any; you choose |

Race optimizers on `morphed_surfaces()`, not on `SURFACES` directly. A fixed landscape can be
learned, and an optimizer tuned to one is not evidence about anything.

## What `SURFACES` leaves out, and why

Both exclusions were measured, not guessed, and `tests/test_objective_registry.py` re-measures them
so the list cannot quietly rot.

**Fixed-dimension surfaces.** Several accept a vector of any length and then read only the first two
or three coordinates. They look fine at every level of inspection short of perturbing one coordinate
at a time, and at `d=100` they silently pose a 2-d problem. `damavandi_on_cube` and `chat_10` use 2
of 100. Listed in `FIXED_DIMENSION`.

**Slow ones.** Above roughly 10ms per evaluation a tournament stops being practical once you
multiply by trials, problems and optimizers. Listed in `SLOW`.

## The rest of this directory

| module | status |
|---|---|
| `classic.py`, `chatgptobjectives.py` | the sources `SURFACES` curates |
| `stochastic_surfaces.py` | the morphing machinery behind `morphed_surfaces()` |
| `portfolio.py` | Markowitz objectives. Real, but 20–60ms per evaluation and portfolio problems rather than surfaces, so not in `SURFACES` |
| `deapobjectives.py` | raw DEAP functions returning fitness *tuples*, not floats. `classic.DEAP_OBJECTIVES` is the usable wrapper |
| `horse.py` | one racing-derived objective, not a formulaic surface |
| `allobjectives.py` | **legacy.** Unscreened aggregate, kept only because `papers/planar_search` reports results over exactly this collection |
| `planar_h1.py`, `planar_h1_optimizer.py` | specific to the planar-search paper |

Removed, having never worked as advertised:

- `bbob_inspired_suite.py` — hard-coded to two dimensions; every function raised above `d=2`
- `enhanced_surfaces.py`, `enhanced_surfaces_working.py` — thin wrappers over `opfunu`, which is not
  a dependency and was not installed, so every call silently took a fallback path. Two copies of the
  same idea, one of which nothing imported
