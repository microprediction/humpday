# Optimizers

Twenty-three derivative-free optimizers, each implemented here rather than wrapped, behind one
calling convention. The wrapper modules this file used to document -- `pysotcube`, `nloptcube`,
`comparisons.eloratings` and the rest -- were removed along with the third-party packages they
depended on; `pip install humpday` now pulls in nothing at all.

### Usage

```python
from humpday import pure_optimize
from humpday.objectives.classic import schwefel_on_cube

best_value, best_x = pure_optimize(schwefel_on_cube, "NelderMead", n_trials=50, n_dim=5)
```

`humpday.ALGORITHM_NAMES` lists what the second argument accepts. For a rectangular domain rather
than the unit cube, use `humpday.minimize(objective, bounds=..., method=...)`, which returns a
scipy-shaped `OptimizeResult`.

### Limitations

- `pure_optimize` takes an objective on the unit hyper-cube. `minimize` transforms for you.
- Single objective only.

For a domain that is a simplex rather than a box -- portfolios, mixtures, allocations -- lift the
objective with `humpday.transforms.cubetosimplex.lift_to_cube`.

### Which one to use

```python
from humpday import suggest

suggest(n_dim=5, n_trials=50)   # ranked, from the recorded tournament
```

The ranking comes from `humpday/data/ratings.json`, a recorded round-robin over two objective
suites at each (dimension, budget) cell, rather than from a rule of thumb. `humpday.ratings`
reads it, and `benchmarks/record_ratings.py` is what writes it. The Elo demo script this file used
to point at, and the three separate rating artifacts behind it, were replaced by that one
tournament.

### Run every optimizer against your problem

```python
from humpday import ALGORITHM_NAMES, pure_optimize
from humpday.objectives.classic import schwefel_on_cube

for name in ALGORITHM_NAMES:
    value, _ = pure_optimize(schwefel_on_cube, name, n_trials=50, n_dim=5)
    print(f"{name:<24} {value:.6f}")
```

Not every optimizer is worth running at every size: `humpday.eligibility.passes_dim` and
`passes_trials` say which ones can use the dimension and budget you have, and `recommend` applies
both before it picks.

### Background

The [Comparison of Global Optimizers](https://www.microprediction.com/blog/optimize) and the
[HumpDay](https://www.microprediction.com/blog/humpday) posts describe the study this package grew
out of. They predate the self-contained implementations and the recorded tournament, so read them
as history rather than as instructions.
