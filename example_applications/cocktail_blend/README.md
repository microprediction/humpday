# Cocktail Blend — an inverse problem on the unit simplex

Mix six ingredients — gin, sweet vermouth, Campari, lime juice, simple
syrup, soda water — into proportions that **sum to 1** so the blend's
flavour profile (sweet, sour, bitter, boozy, herbal, citrus) matches a
target spritz as closely as possible.

## Why it's here

The decision variable is a *composition* — a point on the 5-simplex —
not a box point. That's a constraint type none of the box-bounded demos
exercise. Rather than reach for a constrained solver, we use the
**cube→simplex bijection** (`humpday.transforms.cubetosimplex`): the
objective HumpDay sees is an ordinary `[0,1]^5` function whose first act
is to lift the cube point onto the simplex. Simplex problems become box
problems for free.

The target is **over-determined** (6 flavour axes, 5 free proportions),
so the optimum is a genuine constrained least-squares blend — an
*interior* point of the simplex, not a single-ingredient vertex and not
an exact zero.

## Running

```bash
python -m example_applications.cocktail_blend.run
```

Every representative optimiser converges to essentially the same blend
(≈ Campari 47% / sweet vermouth 21% / gin 13%, RMS flavour error ≈ 0.15):
the lifted landscape is smooth and effectively unimodal, so this is an
*easy* simplex problem. It demonstrates the bijection cleanly; it does
**not** discriminate between optimisers — a harder, multimodal simplex
task would be needed for that.

## A note on the bijection's scale

The cube→simplex map has a scale parameter (`STD_L`, and more generally
the `(scale, tail-warp)` family explored in
[`papers/dfo_recommender/bijection_hyperopt.py`](../../papers/dfo_recommender/bijection_hyperopt.py)).
It controls how much of the simplex the cube can reach: too large and the
whole cube collapses near the centroid; the principled default places the
corners within reach. Tuning it is a *preconditioner* for simplex-valued
optimisation — see that script for the amortized experiment.
