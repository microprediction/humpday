# Portfolio Frontier — non-convex allocation on the unit simplex

Allocate capital across eight assets — two tech names, bonds, gold,
energy, utilities, EM equity, a REIT — as **long-only weights that sum
to 1**, to minimise a mean-variance utility *plus a fixed cost per asset
held*. The weights are a composition (a point on the 7-simplex), lifted
from a plain `[0,1]^7` cube point through the **cube→simplex bijection**
(`humpday.transforms.cubetosimplex`).

## Why it's here — and why it's harder than `cocktail_blend`

`cocktail_blend` is a smooth, effectively unimodal simplex problem: every
optimiser nails the same blend, so it demonstrates the bijection but does
not discriminate methods. This one is deliberately **non-convex**.

Classic Markowitz utility `-mu·w + (gamma/2) wᵀΣw` is convex and has a
single optimum. We add a **smoothed cardinality penalty** — a fixed cost
`kappa` per asset actually held — which is concave near zero and so
carves the landscape into competing basins, one per *subset of assets you
choose to own*. The fixed-cost-per-holding is what real funds face
(custody, monitoring, ticket costs), and it is exactly what makes
portfolio selection combinatorial in practice.

## What you see

```bash
python -m example_applications.portfolio_frontier.run
```

The table shows genuine spread, and it is **bimodal**:

- **Single-asset corner trap** — local and Bayesian methods (NelderMead,
  Powell-sometimes, PRIMA_BOBYQA, BayesianOpt, SimulatedAnnealing) anchor
  on the all-`utility` defensive corner (objective ≈ −0.031).
- **Diversified basin** — population/global methods (ParticleSwarm,
  CMA-ES, Differential Evolution) and Powell-when-lucky escape to the
  better blend (objective ≈ −0.041): roughly tech 30–45% / utility 30% /
  gold + REIT for the rest, ~9–10% expected return at ~13–16% vol.

So this is a simplex demo that actually *ranks* optimisers — useful
fodder for the recommender, where the smooth cocktail problem is not.

## Knobs

- `GAMMA` — risk aversion. Raising it traces the efficient frontier toward
  lower-vol portfolios; it is tuned (3.0) so the optimum is a diversified
  *interior* blend rather than a single low-vol corner.
- `KAPPA` — fixed cost per holding; raising it concentrates the optimum
  into fewer names and deepens the multimodality.
- The bijection's scale is itself a tunable preconditioner — see
  [`papers/dfo_recommender/bijection_hyperopt.py`](../../papers/dfo_recommender/bijection_hyperopt.py).
