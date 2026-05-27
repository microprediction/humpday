# HumpDay Example Applications

Real-world(ish) case studies that exercise HumpDay's pure-Python optimisers
on the kind of objectives derivative-free optimization is actually for —
not the classical analytic benchmarks (Rastrigin, Griewank, Salomon) but
problems whose pathologies (constraints, noise, non-stationarity,
expensive evaluation) are why anyone reaches for black-box methods in
the first place.

Each subfolder is self-contained:

| Folder | Domain | What it stresses |
|---|---|---|
| [`welded_beam/`](welded_beam/)             | Structural engineering            | Constraint handling via penalty functions; navigation of narrow feasible regions on disparate dimension scales. |
| [`cart_pole_policy/`](cart_pole_policy/)   | Reinforcement learning            | Direct policy search; tolerance to stochastic episode noise; medium-dimensional parameter spaces. |
| [`algo_trading/`](algo_trading/)           | Quantitative finance              | Walk-forward optimisation; in-sample vs out-of-sample generalisation; the danger of overfitting sharp peaks. |
| [`airfoil_shape/`](airfoil_shape/)         | Surrogate-based aerodynamics      | Severely limited evaluation budgets; landscapes where Bayesian methods dominate evolutionary ones. |

## The HumpDay convention

Every objective in HumpDay maps the unit hypercube `[0, 1]^n` to a
real-valued cost to minimise. So each example's `problem.py` does two
things:

1. **Scale** the `[0, 1]^n` hypercube point to whatever physical units
   the application naturally lives in.
2. **Evaluate** the application-specific cost (with constraint penalties,
   episode rollouts, backtests, surrogate calls, etc).

The `run.py` in each folder calls a handful of representative
optimisers from `humpday.optimizers.alloptimizers.PURE_OPTIMIZERS` —
typically one local (NelderMead), one population-based (DifferentialEvolution
or CMA), one Bayesian (BayesianOpt), and one trust-region (PRIMA_BOBYQA) —
and prints a small comparison table.

## Running an example

From the repo root:

```bash
python -m example_applications.welded_beam.run
python -m example_applications.cart_pole_policy.run
python -m example_applications.algo_trading.run
python -m example_applications.airfoil_shape.run
```

No external dependencies beyond what HumpDay itself uses — these are
intentionally pure-Python so they work in browser/Pyodide too.

## Why these four?

Together they span the typological cross-section that the HumpDay
recommender system needs to be calibrated against:

- **Constraints with hard physical limits** (welded beam): penalty cliffs
  the optimiser must descend without falling off.
- **Stochastic objective** (cart pole): a single evaluation gives a noisy
  estimate; an algorithm that over-trusts a single low value will be
  punished out-of-sample.
- **Non-stationary objective** (algo trading): the in-sample optimum is
  not the out-of-sample optimum; this distinguishes algorithms that hunt
  sharp peaks from those that prefer broad flats.
- **Expensive objective** (airfoil): every evaluation is precious; the
  algorithm must extract maximum information from few samples.

These four are the ones an industrial user would actually face. The
Elo rankings derived from these tasks will be more useful than those
derived from purely analytic functions.
