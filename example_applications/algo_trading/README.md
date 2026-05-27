# Walk-Forward Algorithmic Trading

Tune a 4-parameter momentum-mean-reversion strategy on a synthetic
price series, then score it on an out-of-sample period. The objective
is **negative in-sample Sharpe ratio**; the table reports both
in-sample and out-of-sample Sharpe so you can see how badly each
optimiser overfits.

## What this stresses

- **Non-stationarity.** The synthetic series is generated with two
  regimes — a momentum regime in the first half (drift dominates) and
  a mean-reversion regime in the second (drift suppressed). Parameters
  that exploit the first regime perfectly will *underperform* on the
  second; the better algorithms find broad, robust parameter regions
  instead of sharp peaks.

- **Penalty for overfitting.** Algorithms that hunt narrow optima will
  show the largest in-sample / out-of-sample gap. Algorithms that prefer
  flat plateaus (Bayesian, evolutionary) will generalise better.

- **Discontinuous landscape.** The strategy uses integer-valued lookback
  windows (rounded from `[0,1]^4`). A 1-bar shift in lookback discretely
  changes which trades happen — there is no smooth gradient.

## The strategy

A z-score channel: at each bar, compute the price's z-score relative
to a rolling window. **Buy** if z falls below `−entry_z`; **exit** the
long when z rises above `+exit_z`. No shorts, no leverage, transaction
cost is a flat `cost_bps`.

Four optimisation parameters (all on `[0, 1]`, decoded by `problem.py`):

| Parameter | Decoded range | Meaning |
|---|---|---|
| `lookback`  | 10 – 200 bars | Rolling window for z-score. |
| `entry_z`   | 0.5 – 3.0     | Z-score threshold to open a long. |
| `exit_z`    | 0.0 – 2.0     | Z-score threshold to close. |
| `cost_bps`  | (constant)    | Round-trip cost held fixed at 5 bps. |

The cost is **negative Sharpe ratio** of the strategy's bar-by-bar
return series on the in-sample window. Sharpe ratio is annualised
assuming ~252 trading days.

## Running

```bash
python -m example_applications.algo_trading.run
```

The output table compares optimisers by **in-sample Sharpe** (what
they were trained on) and **out-of-sample Sharpe** (what they should
actually be judged by). A large positive in-sample / near-zero
out-of-sample gap means the optimiser found an overfit corner of
parameter space.

## Caveats

This example uses a deterministic synthetic price series so every
run produces the same result — useful for examples, less useful as a
real benchmark. Replacing `_generate_prices` with a different seed
will change the optimiser ranking. In production benchmarking you'd
average across many regimes.
