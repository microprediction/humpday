# Tetris Brain — Noisy Heuristic Tuning

Tune the **four evaluation weights** of a greedy Tetris bot to clear the most
lines. For the current piece the bot tries every rotation and column, simulates
the drop, and rates the board by

    w1·aggregate_height + w2·lines + w3·holes + w4·bumpiness

playing the best landing. HumpDay's `[0,1]^4` cube maps to the four weights in
`[-1,1]^4`; the objective is the **negative mean lines** over a few games on a
stylised 7×16 board.

## What this stresses

- **Noisy objective.** Piece order is random (7-bag), so the same weights score
  differently every game. Optimisers that over-trust one lucky game get fooled.
- **In-sample vs out-of-sample.** `run.py` reports both the training score and a
  held-out cohort; the gap is the noise-overfit tax. (The companion experiment in
  this session found the real cure is *re-validating* top candidates on fresh
  seeds before committing — not seeking flat optima.)
- **A non-obvious optimum.** The optimiser does *not* rediscover the textbook
  "reward lines, punish holes" weighting. Survivability beats greed, so odd
  vectors — sometimes even penalising line-clears — score well.

## Running

```bash
python -m example_applications.tetris_weights.run
```

Expect the table sorted by held-out mean, with several optimisers near ~70 lines
and a clear train-vs-test gap. Mirrors the browser demo
[`docs/applications/tetris.html`](../../docs/applications/tetris.html).
