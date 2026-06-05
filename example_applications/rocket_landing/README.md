# Rocket Landing — Multimodal Throttle Schedule

Land a falling booster softly with fuel to spare. HumpDay's `[0,1]^12` cube is a
piecewise-constant throttle schedule; the objective simulates the 1-D vertical
descent and scores a soft touchdown (perfect v ≈ 0 = 100, falling to 0 at 80 px/s)
plus up to +10 for leftover fuel.

## What this stresses

- **Several distinct local optima.** A gradual descent that drains the tank early
  versus a late "suicide burn" that waits then dumps thrust — different basins
  with very different scores.
- **A tight resource budget.** The tank holds only ~4 s of full burn, so the
  schedule must spend it at the right moment.
- **A cliff.** Burn too late and the rocket slams in (score crashes); too early
  and it runs out and falls the rest of the way.

## Running

```bash
python -m example_applications.rocket_landing.run
```

A score above 100 means a soft landing with fuel left over. Mirrors the browser
demo [`docs/applications/rocket-landing.html`](../../docs/applications/rocket-landing.html).
