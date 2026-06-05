# Free Kick — Bend It Past the Wall and the Keeper

Score a free kick by tuning four numbers — aim, loft, power and curve (sidespin).
A pure-Python 3-D ball-flight simulator kicks the ball and integrates its path
under gravity, drag and a Magnus side-force, past a defensive wall and a diving
goalkeeper. HumpDay's `[0,1]^4` cube sets the four controls; the objective returns
the negative score (a clean goal ~100+, a block or a save low).

## What this stresses

- **A multimodal, outcome-driven landscape.** GOAL / saved / blocked / wide /
  over are distinct regimes; the ball must rise over (or curl around) the wall
  *and* dip under the bar *and* reach a corner the keeper can't get to.
- **Coupled controls.** Curve and loft trade off — more sidespin bends it round
  the wall but changes where it crosses the line; the optimiser must co-ordinate
  them.
- **A shaped reward.** Near-misses, woodwork and keeper stretches give a gradient,
  so optimisers can climb toward the goal rather than seeing a flat 0/1.

## Running

```bash
python -m example_applications.free_kick.run
```

A score above 100 means a goal the keeper couldn't reach. Mirrors the browser
demo [`docs/applications/free-kick.html`](../../docs/applications/free-kick.html).
