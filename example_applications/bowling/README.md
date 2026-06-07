# Bowling — Chain Reactions and Sensitivity

Maximise the number of pins knocked down from a dense 105-pin triangle. A
**faithful** pure-Python port of the browser demo's from-scratch rigid-body
simulator (no physics engine): a heavy ball plows through the formation and
impulse-based ball-pin and pin-pin collisions propagate the strike. HumpDay's
`[0,1]^4` cube sets ball speed, launch angle, spin (Magnus curve) and release
position; the objective returns the negative pin count.

## What this stresses

- **No analytic objective.** You have to run the collision sim — there's no
  formula. This is the simulation-in-the-loop regime.
- **A rough, sensitive landscape.** Chain reactions mean tiny changes in entry
  angle or spin cascade into very different pin counts, so the surface is bumpy
  and near-discontinuous.
- **A near-miss gradient.** A small term rewards passing close to standing pins,
  so optimisers get a signal even when a throw leaves pins up.

## Running

```bash
python -m example_applications.bowling.run
```

The maximum is 105 pins. Expect the best optimisers into the high-90s/low-100s.
Mirrors the browser demo
[`docs/applications/bowling.html`](../../docs/applications/bowling.html).
