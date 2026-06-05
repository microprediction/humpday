# Tuned Mass Damper — Mixed-Integer Seismic Design

Cut a tower's earthquake sway with a single **tuned mass damper** — a heavy block
on a spring attached to one floor. HumpDay's `[0,1]^4` cube decodes into the
damper's mass, frequency tuning, damping ratio, and the **integer floor** it sits
on. Each design re-runs a full synthetic earthquake (Newmark-β time integration
of a 6-storey shear building) and the objective returns the negative **percentage
reduction in peak roof sway**.

## What this stresses

- **Mixed-integer.** Three continuous knobs plus one discrete choice (the floor),
  so the search space is part-continuous, part-combinatorial.
- **Dynamics in the loop.** Every evaluation integrates 1600 time steps of a
  structural model; this is the expensive, physics-in-the-objective regime.
- **Resonance.** The ground motion is narrowband near the building's first mode,
  so the bare tower resonates and the tuning genuinely matters — good designs
  tune near the first mode and sit high on the tower.

## Running

```bash
python -m example_applications.tuned_mass_damper.run
```

Uses numpy for the modal eigenvalues and the per-step solves. Expect the best
damper to cut peak roof sway by ~45%. Mirrors the browser demo
[`docs/applications/tuned-mass-damper.html`](../../docs/applications/tuned-mass-damper.html).
