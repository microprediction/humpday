# Walking Creature — Evolved Locomotion

Evolve a gait that walks as far as possible. A pure-Python kinematic walker drives
a two-legged body with leg oscillators: a planted foot grips the ground and its
backward sweep carries the body forward; lift both feet at once and it falls.
HumpDay's `[0,1]^6` cube maps to six gait numbers — frequency, stride, the
**phase offset** between the legs, lift timing, lift duty, and a forward lean —
and the objective is the **negative distance** walked in 8 seconds.

## What this stresses

- **Emergent structure from a scalar reward.** Nobody tells the optimiser that
  legs should alternate; it reliably rediscovers it. Watch the phase-offset of
  the best gait converge near half a cycle (≈ 1.0 π).
- **A broad, forgiving basin — with traps.** The walking region is wide enough
  that even Random Search shuffles forward, so the structured methods win on
  *quality*, not just feasibility. But `PRIMA_NEWUOA` can get trapped near its
  starting point and never learn to walk at all.
- **Deterministic but non-convex.** No noise here; the difficulty is the
  oscillator-coupling landscape itself.

## Running

```bash
python -m example_applications.walking_creature.run
```

Expect most optimisers around ~70 body-lengths with a phase offset near 1.0 π,
and Random Search well behind. Mirrors the browser demo
[`docs/applications/creature.html`](../../docs/applications/creature.html).
