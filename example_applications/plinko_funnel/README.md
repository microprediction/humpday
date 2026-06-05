# Plinko Funnel — Steering a Stochastic Process

Tune the **lean profile** of a Galton board so its bouncing balls — which
normally pile up in the middle — funnel into an *off-centre* target bin. A
pure-Python cascade drops 400 random balls down 14 rows of pegs; each region of
the board has a tunable lean that biases the left/right coin flip. HumpDay's
`[0,1]^7` cube maps to the seven-point lean profile; the objective is the
**negative percentage** of balls landing in the target bin.

## What this stresses

- **Inherent noise.** Every ball is a fresh coin-flip cascade, so the score is a
  noisy estimate — the optimiser is steering a *random process*, not a
  deterministic one. This is the pure stochastic-optimisation regime.
- **A funnel, not a point.** A good profile pushes balls across from the centre,
  then catches and concentrates them over the target — an emergent shape the
  optimiser discovers from a single scalar reward.
- **Easy to beat random, hard to perfect.** Random Search reaches only ~a third;
  structured optimisers funnel more than half the balls onto the target.

## Running

```bash
python -m example_applications.plinko_funnel.run
```

Expect structured methods around ~54–55% in the target bin versus ~33% for
Random Search. Mirrors the browser demo
[`docs/applications/plinko.html`](../../docs/applications/plinko.html).
