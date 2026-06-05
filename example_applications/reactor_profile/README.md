# Reactor T-Profile — Optimal Control of a Series Reaction

Maximise the yield of an intermediate product. A pure-Python plug-flow reactor
runs two first-order reactions in series, **A → B → C**, each Arrhenius in
temperature. B is what you want; too hot and it runs straight on to worthless C.
HumpDay's `[0,1]^10` cube sets the temperature in each of ten zones (300–480 K);
the objective integrates the kinetics and returns the negative final yield of B.

## What this stresses

- **The optimum is a profile, not a point.** Because B → C is far more
  temperature-sensitive than A → B, the best strategy runs hot early to convert A
  fast, then cools to stop B decomposing — a calculus-of-variations flavour.
- **A shaped schedule beats the best constant.** `run.py` prints the best
  isothermal yield for reference; the optimisers find a profile that beats it.
- **Smooth, moderate dimension (10-D).** A clean test where most methods make
  progress and you can compare how much shaping each recovers.

## Running

```bash
python -m example_applications.reactor_profile.run
```

Mirrors the browser demo
[`docs/applications/reactor-tprofile.html`](../../docs/applications/reactor-tprofile.html).
