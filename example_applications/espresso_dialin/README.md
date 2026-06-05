# Espresso Dial-In — Sample-Efficient, Noisy Optimisation

Find the "god shot" in as few pulls as possible. A pure-Python response surface
maps HumpDay's `[0,1]^4` cube to the four dials — grind, dose, temperature, time.
A great shot needs the **extraction yield** near 20% **and** the **brew ratio**
near 1:2; the two are coupled, so the sweet spot is small. Each pull is **noisy**,
and the objective returns the negative shot score.

## What this stresses

- **Sample efficiency.** The budget is only a couple-dozen pulls — nobody pulls
  500 espressos. This is the expensive-evaluation regime: interpolation /
  trust-region methods (`PRIMA_BOBYQA`) and `BayesianOpt` nail the sweet spot
  while population methods are still warming up.
- **Noise.** Real shots vary, so a single tasty-looking pull can be luck; the
  `run.py` re-evaluates the chosen recipe *without* noise to show its true quality.
- **A small coupled target.** Two conditions that must hold at once make the
  feasible region tight — easy to wander past.

## Running

```bash
python -m example_applications.espresso_dialin.run
```

At 24 pulls expect Nelder-Mead / PRIMA_BOBYQA / Bayesian methods near a true
score of 90+, with population methods well behind — the reverse of the
high-dimensional examples. Mirrors the browser demo
[`docs/applications/espresso.html`](../../docs/applications/espresso.html).
