# Lens Design — Rugged Needle in a Smooth Bowl

Focus a collimated beam of light to the tightest possible point by bending
**four glass surfaces**. A pure-Python 2-D sequential ray trace sends 21 parallel
rays through two glass elements, refracting each by Snell's law, and measures the
**RMS spot size** on a fixed focal plane. HumpDay's `[0,1]^4` cube maps to the four
surface curvatures; minimising the spot sharpens the focus.

## What this stresses

- **Deceptive, multimodal landscape.** Real Snell refraction produces spherical
  aberration, so the sharp-focus designs sit in a *narrow needle* surrounded by
  blur. The good region is tiny — you cannot sample your way into it.
- **Low dimension, but precision-critical.** Only 4-D, yet Random Search stalls
  a hundred-fold short of the methods that follow the spot down into the valley.
- **A reputation reversal.** Interpolation / local methods (`PRIMA_BOBYQA`,
  `NelderMead`) *win* here — the mirror image of the high-dimensional
  [`genetic_art/`](../genetic_art/) example, where those same methods come last.
  No optimiser is best everywhere.

## Running

```bash
python -m example_applications.lens_design.run
```

Expect Nelder-Mead and PRIMA_BOBYQA to drive the RMS spot below ~0.005 while
Random Search languishes above 0.1. Mirrors the browser demo
[`docs/applications/lens.html`](../../docs/applications/lens.html).
