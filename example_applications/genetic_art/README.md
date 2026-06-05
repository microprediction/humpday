# Genetic Art — High-Dimensional Image Fit

Approximate a target picture using a handful of **translucent triangles**. A
pure-Python software rasteriser paints the triangles back-to-front onto a small
canvas; each triangle has 10 genes (3 vertices, RGB, alpha). HumpDay's
`[0,1]^(10·N)` cube *is* the painting, and the objective is the **pixel RMS
error** to a fixed target image. The default 6 triangles makes it 60-D.

## What this stresses

- **High dimensionality.** Even at 60-D the search is large and multimodal —
  countless arrangements score about the same.
- **A reputation reversal.** `NelderMead`, which *wins* the low-D
  [`lens_design/`](../lens_design/) example, collapses here, while population
  methods cope easily — the same method, opposite verdict. At 60-D PRIMA_BOBYQA
  still copes; push `N_TRIANGLES` toward the 300-D browser scale and the
  interpolation methods fall behind too. The No-Free-Lunch theorem in miniature.
- **The CMA cost.** CMA-ES carries an *n×n* covariance matrix; at 60-D it is
  fine, but raise `N_TRIANGLES` and feel it grow — the full 300-D browser version
  is impractical for exactly this reason.

## Running

```bash
python -m example_applications.genetic_art.run
```

It will never be photographic — that is the low-poly charm. Expect the population
methods to reach the lowest RMS error. Mirrors the browser demo
[`docs/applications/genetic-art.html`](../../docs/applications/genetic-art.html).
