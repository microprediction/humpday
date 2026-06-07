# Antenna Array — Optimiser Beats Intuition

Place seven radiating elements on a line to maximise **forward gain**. A
pure-Python reduced-order array-factor model computes the beam; HumpDay's
`[0,1]^7` cube sets the element positions (over a 4-wavelength span), and the
objective returns the negative forward directivity in dBi.

## What this stresses

- **Optimiser beats intuition.** A human spaces the elements evenly — about
  8 dBi. The optimiser finds an *irregular* spacing that concentrates more
  energy forward and beats the uniform array by a couple of decibels.
- **A wiggly landscape.** Interference between elements creates many local
  lobes, so the objective is multimodal and rewards a real search over guessing.
- **Moderate dimension (7-D).** Tractable, but enough room for the spacing to
  matter.

## Running

```bash
python -m example_applications.antenna_array.run
```

The table reports each optimiser's gain versus the even-spacing baseline. Mirrors
the browser demo [`docs/applications/antenna.html`](../../docs/applications/antenna.html).
