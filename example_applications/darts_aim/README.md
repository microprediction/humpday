# Darts Aim

Where should you aim on a dartboard if your throws scatter?

You pick an aim point `(x, y)` in millimetres. The dart lands there plus
an isotropic Gaussian error with `SIGMA = 35 mm` (an amateur). The
objective is the **negative expected score**, so minimising it maximises
your expected points. It is computed deterministically by averaging the
standard dartboard score over a fixed bank of ~600 noise offsets
precomputed at import, making the objective a repeatable function of the
aim.

- `n_dim = 2`: `u ∈ [0,1]^2` maps to `x = (u0-0.5)*340`, `y = (u1-0.5)*340` mm.
- **Pathology:** deterministic but strongly multimodal. The board's
  big-next-to-small pinwheel layout makes the score surface jagged;
  smoothing it with the 35 mm Gaussian leaves several competing humps
  (treble-20 at top, the 19/16 cluster lower-left, the centre), so
  hill-climbers easily settle on the wrong one.
- **Famous result** (Tibshirani, Price & Taylor, 2011): a perfect player
  aims at treble-20, but as throw noise grows the optimal aim slides down
  and to the lower-left of centre.

Run it:

```
python -m example_applications.darts_aim.run
```
