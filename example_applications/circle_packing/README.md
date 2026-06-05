# Circle Packing — Maximise the Minimum

Pack six equal circles into the unit square as tightly as possible. HumpDay's
`[0,1]^12` cube places the six centres; the circles then grow to the largest
shared radius that fits without overlapping or leaving the square. The objective
returns the negative radius (scored against the best-known pack, r* ≈ 0.1875).

## What this stresses

- **A non-smooth objective.** The achievable radius is the *minimum* over every
  pairwise gap and every wall clearance, so the surface has sharp ridges where
  the binding constraint switches — exactly the kind of kink that defeats
  gradient methods.
- **No infeasible points.** Every layout is valid; the difficulty is purely the
  rugged "maximise the minimum" structure.
- **A known optimum.** For six circles the answer is two offset columns of three.

## Running

```bash
python -m example_applications.circle_packing.run
```

It's a genuinely hard pack — expect the best optimisers around 70–80% of the
optimal radius at a few hundred trials, well above random. Mirrors the browser
demo [`docs/applications/packing.html`](../../docs/applications/packing.html).
