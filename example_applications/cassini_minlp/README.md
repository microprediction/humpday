# Cassini-style MGA Trajectory — mixed-integer flyby selection

Fly from Earth to Saturn with four gravity-assist flybys, paying total Δv
(launch burn + a powered-flyby correction at each swing-by + Saturn arrival
burn). You choose **both** the continuous timing **and** the discrete flyby
planets:

- **6 continuous** — launch epoch + five leg times-of-flight;
- **4 discrete** — which planet to swing by at each flyby (from a candidate set).

```bash
python -m example_applications.cassini_minlp.run
```

## Why it's here

A pathology pair the suite under-samples:

- **Mixed-integer** — four discrete planet choices riding on top of the
  continuous timing (like `tuned_mass_damper`, but combinatorial in several
  slots at once).
- **Multimodal with deceptive near-ties** — distinct flyby *sequences* reach
  almost the same Δv, so a search can't tell which sequence is truly best from a
  handful of samples, and picking the wrong one strands it in a basin no amount
  of timing tweaks escapes.

The run table is the lesson: every optimiser returns a **different** flyby
sequence (`E-E-M-M-E`, `E-V-V-E-E`, `E-E-J-J-J`, …) at near-tied Δv — they
disagree on the discrete choice, which is the combinatorial trap. (Pleasingly,
random search and several methods gravitate to Earth-heavy sequences, echoing
the real GTOPX Cassini1-MINLP, whose deceptive local optimum is the flyby
sequence `{Earth, Earth, Earth, Jupiter}`.)

## Model

Reduced-order: Sun-centred, all bodies on coplanar circular orbits, canonical
units (μ=1, AU, Earth period 2π). A universal-variable Lambert solver (bisection,
no dependencies) links the legs; each flyby is an idealised powered swing-by —
Δv = relative-speed mismatch + a soft turn-angle penalty.

*Simplified — coplanar circular orbits, single-revolution Lambert, idealised
gravity assists. **Not** a numeric replica of GTOPX Cassini1; it reproduces the
mixed-integer / deceptive-sequence structure, not the exact best-known Δv.*

## References

- ESA GTOP / GTOPX Cassini1-MINLP (arXiv 2010.07517).
- Curtis, *Orbital Mechanics for Engineering Students*, Lambert's problem.
