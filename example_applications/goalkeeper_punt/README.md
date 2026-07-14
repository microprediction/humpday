# Goalkeeper Punt — Two Steps, Hit the Overhanging Wire

Every keeper has tried it at an empty training ground: punt the ball straight
up into the TV spider-cam wire. Here you tune four numbers — two run-up step
lengths, loft and leg power. A pure-Python side-view flight simulator walks
the keeper forward, punts the ball from the hands, and integrates the flight
under gravity and drag. The wire hangs 30 m downfield and 13 m up; the ball
must pass within 25 cm of it. HumpDay's `[0,1]^4` cube sets the four controls;
the objective returns the negative score (a wire strike 100+, misses shaped
by closest approach).

## What this stresses

- **A constrained boundary optimum.** Longer strides add run-up speed to the
  punt, but the ball must be released inside the penalty area — overstep and
  it's a foul. The optimum takes the longest strides the box allows
  (stride sum 2.899 m against a 2.9 m limit).
- **Coupled controls.** Contact quality peaks when the second step is ~0.25 m
  longer than the first (an accelerating rhythm), so the two step dimensions
  couple with each other and with power.
- **Two hit manifolds.** The wire can be clipped on the way up (flat, fast)
  or on the way down (high, dropping) — hits exist at any loft from ~33° to
  70°, but the impact-speed bonus makes the flat rising strike the true
  optimum (~109.4 at loft ≈ 32°, power maxed).
- **A shaped reward.** Misses score `90·exp(−d/2)` in closest-approach
  distance, so optimisers can climb toward the wire rather than seeing a
  flat 0/1. Only ~1.2% of random punts hit.

## Running

```bash
python -m example_applications.goalkeeper_punt.run
```

A score above 100 means the ball hit the wire. Mirrors the browser demo
[`docs/applications/punt-the-wire.html`](../../docs/applications/punt-the-wire.html).
