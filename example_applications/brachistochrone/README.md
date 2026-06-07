# Brachistochrone — The Fastest Slide

Shape a ramp so a marble slides from the top-left to the bottom-right in the
least time. HumpDay's `[0,1]^8` cube sets the heights of eight control points;
the objective returns the **descent time** under gravity (by energy
conservation). The famous answer is *not* the straight line — it's the cycloid,
which dips steeply early to build speed before a shallow run-out.

## What this stresses

- **A calculus-of-variations classic.** The optimum is a known curve, so you can
  see the optimiser rediscover the early-dip shape.
- **A counter-intuitive optimum.** The shortest path (straight line) is slower
  than a longer, faster one — easy for a naive search to miss.
- **Smooth, moderate dimension (8-D).** A clean test where most methods make
  progress and you can compare how close each gets to the cycloid.

## Running

```bash
python -m example_applications.brachistochrone.run
```

The straight-line baseline is printed for comparison; the optimised ramps are
clearly faster. (This port scores by energy conservation; the browser demo
[`docs/applications/brachistochrone.html`](../../docs/applications/brachistochrone.html)
rolls a Matter.js marble — same cycloid optimum.)
