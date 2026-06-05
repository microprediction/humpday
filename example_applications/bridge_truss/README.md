# Bridge Truss — Constrained Structural Optimisation

Size the members of a bridge for minimum weight without yielding or buckling. A
pure-Python finite-element model solves a 6-node, 10-member truss under a midspan
load. HumpDay's `[0,1]^10` cube sets each member's cross-sectional area (log
scale); the objective returns **weight plus a penalty** for any member that
overstresses or buckles. Uses numpy for the FEM solve.

## What this stresses

- **Hard constraints on the boundary.** Bigger members are heavier but stronger,
  so the lightest safe design sits *exactly* on the stress/buckling limit — the
  optimiser must descend the penalty cliffs without falling off.
- **Statically indeterminate.** The X-braced bay is redundant, so the force flow
  *shifts* as you resize members — resizing one changes the load on others.
- **Two failure modes.** Tension members yield; compression members buckle
  (Euler) — different limits that bound different parts of the design.

## Running

```bash
python -m example_applications.bridge_truss.run
```

Lightest feasible truss wins; methods that get stuck leave a member unsafe.
Mirrors the browser demo
[`docs/applications/bridge-truss.html`](../../docs/applications/bridge-truss.html).
