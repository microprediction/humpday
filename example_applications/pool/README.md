# Pool — Reduced-Order Cut

A **reduced-order** pure-Python objective with the same parameterisation and
target as the browser demo. The interactive version
([`docs/applications/pool.html`](../../docs/applications/pool.html)) uses the
**Matter.js** rigid-body engine; porting that to pure Python would mean porting a
physics engine, so this is a deliberately simplified physics model — a faithful
optimisation *problem* to run against the HumpDay optimisers, not a bit-identical
sim. See the module docstring in `problem.py` for the exact model.

## Running

```bash
python -m example_applications.pool.run
```
