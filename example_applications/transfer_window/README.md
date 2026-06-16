# Interplanetary Transfer Window — disjoint islands (the porkchop)

Pick a **departure date** and a **time-of-flight** for an Earth→Mars transfer and
pay the total Δv (departure burn + arrival burn). Solving Lambert's problem for
the transfer arc over a grid of (departure, time-of-flight) is the classic
**porkchop plot**.

```bash
python -m example_applications.transfer_window.run
```

## Why it's here

The landscape **type** the suite most needs: **disjoint feasible islands /
multi-basin**. Good launch windows are isolated low-Δv valleys, one per synodic
alignment, separated by tall ridges of impossible or expensive transfers. A
local or trust-region method converges to whichever island it started in and
never discovers the others — this is basin-hopping / CMA / population territory.
It's the pure-Python reduced form of ESA's GTOP / GTOPX trajectory benchmarks.

The run table is the lesson: every optimiser reaches essentially the **same**
optimal Δv ≈ 0.188, but at **different departure epochs (~0, ~13, ~27)** — three
distinct launch windows, one synodic period apart. Same cost, different islands.

## Model & validation

Reduced-order: Sun-centred, both planets on coplanar circular orbits, canonical
units (μ=1, AU, Earth period 2π). A universal-variable Lambert solver (Curtis
Alg. 5.2, solved by **bisection** — no dependencies) returns the transfer
velocities; Δv is summed against the circular planet velocities. Validated
against the analytic Hohmann transfer: the global-best Δv ≈ **0.1881** matches
the closed-form Hohmann **0.1878** to four significant figures.

*Simplified — coplanar circular orbits, single-revolution Lambert.*

## References

- ESA GTOP / GTOPX interplanetary trajectory benchmarks (arXiv 2010.07517).
- Curtis, *Orbital Mechanics for Engineering Students*, Lambert's problem (Alg. 5.2).
