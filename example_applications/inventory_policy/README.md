# (s, S) Inventory Policy

A classic operations-research policy-tuning problem with a **stochastic**
objective (like `cart_pole_policy`).

Tune an `(s, S)` periodic-review reorder policy: when end-of-period
inventory drops to or below the reorder point `s`, order up to the
order-up-to level `S`. We simulate `T=40` periods of random demand
(`d ~ max(0, round(gauss(20, 7)))`) and accumulate ordering, holding,
and shortage costs. The objective is the total cost of one rollout, so
**repeated evaluations of the same policy differ** — the true landscape
is a smooth expected-cost bowl buried under sampling noise.

- **N_DIM:** 2 — `s ∈ [0, 60]`, `S ∈ [0, 120]` (S clamped up to s so the
  policy is always valid).
- **Pathology:** genuine evaluation noise. An optimiser that over-trusts
  a single lucky rollout is punished.

Run it:

```bash
python -m example_applications.inventory_policy.run
```
