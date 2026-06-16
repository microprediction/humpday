# Multi-Exponential Fit — the ill-conditioned valley

Recover a sum of two decaying exponentials from noisy data:

```
y(t) = A1·exp(-k1·t) + A2·exp(-k2·t)
```

the canonical inverse problem of spectroscopy, fluorescence-lifetime imaging,
and pharmacokinetics. The objective is the sum of squared residuals over the
decay curve; the four decision variables (two amplitudes, two log-scaled rates)
map from `[0,1]^4`.

## Why it's here

This is the **genuinely ill-conditioned** landscape the rest of the suite lacks —
a long, *curving, degenerate valley* rather than a multimodal field. When the two
rates `k1, k2` are close, the two exponentials become nearly indistinguishable and
an entire ridge of `(A, k)` combinations fits the data almost equally well. The
Jacobian's condition number diverges as `k1 → k2` (the normal equations go
singular) — the textbook "sloppy", ill-posed problem.

```bash
python -m example_applications.multi_exponential_fit.run
```

The table is the lesson: optimisers reach near-identical residuals (~the noise
floor) while reporting **wildly different rates** — e.g. recovering `(k1,k2)` as
`(1.9, 0.9)`, `(4.4, 1.2)`, `(1.5, 0.6)`, `(0.8, 1.7)`. They're all sitting at
different points along the same flat-bottomed valley. There is also a label-swap
symmetry — `(A1,k1,A2,k2)` and `(A2,k2,A1,k1)` are the same fit — so the global
optimum is two-fold degenerate.

## References

- Sethna (Cornell), *Fitting Exponentials* — "a sloppy, ill-posed problem."
- Transtrum, Machta & Sethna, *sloppy models* (broad flat/curving valleys).
- *Nature Sci. Rep.* 2022, s41598-022-08638-7 — condition number → ∞ as rates converge.
