"""
candidate_v2 — SUPERSEDED.

This was the prototype that added a separable quadratic trust-region jump on top
of the base DE/ES hybrid, to test (IDEAS.md §A.1) whether a cheap NEWUOA-like
model step lowers regret. The ablation (2026-06-17) confirmed it does — disabling
the jump on the evolved genome more than doubled regret (0.112 -> 0.238) — so the
mechanism has been **folded into the main template** `algo_dev.make_candidate`,
with `p_surrogate` / `r2_min` as standard genes 12..13.

Kept only as a compatibility shim so older drivers still import a working symbol.
New code should call `algo_dev.make_candidate` directly.
"""

from __future__ import annotations

from algo_dev import (  # noqa: F401 — re-exported for backward compatibility
    _fit_separable_quadratic,
    make_candidate as make_candidate_v2,
)
