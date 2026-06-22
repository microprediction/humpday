# Overnight test battery (designed 2026-06-16)

Two threads, both pushed to the scale that earlier small runs couldn't support.
All jobs checkpoint to `runs/` so partial results survive a kill.

## Thread 1 — Schur/CMA-ES, definitive (free: numpy/scipy, no API)

`overnight_schur.py` → `runs/overnight_schur.json`

Matrix: **all ~51 demos (bowling skipped) × 5 seeds × 3 budgets {60,120,240} ×
6 damping variants**, bucketed by dimension (low <10, high ≥10).

Variants: full (γ=1), blind diagonal (γ=0.25, γ=0), seriated HRP-discovered
blocks (γ=.5/.0 at distance-cut 0.5, γ=.5 at 0.35).

Settles:
1. Does *any* damping beat full-covariance CMA, and in which (budget, dim) cell?
2. Is the effect **budget-dependent** — damping should help most where the
   covariance is most undersampled (high dim AND low budget). The 3-budget ×
   2-bucket grid is built to expose exactly that interaction.
3. Does **seriated** (discovered) block structure beat **blind** diagonal
   damping? (The wind_farm result said assumed contiguous blocks were backwards;
   seriation is the fix — this tests whether discovery rescues it.)

Companion: `seriation_hidim.py` → `/tmp/seriation.log` (per-demo table on the 9
high-dim demos, incl. a naive-contiguous-block column for the structured ones).

## Thread 2 — Simplex semantic-mixing, regret-surface map (bounded API)

`simplex_blend.py --points 30` → `runs/simplex_overnight.json` (+ generated code).

30 simplex points (random interior via the cube→simplex bijection) on ~10 demos
(mix incl. high-dim), each point generating a fresh optimizer via Claude with the
**new host+graft prompt** (dominant weight = host architecture, minority weights =
grafted ideas — the asymmetric "70% A, 30% B = A borrowing from B" semantics).

Settles the one empirical question that isn't a literature question:
**is the simplex interior *rich*?** I.e. does regret vary meaningfully across the
simplex (is there a blend that beats the pure vertices?), or do nearby points
collapse to near-identical code (the plateau risk)? Bounded to ~30 LLM calls.

## Thread 3 — Schur WAVE 2: reliability-adaptive γ* (free, added later 2026-06-16)

`overnight_schur2.py` → `runs/overnight_schur2.json`

Tests the writeup's #1 follow-up: **`adaptive γ*`** — per-correlation noise-floor
shrinkage `s = ρ²/(ρ² + c/λ)`, no fixed dial — vs full / blind / seriated at
tighter cuts (0.35, 0.20), across budgets {120,240,**480**} × dim, 5 seeds.
Two questions: does adaptive γ* win in *every* (budget,dim) cell (the self-tuning
rule), and does damping keep pulling ahead in high-dim as budget → 480?

## What I'll synthesize when results land
- Schur: a budget×dim table of which variant wins each cell; a clear yes/no on
  "damping helps CMA, and where", and whether seriation beats blind.
- Simplex: the regret surface — best blend vs best vertex, and whether the
  interior holds anything the corners don't.
- Honest verdict on both, no spin.
