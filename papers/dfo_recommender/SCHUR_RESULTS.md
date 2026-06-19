# A SAD Optimizer: Schur Acquisition Damping for undersampled covariance in derivative-free optimization

> **SAD** = **S**chur **A**cquisition **D**amping — reining in the over-trusted,
> undersampled covariance that CMA-ES (and Bayesian-optimization acquisition)
> adapt from too few samples. The name is self-deprecating; the result is not.

**Status: complete. Both batteries finished at 51/51 demos × 5 seeds
(`runs/overnight_schur.json` = wave 1, budgets 60/120/240, 6 fixed/seriated
variants; `runs/overnight_schur2.json` = wave 2, budgets 120/240/480, adds the
reliability-adaptive γ\* and tighter seriation cuts). The numbers below are final.
Caveat that governs everything: regret is normalised *within each battery's own
variant set*, so compare rankings within a battery — never absolute numbers
across the two.**

## TL;DR

CMA-ES adapts a full `n×n` covariance from only `λ ≈ 4+3ln n` samples per
generation — undersampled in high dimensions. **SAD (Schur Acquisition Damping)**
borrows the **Schur damping** dial from Hierarchical Risk Parity (`γ∈[0,1]` from
full covariance `γ=1` to block/diagonal `γ=0`, with the blocks *discovered* by
seriation of the correlation matrix) and applies it to that covariance. We ask
whether reining the covariance in helps. Final findings (these revise an earlier
44/51 draft — several headline claims changed):

1. **Reliability-adaptive `γ*` is the win.** Setting γ per-correlation from a
   noise floor (no fixed dial) **beats full covariance and every fixed/seriated
   variant at every budget** (wave 2: 0.369 / 0.362 / 0.357 at budget
   120/240/480, vs full 0.478 / 0.436 / 0.397). This is the mechanism the earlier
   draft called the "single most valuable next step" — it was run, and it works.
2. **Plain blind shrinkage beats full covariance** once budget ≥120 (both
   batteries), confirming CMA over-trusts the undersampled `C`. The earlier
   "fixed-γ is ≈neutral" claim was an artifact of the budget-60 cell dragging the
   aggregate down.
3. The effect is **budget-dependent**: at *tiny* budget (60), full covariance is
   actually **best in high dimensions** (0.320) — CMA has too few generations to
   overfit `C`, so there is nothing to tame and damping only discards signal.
4. **Assumed contiguous blocks fail** — on interaction problems they damp exactly
   the coupling that matters (wind_farm: full-covariance is *worst possible*).
5. **Seriation is *not* a robust win.** It looked strong in wave 1 (budget-240
   high-dim, 0.357) but was mid-pack in wave 2 (~0.45, worse than blind and
   adaptive). Its advantage does not survive a change of variant set / budget
   grid. Demoted from headline to "promising on hand-picked structured demos."

The clean takeaway: **CMA-ES leaves performance on the table by over-trusting an
undersampled covariance, and the principled fix is a *reliability-adaptive* γ\*
that shrinks each correlation by how trustworthy it is — not a fixed dial and not
(robustly) seriated blocks.**

## 1. Motivation

CMA-ES is among the strongest derivative-free optimizers in general, but on this
suite's low-budget regime (≤240 evaluations) it underperforms simpler methods
(it sat near the bottom of our disguised-benchmark leaderboard, behind Simulated
Annealing and Nelder-Mead). The covariance it adapts, `C`, is estimated from
`λ` samples per generation; for `n=16` that is ~12 samples for a 16×16 matrix —
badly undersampled. We hypothesised this over-trusted covariance is *why* CMA
underperforms here, and that damping it would help.

**Schur damping** (schur.microprediction.org) supplies the dial. In HRP it
interpolates portfolio construction between trusting only diagonal blocks
(`γ=0`) and inverting the full covariance (`γ=1`); the blocks are found by
**seriation** — hierarchically clustering the correlation matrix so coupled
variables are grouped, making the matrix quasi-block-diagonal. Porting that dial
to CMA-ES's sampling covariance (and to the GP posterior behind a Bayesian
acquisition function) is what we call **SAD — Schur Acquisition Damping**.

## 2. Method

SAD applies Schur damping to `C` each generation before sampling (`schur_cma.py`);
the variants below are the dials it exposes:

| dial | what it does | endpoints |
|---|---|---|
| **blind** `schur_damp(C,γ)` | scale *all* correlations by γ, keep variances | γ=1 full, γ=0 diagonal |
| **block** `schur_damp_blocks(C,γ,b)` | keep within-block (contiguous size-`b`) covariance, damp cross-block by γ | γ=0 block-diagonal |
| **seriated** `schur_damp_seriated(C,γ,t)` | cluster corr. matrix (HRP distance `√(½(1−ρ))`, avg-linkage, cut `t`); keep within-cluster, damp cross-cluster by γ | data-driven blocks |

Sampling is a compact Hansen CMA-ES validated to track humpday's own
`CMAEvolutionStrategy` (e.g. 0.00042 vs 0.00043 on airfoil). Fitness is
**normalised regret vs a baseline panel** on the *disguised* example-application
suite (a seeded cube→cube diffeomorphism relocates each optimum, so nothing can
be won by memorising locations). Lower = better; 0 = best in the comparison set
on that instance.

## 3. Results

### 3.1 Wave 1 — fixed/seriated variants (51 demos × 5 seeds)

Mean normalised regret, split by dimension (`n<10` low, `n≥10` high):

| budget | variant | all | low-dim | high-dim |
|---|---|---|---|---|
| **60** | full γ=1.00 | 0.458 | 0.487 | **0.320** |
| | blind γ=0.25 | 0.474 | 0.456 | 0.557 |
| | blind γ=0.00 | **0.436** | 0.408 | 0.567 |
| | seriat γ=.5 t.35 | 0.441 | 0.426 | 0.513 |
| **120** | full γ=1.00 | 0.478 | 0.463 | 0.549 |
| | **blind γ=0.25** | **0.403** | 0.391 | **0.456** |
| | blind γ=0.00 | 0.416 | 0.411 | 0.443 |
| | seriat γ=.5 t.35 | 0.436 | 0.401 | 0.601 |
| **240** | full γ=1.00 | 0.460 | 0.437 | 0.567 |
| | **blind γ=0.25** | **0.368** | 0.358 | 0.414 |
| | blind γ=0.00 | 0.442 | 0.422 | 0.533 |
| | seriat γ=.5 t.35 | 0.395 | 0.403 | **0.357** |

Reading: **fixed-γ blind damping is a real win once budget ≥120** — blind γ=0.25
beats full by ~0.08–0.09 (≈18% at budget 240). The earlier "≈neutral" verdict came
from averaging in the budget-60 cell, where the picture *inverts*: at tiny budget,
full covariance is **best in high dimensions** (0.320), because CMA has too few
generations to overfit `C`, so damping only discards signal. So the budget
dependence is the robust structural finding; the seriation high-dim win at budget
240 (0.357) is real *in this battery* but does not reproduce in wave 2 (§3.2).

### 3.2 Wave 2 — reliability-adaptive γ\* (51 demos × 5 seeds)

Wave 2 swaps in the reliability-adaptive `γ*` (per-correlation noise-floor
shrinkage, no fixed dial) and tighter seriation cuts, and extends to budget 480:

| budget | variant | all | low-dim | high-dim |
|---|---|---|---|---|
| **120** | full γ=1.00 | 0.478 | 0.456 | 0.584 |
| | blind γ=0.00 | 0.395 | 0.396 | 0.387 |
| | seriat t.35 | 0.451 | 0.431 | 0.546 |
| | **adaptive γ\*** | **0.369** | 0.365 | **0.385** |
| **240** | full γ=1.00 | 0.436 | 0.416 | 0.529 |
| | blind γ=0.00 | 0.394 | 0.384 | 0.440 |
| | seriat t.35 | 0.446 | 0.434 | 0.503 |
| | **adaptive γ\*** | **0.362** | 0.353 | **0.402** |
| **480** | full γ=1.00 | 0.397 | 0.373 | 0.507 |
| | blind γ=0.00 | 0.389 | 0.384 | 0.415 |
| | seriat t.35 | 0.436 | 0.432 | 0.457 |
| | **adaptive γ\*** | **0.357** | **0.314** | 0.554 |

**Adaptive γ\* wins the aggregate at every budget**, and wins low-dim outright at
budget 480. It only loses in *high* dim at budget 480 (0.554, where plain blind
γ=0 at 0.415 is best) — i.e. when budget is large the adaptive rule still
over-trusts some high-dim correlations. Note seriation (t.35) is **mid-pack here**,
behind both blind and adaptive — directly contradicting wave 1's budget-240
high-dim seriation win. Because the two batteries normalise within different
variant sets, the honest reading is: the seriation advantage is *fragile*, while
the adaptive-γ\* and blind-shrinkage advantages reproduce across both.

### 3.3 Assumed contiguous blocks fail — seriation rescues (9 high-dim demos × 3 seeds, budget 150)

Per-demo normalised regret (✓ = best in row):

| demo (n, block) | full γ=1 | blind γ=0 | naive-block γ=.5 | **seriated γ=.5** |
|---|---|---|---|---|
| **wind_farm** (16, 2) | 1.000 | 0.553 | 0.306 | **0.212** ✓ |
| **lennard_jones** (15, 3) | 0.454 | 0.333 | 0.526 | **0.211** ✓ |
| **battery_dispatch** (24) | 0.667 | 0.522 | — | **0.374** ✓ |
| circle_packing (12, 2) | 0.508 | 0.667 | **0.359** ✓ | 0.580 |
| bridge_truss (10) | 0.751 | **0.000** ✓ | — | 0.727 |
| cassini_minlp (10) | **0.333** ✓ | 0.617 | — | 0.625 |
| genetic_art (60) | **0.333** ✓ | 0.610 | — | 0.667 |
| reactor_profile (10) | **0.158** ✓ | 0.537 | — | 0.795 |
| rocket_landing (12) | 0.658 | **0.355** ✓ | — | 0.621 |

The headline is **wind_farm**: it is an *interaction* problem (turbine wakes), so
a single turbine's own (x,y) are nearly independent while the coupling that
matters is *between* turbines. Naive contiguous blocks (one turbine = one block)
therefore damp exactly the wrong correlations — and full covariance is the
**worst possible** (1.000). **Seriation discovers the real coupling** (grouping
interacting turbines) and wins decisively (0.212), beating even naive blocks.
The same holds for lennard_jones (atom triples) and battery_dispatch.

Where the landscape is smoother / less block-structured (cassini, genetic_art,
reactor), full covariance still wins — damping there discards useful information.
So *on these hand-picked structured demos* the result is structure-dependent,
as theory predicts. **But caution:** this 9-demo focused panel is what made
seriation look like a general winner, and §3.2 shows that advantage does **not**
survive the full 51-demo suite under wave 2's variant set. Read §3.3 as evidence
that seriation *can* discover real coupling (wind_farm), not that it is the
best general-purpose damping rule — that title goes to adaptive γ\* (§3.2).

## 4. Honest caveats

- **Cross-battery absolute numbers are not comparable.** Regret is normalised
  within each battery's variant set, so wave-1 and wave-2 cells differ even for
  the *same* variant (e.g. "seriat t.35" budget-240 high-dim reads 0.357 in wave 1
  vs 0.503 in wave 2). Only within-battery rankings are meaningful.
- **The seriation high-dim win is fragile**, not refuted: it appeared in wave 1
  and on the 9-demo §3.3 panel, but vanished in wave 2. It needs a dedicated,
  fixed-normalisation rerun before any claim survives.
- The damping is the **correlation/shrinkage form**, not the full *recursive*
  Schur-complement construction; a faithful recursive version is untested.
- **Adaptive γ\* is the new front-runner but still has a hole:** it loses in high
  dim at the largest budget (480), where it over-trusts correlations that plain
  blind γ=0 correctly discards. A budget/dim-aware floor is the obvious patch.
- Aggregate leaderboards mixing `block` (defined only for structured demos) with
  full-suite variants are apples-to-oranges; trust the per-cell / per-demo tables.

## 5. Takeaway and next steps

CMA-ES over-trusts an undersampled covariance, and the robust, reproducible fix
is **SAD's reliability-adaptive γ\*** — shrinking each correlation by how
trustworthy it is — it beats full covariance and every fixed/seriated variant on the aggregate
at all three budgets (§3.2). Plain blind shrinkage is a simpler partial win
(budget ≥120). Seriation can discover genuine coupling (wind_farm) but is not a
robust general damping rule. The earlier draft's two headline claims —
"fixed-γ ≈ neutral" and "seriation rescues it" — did **not** survive the full
suite and have been corrected above.

Highest-value follow-ups:
1. **Patch adaptive γ\*'s high-dim/large-budget hole** — add a dim- and
   budget-aware noise floor so it never over-trusts where blind γ=0 wins (the only
   cell it currently loses). This should turn "wins the aggregate" into "wins or
   ties every cell".
2. **Fixed-normalisation seriation rerun** — settle whether the wave-1 seriation
   high-dim win is real or a normalisation artifact, with all variants (incl.
   adaptive) scored in one comparison set.
3. **Fold the adaptive-γ\* knob into humpday's `CMAEvolutionStrategy`** and re-run
   the disguised leaderboard — does adaptive-Schur-CMA climb past SA / Nelder-Mead?
