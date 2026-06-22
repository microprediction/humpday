# Weekend experiment suite (designed 2026-06-18)

Goal: run unattended Fri-eve → Mon, addressing the highest-value open questions from
this session. **Every experiment writes atomic per-step JSON checkpoints** (power-
outage lesson) and a top-level `runs/weekend_manifest.json` records which finished, so
a crash/outage resumes instead of restarting. A `weekend_runner.py` runs Tier A
sequentially (CPU-only, free, can run for days); Tier B (LLM/Opus, costs money) is
opt-in and budget-capped.

## Tier A — CPU-only backbone (no API spend, safe to run freely)

### E1 — Benchmark-validity rank correlation  ⭐ the paper spine
**Hypothesis:** the optimizer leaderboard on *synthetic analytic* functions has low
Kendall-τ with the leaderboard on *disguised real-world* objectives — i.e. synthetic
benchmarks mis-rank optimizers for real problems.
**Method:** panel = 22 humpday optimizers + ngCMA(pycma). Synthetic suite = nevergrad
`ArtificialFunction` (sphere/ellipsoid/cigar/rastrigin/rosenbrock, rotated) at dims
{5,20,40}; real suite = disguised demos (dim-spread subset, ~30 of 66). For each
(suite, budget∈{60,120,240}): per-instance normalised scores → per-optimizer mean
rank → leaderboard. Report **Kendall-τ + Spearman between the synthetic and real
leaderboards per budget**, plus per-dim-bucket. Low τ = headline result.
**Output:** `runs/rankcorr.json`. **New harness:** `rankcorr.py`. **~½ day.**

### E2 — Discovered optimizers, hardened (multi-budget, held-out)
**Hypothesis:** the three "discovered" optimizers — the LLM **centroid**, the evolved
**algo_dev surrogate** genome, and the **best unstructured** Opus optimizer — beat the
classic panel on held-out demos *across budgets*, and we can characterise where each
wins.
**Method:** held-out demos (complement of all selection sets) × seeds 0-4 × budgets
{60,120,240,480} × panel {NM, DE, CMA, ngCMA, centroid, surrogate, unstructured}.
Mean rank + win-rate per (budget, dim-bucket). Generalises `centroid_eval.py`.
**Output:** `runs/discovered_hardened.json`. **~½ day.**

### E3 — Crossover / blend harness  (Phase 5 of the ask/tell work)
**Hypothesis:** runtime-blended optimizers built on the new ask/tell interface beat
their individual components on the disguised suite.
**Method:** using `suggest_next`/`receive_update`, build (a) **round-robin interleave**
of optimizer pairs sharing one budget + incumbent, and (b) a **k-portfolio** that
splits budget across k optimizers and seeds each from the shared best. Compare blends
vs their components × disguised demos × seeds. This is both an experiment and the
Phase-5 deliverable.
**Output:** `runs/crossover.json`. **New harness:** `crossover_harness.py`. **~½ day.**

### E4 — SAD adaptive-γ*, fixed-normalisation rerun on enriched high-dim
**Hypothesis:** adaptive γ* beats full/fixed-blind/seriated across budget×dim when ALL
variants are scored in one comparison set on the enriched high-dim demos; and the
wave-1 seriation win does not reproduce (settles the SCHUR_RESULTS fragility caveat).
**Method:** reuse `overnight_schur*` with one unified variant set incl. adaptive, on
the now-17-strong n≥16 pool, budgets {120,240,480}. **Output:** `runs/schur_final.json`.
**~few hrs.**

### E5 — Surrogate template generalisation (train/test split)
**Hypothesis:** a 14-gene genome evolved on demo-set A still wins on held-out demo-set
B — the surrogate template generalises (closes the EVOLVED_OPTIMIZER.md §7 no-split
gap). **Method:** GA-evolve on A (seeds 0,1), score the winner on disjoint B (seeds
2,3,4) vs the panel. **Output:** `runs/surrogate_generalisation.json`. **~few hrs.**

## Tier B — LLM/Opus (bounded, opt-in, costs money)

### E6 — Re-warm-started simplex (around the centroid) + matched-sample ablation
**Hypothesis:** warm-starting the simplex search at the *centroid* region (last run's
warm start was misdirected at the overfit PatternSearch corner) finds blends below
0.0866; and a **matched** best-of-15 unstructured vs best-of-15 simplex settles whether
the blend structure adds value beyond more shots-on-goal.
**Method:** `simplex_blend.py --points warm` with `WARM_CENTER` re-pointed to the equal
centroid; `ablation_unstructured.py --attempts 15`. **~40 Opus calls (~$).**
**Output:** `runs/simplex_centroidwarm.json`, `runs/ablation_matched.json`.

### E7 — Multi-seed simplex regret surface (lower priority)
Repeat the simplex map over disguise seeds {0,1,2} to get variance bars on the
interior-vs-vertex finding. **~50 Opus calls.**

## Orchestration & safety
- `weekend_runner.py` runs **E1→E5 in sequence** (one at a time — avoids CPU thrash;
  each is itself internally parallel where cheap). Writes `runs/weekend_manifest.json`
  after each finishes; on restart it skips completed experiments.
- All runs use the repo venv and write under `papers/dfo_recommender/runs/`.
- Tier B launched separately only if opted in, with the `--attempts`/points caps above
  bounding spend.
- Each experiment prints a final summary block to its `.log` for quick Monday triage.

## Priority order
E1 (paper spine) → E3 (new capability) → E2 (hardening) → E4 → E5; Tier B if opted in.
