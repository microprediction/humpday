# Ideas backlog — discovering better black-box optimizers

Context: `algo_dev.py` evolves a 12-gene DE/ES-hybrid genome; fitness = normalised
regret vs a baseline panel across *disguised* real-world demos. Two structural
limits became clear this session:

1. **Mutation-only ES stalls** at the warm start (deterministic fitness, no way to
   combine ideas across genomes). → Fixed with a `(μ+λ)` GA + crossover.
2. **The genome template caps what's discoverable.** All 12 knobs are DE/ES
   parameters. The candidate has *no curvature model* — its smartest moves are DE
   difference vectors and isotropic Gaussian steps. On the smooth, low-to-moderate
   dim engineering demos (airfoil, gear ratios, PID, lens, pressure vessel…) the
   panel's CMA-ES / NelderMead win precisely because they exploit curvature. To
   *beat* them we likely need new mechanisms, not just re-tuned old ones.

## A. New mechanisms to add to the candidate template (combinable "ideas")

Ranked by expected leverage on this demo suite.

1. **Local quadratic trust-region jump (separable surrogate).** ⭐ highest leverage.
   Fit a diagonal quadratic `f ≈ c + Σ bᵢxᵢ + Σ aᵢxᵢ²` to the nearest ~2n+1
   evaluated points, jump to its per-coordinate minimizer `xᵢ* = -bᵢ/(2aᵢ)`,
   clipped to an adaptive trust radius. A "poor-man's NEWUOA" — O(n) cost, no
   matrix inversion, gives the candidate the curvature exploitation it lacks.
   Prototyped in `candidate_v2.py`.
2. **Per-coordinate adaptive step (sep-CMA-lite).** Replace the isotropic local
   Gaussian with per-coordinate σᵢ learned from successful step magnitudes.
   Handles axis-aligned ill-conditioning cheaply (no full covariance).
3. **Low-discrepancy init (Halton/Sobol).** Replace uniform random init with a
   Halton sequence — better cube coverage, matters most at low trial budgets.
   Pure-Python Halton is ~10 lines.
4. **Opposition-based learning.** For each random point also consider its
   reflection `1 - x`; cheap DE booster, helps escape bad init.
5. **JADE/SHADE external archive.** Keep replaced solutions; draw difference
   vectors from population ∪ archive for extra diversity without extra evals.
6. **IPOP-style restart.** Grow population (and reset σ) on stagnation instead of
   only reinitialising a fraction — the canonical fix for premature convergence.
7. **Final elite polish.** Spend the last ~5% of budget on coordinate line search
   around the incumbent.

## B. Meta-level search (search over genomes)

- **Dogfood CMA-ES as the meta-optimizer.** The genome is `[0,1]¹²` and
  `candidate_fitness` is a black-box `[0,1]¹² → regret`. Just call humpday's own
  `CMAEvolutionStrategy` on it — elegant, and CMA-ES is ideal for ~12-dim
  continuous tuning. Compare against the GA. Prototyped wrapper idea in
  `meta_cmaes.py` (TODO).
- **Island model** — several sub-populations + occasional migration to keep
  diversity and dodge the warm-start basin.
- **Multi-objective** — optimise (mean regret, robustness=variance across
  instances) jointly; surface a Pareto front of specialist vs generalist
  optimizers instead of one scalar winner.

## C. Fitness / methodology (make wins real, not overfit)

- **Train/validation split.** ⭐ important. There is currently NO held-out set —
  a "winning" genome may just be tuned to these 15×2 = 30 instances. Evolve on a
  train split (demos × seeds), report final fitness on *held-out demos and unseen
  disguise seeds*. Without this we can't claim a discovered optimizer generalises.
- **Anytime / budget-aware fitness.** Score at several budgets (30/100/200
  trials) so we don't overfit to exactly-100-trial behaviour.
- **Log-regret or "fraction-of-panel-beaten".** The current min-max regret is
  sensitive to panel spread on a given instance; a rank-fraction or log metric is
  more robust.

## E. Schur-damping for covariance-based optimizers (CMA-ES, BayesOpt)

Source: schur.microprediction.org. Schur damping is a dial **γ∈[0,1]** that
interpolates between using only diagonal/block covariance (γ=0, robust under
noise) and the full inverse covariance (γ=1), by augmenting each block's
covariance with the **Schur complement** of the others. A reliability parameter
**γ\*** sets the optimal damping when the covariance is *undersampled*. Born in
portfolio construction (HRP↔MVP), but the mechanism is generic to any
covariance/precision matrix.

Two concrete optimizer connections worth testing:

1. **CMA-ES — reliability-aware sep↔full dial.** ⭐ CMA-ES adapts a covariance
   `C` that is noisily estimated precisely when the population λ is small vs the
   dimension n — its known weak regime (and exactly the low-budget DFO setting
   here). Today the choice is binary: `sep-CMA` (diagonal, robust, cheap) vs full
   covariance. Schur-damping replaces the either/or with a principled γ-dial over
   coordinate blocks, with **γ\* driven by an undersampling/reliability estimate**
   (λ vs n, condition number of `C`). Hypothesis: a Schur-damped CMA-ES beats both
   sep-CMA and full-CMA in the low-λ / moderate-n band the disguised demos cover.
2. **BayesOpt (GP) — principled jitter from sparse data.** The GP posterior
   covariance *is* a Schur complement (`K** − K*ₓ Kₓₓ⁻¹ Kₓ*`). Early BayesOpt has
   very few points → `Kₓₓ` ill-conditioned → practitioners add an ad-hoc nugget.
   Schur-damping gives a principled γ\* for that conditioning, tied to how
   undersampled the kernel matrix is. The source page notes the same damping
   appears in pseudo-likelihood GP covariance estimation — direct support.
3. **Others / connection to the repo's preconditioner theme.** The cube→simplex
   θ-preconditioner is itself a "damping dial"; worth asking whether γ and θ are
   the same knob in different clothes. Any method with a precision matrix
   (some surrogate steps in `candidate_v2`) could take a γ.

Test plan: add a γ knob to humpday's CMAEvolutionStrategy (Schur-damped block
covariance), sweep γ∈{0,0.25,0.5,0.75,1} + a reliability-set γ\* on the disguised
suite at several (n_dim, λ, n_trials) points; same for a damped GP nugget in
BayesOpt. Win condition: the γ\* variant dominates both endpoints across the band.

## D. Quick experiments to run (after the live GA finishes, to avoid CPU contention)

1. Solo head-to-head: base candidate vs `candidate_v2` (surrogate) vs panel best,
   on the smooth demos. Does the surrogate jump actually win where we expect?
2. GA with the richer (surrogate-enabled) template — more genes, more ideas to
   recombine. Does crossover exploit the new mechanisms?
3. Meta-CMA-ES vs GA on the same genome space — which finds lower regret per
   genome-eval budget?
4. Re-score the GA winner on a held-out demo split — does it generalise?

## Signed compositions: thinking outside the simplex (user idea, 2026-07-06)

Extend barycentric recipes to signed/affine ones: weight -1 on algorithm A
and +2 on B means run B (and A in shadow) but steer AWAY from where A's
logic would sample — A becomes an anti-inspiration. Mechanically: the
generated program runs A's proposal logic without spending budget on it and
applies a repulsion penalty (or rejection radius) around A's suggestions.
Relatives: tabu search (avoid visited regions), repulsive PSO, negative
correlation in ensembles, DPP diversity. The coordinate space becomes a
larger polytope containing the simplex; the simplex is the all-positive
face. Slot semantics: a negatively-weighted vertex owns "avoidance" slots
rather than contributing mechanisms. Open questions: normalisation (sum of
absolute weights?), whether anti-inspiration helps only on multimodal
objectives, and whether the E6-style validation shows any signed recipe
beating the best unsigned one.
