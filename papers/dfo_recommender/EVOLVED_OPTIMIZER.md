# Discovering a black-box optimizer by evolution against a memorisation-proof benchmark

**Status: base-template GA complete (`runs/ga_20260616_135951.json`): best
normalised regret 0.191, mean rank 1.73 vs the NM/DE/CMA panel. A follow-up added
a separable quadratic trust-region jump to the template (now 14 knobs); see §6.**

## 1. The question

Can we *discover* a derivative-free optimizer — rather than hand-design one — by
searching a space of optimizer behaviours against a benchmark that an optimizer
cannot game? The HumpDay suite gives us ~30 real-world objectives (airfoil shape,
gear ratios, PID tuning, lens design, pressure-vessel weight, rocket landing,
Lennard-Jones clusters, …). The discovered optimizer should be competitive with,
or beat, a panel of established baselines (NelderMead, Differential Evolution,
CMA-ES) across all of them.

## 2. The space of optimizers (the "genome")

Rather than search arbitrary code, we parameterise a single **unified DE/ES-hybrid
template** with 14 behavioural knobs (12 original + 2 added in §6), each in [0,1].
A genome decodes to a concrete optimizer that blends, in tunable proportion:

| knob | meaning |
|---|---|
| `pop` | population size (4–30) |
| `F0`, `CR0` | base DE differential weight / crossover rate |
| `ctb` | P(current-to-best/1) vs rand/1 DE move |
| `p_local` | P(local refinement move) vs DE move |
| `sigma0`, `adapt_sigma` | local step size + 1/5-success-rule adaptation |
| `p_pattern` | within a local move, P(coordinate pattern step) vs Gaussian |
| `adapt_fcr` | SHADE-style self-adaptation of F/CR from successful trials |
| `temp0` | simulated-annealing acceptance temperature (0 = greedy) |
| `stagnate`, `restart_frac` | restart trigger + fraction reinitialised |
| `p_surrogate` | P(a generation fires a quadratic trust-region jump) — §6 |
| `r2_min` | min model R² to trust that jump (gates it off on bad fits) — §6 |

The template deliberately spans several algorithm ideas (DE, ES, pattern search,
simulated annealing, SHADE adaptation, restarts) so that *combinations* of them
are reachable. A single genome is therefore a recipe for "how much of each idea,
and how they interact."

## 3. The discovery loop

A **(μ+λ) genetic algorithm** evolves a population of genomes:

- **Diverse initialisation** — a couple of copies of a sane warm-start genome plus
  random genomes, so there are genuinely different strategies to recombine.
- **Crossover** — gene-by-gene, each knob is inherited wholesale from one parent
  (discrete recombination: a child can take the SA-temperature of one parent and
  the restart policy of another intact) or as an arithmetic blend.
- **Mutation** — per-gene Gaussian perturbation with a self-adapting step.
- **Elitist survival** — the best μ of parents+offspring carry forward.
- Fitness is deterministic, so it is cached: surviving parents are never
  re-evaluated.

> An earlier mutation-only (1+λ) evolution strategy **stalled** at the warm start
> (regret flat at 0.337 for 6 generations) — with no crossover it could only jitter
> one genome and never *combine* good traits. Switching to the crossover GA broke
> the plateau immediately (see §5).

## 4. Why the benchmark can't be gamed (disguised instances)

If we scored optimizers on the raw objectives, a meta-search could "win" by
memorising where each optimum sits. To prevent that, every objective is wrapped in
a **seeded cube→cube diffeomorphism** (`humpday.transforms.cube_disguise`): the
landscape's difficulty, critical-point structure and optimal value are preserved,
but the optimum is relocated to an instance-specific, unpredictable point. Each
problem becomes several disguised instances. An optimizer can only do well by
*genuinely searching*, not by remembering locations.

**Fitness** = normalised regret vs the baseline panel, averaged over all disguised
instances (0 = best on every instance, 1 = worst). It is continuous (not integer
rank) so the search sees a gradient even when a candidate is close behind the panel.

## 5. What evolution found (in progress)

Regret trajectory of the best genome:

| generation | 0 | 1 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| best regret | 0.337 | 0.321 | 0.305 | 0.223 | 0.223 |

A 34% regret reduction by generation 5, with the whole population converging
downward. The gen-3→4 drop is a crossover win — recombining knobs from two genomes
produced a strategy neither parent had.

**The current best optimizer's behaviour** (gen 5; decoded knobs):

> Explore broadly with **rand/1 DE** (`ctb≈0.09`, almost no greedy current-to-best),
> refine heavily **along coordinate axes with pattern search** (`p_local≈0.6`,
> `p_pattern≈0.85`), accept **uphill moves via simulated annealing** (`temp0≈0.24`),
> and **restart half the population** after 10 stagnant generations
> (`restart_frac≈0.53`). Moderate population (16), aggressive DE (`F0≈0.69`),
> SHADE-style F/CR adaptation on.

A notable detail: the simulated-annealing acceptance (`temp0>0`) only became
*reachable* after a bug fix this session — the template used `math.exp` without
importing `math`, so every SA-enabled genome silently raised and scored
worst-possible. The whole SA axis was unexplored before. Evolution has now selected
a non-zero temperature, i.e. the discovered optimizer actively relies on a behaviour
that was dead code hours ago.

## 6. Is this "a new algorithm"? — honest framing

What evolution yields here is a **tuned configuration of an existing hybrid
template** — a new optimizer *instance*, not a new algorithm *family*. That is
useful (a single configuration that beats the panel across 30 disparate real
problems is a real result) but should not be oversold.

The genuinely *novel-mechanism* work added a **local separable-quadratic
trust-region jump** — a curvature model the base template lacked (its smartest
moves were DE difference vectors and isotropic Gaussian noise). On the smooth
engineering problems the panel's CMA-ES/NelderMead win precisely by exploiting
curvature, so giving the candidate a cheap (O(n), R²-gated) NEWUOA-like model step
is the most principled route to beating them.

**It earned its keep, and is now folded into the main template** (`make_candidate`,
genes `p_surrogate`/`r2_min`). Evolving the 14-gene template (`evolve_v2.py`,
`runs/v2.json`) reached **0.112** regret vs the base-template GA's **0.191** at the
same scale. A clean ablation on the evolved genome — same genome, only `p_surrogate`
toggled — isolates the mechanism:

| `p_surrogate` | regret |
|---|---|
| 0.625 (evolved) | **0.112** |
| 0.000 (jump off) | 0.238 |
| 1.000 (always)  | 0.129 |

Disabling the jump **more than doubles regret**, and — tellingly — leaves the
genome (0.238) *worse than the base GA* (0.191): the base knobs **co-adapted** to
rely on the surrogate rather than treating it as a bolt-on. Always-firing (0.129)
is worse than the evolved ~0.62 probability, so the R²-gate + probabilistic firing
matter. The benefit is therefore unlocked by *evolving with* the surrogate gene,
not by hardcoding a jump — which is why it is a gene, and why the warm-start that
hasn't co-adapted shows only a ~neutral on/off difference. **Scope:** measured on
the disguised real-world suite at 100 trials; per the Nevergrad benchmark-validity
check, do not assume the same gain on smooth synthetic functions.

## 7. Limitations / next steps

- **No train/test split yet.** Fitness is measured on the same 15 demos × 2 seeds
  the GA trains on, so the winning genome could be overfitting those 30 instances.
  Before claiming generalisation, re-score the winner on *held-out demos and unseen
  disguise seeds*. (Highest-priority methodological fix.)
- **Budget specificity.** Fitness is measured at exactly 100 trials; an anytime /
  multi-budget fitness would avoid overfitting to one budget.
- **Surrogate folded in** (§6) — done. Next: re-run the main `algo_dev --mode ga`
  with the 14-gene genome to confirm it re-discovers the co-adapted win end-to-end.
- **Template ceiling.** If both GAs plateau short of beating the panel, the answer
  is to add mechanisms to the template (curvature model, low-discrepancy init,
  JADE/SHADE archive — see `IDEAS.md`), not to keep re-tuning the existing knobs.
