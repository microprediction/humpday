# New demo shortlist — landscape pathologies to fill

Candidate `example_applications` demos surfaced by a deep-research sweep of real-world
derivative-free / black-box optimization (26 sources, 24 verified claims). Each targets
a **landscape pathology the existing suite under-samples**. Ordered by build priority.

Status: ✅ built · 🚧 in progress · ⬜ todo · 🔬 deferred (scope risk / wrong home)

| # | Demo | Domain | Pathology (the gap) | n | Known optimum? | Status |
|---|------|--------|---------------------|---|----------------|--------|
| 1 | `multi_exponential_fit` | spectroscopy / pharmacokinetics | **Ill-conditioned curving "sloppy" valley** — Jacobian cond. number → ∞ as two rates converge; many distinct fits look identical | 4 | ≈ noise floor at true params (± label-swap) | ✅ |
| 2 | `speed_reducer` | mechanical design (Golinski gearbox) | **Nonconvex + 11 nonlinear constraints + mixed-integer** (pinion teeth); metaheuristics can't *guarantee* the optimum | 7 | yes — weight ≈ **2994.4712** | ✅ |
| 3 | `gear_ratios` | mechanical design (Sandgren gear train) | **Discreteness / plateaus** — objective is piecewise-constant on the integer tooth lattice; no gradient to follow | 4 | yes — `f≈2.7e-12` at teeth (19,16,43,49) | ✅ |
| 4 | `transfer_window` | aerospace (Earth→Mars Δv) | **Disjoint feasible islands / multi-basin** (porkchop) — local search sees only its own launch window | 2 | yes — Δv≈0.188 (= analytic Hohmann; validated) | ✅ |
| 5 | `cassini_minlp` | aerospace (flyby sequence) | **Mixed-integer + deceptive near-tied sequences** — methods disagree on the discrete flyby planets | 6+4 | reduced-order (structure, not GTOPX's exact 3.5007) | ✅ |

## Wave 2 — broadening the benchmark (toward goal-based algorithm development)

A second batch built to grow a large, diverse real-world suite. All ✅ built, pure-Python,
auto-collected, verified against known optima where one exists.

| Demo | Domain | Pathology | n | Known optimum? |
|------|--------|-----------|---|----------------|
| `pressure_vessel` | mechanical | constrained, gauge-quantised mixed-integer | 4 | ✅ ≈ 6059.7 |
| `tension_spring` | mechanical | constrained; tiny objective vs O(1) violations | 3 | ✅ ≈ 0.01267 |
| `cantilever_beam` | structural | single curved constraint, optimum on the boundary | 5 | ✅ ≈ 1.340 |
| `lennard_jones_cluster` | physics | extreme multimodality (≈exp many minima) | 15 | ✅ LJ₅ ≈ −9.104 |
| `facility_location` | operations | non-smooth `min`, multimodal assignment | 6 | computed |
| `kmeans_clustering` | ML | multimodal (k-means local minima) | 6 | computed |
| `enzyme_kinetics` | biochem | mildly ill-conditioned curved valley | 2 | ✅ noise floor |
| `darts_aim` | games / decision-under-noise | deterministic multimodal expected-score | 2 | computed (Tibshirani) |
| `pid_tuning` | control | ill-conditioned valley + instability cliff | 3 | computed |
| `economic_dispatch_valve` | power systems | valve-point ripples (multimodal, non-smooth) + equality | 3 | ref ≈ 8234 |
| `inventory_policy` | operations | stochastic / noisy objective (expected-cost bowl) | 2 | computed |
| `kalman_tuning` | estimation | ill-conditioned diagonal (q/r) valley, log-scaled | 2 | computed |
| `radiation_therapy` | medical physics | simplex allocation + competing dose objectives | 4 | computed |
| `sensor_localization` | robotics / signal | multimodal flip/fold ambiguities | 8 | noise floor |

**Disguise harness:** `humpday/transforms/cube_disguise.py` wraps any objective in a seeded
random cube→cube diffeomorphism, relocating the optimum per seed so algorithm development
against these demos can't memorise optimum *locations*. Wired into the benchmark via
`papers/dfo_recommender/example_demos.py`: `disguised_demos()` expands the 52 problems into
260 non-memorisable instances (5 seeds each by default); `disguise_demo(demo, seed)` for one.
Run `python papers/dfo_recommender/example_demos.py` to see the relocation verification.

## Deferred / different home

- **`step_function` (De Jong f3)** — the flat-plateau pathology is now covered by
  `gear_ratios` as a *real application* (discrete integer gear train), which fits the
  suite's "not the classical analytic benchmarks" ethos. The raw analytic step function
  is superseded here; keep it for a pure-benchmark/objectives suite if ever wanted. 🔬
- **`schwefel`** — classic *analytic* deceptive-optimum benchmark (global min far from the
  next-best minima, no guiding slope). Real pathology, but analytic, so it belongs in a
  benchmark/objectives suite rather than `example_applications`. Logged. 🔬

## Still-open gap

- **Heteroscedastic noise** (noise magnitude varying across the domain) — the research did
  not surface a citable real-world case. A synthetic `sensor_placement` demo could fill it,
  but it would be invented rather than ported. ⬜

## Sources
Deep-research report (cited): GTOPX (arXiv 2010.07517), NASA NTRS 20150020817, ESA GTOP;
Sethna/Cornell "Fitting Exponentials" + Nature Sci Rep s41598-022-08638-7; Golinski
speed-reducer (arXiv 2505.03512); PLOS ONE pone.0198048 (gear trains); Naser 2025 WIREs
(doi:10.1002/wics.70028, 315-function catalog).
