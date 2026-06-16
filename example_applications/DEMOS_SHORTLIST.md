# New demo shortlist — landscape pathologies to fill

Candidate `example_applications` demos surfaced by a deep-research sweep of real-world
derivative-free / black-box optimization (26 sources, 24 verified claims). Each targets
a **landscape pathology the existing suite under-samples**. Ordered by build priority.

Status: ✅ built · 🚧 in progress · ⬜ todo · 🔬 deferred (scope risk / wrong home)

| # | Demo | Domain | Pathology (the gap) | n | Known optimum? | Status |
|---|------|--------|---------------------|---|----------------|--------|
| 1 | `multi_exponential_fit` | spectroscopy / pharmacokinetics | **Ill-conditioned curving "sloppy" valley** — Jacobian cond. number → ∞ as two rates converge; many distinct fits look identical | 4 | ≈ noise floor at true params (± label-swap) | ✅ |
| 2 | `speed_reducer` | mechanical design (Golinski gearbox) | **Nonconvex + 11 nonlinear constraints + mixed-integer** (pinion teeth); metaheuristics can't *guarantee* the optimum | 7 | yes — weight ≈ **2994.4712** | ✅ |
| 3 | `gear_ratios` | power transmission | **Allocation on a (log-)simplex** — stage ratios must multiply to N; balanced split is optimal | 4 | yes — equal ratios `N^(1/n)` | ✅ |
| 4 | `transfer_window` | aerospace (interplanetary Δv) | **Disjoint feasible islands / multi-basin** with brutal intra-basin scale (porkchop plot) — basin-hopping/CMA territory, local search dies | 4–6 | best-known published (GTOPX) | ⬜ |
| 5 | `cassini_minlp` | aerospace (flyby sequence) | **Mixed-integer + razor-thin deceptive tie** (local 3.6307 vs global 3.5007) | 6+4 | best-known 3.5007 | ⬜ |

## Deferred / different home

- **`schwefel`, `step_function` (De Jong f3)** — classic *analytic* benchmarks (deceptive
  far-optimum; flat plateaus). Real pathologies, but `example_applications` is explicitly
  "not the classical analytic benchmarks" — these belong in a benchmark/objectives suite,
  not here. Logged for that purpose. 🔬

## Still-open gap

- **Heteroscedastic noise** (noise magnitude varying across the domain) — the research did
  not surface a citable real-world case. A synthetic `sensor_placement` demo could fill it,
  but it would be invented rather than ported. ⬜

## Sources
Deep-research report (cited): GTOPX (arXiv 2010.07517), NASA NTRS 20150020817, ESA GTOP;
Sethna/Cornell "Fitting Exponentials" + Nature Sci Rep s41598-022-08638-7; Golinski
speed-reducer (arXiv 2505.03512); PLOS ONE pone.0198048 (gear trains); Naser 2025 WIREs
(doi:10.1002/wics.70028, 315-function catalog).
