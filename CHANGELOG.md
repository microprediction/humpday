# Changelog

Notable changes to `humpday`. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.23.0] — unreleased

The theme is that the recommenders claimed more evidence than they had. Several of these change
what an existing call returns, so read Changed before upgrading.

### Changed — behaviour

- **`suggest()` no longer returns invented numbers.** It returned `score = 1000 + 100*i` and
  `time = 0.1*i`, both computed from a position in a list and labelled "Fake ... for compatibility"
  in the source, while arriving at the caller as `(score, time, name)` — the shape of a
  measurement. `score` is now a measured Elo rating, or `nan` where none exists. `time` is always
  `nan`: humpday records no timing evidence, and inventing one was the larger part of the problem.
  The tuple shape is unchanged.
- **`suggest()` takes a `smooth` argument.** `True` ranks on morphed analytic surfaces, `False` on
  the engineering demos, `None` (default) by *worst* rank across both, so the leader is the
  optimizer that is never terrible rather than one that wins a suite and collapses on the other.
  `PRIMA_UOBYQA` averages rank 3.0 on surfaces and 15.9 on the demos; that is the recommendation
  the default exists to avoid.
- **`minimize()` picks differently.** It routes through `eligibility.recommend`, which now consults
  a per-dimension tournament before the surface-ranked grid. Measured on the engineering demos, old
  pick against new, one problem at a time: better on 24, worse on 15, tied on 2. Eligibility
  filters are unchanged, so nothing unsuitable for the dimension or budget became reachable.
- **Ratings are never stretched across dimensions.** A rating measured at one dimension is evidence
  about that dimension only. Optimizer performance is not smooth in dimension: a Bayesian method can
  work well at five and blow up at ten. Dimensions with no tournament fall back to the hand-written
  ordering and report no ratings.

### Added

- `humpday.objectives.SURFACES` — one curated list of classic formulaic surfaces, screened so that
  every member is defined at any dimension and fast enough to race.
- `humpday.objectives.physics_objectives()` — the engineering demos under `example_applications`,
  each at the dimension its problem actually has.
- `humpday.objectives.morphed_surfaces()` — randomly shifted, rotated, rescaled and noised
  instances, for tournaments where a fixed landscape could otherwise be learned.
- `humpday.elo_ratings()` and `humpday.elo_by_dimension()` — the measured tables, which now ship
  inside the package rather than sitting in `benchmarks/` where nothing could read them.
- `benchmarks/record_elo_by_dimension.py` — records per dimension and per suite, checkpointing
  after every problem so an interrupted run keeps what it has paid for.

### Removed — breaking

- `humpday.objectives.bbob_inspired_suite` — hard-coded to two dimensions; every function raised
  above `d=2`.
- `humpday.objectives.enhanced_surfaces`, `humpday.objectives.enhanced_surfaces_working` — two
  copies of a wrapper around `opfunu`, which is not a dependency and was not installed, so every
  call silently took a fallback path. Nothing imported the second copy.
- `humpday.analysis.advanced_categorization`, `humpday.transforms.zcurves`,
  `humpday.transforms.zcurvehelper` — 861 lines with no reference anywhere in the package, the
  tests, the docs or the JS port. `humpday/analysis` had no `__init__.py` and was never importable.

### Fixed

- Module-level `dict | None` annotations in `humpday/__init__.py` raised `TypeError` on Python 3.9,
  which `requires-python` still allows.
- `test_prima_vs_real_prima` failed rather than skipped when `pdfo` is installed but built against
  a different numpy ABI, which is what anyone on numpy 2 hits.
- `ruff` is pinned to a minor series. An open bound on a *formatter* means CI starts failing on
  untouched files whenever a new style ships, which is how it came to be red for two days.

### Deprecated

- `humpday.objectives.allobjectives` — unscreened, and several members read only their leading
  coordinates whatever the dimension. Kept because `papers/planar_search` reports results over
  exactly that collection. Prefer `SURFACES`.
