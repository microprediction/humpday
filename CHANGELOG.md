# Changelog

Notable changes to `humpday`. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

Benchmarking was three artifacts, built by three scripts at three times over different objectives,
feeding different parts of the library. `minimize` and `suggest` could disagree about the same
problem because they read different tables. This is one table, one aggregation, one consumer path.

### Changed — behaviour

- **One rating artifact, indexed by dimension, budget and suite.** `humpday/data/ratings.json`,
  recorded by `benchmarks/record_ratings.py`. Budget joins the index because it changes the answer
  as much as dimension does: NEWUOA and BOBYQA want roughly `2n+1` points before proposing
  anything and UOBYQA wants `(n+1)(n+2)/2`, which is 5,151 at a hundred variables, so a budget that
  is generous at ten is nothing at a hundred. Budget is interpolated downward only; dimension is
  never interpolated.
- **`suggest()` and `minimize()` read the same table.** `humpday.ratings.robust_order` is the one
  ordering function, and `eligibility.recommend` consults it before the Borda grid. They were
  written separately and were free to drift.
- **`suggest()` reports no score under the default.** Ranking by worst position across two suites
  is not a rating, and Elo from two tournaments is not on one scale. Passing `smooth=True` or
  `smooth=False` selects one suite and returns its measured ratings, as before.
- **An optimizer that cannot return is ranked last, not silently dropped.** Each run is capped at a
  wall clock set to the larger of 25x what the objective's own evaluations cost and 20 ms per
  evaluation, so the allowance follows the problem and a disqualification is a finding about the
  method. Two overruns in a cell disqualify it there, recorded under `timed_out` and readable via
  `humpday.ratings.timed_out`.

### Added

- `humpday.ratings` — the one place ratings are read: `robust_order`, `suite_order`, `rating`,
  `timed_out`, `cells_at`, `recorded_dimensions`.
- `humpday.problems_recorded()` — how many problems back each cell, since that is what makes a
  rating worth anything and what the rank shrinkage weights by.
- `benchmarks/record_ratings.py` — shards one file per cell under `benchmarks/ratings_cells/`, runs
  cells in parallel, resumes rather than restarts, and skips a cell already deep enough.

### Removed — breaking

- `humpday.elo_ratings()`. It read a sweep of sphere and Rosenbrock variants in **two dimensions**,
  which is one of the relics deleted here. There is no honest one-number replacement: that ratings
  depend on dimension, budget and suite is the finding, not an inconvenience.
- `benchmarks/elo_ratings.json`, `humpday/data/elo_ratings.json`, `humpday/data/elo_by_dimension.json`,
  `benchmarks/record_elo.py`, `benchmarks/record_elo_by_dimension.py`.
- `humpday.elo_by_dimension()` is **deprecated**, not removed: it is now a view on the one table and
  can only show one budget at a time.

## [0.23.0] — 2026-09-13

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
