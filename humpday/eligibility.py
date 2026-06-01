"""
Algorithm eligibility + auto-recommendation for humpday.minimize().

Two filters decide which optimizers can sensibly run a given problem:

  1. **Dimensional cap** — some optimizers are structurally unfit above a
     certain n_dim (GridSearch is exponential, BayesianOpt's GP is
     cubic-in-observations and the RBF kernel degenerates, UOBYQA needs
     (n+1)(n+2)/2 interpolation points, ...).

  2. **Per-iteration overhead** — every optimizer carries its own
     bookkeeping cost. A GP fit or eigendecomposition is rounding error
     when each objective call costs a second, but it crushes wall-clock
     when each call costs a microsecond. We tag each algorithm with an
     **overhead tier** and only allow it when the timed objective is
     expensive enough that HumpDay's own work stays a small fraction of
     total time.

The recommender (`recommend`) consults both filters, then picks the top
candidate from the existing rule-based ordering in `suggest_pure`. When
a benchmarks-driven grid is available (see `benchmarks/build_recommendation_grid.py`)
it will be consulted first; until then the rule fallback is used.

This module deliberately has no dependency on humpday.optimizers — it
operates on algorithm *names* only — so unit tests can exercise the
filter logic without instantiating any optimizers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


# -----------------------------------------------------------------------------
# Overhead tiers
# -----------------------------------------------------------------------------
# Each tier is a rough order of magnitude of the optimizer's own per-iteration
# work, in seconds. An algorithm is considered eligible for a given problem
# only when the measured eval_time is at least MIN_EVAL_TIME_FOR_TIER[tier]:
# this caps HumpDay overhead at roughly ~10% of total wall-clock.

TIER_TRIVIAL = 0   # ~us per iter   — pure RNG draws or single componentwise step
TIER_LIGHT = 1     # ~100us per iter — small per-iter loops, no linear algebra
TIER_MEDIUM = 2    # ~1ms per iter   — small linear-algebra updates
TIER_HEAVY = 3     # ~10ms per iter  — O(n^3) per generation
TIER_VERY_HEAVY = 4  # ~100ms per iter — O(n_obs^3) GP fit per iter


# Per-algorithm overhead tier. Values come from rough profiling at n=10;
# Stage B will measure these systematically and may shift entries.
TIER: dict[str, int] = {
    # Trivial
    "RandomSearch": TIER_TRIVIAL,
    "GridSearch": TIER_TRIVIAL,
    "HillClimbing": TIER_TRIVIAL,
    "Rechenberg": TIER_TRIVIAL,
    # Light
    "NelderMead": TIER_LIGHT,
    "Powell": TIER_LIGHT,
    "DifferentialEvolution": TIER_LIGHT,
    "ParticleSwarm": TIER_LIGHT,
    "SimulatedAnnealing": TIER_LIGHT,
    "GeneticAlgorithm": TIER_LIGHT,
    "EvolutionStrategy": TIER_LIGHT,
    "FireflyAlgorithm": TIER_LIGHT,
    "HarmonySearch": TIER_LIGHT,
    "AntColonyOpt": TIER_LIGHT,
    "LBFGSB": TIER_LIGHT,
    "PatternSearch": TIER_LIGHT,
    "CoordinateDescent": TIER_LIGHT,
    # Medium
    "PRIMA_BOBYQA": TIER_MEDIUM,
    "PRIMA_NEWUOA": TIER_MEDIUM,
    # Heavy
    "PRIMA_UOBYQA": TIER_HEAVY,
    "CMAEvolutionStrategy": TIER_HEAVY,
    # Very heavy
    "BayesianOpt": TIER_VERY_HEAVY,
}


# Minimum objective eval-time (seconds) at which each tier is worth using.
# Below the threshold, that tier's overhead dominates wall-clock.
MIN_EVAL_TIME_FOR_TIER: dict[int, float] = {
    TIER_TRIVIAL: 0.0,
    TIER_LIGHT: 1e-5,    # 10 us — almost always allowed
    TIER_MEDIUM: 1e-4,   # 100 us
    TIER_HEAVY: 1e-3,    # 1 ms
    TIER_VERY_HEAVY: 1e-2,  # 10 ms — GP fit dominates below this
}


# -----------------------------------------------------------------------------
# Dimensional caps
# -----------------------------------------------------------------------------
# Structural upper bounds on n_dim, independent of eval_time. Algorithms not
# listed here are treated as having no cap (linear-in-dim or better).

DIM_CAP: dict[str, int] = {
    "GridSearch": 4,        # n_per_axis ** n_dim
    "BayesianOpt": 10,      # GP cost + RBF kernel degeneracy
    "PRIMA_UOBYQA": 12,     # (n+1)(n+2)/2 interpolation set
    "NelderMead": 20,       # simplex degeneracy past ~20 dims
    "PRIMA_NEWUOA": 25,     # underdetermined quadratic model fit
    "PRIMA_BOBYQA": 30,     # same model machinery, bounds-aware — past
                            # n=30 a single optimize() call takes minutes
                            # because of the geometry/BIGLAG steps
    "Powell": 30,           # direction-set line searches
    "PatternSearch": 30,    # 2n directions per pattern
    "FireflyAlgorithm": 50, # O(n_pop^2) attractions per gen
    "AntColonyOpt": 50,     # archive-sample interactions
    "CMAEvolutionStrategy": 100,  # eigendecomp O(n^3) per gen
}


# Minimum n_trials each algorithm needs to be worth spinning up. Below this
# the algorithm either can't initialize (PRIMA needs (n+1)(n+2)/2 model
# points) or doesn't run long enough for its expensive parts to amortize
# (BayesianOpt's GP fits don't pay off until you've drawn enough samples).
def min_trials(name: str, n_dim: int) -> int:
    if name == "PRIMA_UOBYQA":
        return (n_dim + 1) * (n_dim + 2) // 2 + 5
    if name in ("PRIMA_NEWUOA", "PRIMA_BOBYQA"):
        return 2 * n_dim + 5
    if name == "BayesianOpt":
        return max(30, 5 * n_dim)
    if name == "CMAEvolutionStrategy":
        return max(20, 4 * n_dim)
    if name == "GridSearch":
        # Need at least 3 points per axis to be meaningfully a grid.
        return max(3 ** n_dim, 8)
    return 10


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------

def passes_dim(name: str, n_dim: int) -> bool:
    return n_dim <= DIM_CAP.get(name, 10_000)


def passes_trials(name: str, n_dim: int, n_trials: int) -> bool:
    return n_trials >= min_trials(name, n_dim)


def passes_eval_time(name: str, eval_time: float) -> bool:
    tier = TIER.get(name, TIER_LIGHT)
    return eval_time >= MIN_EVAL_TIME_FOR_TIER[tier]


def filter_by_dim(names: Iterable[str], n_dim: int) -> list[str]:
    return [n for n in names if passes_dim(n, n_dim)]


def filter_by_trials(names: Iterable[str], n_dim: int, n_trials: int) -> list[str]:
    return [n for n in names if passes_trials(n, n_dim, n_trials)]


def filter_by_eval_time(names: Iterable[str], eval_time: float) -> list[str]:
    return [n for n in names if passes_eval_time(n, eval_time)]


def eligible(
    names: Iterable[str],
    n_dim: int,
    n_trials: int,
    eval_time: Optional[float] = None,
) -> list[str]:
    """Apply all three filters. eval_time=None skips the overhead filter."""
    candidates = filter_by_dim(names, n_dim)
    candidates = filter_by_trials(candidates, n_dim, n_trials)
    if eval_time is not None:
        candidates = filter_by_eval_time(candidates, eval_time)
    return candidates


# -----------------------------------------------------------------------------
# Objective timing
# -----------------------------------------------------------------------------

@dataclass
class TimingResult:
    eval_time: float          # seconds per call (median of samples)
    samples: list[float]      # all measured times
    used_for_recommendation: bool  # True if we let timing decide eligibility


def time_objective(
    f: Callable,
    x_sample,
    n_warmup: int = 1,
    n_measure: int = 3,
) -> TimingResult:
    """Measure how long one objective evaluation takes.

    Runs `n_warmup` calls to absorb any first-call JIT/cache effects, then
    `n_measure` timed calls and takes the median to be robust against an
    occasional GC pause. Total cost is `n_warmup + n_measure` evaluations,
    typically 4. Caller is responsible for accounting that against the trial
    budget if they care.
    """
    for _ in range(n_warmup):
        f(x_sample)
    samples: list[float] = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        f(x_sample)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    median = samples[len(samples) // 2]
    return TimingResult(
        eval_time=median,
        samples=samples,
        used_for_recommendation=True,
    )


# -----------------------------------------------------------------------------
# Recommendation
# -----------------------------------------------------------------------------

def _rule_based_ranking(n_dim: int, n_trials: int) -> list[str]:
    """The pre-existing ordering from suggest_pure, copy-pasted here so the
    eligibility module stays import-light. Kept in sync with
    humpday.optimizers.alloptimizers.suggest_pure.
    """
    if n_dim <= 2:
        return [
            "NelderMead",
            "PRIMA_UOBYQA",
            "PRIMA_NEWUOA",
            "Powell",
            "LBFGSB",
            "HillClimbing",
            "PatternSearch",
            "CoordinateDescent",
        ]
    if n_dim <= 10:
        return [
            "DifferentialEvolution",
            "CMAEvolutionStrategy",
            "ParticleSwarm",
            "PRIMA_BOBYQA",
            "BayesianOpt",
            "HarmonySearch",
            "GeneticAlgorithm",
            "PatternSearch",
            "EvolutionStrategy",
        ]
    if n_dim <= 50:
        return [
            "CMAEvolutionStrategy",
            "DifferentialEvolution",
            "EvolutionStrategy",
            "ParticleSwarm",
            "Rechenberg",
            "FireflyAlgorithm",
            "AntColonyOpt",
            "RandomSearch",
            "GeneticAlgorithm",
            "SimulatedAnnealing",
        ]
    return [
        "Rechenberg",
        "RandomSearch",
        "ParticleSwarm",
        "DifferentialEvolution",
        "HillClimbing",
        "CoordinateDescent",
        "SimulatedAnnealing",
        "EvolutionStrategy",
        "GeneticAlgorithm",
    ]


# -----------------------------------------------------------------------------
# Benchmarks-driven grid
# -----------------------------------------------------------------------------
# `benchmarks/build_recommendation_grid.py` writes a JSON file mapping each
# (n_dim, n_trials) cell to {algorithm: median_best_value}. When that file is
# present, we use it to break ties between eligible algorithms — picking the
# one with the smallest median_best. If absent or unreadable we fall back to
# the rule-based ranking (which is what shipped before the grid existed).

_GRID_PATH_DEFAULT = (
    Path(__file__).parent.parent / "benchmarks" / "recommendation_grid.json"
)
_grid_cache: Optional[dict] = None
_grid_cache_path: Optional[Path] = None


def _load_grid(path: Path) -> Optional[dict]:
    """Return the parsed grid JSON, or None if the file is missing/unreadable.

    Cached per-path so repeated calls to ``recommend`` don't re-parse the JSON.
    """
    global _grid_cache, _grid_cache_path
    if _grid_cache_path == path and _grid_cache is not None:
        return _grid_cache
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            grid = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    _grid_cache = grid
    _grid_cache_path = path
    return grid


def _snap_to_grid_cell(
    grid: dict, n_dim: int, n_trials: int
) -> Optional[dict[str, dict]]:
    """Find the closest available cell to (n_dim, n_trials) in the grid.

    We snap to the nearest n_dim that is ≤ caller's n_dim (so we don't make
    recommendations from a higher-dim sweep where the easy algorithms got
    eliminated), then to the largest n_trials ≤ caller's budget. Returns the
    cell's {algorithm: stats} dict, or None when the grid has no usable cell.
    """
    cells = grid.get("cells", {})
    if not cells:
        return None

    def parse_key(k: str) -> tuple[int, int]:
        d, t = k.split("/")
        return int(d), int(t)

    candidates = [parse_key(k) for k in cells]
    feasible = [(d, t) for d, t in candidates if d <= n_dim and t <= n_trials]
    if not feasible:
        # Fall back to nearest by L1 distance — used when caller is below
        # the grid's smallest cell (e.g. n_dim=1 with no n_dim=1 sweep).
        feasible = candidates
        feasible.sort(key=lambda dt: abs(dt[0] - n_dim) + abs(dt[1] - n_trials))
    else:
        # Pick the cell closest to (n_dim, n_trials) under the ≤ constraint.
        feasible.sort(
            key=lambda dt: (abs(dt[0] - n_dim), abs(dt[1] - n_trials))
        )
    d, t = feasible[0]
    return cells.get(f"{d}/{t}")


def recommend(
    n_dim: int,
    n_trials: int,
    eval_time: Optional[float] = None,
    available: Optional[Iterable[str]] = None,
    grid_path: Optional[Path] = None,
) -> str:
    """Pick the best algorithm name for (n_dim, n_trials, eval_time).

    ``available`` lets a caller restrict to a known optimizer registry; when
    None we consider every algorithm we have a tier for.

    ``grid_path`` overrides the default benchmarks/recommendation_grid.json
    path (handy for tests). When a grid is available, the chosen algorithm is
    the eligible candidate with the smallest median_best on the nearest
    (n_dim, n_trials) cell. When no grid is present we fall through to the
    rule-based ranking that mirrors :func:`suggest_pure`.

    If nothing passes the filters (e.g. n_trials too small for any sane
    algorithm), falls back to RandomSearch — it works at any
    (n_dim, n_trials) and gives a defensible baseline.
    """
    universe = list(available) if available is not None else list(TIER.keys())
    candidates = set(eligible(universe, n_dim, n_trials, eval_time))

    # Grid-driven pick when available.
    grid = _load_grid(grid_path or _GRID_PATH_DEFAULT)
    if grid is not None:
        cell = _snap_to_grid_cell(grid, n_dim, n_trials)
        if cell:
            scored = [
                (cell[name]["median_best"], name)
                for name in candidates
                if name in cell
                and cell[name].get("median_best") not in (None, float("inf"))
            ]
            if scored:
                scored.sort()
                return scored[0][1]

    # Rule-based fallback.
    for name in _rule_based_ranking(n_dim, n_trials):
        if name in candidates:
            return name
    if candidates:
        return next(iter(candidates))
    return "RandomSearch"


def _clear_grid_cache() -> None:
    """Test hook — drop any cached grid so a fresh _load_grid() re-reads disk."""
    global _grid_cache, _grid_cache_path
    _grid_cache = None
    _grid_cache_path = None


__all__ = [
    "TIER",
    "DIM_CAP",
    "MIN_EVAL_TIME_FOR_TIER",
    "TIER_TRIVIAL",
    "TIER_LIGHT",
    "TIER_MEDIUM",
    "TIER_HEAVY",
    "TIER_VERY_HEAVY",
    "TimingResult",
    "min_trials",
    "passes_dim",
    "passes_trials",
    "passes_eval_time",
    "filter_by_dim",
    "filter_by_trials",
    "filter_by_eval_time",
    "eligible",
    "time_objective",
    "recommend",
    "_clear_grid_cache",
]
