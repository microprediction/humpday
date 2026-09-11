"""
Humpday: Lightweight derivative-free optimization
Pure Python implementations with no external dependencies (beyond numpy/scipy)
"""

# Module-level annotations are evaluated at import time, so `dict | None` here would be a
# TypeError on Python 3.9, which pyproject still supports. This keeps them unevaluated.
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("humpday")
except PackageNotFoundError:
    # Running from a source checkout without `pip install` — fall back so
    # `humpday.__version__` is always a string. The CI publish workflow's
    # `python -c "import humpday; print(humpday.__version__)"` step relies on
    # this attribute existing.
    __version__ = "0.0.0+source"

import json

from humpday.optimizers.adaptive_optimizer import (
    EloRatingSystem,
    adaptive_optimize,
    rosenbrock_variants_generator,
    sphere_variants_generator,
    suggest_algorithm_from_elo,
)
from humpday.optimizers.alloptimizers import (
    ALGORITHM_NAMES,
    OPTIMIZERS,
    PURE_OPTIMIZERS,
    get_optimizer,
    pure_optimize,
    suggest_pure,
)

# Rectangular bounds interface
from humpday.optimizers.scipy_interface import (
    # Utilities
    OptimizeResult,
    cube_cma_es,
    cube_differential_evolution,
    # Explicit cube-based interface
    cube_minimize,
    cube_minimize_scalar,
    cube_nelder_mead,
    cube_particle_swarm,
    cube_prima_uobyqa,
    # Clean everyday interface
    minimize,
    minimize_scalar,
    transform_from_unit_cube,
    transform_to_unit_cube,
    unbounded_to_unit_cube,
    unit_cube_to_unbounded,
)

_ELO_CACHE: dict | None = None
_ELO_BY_DIM_CACHE: dict | None = None


def elo_ratings() -> dict:
    """Overall Elo ratings, optimizer name -> rating. Empty if the table is unavailable.

    Recorded by ``benchmarks/record_elo.py`` over 6930 matches in **two dimensions** on sphere and
    Rosenbrock variants. Prefer :func:`elo_by_dimension`, which records per dimension and on both
    analytic surfaces and engineering demos; this remains for the overall picture and for callers
    that predate it.
    """
    global _ELO_CACHE
    if _ELO_CACHE is None:
        try:
            from pathlib import Path

            raw = json.loads((Path(__file__).parent / "data" / "elo_ratings.json").read_text())
            _ELO_CACHE = dict(raw.get("ratings", {}))
        except Exception:  # pragma: no cover - the table is optional
            _ELO_CACHE = {}
    return _ELO_CACHE


def elo_by_dimension() -> dict:
    """Measured ratings keyed by problem dimension, ``{n_dim: {optimizer: rating}}``.

    Recorded by ``benchmarks/record_elo_by_dimension.py`` on randomly morphed surfaces, with the
    evaluation budget scaled to the dimension so that model-based methods can actually build a
    model before being scored. Empty if nothing has been recorded.
    """
    global _ELO_BY_DIM_CACHE
    if _ELO_BY_DIM_CACHE is None:
        try:
            from pathlib import Path

            raw = json.loads(
                (Path(__file__).parent / "data" / "elo_by_dimension.json").read_text()
            )
            _ELO_BY_DIM_CACHE = {
                int(k): {
                    suite: dict(entry.get("ratings", {})) for suite, entry in suites.items()
                }
                for k, suites in raw.get("by_dimension", {}).items()
            }
        except Exception:  # pragma: no cover - the table is optional
            _ELO_BY_DIM_CACHE = {}
    return _ELO_BY_DIM_CACHE


def _ratings_for(n_dim: int, suite: str):
    """Ratings recorded for this suite **at this exact dimension**, or None.

    No interpolation and no nearest-neighbour. Optimizer performance is not smooth in dimension: a
    Bayesian method can work well at five and blow up at ten, and the trust-region methods here top
    the analytic table at twenty-five while being unable to build their interpolation set at a
    hundred on a comparable budget. A rating measured at one dimension is evidence about that one.
    """
    return elo_by_dimension().get(n_dim, {}).get(suite)


def _never_terrible(n_dim: int):
    """Order by worst rank across the suites, so the leader is the one that is never bad.

    Best-on-average picks a specialist. The analytic surfaces are smooth and smoothness rewards
    local search, so the trust-region methods sweep them; on the engineering demos at the same
    dimensions they are displaced. An optimizer that wins one and places near the bottom of the
    other is a worse default than one that is respectable on both, because the caller usually does
    not know which kind of objective they have.
    """
    suites = elo_by_dimension().get(n_dim, {})
    if len(suites) < 2:
        return None
    worst = {}
    for ratings in suites.values():
        order = sorted(ratings, key=lambda n: -ratings[n])
        for rank, name in enumerate(order, start=1):
            worst[name] = max(worst.get(name, 0), rank)
    for name in list(worst):  # only rank what every suite actually scored
        if not all(name in r for r in suites.values()):
            del worst[name]
    return sorted(worst, key=lambda n: (worst[n], n)) or None


# Simple suggest function using pure algorithms
def suggest(
    n_dim: int, n_trials: int = 100, n_seconds: float = None, smooth: bool = None
):
    """
    Suggest optimizers for your problem, best first.

    Args:
        n_dim: Problem dimension. Ratings are used only where a tournament was recorded at exactly
            this dimension.
        n_trials: Evaluation budget. Recorded for reference; the ordering does not vary with it.
        n_seconds: Ignored (for compatibility).
        smooth: What kind of objective you have, if you know.

            * ``True`` -- smooth and locally well-behaved. Ranked on morphed analytic surfaces,
              where local model-based search does well.
            * ``False`` -- rugged, or realistic in the way an engineering problem is. Ranked on the
              physics and engineering demos.
            * ``None`` (default) -- you do not know, or you want a safe choice. Ranked by *worst*
              position across both, so the leader is the optimizer that is never terrible rather
              than the one that wins a suite and collapses on the other.

            Two suites are a coarse hint and not a taxonomy. "Globally lumpy, locally smooth" is one
            class of objective; something like exp(OU) is not in it, and neither suite speaks for it.

    Returns:
        List of (score, time, name) tuples. ``score`` is the measured Elo rating when a single
        suite was selected, and ``nan`` under the default, where the ordering is by worst rank and
        no single rating describes it. ``time`` is always ``nan``: humpday records no timing
        evidence.
    """
    if smooth is None:
        robust = _never_terrible(n_dim)
        if robust:
            return [(float("nan"), float("nan"), n) for n in robust]
        # Only one suite recorded here: fall through and use whichever it is.
        for candidate in ("physics", "smooth"):
            ratings = _ratings_for(n_dim, candidate)
            if ratings:
                names = sorted(ratings, key=lambda n: -ratings[n])
                return [(ratings[n], float("nan"), n) for n in names]
    else:
        ratings = _ratings_for(n_dim, "smooth" if smooth else "physics")
        if ratings:
            names = sorted(ratings, key=lambda n: -ratings[n])
            return [(ratings[n], float("nan"), n) for n in names]

    # Nothing recorded at this dimension: fall back to the hand-written ordering, and do not dress
    # it up with scores it has not earned.
    return [(float("nan"), float("nan"), n) for n in suggest_pure(n_dim, n_trials)]


def minimize_unit_cube(
    objective, n_dim: int = 2, n_trials: int = 100, algorithm: str = None
):
    """
    Minimize an objective function on unit hypercube [0,1]^n.

    Args:
        objective: Function to minimize (takes array in [0,1]^n)
        n_dim: Problem dimension
        n_trials: Number of evaluations
        algorithm: Algorithm name. Auto-selected via :func:`suggest` when None, which means the
            measured best at this dimension where a tournament has been recorded near it, and the
            hand-written ordering otherwise.

    Returns:
        (best_value, best_point) tuple
    """
    if algorithm is None:
        algorithm = suggest(n_dim, n_trials)[0][2]

    return pure_optimize(objective, algorithm, n_trials, n_dim)


# Backward compatibility
recommend = suggest

__all__ = [
    "elo_ratings",
    "elo_by_dimension",
    # Package metadata
    "__version__",
    # Core interface
    "OPTIMIZERS",
    "PURE_OPTIMIZERS",
    "pure_optimize",
    "suggest",
    "recommend",
    "get_optimizer",
    "ALGORITHM_NAMES",
    # Minimization interfaces
    "minimize",  # Clean interface with rectangular bounds
    "minimize_scalar",  # 1D minimization
    "minimize_unit_cube",  # Original unit cube interface
    # Adaptive optimization
    "adaptive_optimize",
    "EloRatingSystem",
    "suggest_algorithm_from_elo",
    "sphere_variants_generator",
    "rosenbrock_variants_generator",
    # Explicit cube-based interface
    "cube_minimize",
    "cube_minimize_scalar",
    "cube_nelder_mead",
    "cube_differential_evolution",
    "cube_particle_swarm",
    "cube_cma_es",
    "cube_prima_uobyqa",
    # Utilities
    "OptimizeResult",
    "transform_to_unit_cube",
    "transform_from_unit_cube",
    "unbounded_to_unit_cube",
    "unit_cube_to_unbounded",
]
