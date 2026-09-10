"""
Humpday: Lightweight derivative-free optimization
Pure Python implementations with no external dependencies (beyond numpy/scipy)
"""

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

import json

_ELO_CACHE: dict | None = None


def elo_ratings() -> dict:
    """Measured Elo ratings, optimizer name -> rating. Empty if the table is unavailable.

    Recorded by ``benchmarks/record_elo.py`` over 6930 matches. Read the provenance before
    leaning on these: the tournament ran in **two dimensions** on sphere and Rosenbrock variants
    with 100 trials per problem, so the ratings say what beat what on smooth low-dimensional
    surfaces and nothing more. They are not evidence about high-dimensional or noisy objectives.
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


# Simple suggest function using pure algorithms
def suggest(n_dim: int, n_trials: int = 100, n_seconds: float = None):
    """
    Suggest optimizers for your problem.

    Args:
        n_dim: Problem dimension. This is what selects the ordering.
        n_trials: Evaluation budget. Recorded for the caller's reference and returned in the
            provenance; the ordering does not currently vary with it.
        n_seconds: Ignored (for compatibility)

    Returns:
        List of (score, time, name) tuples, best first.

        ``score`` is the optimizer's **measured Elo rating** where one exists, and ``nan``
        where it does not -- ``Rechenberg`` is unrated, and it leads the n_dim > 50 ordering.
        ``time`` is always ``nan``: humpday records no timing evidence, and the field is kept
        only so the tuple shape does not change.

        The *ordering* is a hand-written rule over ``n_dim`` (see ``suggest_pure``), not a
        ranking derived from the ratings, because the ratings were measured in two dimensions
        only. Treat it as a starting point, not a measurement.
    """
    algorithm_names = suggest_pure(n_dim, n_trials)
    ratings = elo_ratings()
    return [
        (ratings.get(name, float("nan")), float("nan"), name) for name in algorithm_names
    ]


def minimize_unit_cube(
    objective, n_dim: int = 2, n_trials: int = 100, algorithm: str = None
):
    """
    Minimize an objective function on unit hypercube [0,1]^n.

    Args:
        objective: Function to minimize (takes array in [0,1]^n)
        n_dim: Problem dimension
        n_trials: Number of evaluations
        algorithm: Algorithm name (auto-selected if None)

    Returns:
        (best_value, best_point) tuple
    """
    if algorithm is None:
        # Auto-select algorithm
        suggestions = suggest_pure(n_dim, n_trials)
        algorithm = suggestions[0]

    return pure_optimize(objective, algorithm, n_trials, n_dim)


# Backward compatibility
recommend = suggest

__all__ = [
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
