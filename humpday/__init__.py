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


def elo_by_dimension(n_trials: int = 100) -> dict:
    """Deprecated view: ``{dimension: {suite: {optimizer: rating}}}`` at one budget.

    Superseded by :mod:`humpday.ratings`, which indexes by budget as well, because which optimizer
    wins depends on the budget at least as strongly as on the dimension. Kept because it was a
    documented return shape; it can only show one budget at a time, and shows the recorded one
    nearest below ``n_trials``.
    """
    from humpday import ratings

    out: dict = {}
    for n_dim in ratings.recorded_dimensions():
        for suite, cell in ratings.cells_at(n_dim, n_trials).items():
            out.setdefault(n_dim, {})[suite] = dict(cell.get("ratings", {}))
    return out


def problems_recorded(n_dim: int = None, n_trials: int = 100) -> dict:
    """How many problems back the ratings: ``{dimension: {suite: count}}``.

    The count is what makes a rating worth anything, so it is readable rather than buried in the
    artifact, and it is what :func:`humpday.ratings.robust_order` shrinks toward the mean by.
    """
    from humpday import ratings

    dims = ratings.recorded_dimensions() if n_dim is None else [n_dim]
    out: dict = {}
    for d in dims:
        for suite, cell in ratings.cells_at(d, n_trials).items():
            out.setdefault(d, {})[suite] = int(cell.get("problems", 0))
    return out


# Simple suggest function using pure algorithms
def _or_nan(value) -> float:
    return float("nan") if value is None else float(value)


def suggest(
    n_dim: int, n_trials: int = 100, n_seconds: float = None, smooth: bool = None
):
    """
    Suggest optimizers for your problem, best first.

    Args:
        n_dim: Problem dimension. Ratings are used only where a tournament was recorded at exactly
            this dimension; performance is not smooth in dimension and is never interpolated.
        n_trials: Evaluation budget. Selects the recorded budget, taking the largest not exceeding
            this one.
        n_seconds: Ignored (for compatibility).
        smooth: What kind of objective you have, if you know.

            * ``True`` -- smooth and locally well behaved. Ranked on the analytic surfaces.
            * ``False`` -- rugged, or realistic in the way an engineering problem is. Ranked on the
              engineering suite.
            * ``None`` (default) -- ranked by worst position across both, so the leader is the
              optimizer that is never terrible rather than one that wins a suite and collapses on
              the other.

            Two suites are a coarse hint rather than a taxonomy.

    Returns:
        List of (score, time, name) tuples. ``score`` is the measured rating when one suite is
        selected, and ``nan`` under the default, where the ordering is by worst rank and no single
        rating describes it. ``time`` is always ``nan``: humpday records no timing evidence.
    """
    from humpday import ratings

    if smooth is None:
        order = ratings.robust_order(n_dim, n_trials)
        if order:
            return [(float("nan"), float("nan"), n) for n in order]
    else:
        suite = "surfaces" if smooth else "engineering"
        order = ratings.suite_order(n_dim, suite, n_trials)
        if order:
            return [
                (ratings.rating(n_dim, suite, n, n_trials), float("nan"), n)
                for n in order
            ]

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
    "elo_by_dimension",
    "problems_recorded",
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
