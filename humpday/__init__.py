"""
Humpday: Lightweight derivative-free optimization
Pure Python implementations with no external dependencies (beyond numpy/scipy)
"""

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


# Simple suggest function using pure algorithms
def suggest(n_dim: int, n_trials: int = 100, n_seconds: float = None):
    """
    Suggest optimizers for your problem.

    Args:
        n_dim: Problem dimension
        n_trials: Number of function evaluations
        n_seconds: Ignored (for compatibility)

    Returns:
        List of (score, time, name) tuples - best algorithms first
    """
    algorithm_names = suggest_pure(n_dim, n_trials)

    # Create compatibility tuples (score, time, name)
    suggestions = []
    for i, name in enumerate(algorithm_names):
        score = 1000 + i * 100  # Fake scores for compatibility
        time_estimate = 0.1 * i  # Fake times for compatibility
        suggestions.append((score, time_estimate, name))

    return suggestions


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
