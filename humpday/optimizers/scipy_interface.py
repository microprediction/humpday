"""
SciPy-style interface with rectangular bounds for Humpday optimizers.

This module provides thin wrappers that allow our unit hypercube [0,1]^n optimizers
to work with arbitrary rectangular bounds, following SciPy conventions.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .alloptimizers import PURE_OPTIMIZERS, pure_optimize


def unbounded_to_unit_cube(
    x_real: np.ndarray, scale: Union[float, np.ndarray] = 1.0
) -> np.ndarray:
    """
    Map from unbounded real space R^n to unit hypercube [0,1]^n.

    Uses inverse tangent transformation: x_unit = (atan(x_real/scale)/π) + 0.5
    This maps (-∞, +∞) to (0, 1) symmetrically around 0.5.

    Parameters:
    -----------
    x_real : array_like
        Point in unbounded space
    scale : float or array_like, optional
        Scaling factor for the transformation. Higher scale concentrates
        more of the unit cube around larger values. Default is 1.0.
    """
    return (np.arctan(x_real / scale) / np.pi) + 0.5


def unit_cube_to_unbounded(
    x_unit: np.ndarray, scale: Union[float, np.ndarray] = 1.0
) -> np.ndarray:
    """
    Map from unit hypercube [0,1]^n to unbounded real space R^n.

    Uses tangent transformation: x_real = scale * tan(π(x_unit - 0.5))
    This maps (0, 1) to (-∞, +∞) symmetrically.

    Parameters:
    -----------
    x_unit : array_like
        Point in unit hypercube
    scale : float or array_like, optional
        Scaling factor for the transformation. Higher scale expands the
        effective search range. Default is 1.0.
    """
    # Avoid singularities at 0 and 1
    x_safe = np.clip(x_unit, 1e-15, 1 - 1e-15)
    return scale * np.tan(np.pi * (x_safe - 0.5))


def create_unbounded_objective(
    objective: Callable, scale: Union[float, np.ndarray] = 1.0
) -> Callable:
    """
    Create objective function that maps from [0,1]^n to unbounded domain R^n.

    Parameters:
    -----------
    objective : callable
        Original objective function on unbounded domain
    scale : float or array_like, optional
        Scaling factor for the transformation. Should match the expected
        scale of the problem. Default is 1.0.
    """

    def unbounded_objective(x_unit):
        # Transform from unit cube to unbounded space
        x_unit = np.asarray(x_unit)
        x_real = unit_cube_to_unbounded(x_unit, scale)
        return objective(x_real)

    return unbounded_objective


def parse_bounds(bounds, n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse bounds specification into lower and upper arrays."""
    if bounds is None:
        # Default to unit hypercube
        return np.zeros(n_dim), np.ones(n_dim)

    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        # Check if it's a single bound pair applied to all dimensions
        if isinstance(bounds[0], (int, float)) and isinstance(bounds[1], (int, float)):
            lower = np.full(n_dim, bounds[0])
            upper = np.full(n_dim, bounds[1])
            return lower, upper

    # Otherwise expect list of (lower, upper) pairs
    if len(bounds) != n_dim:
        raise ValueError(f"Expected {n_dim} bound pairs, got {len(bounds)}")

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    # Validate bounds
    if np.any(lower >= upper):
        raise ValueError("Lower bounds must be less than upper bounds")

    return lower, upper


def create_bounded_objective(
    objective: Callable, lower: np.ndarray, upper: np.ndarray
) -> Callable:
    """Create objective function that maps from [0,1]^n to rectangular domain."""

    def bounded_objective(x_unit):
        # Transform from unit hypercube to rectangular domain
        x_unit = np.asarray(x_unit)
        x_real = lower + x_unit * (upper - lower)
        return objective(x_real)

    return bounded_objective


def transform_solution(
    x_unit: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Transform solution from unit hypercube back to rectangular domain."""
    return lower + x_unit * (upper - lower)


def cube_minimize(
    fun: Callable,
    x0: Optional[np.ndarray] = None,
    args: Tuple = (),
    method: str = "NelderMead",
    bounds: Optional[Union[List[Tuple[float, float]], Tuple[float, float]]] = None,
    scale: Optional[Union[float, np.ndarray]] = None,
    options: Optional[dict] = None,
) -> object:
    """
    Cube-based minimization interface for Humpday optimizers.

    Transforms arbitrary rectangular domains to unit hypercube [0,1]^n internally.

    Parameters:
    -----------
    fun : callable
        Objective function to minimize. Should accept numpy array and return scalar.
    x0 : array_like, optional
        Initial guess (currently ignored - our algorithms use random initialization)
    args : tuple, optional
        Extra arguments passed to objective function (currently not supported)
    method : str, optional
        Optimization algorithm name. Must be one of the 22 available algorithms.
        Default is 'NelderMead'.
    bounds : sequence or tuple, optional
        Bounds for variables. Either:
        - List of (min, max) tuples for each dimension: [(x1_min, x1_max), (x2_min, x2_max), ...]
        - Single (min, max) tuple applied to all dimensions: (min, max)
        - None for unbounded optimization on entire real space R^n (uses tan(π(x-0.5)) mapping)
    scale : float or array_like, optional
        Indicative scale for unbounded optimization (ignored if bounds provided).
        Should match the expected magnitude of the solution. For example:
        - scale=1.0 (default): Solution expected around [-1, 1]
        - scale=100.0: Solution expected around [-100, 100]
        - scale=[10, 1, 100]: Different scales per dimension
        This concentrates more optimization effort around the expected scale.
    options : dict, optional
        Solver options. Supported:
        - 'maxiter': int, maximum number of function evaluations (default 1000)

    Returns:
    --------
    OptimizeResult
        Result object with attributes:
        - x: Solution array
        - fun: Function value at solution
        - nfev: Number of function evaluations
        - success: Whether optimization succeeded
        - message: Description of termination
    """

    # Handle arguments
    if args:
        raise NotImplementedError(
            "Extra arguments to objective function not yet supported"
        )

    # Parse options
    if options is None:
        options = {}
    maxiter = options.get("maxiter", 1000)

    # Determine problem dimension
    if bounds is None and x0 is None:
        raise ValueError(
            "Must specify either bounds or x0 to determine problem dimension"
        )

    if x0 is not None:
        n_dim = len(x0)
    elif bounds is None:
        # This shouldn't happen due to check above, but just in case
        raise ValueError("Cannot determine dimension without bounds or x0")
    else:
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            if isinstance(bounds[0], (int, float)):
                # Single bound pair - need to infer dimension from x0 or raise error
                if x0 is None:
                    raise ValueError(
                        "Cannot infer dimension from single bound pair without x0"
                    )
                n_dim = len(x0)
            else:
                n_dim = len(bounds)
        else:
            n_dim = len(bounds)

    # Validate method
    if method not in PURE_OPTIMIZERS:
        available = ", ".join(list(PURE_OPTIMIZERS.keys())[:10])
        raise ValueError(f"Unknown method '{method}'. Available: {available}...")

    # Handle bounded vs unbounded optimization
    if bounds is None:
        # Unbounded optimization: map R^n to [0,1]^n
        if scale is None:
            scale = 1.0
        unbounded_obj = create_unbounded_objective(fun, scale)
        best_value, best_x_unit = pure_optimize(unbounded_obj, method, maxiter, n_dim)

        # Transform solution back to unbounded space
        best_x = unit_cube_to_unbounded(np.array(best_x_unit), scale)
    else:
        # Bounded optimization: map rectangular bounds to [0,1]^n
        lower, upper = parse_bounds(bounds, n_dim)
        bounded_obj = create_bounded_objective(fun, lower, upper)
        best_value, best_x_unit = pure_optimize(bounded_obj, method, maxiter, n_dim)

        # Transform solution back to original domain
        best_x = transform_solution(np.array(best_x_unit), lower, upper)

    # Create result object
    result = OptimizeResult(
        x=best_x,
        fun=best_value,
        nfev=maxiter,  # Our optimizers don't currently track exact evaluations
        success=True,  # We always return the best found solution
        message="Optimization completed successfully",
    )

    return result


class OptimizeResult:
    """Result object matching SciPy's OptimizeResult interface."""

    def __init__(self, x, fun, nfev, success, message):
        self.x = x
        self.fun = fun
        self.nfev = nfev
        self.success = success
        self.message = message

    def __repr__(self):
        return (
            f"OptimizeResult(x={self.x}, fun={self.fun:.6e}, "
            f"nfev={self.nfev}, success={self.success})"
        )


def cube_minimize_scalar(
    fun: Callable,
    bounds: Optional[Tuple[float, float]] = None,
    args: Tuple = (),
    method: str = "NelderMead",
    options: Optional[dict] = None,
) -> object:
    """
    Cube-based scalar minimization (1D problems).

    Parameters:
    -----------
    fun : callable
        Objective function to minimize. Should accept scalar and return scalar.
    bounds : tuple, optional
        Bounds for the variable: (min, max). Default is (0, 1).
    args : tuple, optional
        Extra arguments (not yet supported)
    method : str, optional
        Optimization algorithm name. Default is 'NelderMead'.
    options : dict, optional
        Solver options including 'maxiter'

    Returns:
    --------
    OptimizeResult
        Result object with scalar solution
    """

    # Wrap scalar function for vector interface
    def vector_fun(x):
        return fun(x[0])

    # Set default bounds for scalar case
    if bounds is None:
        bounds = [(0.0, 1.0)]
    else:
        bounds = [bounds]  # Convert to list format

    # Call vector cube_minimize
    result = cube_minimize(vector_fun, bounds=bounds, method=method, options=options)

    # Extract scalar solution
    result.x = result.x[0]

    return result


def minimize(
    fun: Callable,
    x0: Optional[np.ndarray] = None,
    method: str = "NelderMead",
    bounds: Optional[Union[List[Tuple[float, float]], Tuple[float, float]]] = None,
    scale: Optional[Union[float, np.ndarray]] = None,
    options: Optional[dict] = None,
) -> object:
    """
    Clean minimization interface with rectangular bounds.

    This is a simplified wrapper around cube_minimize() for everyday use.
    Automatically transforms rectangular domains to unit hypercube internally.

    Parameters:
    -----------
    fun : callable
        Objective function to minimize
    x0 : array_like, optional
        Initial guess (currently ignored)
    method : str, optional
        Optimization algorithm name (default: 'NelderMead')
    bounds : sequence or tuple, optional
        Bounds for variables (None for unbounded optimization)
    scale : float or array_like, optional
        Indicative scale for unbounded optimization (ignored if bounds provided)
    options : dict, optional
        Solver options

    Returns:
    --------
    OptimizeResult
        Optimization result object

    Examples:
    --------
    >>> from humpday import minimize
    >>> # Bounded optimization
    >>> def objective(x): return (x[0]-1)**2 + (x[1]-2)**2
    >>> result = minimize(objective, bounds=[(-5,5), (-5,5)])
    >>> print(result.x)  # Should be close to [1, 2]
    >>>
    >>> # Unbounded optimization with scale hint
    >>> result = minimize(objective, x0=[0, 0], scale=10.0)  # Expect solution around scale 10
    >>> print(result.x)  # Should be close to [1, 2]
    """
    return cube_minimize(
        fun=fun, x0=x0, method=method, bounds=bounds, scale=scale, options=options
    )


def minimize_scalar(
    fun: Callable,
    bounds: Optional[Tuple[float, float]] = None,
    method: str = "NelderMead",
    options: Optional[dict] = None,
) -> object:
    """
    Clean scalar minimization interface.

    Simplified wrapper around cube_minimize_scalar() for everyday use.

    Parameters:
    -----------
    fun : callable
        Objective function to minimize (1D)
    bounds : tuple, optional
        Bounds for the variable: (min, max)
    method : str, optional
        Optimization algorithm name
    options : dict, optional
        Solver options

    Returns:
    --------
    OptimizeResult
        Optimization result with scalar solution
    """
    return cube_minimize_scalar(fun=fun, bounds=bounds, method=method, options=options)


# Convenience functions for specific algorithms
def cube_nelder_mead(fun: Callable, bounds=None, options=None) -> object:
    """Minimize using Nelder-Mead algorithm with cube transformation."""
    return cube_minimize(fun, method="NelderMead", bounds=bounds, options=options)


def cube_differential_evolution(fun: Callable, bounds=None, options=None) -> object:
    """Minimize using Differential Evolution with cube transformation."""
    return cube_minimize(
        fun, method="DifferentialEvolution", bounds=bounds, options=options
    )


def cube_particle_swarm(fun: Callable, bounds=None, options=None) -> object:
    """Minimize using Particle Swarm Optimization with cube transformation."""
    return cube_minimize(fun, method="ParticleSwarm", bounds=bounds, options=options)


def cube_cma_es(fun: Callable, bounds=None, options=None) -> object:
    """Minimize using CMA Evolution Strategy with cube transformation."""
    return cube_minimize(
        fun, method="CMAEvolutionStrategy", bounds=bounds, options=options
    )


def cube_prima_uobyqa(fun: Callable, bounds=None, options=None) -> object:
    """Minimize using PRIMA UOBYQA algorithm with cube transformation."""
    return cube_minimize(fun, method="PRIMA_UOBYQA", bounds=bounds, options=options)


# Domain transformation utilities for advanced users
def transform_to_unit_cube(
    x: np.ndarray, bounds: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Transform point from rectangular domain to unit hypercube.

    Parameters:
    -----------
    x : array_like
        Point in rectangular domain
    bounds : list of tuples
        Bounds specification [(x1_min, x1_max), ...]

    Returns:
    --------
    x_unit : ndarray
        Point in unit hypercube [0,1]^n
    """
    x = np.asarray(x)
    lower, upper = parse_bounds(bounds, len(x))
    return (x - lower) / (upper - lower)


def transform_from_unit_cube(
    x_unit: np.ndarray, bounds: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Transform point from unit hypercube to rectangular domain.

    Parameters:
    -----------
    x_unit : array_like
        Point in unit hypercube [0,1]^n
    bounds : list of tuples
        Bounds specification [(x1_min, x1_max), ...]

    Returns:
    --------
    x : ndarray
        Point in rectangular domain
    """
    x_unit = np.asarray(x_unit)
    lower, upper = parse_bounds(bounds, len(x_unit))
    return lower + x_unit * (upper - lower)


__all__ = [
    # Clean everyday interface
    "minimize",
    "minimize_scalar",
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
