"""
SciPy-style interface with rectangular bounds for Humpday optimizers.

This module provides thin wrappers that allow our unit hypercube [0,1]^n optimizers
to work with arbitrary rectangular bounds, following SciPy conventions.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

from humpday import _array as _A
from humpday import eligibility as _E

from .alloptimizers import PURE_OPTIMIZERS, pure_optimize


def unbounded_to_unit_cube(x_real, scale: Union[float, Any] = 1.0):
    """Map ℝⁿ → (0, 1)ⁿ via `x_unit = atan(x_real / scale) / π + 0.5`.
    Symmetric around 0.5; `scale` controls how much of the cube is
    spent on which range of values."""
    # Elementwise division when x_real is a vector and scale is a scalar
    # works on both _Vec (via __truediv__) and numpy ndarray.
    return (_A.atan(x_real / scale) / _A.pi) + 0.5


def unit_cube_to_unbounded(x_unit, scale: Union[float, Any] = 1.0):
    """Map (0, 1)ⁿ → ℝⁿ via `x_real = scale * tan(π (x_unit - 0.5))`.
    Inverse of `unbounded_to_unit_cube`."""
    x_safe = _A.clip(x_unit, 1e-15, 1 - 1e-15)
    return scale * _A.tan(_A.pi * (x_safe - 0.5))


def create_unbounded_objective(
    objective: Callable, scale: Union[float, Any] = 1.0
) -> Callable:
    """Wrap an objective defined on ℝⁿ so it can be called on a unit-cube
    point — the wrapper applies `unit_cube_to_unbounded` internally."""

    def unbounded_objective(x_unit):
        x_unit = _A.asarray(x_unit)
        x_real = unit_cube_to_unbounded(x_unit, scale)
        return objective(x_real)

    return unbounded_objective


def parse_bounds(bounds, n_dim: int):
    """Parse a `bounds` specification into (lower, upper) vectors. Returns
    either `_Vec` or numpy.ndarray pair depending on the active backend."""
    if bounds is None:
        return _A.zeros(n_dim), _A.ones(n_dim)

    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        # Single (lo, hi) pair, applied to every dimension.
        if isinstance(bounds[0], (int, float)) and isinstance(bounds[1], (int, float)):
            lower = _A.full(n_dim, bounds[0])
            upper = _A.full(n_dim, bounds[1])
            return lower, upper

    # Otherwise expect a sequence of (lower, upper) per-dimension pairs.
    if len(bounds) != n_dim:
        raise ValueError(f"Expected {n_dim} bound pairs, got {len(bounds)}")

    lower = _A.asarray([b[0] for b in bounds])
    upper = _A.asarray([b[1] for b in bounds])

    if any(float(lower[i]) >= float(upper[i]) for i in range(n_dim)):
        raise ValueError("Lower bounds must be less than upper bounds")

    return lower, upper


def create_bounded_objective(objective: Callable, lower, upper) -> Callable:
    """Wrap an objective defined on [lower, upper] so it can be called on
    a unit-cube point — the wrapper applies the affine map internally."""

    def bounded_objective(x_unit):
        x_unit = _A.asarray(x_unit)
        x_real = lower + x_unit * (upper - lower)
        return objective(x_real)

    return bounded_objective


def transform_solution(x_unit, lower, upper):
    """Map a unit-cube point back to the [lower, upper] box."""
    return lower + x_unit * (upper - lower)


def cube_minimize(
    fun: Callable,
    x0: Optional[Any] = None,
    args: Tuple = (),
    method: Optional[str] = None,
    bounds: Optional[Union[List[Tuple[float, float]], Tuple[float, float]]] = None,
    scale: Optional[Union[float, Any]] = None,
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
        Optimization algorithm name. Must be one of the available algorithms.
        If None (default), the algorithm is auto-selected by
        `humpday.eligibility.recommend` using three filters:

          1. Dimensional cap (GridSearch ≤ 4, BayesianOpt ≤ 10, ...).
          2. Minimum trials needed to initialize / amortize.
          3. Overhead tier vs. measured objective eval-time: when each
             objective call is microseconds, we avoid CMA-ES (eigendecomp)
             and BayesianOpt (GP fit); when each call is seconds, the
             sample-efficient algorithms become eligible.

        The timing step costs 4 objective evaluations at startup. Disable
        with `options={'auto_timing': False}` for stochastic objectives or
        when reproducibility matters more than expense-aware selection.
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

    # Build the cube-space objective once. Auto-selection (next block) wants
    # to time this same wrapped function so the measurement reflects what the
    # optimizer will actually call.
    if bounds is None:
        if scale is None:
            scale = 1.0
        cube_obj = create_unbounded_objective(fun, scale)
        lower = upper = None  # set below if bounded
    else:
        lower, upper = parse_bounds(bounds, n_dim)
        cube_obj = create_bounded_objective(fun, lower, upper)

    # Auto-select method via the eligibility module: time one call to learn
    # how expensive the objective is, then pick the highest-ranked algorithm
    # that (a) is structurally suited to n_dim, (b) has enough trials to
    # initialize, and (c) carries overhead small relative to eval_time.
    #
    # Callers can disable the timing call with options={'auto_timing': False}
    # — useful for stochastic objectives where one extra eval changes seeds.
    eval_time_used: Optional[float] = None
    if method is None:
        auto_timing = options.get("auto_timing", True)
        if auto_timing:
            try:
                x_sample = _A.asarray([0.5] * n_dim)
                timing = _E.time_objective(cube_obj, x_sample, n_warmup=1, n_measure=3)
                eval_time_used = timing.eval_time
                # Account for the timing calls against the budget so the user
                # gets the n_trials they asked for, not n_trials + 4.
                maxiter = max(maxiter - 4, 1)
            except Exception:
                # If the objective throws on a random feasible point, skip
                # timing and let recommend() fall back to dim/trials only.
                eval_time_used = None
        method = _E.recommend(
            n_dim=n_dim,
            n_trials=maxiter,
            eval_time=eval_time_used,
            available=list(PURE_OPTIMIZERS.keys()),
        )

    # Validate method
    if method not in PURE_OPTIMIZERS:
        available = ", ".join(list(PURE_OPTIMIZERS.keys())[:10])
        raise ValueError(f"Unknown method '{method}'. Available: {available}...")

    # Run the optimizer on the prepared cube objective.
    best_value, best_x_unit = pure_optimize(cube_obj, method, maxiter, n_dim)

    # Transform solution back to caller's coordinate system.
    if bounds is None:
        best_x = unit_cube_to_unbounded(_A.asarray(best_x_unit), scale)
    else:
        best_x = transform_solution(_A.asarray(best_x_unit), lower, upper)

    # Create result object
    result = OptimizeResult(
        x=best_x,
        fun=best_value,
        nfev=maxiter,  # Our optimizers don't currently track exact evaluations
        success=True,  # We always return the best found solution
        message="Optimization completed successfully",
        method=method,
        eval_time_measured=eval_time_used,
        tier=_E.TIER.get(method),
    )

    return result


class OptimizeResult:
    """Result of a humpday.minimize() / cube_minimize() call.

    Mirrors `scipy.optimize.OptimizeResult` (``x``, ``fun``, ``nfev``,
    ``success``, ``message``) and adds three humpday-specific fields:

    - ``method``: the algorithm name that was actually run. Equals the
      ``method=`` argument when the caller picked one explicitly, or the
      name chosen by :func:`humpday.eligibility.recommend` when ``method``
      was None.
    - ``eval_time_measured``: seconds-per-objective-call as timed at
      ``minimize`` entry, or None when auto-timing was off or the timing
      call raised. Useful for understanding why a given algorithm was
      picked.
    - ``tier``: the overhead tier of the chosen algorithm (0 trivial …
      4 GP-fit-heavy). Indexes into :data:`humpday.eligibility.TIER`.

    The convenience method :meth:`tuple` returns ``(fun, x)`` for the
    callers that prefer the original two-tuple shape used by
    :func:`humpday.minimize_unit_cube`.
    """

    def __init__(
        self,
        x,
        fun,
        nfev,
        success,
        message,
        method: Optional[str] = None,
        eval_time_measured: Optional[float] = None,
        tier: Optional[int] = None,
    ):
        self.x = x
        self.fun = fun
        self.nfev = nfev
        self.success = success
        self.message = message
        self.method = method
        self.eval_time_measured = eval_time_measured
        self.tier = tier

    def tuple(self) -> Tuple[float, Any]:
        """Return ``(fun, x)`` for callers that want the legacy two-tuple shape."""
        return self.fun, self.x

    def __repr__(self):
        method = f", method={self.method!r}" if self.method else ""
        et = (
            f", eval_time_measured={self.eval_time_measured:.2e}"
            if self.eval_time_measured is not None
            else ""
        )
        return (
            f"OptimizeResult(x={self.x}, fun={self.fun:.6e}, "
            f"nfev={self.nfev}, success={self.success}{method}{et})"
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
    x0: Optional[Any] = None,
    method: Optional[str] = None,
    bounds: Optional[Union[List[Tuple[float, float]], Tuple[float, float]]] = None,
    scale: Optional[Union[float, Any]] = None,
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
        Optimization algorithm name. If None (the default), the method is
        chosen automatically by `humpday.eligibility.recommend`, which
        filters by problem dimension, trial budget, and the measured cost
        of one objective call. See `cube_minimize` for the full description.
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


# Domain transformation utilities for advanced users.
def transform_to_unit_cube(x, bounds: List[Tuple[float, float]]):
    """Affine map from a rectangular box to the unit cube."""
    x = _A.asarray(x)
    lower, upper = parse_bounds(bounds, len(x))
    return (x - lower) / (upper - lower)


def transform_from_unit_cube(x_unit, bounds: List[Tuple[float, float]]):
    """Inverse of `transform_to_unit_cube`."""
    x_unit = _A.asarray(x_unit)
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
