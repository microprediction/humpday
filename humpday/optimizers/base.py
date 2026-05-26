"""
Base class for all pure optimization algorithms.

Provides common functionality for objective evaluation, best value tracking,
and path visualization support.

Numpy-optional
--------------
This module reads array primitives from `humpday._array` rather than from
numpy directly. The shim transparently selects a numpy backend when numpy
is installed (no performance penalty — the operations re-export numpy
directly) and a pure-Python backend otherwise.

External API is unchanged: callers still get `.best_x`, `.best_value`,
`.evaluations`, `.path`, and `.track_path` with the same semantics.
The runtime type of `.best_x` is `numpy.ndarray` under the numpy backend
and `humpday._array_pure._Vec` (a `list` subclass) under the pure
backend; both are iterable, indexable, and support arithmetic.
"""

from typing import Callable

from humpday import _array as _A


class BaseOptimizer:
    """Base class for all pure optimization algorithms."""

    def __init__(self, objective: Callable, n_trials: int, n_dim: int):
        self.objective = objective
        self.n_trials = n_trials
        self.n_dim = n_dim
        self.evaluations = 0
        self.best_value = float("inf")
        self.best_x = _A.random_uniform(n_dim)
        self.track_path = False
        self.path = []

    def evaluate(self, x) -> float:
        """Evaluate objective with tracking. `x` may be a numpy array, a
        `_Vec`, or any sequence of floats — `_A.clip` normalises it."""
        self.evaluations += 1
        x_clipped = _A.clip(x, 0, 1)
        value = self.objective(x_clipped)

        # Track path for visualization. Sample at ~20 evenly-spaced points
        # across the run by default; always record the first evaluation.
        if self.track_path and (
            self.evaluations % max(1, self.n_trials // 20) == 0 or self.evaluations == 1
        ):
            # `.copy()` exists on both numpy.ndarray and `_Vec` (which is a
            # list subclass). Either yields an independent snapshot.
            self.path.append(x_clipped.copy())

        if value < self.best_value:
            self.best_value = value
            self.best_x = x_clipped.copy()

        return value
