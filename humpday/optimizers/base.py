"""
Base class for all pure optimization algorithms.

Provides common functionality for objective evaluation, best value tracking,
and path visualization support.
"""

import random
from typing import Callable, List, Tuple

import numpy as np


class BaseOptimizer:
    """Base class for all pure optimization algorithms."""

    def __init__(self, objective: Callable, n_trials: int, n_dim: int):
        self.objective = objective
        self.n_trials = n_trials
        self.n_dim = n_dim
        self.evaluations = 0
        self.best_value = float("inf")
        self.best_x = np.random.random(n_dim)
        self.track_path = False
        self.path = []

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate objective with tracking."""
        self.evaluations += 1
        x_clipped = np.clip(x, 0, 1)
        value = self.objective(x_clipped)

        # Track path for visualization
        if self.track_path and (
            self.evaluations % max(1, self.n_trials // 20) == 0 or self.evaluations == 1
        ):
            self.path.append(x_clipped.copy())

        if value < self.best_value:
            self.best_value = value
            self.best_x = x_clipped.copy()

        return value
