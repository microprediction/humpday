"""
Search algorithm implementations.

These algorithms perform systematic or guided search through the solution space,
including methods like coordinate descent, pattern search, and adaptive random search.
They are particularly effective for local optimization and structured exploration.
"""

from typing import Tuple

import numpy as np

from .base import BaseOptimizer


class AdaptiveRandomSearch(BaseOptimizer):
    """Adaptive Random Search with step size adaptation."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        x = np.random.random(self.n_dim)
        f = self.evaluate(x)
        step_size = 0.1
        success_rate = 0.5

        while self.evaluations < self.n_trials:
            # Random step
            direction = np.random.randn(self.n_dim)
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            x_new = np.clip(x + step_size * direction, 0, 1)

            if self.evaluations < self.n_trials:
                f_new = self.evaluate(x_new)

                if f_new < f:
                    x = x_new
                    f = f_new
                    success_rate = 0.8 * success_rate + 0.2 * 1.0
                else:
                    success_rate = 0.8 * success_rate + 0.2 * 0.0

                # Adapt step size
                if success_rate > 0.2:
                    step_size = min(0.3, step_size * 1.1)
                else:
                    step_size = max(0.01, step_size * 0.9)

        return self.best_value, self.best_x


class CoordinateDescent(BaseOptimizer):
    """Coordinate Descent optimization."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        x = np.random.random(self.n_dim)
        f = self.evaluate(x)
        step_size = 0.1

        while self.evaluations < self.n_trials:
            improved = False

            # Cycle through coordinates
            for i in range(self.n_dim):
                if self.evaluations >= self.n_trials:
                    break

                best_x = x[i]
                best_f = f

                # Try steps in both directions
                for direction in [-1, 1]:
                    x_trial = x.copy()
                    x_trial[i] = np.clip(x[i] + direction * step_size, 0, 1)

                    if self.evaluations < self.n_trials:
                        f_trial = self.evaluate(x_trial)
                        if f_trial < best_f:
                            best_x = x_trial[i]
                            best_f = f_trial
                            improved = True

                x[i] = best_x
                f = best_f

            if not improved:
                step_size *= 0.8
                if step_size < 1e-6:
                    step_size = 0.05  # Reset

        return self.best_value, self.best_x


class PatternSearch(BaseOptimizer):
    """Pattern Search algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        x = np.random.random(self.n_dim)
        f = self.evaluate(x)
        step_size = 0.1

        while self.evaluations < self.n_trials:
            improved = False

            # Pattern directions (coordinate directions + diagonals)
            directions = []
            # Coordinate directions
            for i in range(self.n_dim):
                direction = np.zeros(self.n_dim)
                direction[i] = 1
                directions.append(direction)
                directions.append(-direction)

            # Try each direction
            for direction in directions:
                if self.evaluations >= self.n_trials:
                    break

                x_trial = np.clip(x + step_size * direction, 0, 1)
                f_trial = self.evaluate(x_trial)

                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
                    break

            if improved:
                step_size = min(0.3, step_size * 1.2)
            else:
                step_size *= 0.5
                if step_size < 1e-6:
                    # Random restart
                    x = np.random.random(self.n_dim)
                    if self.evaluations < self.n_trials:
                        f = self.evaluate(x)
                    step_size = 0.1

        return self.best_value, self.best_x