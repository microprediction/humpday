"""
SciPy-based algorithm implementations: Nelder-Mead, Powell, and L-BFGS-B.

These are pure Python implementations of classical optimization algorithms
that are commonly found in SciPy. Well-established and reliable methods
for derivative-free and gradient-based optimization.

Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html
"""

from typing import Tuple

import numpy as np

from .base import BaseOptimizer


class NelderMead(BaseOptimizer):
    """Nelder-Mead simplex algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n = self.n_dim

        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        values = np.zeros(n + 1)

        # Random initial simplex
        simplex[0] = np.random.random(n)
        values[0] = self.evaluate(simplex[0])

        for i in range(1, n + 1):
            simplex[i] = simplex[0].copy()
            simplex[i][i - 1] = min(1.0, simplex[i][i - 1] + 0.1)
            values[i] = self.evaluate(simplex[i])

        # Nelder-Mead parameters
        alpha, gamma, beta, sigma = 1.0, 2.0, 0.5, 0.5

        while self.evaluations < self.n_trials:
            # Sort simplex
            indices = np.argsort(values)
            simplex = simplex[indices]
            values = values[indices]

            # Centroid of best n points
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            reflected = centroid + alpha * (centroid - simplex[-1])
            reflected = np.clip(reflected, 0, 1)

            if self.evaluations >= self.n_trials:
                break

            f_reflected = self.evaluate(reflected)

            if values[0] <= f_reflected < values[-2]:
                simplex[-1] = reflected
                values[-1] = f_reflected
            elif f_reflected < values[0]:
                # Expansion
                expanded = centroid + gamma * (reflected - centroid)
                expanded = np.clip(expanded, 0, 1)
                if self.evaluations < self.n_trials:
                    f_expanded = self.evaluate(expanded)
                    if f_expanded < f_reflected:
                        simplex[-1] = expanded
                        values[-1] = f_expanded
                    else:
                        simplex[-1] = reflected
                        values[-1] = f_reflected
            else:
                # Contraction
                if f_reflected < values[-1]:
                    contracted = centroid + beta * (reflected - centroid)
                else:
                    contracted = centroid + beta * (simplex[-1] - centroid)

                contracted = np.clip(contracted, 0, 1)
                if self.evaluations < self.n_trials:
                    f_contracted = self.evaluate(contracted)
                    if f_contracted < min(f_reflected, values[-1]):
                        simplex[-1] = contracted
                        values[-1] = f_contracted
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], 0, 1)
                            if self.evaluations < self.n_trials:
                                values[i] = self.evaluate(simplex[i])

        return self.best_value, self.best_x


class Powell(BaseOptimizer):
    """Powell's conjugate direction method."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n = self.n_dim
        x = np.random.random(n)
        f = self.evaluate(x)

        # Initial direction set (coordinate directions)
        directions = np.eye(n)
        step_size = 0.1

        while self.evaluations < self.n_trials:
            x_start = x.copy()

            # Line searches along each direction
            for i in range(n):
                if self.evaluations >= self.n_trials:
                    break

                direction = directions[i]

                # Simple line search
                best_step = 0
                best_val = f

                for step in [-step_size, step_size]:
                    x_trial = np.clip(x + step * direction, 0, 1)
                    if self.evaluations < self.n_trials:
                        f_trial = self.evaluate(x_trial)
                        if f_trial < best_val:
                            best_val = f_trial
                            best_step = step

                if best_step != 0:
                    x = np.clip(x + best_step * direction, 0, 1)
                    f = best_val

            # Update direction set
            if not np.allclose(x, x_start):
                new_direction = x - x_start
                new_direction = new_direction / (np.linalg.norm(new_direction) + 1e-10)
                # Replace last direction
                directions[-1] = new_direction

        return self.best_value, self.best_x


class LBFGSB(BaseOptimizer):
    """L-BFGS-B algorithm (simplified)."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        x = np.random.random(self.n_dim)
        f = self.evaluate(x)

        # Simple gradient-based optimization
        step_size = 0.01
        momentum = np.zeros(self.n_dim)
        beta = 0.9

        while self.evaluations < self.n_trials:
            # Finite difference gradient
            grad = np.zeros(self.n_dim)
            eps = 1e-6

            for i in range(self.n_dim):
                if self.evaluations >= self.n_trials:
                    break

                x_plus = x.copy()
                x_plus[i] = min(1.0, x_plus[i] + eps)
                f_plus = self.evaluate(x_plus)

                grad[i] = (f_plus - f) / eps

            # Update with momentum
            momentum = beta * momentum - step_size * grad
            x_new = np.clip(x + momentum, 0, 1)

            if self.evaluations < self.n_trials:
                f_new = self.evaluate(x_new)
                if f_new < f:
                    x = x_new
                    f = f_new
                else:
                    step_size *= 0.8

        return self.best_value, self.best_x