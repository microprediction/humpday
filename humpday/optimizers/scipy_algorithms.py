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
    """Powell's method with golden section line search."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n = self.n_dim
        x = 0.3 + 0.4 * np.random.random(n)  # Start in interior
        f = self.evaluate(x)

        # Initial direction set (coordinate directions)
        directions = np.eye(n)

        max_iterations = min(20, self.n_trials // (n + 1))

        for iteration in range(max_iterations):
            if self.evaluations >= self.n_trials:
                break

            x_start = x.copy()
            f_start = f

            # Line searches along each direction
            for i in range(n):
                if self.evaluations >= self.n_trials:
                    break

                direction = directions[i]
                x_new, f_new = self._golden_section_line_search(x, f, direction)
                x, f = x_new, f_new

            # Check for convergence
            if np.linalg.norm(x - x_start) < 1e-8:
                break

            # Update directions (Powell's conjugate direction update)
            if not np.allclose(x, x_start):
                new_direction = x - x_start
                new_direction = new_direction / (np.linalg.norm(new_direction) + 1e-12)

                # Replace the direction that contributed least
                directions = np.vstack([directions[1:], new_direction.reshape(1, -1)])

                # Additional line search along new conjugate direction
                if self.evaluations < self.n_trials:
                    x_new, f_new = self._golden_section_line_search(x, f, new_direction)
                    x, f = x_new, f_new

        return self.best_value, self.best_x

    def _golden_section_line_search(self, x, f, direction, max_step=0.5):
        """Golden section line search along direction."""
        # Golden ratio constants
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi

        # Find bracketing interval [a, b] where minimum lies
        a, b = self._bracket_minimum(x, f, direction, max_step)

        if abs(b - a) < 1e-10:
            return x, f

        # Golden section search
        tol = 1e-6
        c = a + resphi * (b - a)
        d = a + (1 - resphi) * (b - a)

        # Evaluate at c and d
        x_c = np.clip(x + c * direction, 0, 1)
        x_d = np.clip(x + d * direction, 0, 1)

        if self.evaluations >= self.n_trials:
            return x, f

        f_c = self.evaluate(x_c)

        if self.evaluations >= self.n_trials:
            return x_c if f_c < f else x, min(f_c, f)

        f_d = self.evaluate(x_d)

        # Main golden section loop
        while abs(b - a) > tol and self.evaluations < self.n_trials - 1:
            if f_c < f_d:
                b = d
                d = c
                f_d = f_c
                c = a + resphi * (b - a)
                x_c = np.clip(x + c * direction, 0, 1)
                f_c = self.evaluate(x_c)
            else:
                a = c
                c = d
                f_c = f_d
                d = a + (1 - resphi) * (b - a)
                x_d = np.clip(x + d * direction, 0, 1)
                f_d = self.evaluate(x_d)

            if self.evaluations >= self.n_trials:
                break

        # Return best point found
        if f_c < f_d:
            best_alpha = c
            best_f = f_c
        else:
            best_alpha = d
            best_f = f_d

        x_best = np.clip(x + best_alpha * direction, 0, 1)
        return (x_best, best_f) if best_f < f else (x, f)

    def _bracket_minimum(self, x, f, direction, max_step):
        """Find bracketing interval [a, b] containing minimum."""
        # Start with small step
        step = 0.01
        a = 0.0  # Start at current point

        # Find initial bracket
        x_step = np.clip(x + step * direction, 0, 1)

        if self.evaluations >= self.n_trials:
            return a, step

        f_step = self.evaluate(x_step)

        if f_step > f:
            # Minimum is between 0 and step, try negative direction
            neg_step = -step
            x_neg = np.clip(x + neg_step * direction, 0, 1)

            if self.evaluations >= self.n_trials:
                return neg_step, a

            f_neg = self.evaluate(x_neg)

            if f_neg < f:
                # Minimum in negative direction
                return self._expand_bracket(x, f, direction, neg_step, -1, max_step)
            else:
                # Minimum between negative step and positive step
                return neg_step, step
        else:
            # Minimum is beyond step, expand in positive direction
            return self._expand_bracket(x, f, direction, step, 1, max_step)

    def _expand_bracket(self, x, f, direction, initial_step, sign, max_step):
        """Expand bracket to find interval containing minimum."""
        a = 0.0 if sign > 0 else initial_step
        b = initial_step if sign > 0 else 0.0
        step = abs(initial_step)

        # Expand bracket until we bracket the minimum
        for _ in range(10):  # Max expansions
            if self.evaluations >= self.n_trials - 1:
                break

            step *= 2
            if step > max_step:
                step = max_step

            test_alpha = sign * step
            x_test = np.clip(x + test_alpha * direction, 0, 1)
            f_test = self.evaluate(x_test)

            if f_test > f or step >= max_step:
                # Found bracket or hit max step
                if sign > 0:
                    return a, test_alpha
                else:
                    return test_alpha, a

            # Update bracket
            if sign > 0:
                a = b
                b = test_alpha
            else:
                b = a
                a = test_alpha

        return a, b


class LBFGSB(BaseOptimizer):
    """L-BFGS-B algorithm - Limited memory BFGS with bounds."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n = self.n_dim
        x = np.random.random(n)
        f = self.evaluate(x)

        # L-BFGS-B parameters
        m = min(5, n)  # Memory limit
        eps = 1e-8  # Finite difference step

        # History storage
        s_hist = []  # x differences
        y_hist = []  # gradient differences
        rho_hist = []  # 1 / (s^T y)

        # Compute initial gradient
        grad = self._finite_difference_gradient(x, f, eps)

        # Initialize inverse Hessian approximation (scaled identity)
        H0_factor = 1.0

        iteration = 0
        max_iterations = min(50, self.n_trials // (n + 1))

        while self.evaluations < self.n_trials and iteration < max_iterations:
            # Compute search direction using L-BFGS two-loop recursion
            q = grad.copy()

            # First loop (backward)
            alphas = []
            for i in range(len(s_hist) - 1, -1, -1):
                alpha = rho_hist[i] * np.dot(s_hist[i], q)
                alphas.insert(0, alpha)
                q -= alpha * y_hist[i]

            # Apply initial Hessian approximation
            r = H0_factor * q

            # Second loop (forward)
            for i in range(len(s_hist)):
                beta = rho_hist[i] * np.dot(y_hist[i], r)
                r += s_hist[i] * (alphas[i] - beta)

            direction = -r

            # Line search with bounds projection
            step_length = self._backtracking_line_search(x, f, grad, direction)

            if step_length is None or step_length < 1e-12:
                break

            # Update
            x_new = np.clip(x + step_length * direction, 0, 1)

            if self.evaluations >= self.n_trials:
                break

            f_new = self.evaluate(x_new)

            if f_new >= f:  # Line search failed
                break

            # Compute new gradient
            grad_new = self._finite_difference_gradient(x_new, f_new, eps)

            # Update history
            s_k = x_new - x
            y_k = grad_new - grad

            # Check curvature condition
            sy = np.dot(s_k, y_k)
            if sy > 1e-10:
                # Add to history
                s_hist.append(s_k)
                y_hist.append(y_k)
                rho_hist.append(1.0 / sy)

                # Maintain memory limit
                if len(s_hist) > m:
                    s_hist.pop(0)
                    y_hist.pop(0)
                    rho_hist.pop(0)

                # Update initial Hessian scaling
                H0_factor = sy / np.dot(y_k, y_k)

            x, f, grad = x_new, f_new, grad_new
            iteration += 1

        return self.best_value, self.best_x

    def _finite_difference_gradient(self, x, f, eps):
        """Compute gradient using finite differences."""
        grad = np.zeros(self.n_dim)

        for i in range(self.n_dim):
            if self.evaluations >= self.n_trials:
                break

            # Forward difference with bound checking
            x_plus = x.copy()
            if x[i] + eps <= 1.0:
                x_plus[i] = x[i] + eps
                f_plus = self.evaluate(x_plus)
                grad[i] = (f_plus - f) / eps
            else:
                # Backward difference near upper bound
                x_minus = x.copy()
                x_minus[i] = max(0.0, x[i] - eps)
                f_minus = self.evaluate(x_minus)
                grad[i] = (f - f_minus) / eps

        return grad

    def _backtracking_line_search(self, x, f, grad, direction, c1=1e-4, alpha_max=1.0):
        """Backtracking line search with Armijo condition."""
        alpha = min(alpha_max, 1.0)
        rho = 0.8  # Backtracking factor

        # Ensure we don't violate bounds
        for i in range(self.n_dim):
            if direction[i] > 0:
                alpha = min(alpha, (1.0 - x[i]) / (direction[i] + 1e-10))
            elif direction[i] < 0:
                alpha = min(alpha, x[i] / (-direction[i] + 1e-10))

        armijo_condition = c1 * np.dot(grad, direction)

        for _ in range(10):  # Max backtracking steps
            if self.evaluations >= self.n_trials:
                return None

            x_new = np.clip(x + alpha * direction, 0, 1)
            f_new = self.evaluate(x_new)

            # Armijo condition
            if f_new <= f + alpha * armijo_condition:
                return alpha

            alpha *= rho

            if alpha < 1e-10:
                break

        return alpha if alpha >= 1e-10 else None
