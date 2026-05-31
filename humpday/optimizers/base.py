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

    # ------------------------------------------------------------------ #
    # Shared L-BFGS polish.                                              #
    # ------------------------------------------------------------------ #
    # Used by DifferentialEvolution (matches scipy.differential_evolution
    # `polish=True`), SimulatedAnnealing (matches scipy.dual_annealing's
    # local-search stage), and BayesianOpt (refines the GP-EI best).
    # All three reference implementations finish with an L-BFGS-B polish;
    # this is the pure-Python equivalent: two-loop recursion with FD
    # gradient + Armijo backtracking. Mirrors the JS port in
    # `docs/js/modules/evolutionary-algorithms.js`.
    def _lbfgs_polish(self):
        n = self.n_dim
        x = self.best_x.copy()
        f = self.best_value
        grad = self._fd_gradient_for_polish(x)

        memory = min(5, n)
        s_list: list = []
        y_list: list = []

        while self.evaluations < self.n_trials - 2 * n:
            direction = [-float(g) for g in grad]
            alpha = [0.0] * len(s_list)
            for i in range(len(s_list) - 1, -1, -1):
                sy = sum(float(s_list[i][k]) * float(y_list[i][k]) for k in range(n))
                if abs(sy) < 1e-30:
                    continue
                rho = 1.0 / sy
                alpha[i] = rho * sum(
                    float(s_list[i][k]) * direction[k] for k in range(n)
                )
                direction = [
                    direction[k] - alpha[i] * float(y_list[i][k]) for k in range(n)
                ]
            for i in range(len(s_list)):
                sy = sum(float(s_list[i][k]) * float(y_list[i][k]) for k in range(n))
                if abs(sy) < 1e-30:
                    continue
                rho = 1.0 / sy
                beta = rho * sum(float(y_list[i][k]) * direction[k] for k in range(n))
                direction = [
                    direction[k] + (alpha[i] - beta) * float(s_list[i][k])
                    for k in range(n)
                ]

            c1 = 1e-4
            gd = sum(float(grad[k]) * direction[k] for k in range(n))
            step = 1.0
            new_x = x.copy()
            new_f = f
            if gd < -1e-30:
                while step > 1e-12:
                    if self.evaluations >= self.n_trials:
                        break
                    candidate = _A.clip(
                        _A.asarray(
                            [float(x[k]) + step * direction[k] for k in range(n)]
                        ),
                        0,
                        1,
                    )
                    cand_f = self.evaluate(candidate)
                    if cand_f <= f + c1 * step * gd:
                        new_x = candidate
                        new_f = cand_f
                        break
                    step *= 0.5
            else:
                # Non-descent direction (numerical drift). Reset memory.
                s_list.clear()
                y_list.clear()

            new_grad = self._fd_gradient_for_polish(new_x)
            if float(_A.norm(new_grad)) < 1e-6:
                break

            s = _A.asarray([float(new_x[k]) - float(x[k]) for k in range(n)])
            y = _A.asarray([float(new_grad[k]) - float(grad[k]) for k in range(n)])
            if float(_A.dot(s, y)) > 1e-12:
                s_list.append(s)
                y_list.append(y)
                if len(s_list) > memory:
                    s_list.pop(0)
                    y_list.pop(0)
            x, f, grad = new_x, new_f, new_grad

    def _fd_gradient_for_polish(self, x):
        """Central-difference gradient with budget guards (mirrors
        LBFGSB._fd_gradient)."""
        n = self.n_dim
        h = 1e-6
        grad = [0.0] * n
        for i in range(n):
            if self.evaluations >= self.n_trials:
                break
            x_plus = x.copy()
            x_plus[i] = min(1.0, float(x[i]) + h)
            f_plus = self.evaluate(x_plus)
            if self.evaluations >= self.n_trials:
                break
            x_minus = x.copy()
            x_minus[i] = max(0.0, float(x[i]) - h)
            f_minus = self.evaluate(x_minus)
            denom = float(x_plus[i]) - float(x_minus[i])
            if denom > 0:
                grad[i] = (f_plus - f_minus) / denom
        return _A.asarray(grad)
