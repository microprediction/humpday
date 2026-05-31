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
    # Shared L-BFGS-B (Byrd-Lu-Nocedal-Zhu 1995, simple-bounds form).    #
    # ------------------------------------------------------------------ #
    # Used as a polish stage by DifferentialEvolution (matches
    # scipy.differential_evolution `polish=True`), SimulatedAnnealing
    # (matches scipy.dual_annealing's local-search stage), and BayesianOpt.
    # Also used directly by the LBFGSB optimizer.
    #
    # This is a faithful pure-Python port of scipy's `_minimize_lbfgsb`
    # with four elements the previous "two-loop + Armijo" did not have:
    #
    #   1. **Bound-aware direction projection** — when x[i] sits on a
    #      bound and the unconstrained direction would push past it,
    #      that component is zeroed (the "Cauchy-point" idea reduced
    #      to its essence for simple bounds).
    #   2. **Projected-gradient convergence test** — terminate when the
    #      sup-norm of the bound-projected gradient falls below `pgtol`,
    #      not the unconstrained gradient norm.
    #   3. **f-tolerance termination** — terminate when the relative
    #      decrease in f falls below `factr * eps_mach`, scipy's
    #      headline convergence criterion.
    #   4. **Feasibility-limited step** — cap the initial step length so
    #      x + step*direction stays in [0,1]^n before backtracking,
    #      avoiding wasted line-search iterations that get clipped.
    #
    # See: Byrd, Lu, Nocedal, Zhu (1995), "A Limited Memory Algorithm
    # for Bound Constrained Optimization", SIAM J. Sci. Comput. 16(5).
    # scipy reference: scipy/optimize/lbfgsb_src/ + _minimize_lbfgsb.

    # scipy's default `factr` is 1e7 (moderate accuracy). 1e2 = "high
    # accuracy" per scipy's docstring; we use the moderate default.
    _LBFGS_FACTR = 1e7
    _LBFGS_PGTOL = 1e-5
    _LBFGS_EPS_MACH = 2.220446049250313e-16
    _LBFGS_MEMORY = 10  # scipy `maxcor` default

    def _lbfgs_polish(self):
        n = self.n_dim
        memory = min(self._LBFGS_MEMORY, max(1, n))

        x = self.best_x.copy()
        f = self.best_value
        grad = self._fd_gradient_for_polish(x)

        s_list: list = []
        y_list: list = []

        # Budget reservation: each iteration costs at least one gradient
        # (2·n evals) plus one or more candidate evaluations. Stop early
        # enough to leave room for the final gradient computation.
        while self.evaluations < self.n_trials - 2 * n:
            # (1) Projected-gradient convergence test.
            pg_inf = self._proj_grad_sup_norm(x, grad)
            if pg_inf < self._LBFGS_PGTOL:
                break

            # (2) Two-loop recursion for the unconstrained L-BFGS direction.
            direction = self._lbfgs_two_loop(grad, s_list, y_list)

            # (3) Project direction at active bounds: zero out components
            # that would push x past the bound it already sits on. This
            # is the simple-bounds reduction of the Cauchy-point step.
            for k in range(n):
                xk = float(x[k])
                if xk <= 0.0 and direction[k] < 0.0:
                    direction[k] = 0.0
                elif xk >= 1.0 and direction[k] > 0.0:
                    direction[k] = 0.0

            gd = sum(float(grad[k]) * direction[k] for k in range(n))
            if gd > -1e-30:
                # Direction isn't a descent (zero curvature, memory drift,
                # or every component clipped). Reset memory and fall back
                # to the projected-gradient steepest-descent direction.
                s_list.clear()
                y_list.clear()
                direction = [-float(g) for g in grad]
                for k in range(n):
                    xk = float(x[k])
                    if xk <= 0.0 and direction[k] < 0.0:
                        direction[k] = 0.0
                    elif xk >= 1.0 and direction[k] > 0.0:
                        direction[k] = 0.0
                gd = sum(float(grad[k]) * direction[k] for k in range(n))
                if gd > -1e-30:
                    break  # truly stuck — projected gradient is zero

            # (4) Cap step length so x + step*direction stays in [0,1]^n.
            step_max = float("inf")
            for k in range(n):
                dk = direction[k]
                if dk > 0.0:
                    step_max = min(step_max, (1.0 - float(x[k])) / dk)
                elif dk < 0.0:
                    step_max = min(step_max, (0.0 - float(x[k])) / dk)
            step = min(1.0, step_max) if step_max > 0.0 else 1.0

            # (5) Armijo backtracking with feasibility-clipped candidates.
            c1 = 1e-4
            new_x = x.copy()
            new_f = f
            accepted = False
            while step > 1e-12:
                if self.evaluations >= self.n_trials:
                    break
                candidate = _A.clip(
                    _A.asarray([float(x[k]) + step * direction[k] for k in range(n)]),
                    0,
                    1,
                )
                cand_f = self.evaluate(candidate)
                if cand_f <= f + c1 * step * gd:
                    new_x = candidate
                    new_f = cand_f
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                break  # line search failed — no further descent

            # (6) f-tolerance termination: stop when the relative decrease
            # falls below scipy's `factr * eps_mach` criterion.
            f_scale = max(abs(f), abs(new_f), 1.0)
            if (f - new_f) < self._LBFGS_FACTR * self._LBFGS_EPS_MACH * f_scale:
                x, f = new_x, new_f
                break

            new_grad = self._fd_gradient_for_polish(new_x)

            # (7) L-BFGS memory update.
            s = _A.asarray([float(new_x[k]) - float(x[k]) for k in range(n)])
            y = _A.asarray([float(new_grad[k]) - float(grad[k]) for k in range(n)])
            if float(_A.dot(s, y)) > 1e-12:
                s_list.append(s)
                y_list.append(y)
                if len(s_list) > memory:
                    s_list.pop(0)
                    y_list.pop(0)
            x, f, grad = new_x, new_f, new_grad

    def _lbfgs_two_loop(self, grad, s_list, y_list):
        """Two-loop recursion for the L-BFGS search direction
        (Nocedal 1980)."""
        n = len(grad)
        direction = [-float(g) for g in grad]
        alpha = [0.0] * len(s_list)
        for i in range(len(s_list) - 1, -1, -1):
            sy = sum(float(s_list[i][k]) * float(y_list[i][k]) for k in range(n))
            if abs(sy) < 1e-30:
                continue
            rho = 1.0 / sy
            alpha[i] = rho * sum(float(s_list[i][k]) * direction[k] for k in range(n))
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
                direction[k] + (alpha[i] - beta) * float(s_list[i][k]) for k in range(n)
            ]
        return direction

    def _proj_grad_sup_norm(self, x, grad):
        """Sup-norm of the bound-projected gradient: ||P(x - g) - x||_inf.

        For simple bounds [0, 1] this reduces to clipping g componentwise
        wherever a bound is active in the steepest-descent direction —
        the standard convergence criterion for box-constrained L-BFGS.
        """
        n = len(grad)
        m = 0.0
        for k in range(n):
            gk = float(grad[k])
            xk = float(x[k])
            if xk <= 0.0 and gk > 0.0:
                continue  # bound active, gradient points outward
            if xk >= 1.0 and gk < 0.0:
                continue  # bound active, gradient points outward
            if abs(gk) > m:
                m = abs(gk)
        return m

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
