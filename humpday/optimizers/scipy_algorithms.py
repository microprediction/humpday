"""
SciPy-based algorithm implementations: Nelder-Mead, Powell, and L-BFGS-B.

These are pure Python implementations of classical optimization algorithms
that are commonly found in SciPy. Well-established and reliable methods
for derivative-free and gradient-based optimization.

Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html
"""

import math

from humpday import _array as _A

from .base import BaseOptimizer


class NelderMead(BaseOptimizer):
    """Nelder-Mead simplex algorithm (faithful to SciPy implementation).

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    The simplex is stored as a Python list of n+1 vectors (each a `_Vec`
    or numpy.ndarray, depending on backend) instead of as a 2-D array;
    row operations become list comprehensions, fancy indexing becomes
    explicit `[sim[i] for i in order]`, and `np.argsort` becomes
    `sorted(range(...), key=fsim.__getitem__)`.
    """

    def optimize(self):
        n = self.n_dim

        # SciPy parameter values.
        rho = 1.0  # reflection
        chi = 2.0  # expansion
        psi = 0.5  # contraction
        sigma = 0.5  # shrinkage

        # SciPy simplex-initialization constants.
        nonzdelt = 0.05
        zdelt = 0.00025

        # Initial simplex: n+1 vertices. Vertex 0 is an interior random
        # point; the rest perturb one coordinate at a time.
        x0 = 0.3 + 0.4 * _A.random_uniform(n)
        sim = [x0.copy()]
        for k in range(n):
            y = x0.copy()
            if y[k] != 0:
                y[k] = (1 + nonzdelt) * y[k]
            else:
                y[k] = zdelt
            sim.append(y)

        # Clip every vertex into the unit cube.
        sim = [_A.clip(v, 0, 1) for v in sim]

        # Evaluate initial simplex.
        fsim = [0.0] * (n + 1)
        for k in range(n + 1):
            if self.evaluations >= self.n_trials:
                break
            fsim[k] = self.evaluate(sim[k])

        # Sort by fitness ascending (best first).
        order = sorted(range(n + 1), key=fsim.__getitem__)
        sim = [sim[i] for i in order]
        fsim = [fsim[i] for i in order]

        # SciPy tolerances.
        xatol = 1e-4
        fatol = 1e-4

        while self.evaluations < self.n_trials:
            # SciPy convergence check, restated without numpy broadcasting:
            # max over all (i, k) of |sim[i][k] - sim[0][k]| <= xatol AND
            # max over i of |fsim[0] - fsim[i]| <= fatol.
            x_max = max(
                abs(sim[i][k] - sim[0][k]) for i in range(1, n + 1) for k in range(n)
            )
            f_max = max(abs(fsim[0] - fsim[i]) for i in range(1, n + 1))
            if x_max <= xatol and f_max <= fatol:
                break

            # Centroid of the best n vertices (sum-of-rows / n).
            xbar = _A.zeros(n)
            for v in sim[:-1]:
                xbar = xbar + v
            xbar = xbar / n

            # Reflection.
            xr = _A.clip((1 + rho) * xbar - rho * sim[-1], 0, 1)
            if self.evaluations >= self.n_trials:
                break
            fxr = self.evaluate(xr)

            if fxr < fsim[0]:
                # Expansion.
                xe = _A.clip((1 + rho * chi) * xbar - rho * chi * sim[-1], 0, 1)
                if self.evaluations >= self.n_trials:
                    break
                fxe = self.evaluate(xe)
                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            elif fxr < fsim[-2]:
                # Accept reflection.
                sim[-1] = xr
                fsim[-1] = fxr
            else:
                # Contraction (outside if fxr < fsim[-1], else inside).
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                else:
                    xc = (1 - psi) * xbar + psi * sim[-1]
                xc = _A.clip(xc, 0, 1)

                if self.evaluations >= self.n_trials:
                    break
                fxc = self.evaluate(xc)

                if fxc < min(fxr, fsim[-1]):
                    sim[-1] = xc
                    fsim[-1] = fxc
                else:
                    # Shrink: every non-best vertex moves toward sim[0].
                    for j in range(1, n + 1):
                        sim[j] = _A.clip(sim[0] + sigma * (sim[j] - sim[0]), 0, 1)
                        if self.evaluations < self.n_trials:
                            fsim[j] = self.evaluate(sim[j])

            # Re-sort by fitness.
            order = sorted(range(n + 1), key=fsim.__getitem__)
            sim = [sim[i] for i in order]
            fsim = [fsim[i] for i in order]

        return self.best_value, self.best_x


class Powell(BaseOptimizer):
    """Powell's method - direct adaptation from SciPy source code.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    The direction set `direc` is stored as a list of rows; row accesses
    are wrapped in `_A.asarray(...)` so they become `_Vec` (pure) or
    ndarray (numpy) and support elementwise arithmetic on either path.
    """

    def optimize(self):
        n = self.n_dim
        x = 0.3 + 0.4 * _A.random_uniform(n)  # Interior start

        # Initial direction set: identity matrix (coordinate directions).
        direc = _A.linalg.eye(n)

        ftol = 1e-4
        maxiter = n * 20

        fval = self.evaluate(x)
        x1 = x.copy()
        iteration = 0

        while iteration < maxiter and self.evaluations < self.n_trials:
            fx = fval
            bigind = 0
            delta = 0.0

            # Line searches along each direction.
            for i in range(n):
                if self.evaluations >= self.n_trials:
                    break

                direc1 = _A.asarray(direc[i])
                fx2 = fval
                fval, x, direc1 = self._linesearch_powell(x, direc1, fval)

                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i

            iteration += 1

            # SciPy convergence check.
            bnd = ftol * (abs(fx) + abs(fval)) + 1e-20
            if 2.0 * (fx - fval) <= bnd:
                break

            if self.evaluations >= self.n_trials:
                break

            # Construct the extrapolated point.
            direc1 = x - x1
            x1 = x.copy()
            x2 = _A.clip(x + direc1, 0, 1)

            if self.evaluations >= self.n_trials:
                break

            fx2 = self.evaluate(x2)

            if fx > fx2:
                t = 2.0 * (fx + fx2 - 2.0 * fval)
                temp = fx - fval - delta
                t *= temp * temp
                temp = fx - fx2
                t -= delta * temp * temp

                if t < 0.0:
                    fval, x, direc1 = self._linesearch_powell(x, direc1, fval)
                    # `np.any(arr)` for a float vector means "any non-zero entry".
                    if any(v != 0 for v in direc1):
                        direc[bigind] = list(direc[-1])
                        direc[-1] = list(direc1)

        return self.best_value, self.best_x

    def _linesearch_powell(self, p, xi, fval):
        """Bounded golden-section line search along direction `xi` from
        point `p`. Returns `(best_f, best_x, scaled_direction)`."""

        def myfunc(alpha):
            x_trial = _A.clip(p + alpha * xi, 0, 1)
            if self.evaluations >= self.n_trials:
                return float("inf")
            return self.evaluate(x_trial)

        # If direction is essentially zero, skip the search.
        if not any(v != 0 for v in xi):
            return fval, p, xi

        # Clamp alpha so p + alpha * xi stays in [0, 1]^n.
        alpha_bounds = []
        for i in range(len(xi)):
            xi_i = float(xi[i])
            if abs(xi_i) > 1e-12:
                bound1 = -float(p[i]) / xi_i
                bound2 = (1.0 - float(p[i])) / xi_i
                alpha_bounds.extend([bound1, bound2])

        if not alpha_bounds:
            return fval, p, xi

        alpha_min = max(min(alpha_bounds), -1.0)
        alpha_max = min(max(alpha_bounds), 1.0)

        if alpha_max <= alpha_min:
            return fval, p, xi

        # Golden-section search.
        golden_ratio = (3.0 - math.sqrt(5.0)) / 2.0
        tol = 1e-6
        max_evals = min(10, self.n_trials - self.evaluations)

        if abs(alpha_min) < abs(alpha_max):
            a, c = alpha_min, alpha_max
        else:
            a, c = alpha_max, alpha_min

        b = a + golden_ratio * (c - a)

        if self.evaluations >= self.n_trials:
            return fval, p, xi
        fa = fval if abs(a) < 1e-10 else myfunc(a)

        if self.evaluations >= self.n_trials:
            return fval, p, xi
        fc = myfunc(c)

        if self.evaluations >= self.n_trials:
            return fval, p, xi
        fb = myfunc(b)

        best_alpha = a
        best_f = fa
        if fb < best_f:
            best_alpha = b
            best_f = fb
        if fc < best_f:
            best_alpha = c
            best_f = fc

        evaluations_used = 3
        while (
            evaluations_used < max_evals
            and abs(c - a) > tol
            and self.evaluations < self.n_trials
        ):
            if c - b > b - a:
                # Larger interval is on the right.
                x = b + golden_ratio * (c - b)
                fx = myfunc(x)
                evaluations_used += 1
                if fx < fb:
                    a, b = b, x
                    fa, fb = fb, fx
                    if fx < best_f:
                        best_alpha = x
                        best_f = fx
                else:
                    c = x
                    fc = fx
            else:
                # Larger interval is on the left.
                x = b - golden_ratio * (b - a)
                fx = myfunc(x)
                evaluations_used += 1
                if fx < fb:
                    c, b = b, x
                    fc, fb = fb, fx
                    if fx < best_f:
                        best_alpha = x
                        best_f = fx
                else:
                    a = x
                    fa = fx

        if best_f < fval:
            x_new = _A.clip(p + best_alpha * xi, 0, 1)
            xi_new = best_alpha * xi
            return best_f, x_new, xi_new
        else:
            return fval, p, _A.zeros(len(xi))


class LBFGSB(BaseOptimizer):
    """L-BFGS-B (simplified): finite-difference gradient + momentum.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    This implementation is the original LBFGSB code path, retained as-is
    apart from the numpy-to-shim rename. It is named after L-BFGS-B but
    is structurally closer to gradient descent with adaptive step size
    and momentum — see the docstring of the legacy class for context.
    """

    def optimize(self):
        x = 0.3 + 0.4 * _A.random_uniform(self.n_dim)  # Interior start
        f = self.evaluate(x)

        step_size = 0.1
        momentum = _A.zeros(self.n_dim)
        beta = 0.9

        # Finite-difference step.
        eps = 1e-5

        while self.evaluations < self.n_trials:
            # Central-difference gradient.
            grad = _A.zeros(self.n_dim)
            for i in range(self.n_dim):
                if self.evaluations >= self.n_trials:
                    break

                x_plus = x.copy()
                x_plus[i] = min(1.0, x[i] + eps)
                f_plus = self.evaluate(x_plus)

                if self.evaluations >= self.n_trials:
                    break

                x_minus = x.copy()
                x_minus[i] = max(0.0, x[i] - eps)
                f_minus = self.evaluate(x_minus)

                grad[i] = (f_plus - f_minus) / (x_plus[i] - x_minus[i])

            # Convergence check.
            grad_norm = _A.norm(grad)
            if grad_norm < 1e-4:
                break

            # Momentum update.
            momentum = beta * momentum - step_size * grad
            x_new = _A.clip(x + momentum, 0, 1)

            if self.evaluations >= self.n_trials:
                break

            f_new = self.evaluate(x_new)

            # Adaptive step size.
            if f_new < f:
                step_size = min(step_size * 1.05, 0.5)
                x = x_new
                f = f_new
            else:
                step_size *= 0.7
                momentum = momentum * 0.5  # damp momentum on rejection

            if step_size < 1e-6:
                break

        return self.best_value, self.best_x
