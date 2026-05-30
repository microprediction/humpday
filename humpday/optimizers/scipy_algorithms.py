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

        # Convergence tolerances. scipy's defaults are 1e-4, but those
        # cause termination well before the budget is exhausted on easy
        # landscapes — on sphere this stops at f ≈ 1e-9 after ~45 evals
        # of a 200-budget. Tightening to 1e-12 lets the algorithm use its
        # budget where the landscape permits, with no downside on hard
        # problems (it just stays in the contraction phase a little
        # longer before terminating on its own at the budget cap).
        xatol = 1e-12
        fatol = 1e-12

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

        # As with NelderMead, scipy's default ftol=1e-4 stops the
        # iteration far before the budget is exhausted on smooth
        # problems. Tightening to 1e-12 lets Powell use its budget.
        ftol = 1e-12
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
        """Bounded Brent-method line search along direction `xi` from
        point `p`. Returns `(best_f, best_x, scaled_direction)`.

        Ported from scipy.optimize._optimize.Brent.optimize and the
        downhill-walk `bracket` routine that initialises it. Compared
        to the previous bounded golden-section, Brent reaches superlinear
        convergence on smooth landscapes by interleaving inverse
        quadratic interpolation with golden-section fallbacks, and is
        what gives scipy's Powell its sharpness on convex problems.

        Adaptations:
          - The bracket-grow step stops at the alpha bounds (so we stay
            inside `[0, 1]^n`) instead of growing freely as scipy does.
          - Every function evaluation is gated by the remaining budget.
          - On budget exhaustion we return the best alpha seen so far,
            which preserves the caller's invariant that the line search
            never worsens `fval`.
        """

        def myfunc(alpha):
            x_trial = _A.clip(p + alpha * xi, 0, 1)
            return self.evaluate(x_trial)

        # If direction is essentially zero, skip the search.
        if not any(v != 0 for v in xi):
            return fval, p, xi

        # Clamp alpha so p + alpha * xi stays in [0, 1]^n.
        # Each non-zero `xi[i]` gives two alpha values where the
        # corresponding coordinate hits 0 or 1; the intersection of all
        # such intervals is [alpha_min, alpha_max].
        alpha_lo = float("-inf")
        alpha_hi = float("inf")
        for i in range(len(xi)):
            xi_i = float(xi[i])
            if abs(xi_i) <= 1e-12:
                continue
            b1 = -float(p[i]) / xi_i
            b2 = (1.0 - float(p[i])) / xi_i
            lo, hi = (b1, b2) if b1 < b2 else (b2, b1)
            if lo > alpha_lo:
                alpha_lo = lo
            if hi < alpha_hi:
                alpha_hi = hi

        if alpha_hi <= alpha_lo:
            return fval, p, xi
        # Cap to a sensible range. The original bounded golden-section
        # used [-1, 1] so the magnitude of `xi_new` stayed comparable to
        # `xi`'s; keep that convention.
        alpha_lo = max(alpha_lo, -1.0)
        alpha_hi = min(alpha_hi, 1.0)
        if alpha_hi <= alpha_lo:
            return fval, p, xi

        # Track the best alpha seen across bracket + Brent so that on
        # budget exhaustion we still return progress.
        best_alpha = 0.0
        best_f = fval

        def evaluate(alpha):
            """Run `myfunc(alpha)` with budget check, updating best_*."""
            nonlocal best_alpha, best_f
            if self.evaluations >= self.n_trials:
                return None
            f = myfunc(alpha)
            if f < best_f:
                best_alpha = alpha
                best_f = f
            return f

        # ---- Bracket the minimum (scipy.optimize.bracket adaptation) ----
        # Start two points apart inside the bounded interval. Use a step
        # one decimal of the interval width — tiny enough that a smooth
        # function shows curvature, large enough not to look constant.
        span = alpha_hi - alpha_lo
        xa = 0.0 if alpha_lo < 0 < alpha_hi else alpha_lo
        xb = xa + min(span * 0.1, 1e-1)
        if xb >= alpha_hi:
            xb = alpha_lo + 0.5 * (alpha_hi - alpha_lo)

        fa = evaluate(xa)
        if fa is None:
            return self._linesearch_result(p, xi, fval, best_alpha, best_f)
        fb = evaluate(xb)
        if fb is None:
            return self._linesearch_result(p, xi, fval, best_alpha, best_f)

        if fa < fb:
            xa, xb = xb, xa
            fa, fb = fb, fa

        # Step in the downhill direction (golden-ratio expansion).
        gold = 1.618033988749895
        xc = xb + gold * (xb - xa)
        if xc > alpha_hi:
            xc = alpha_hi
        if xc < alpha_lo:
            xc = alpha_lo

        fc = evaluate(xc)
        if fc is None:
            return self._linesearch_result(p, xi, fval, best_alpha, best_f)

        # Grow the bracket until f starts increasing on the far side
        # or we hit a bound. Capped at a small constant to keep the
        # cost predictable.
        for _ in range(20):
            if fc >= fb:
                break
            # We have a downhill triple — keep walking.
            new_xc = xc + gold * (xc - xb)
            if new_xc > alpha_hi or new_xc < alpha_lo:
                new_xc = alpha_hi if (xc - xb) > 0 else alpha_lo
                if new_xc == xc:
                    break
            xa, xb = xb, xc
            fa, fb = fb, fc
            xc = new_xc
            fc = evaluate(xc)
            if fc is None:
                return self._linesearch_result(p, xi, fval, best_alpha, best_f)

        # If we couldn't bracket (e.g. flat or boundary-attached), bail
        # out gracefully with whatever the best evaluation was.
        if not ((xa < xb < xc) or (xc < xb < xa)) or not (fb <= fa and fb <= fc):
            return self._linesearch_result(p, xi, fval, best_alpha, best_f)

        # ---- Brent's method inside the bracket --------------------------
        if xa > xc:
            a_, b_ = xc, xa
        else:
            a_, b_ = xa, xc

        x = w = v = xb
        fx = fw = fv = fb
        deltax = 0.0
        rat = 0.0
        cg = 0.3819660  # 1 - 1/phi (scipy's _cg)
        brent_tol = 1.48e-3
        mintol = 1.0e-11

        for _ in range(50):
            tol1 = brent_tol * abs(x) + mintol
            tol2 = 2.0 * tol1
            xmid = 0.5 * (a_ + b_)
            if abs(x - xmid) < (tol2 - 0.5 * (b_ - a_)):
                break

            if abs(deltax) <= tol1:
                # Take a golden-section step.
                deltax = (a_ - x) if x >= xmid else (b_ - x)
                rat = cg * deltax
            else:
                # Try an inverse parabolic step.
                tmp1 = (x - w) * (fx - fv)
                tmp2 = (x - v) * (fx - fw)
                p_ = (x - v) * tmp2 - (x - w) * tmp1
                tmp2 = 2.0 * (tmp2 - tmp1)
                if tmp2 > 0.0:
                    p_ = -p_
                tmp2 = abs(tmp2)
                dx_temp = deltax
                deltax = rat
                if (
                    (p_ > tmp2 * (a_ - x))
                    and (p_ < tmp2 * (b_ - x))
                    and (abs(p_) < abs(0.5 * tmp2 * dx_temp))
                ):
                    rat = p_ / tmp2
                    u = x + rat
                    if (u - a_) < tol2 or (b_ - u) < tol2:
                        rat = tol1 if (xmid - x) >= 0 else -tol1
                else:
                    deltax = (a_ - x) if x >= xmid else (b_ - x)
                    rat = cg * deltax

            # Ensure the step is at least tol1.
            if abs(rat) < tol1:
                u = x + (tol1 if rat >= 0 else -tol1)
            else:
                u = x + rat

            fu = evaluate(u)
            if fu is None:
                break

            if fu > fx:
                if u < x:
                    a_ = u
                else:
                    b_ = u
                if fu <= fw or w == x:
                    v, fv = w, fw
                    w, fw = u, fu
                elif fu <= fv or v == x or v == w:
                    v, fv = u, fu
            else:
                if u >= x:
                    a_ = x
                else:
                    b_ = x
                v, fv = w, fw
                w, fw = x, fx
                x, fx = u, fu

        return self._linesearch_result(p, xi, fval, best_alpha, best_f)

    def _linesearch_result(self, p, xi, fval, best_alpha, best_f):
        """Translate the best (alpha, f) pair from the line search back
        into the (f, x, direction) tuple Powell's outer loop expects."""
        if best_f < fval:
            x_new = _A.clip(p + best_alpha * xi, 0, 1)
            xi_new = best_alpha * xi
            return best_f, x_new, xi_new
        else:
            return fval, p, _A.zeros(len(xi))


class LBFGSB(BaseOptimizer):
    """L-BFGS-B with a finite-difference gradient (Byrd–Lu–Nocedal–Zhu).

    Limited-memory BFGS with simple bound constraints — the
    derivative-free workflow uses central-difference gradients (cost:
    2·n_dim evals per iteration) since HumpDay's contract is to take
    a black-box objective. The L-BFGS update itself is the standard
    two-loop recursion of Nocedal (1980) with a 5-pair memory.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Ports the existing JavaScript L-BFGS-B implementation in
    `docs/js/modules/scipy-algorithms.js::LBFGSB` line-for-line.

    Before this rewrite, humpday's `LBFGSB` was a finite-difference
    gradient + Polyak-momentum baseline — not L-BFGS at all. The
    snapshot at `benchmarks/reference_alignment.json` showed it ~6.6e+06×
    worse than scipy's L-BFGS-B on the sphere; the rewrite closes
    that gap to within a few orders of magnitude (the residual is
    HumpDay's FD gradient cost — scipy uses analytical-or-FD with
    cleaner step control).
    """

    def optimize(self):
        n = self.n_dim
        x = _A.random_uniform(n)
        f = self.evaluate(x)
        grad = self._fd_gradient(x)

        memory = min(5, n)
        s_list: list = []  # step vectors x_{k+1} - x_k
        y_list: list = []  # gradient differences ∇f_{k+1} - ∇f_k

        # Reserve a small budget at the end so the per-iteration
        # `numeric_gradient` (which costs 2·n_dim evals) doesn't
        # overshoot the budget. Same guard the JS port uses.
        while self.evaluations < self.n_trials - 2 * n:
            # L-BFGS two-loop recursion to get search direction.
            direction = [-float(gi) for gi in grad]
            alpha = [0.0] * len(s_list)
            for i in range(len(s_list) - 1, -1, -1):
                sy = _A.dot(s_list[i], y_list[i])
                if abs(float(sy)) < 1e-30:
                    continue
                rho = 1.0 / float(sy)
                alpha[i] = rho * float(_A.dot(s_list[i], direction))
                direction = [
                    direction[j] - alpha[i] * float(y_list[i][j]) for j in range(n)
                ]

            for i in range(len(s_list)):
                sy = _A.dot(s_list[i], y_list[i])
                if abs(float(sy)) < 1e-30:
                    continue
                rho = 1.0 / float(sy)
                beta = rho * float(_A.dot(y_list[i], direction))
                direction = [
                    direction[j] + (alpha[i] - beta) * float(s_list[i][j])
                    for j in range(n)
                ]

            # Backtracking line search with Armijo (sufficient-decrease)
            # condition. Starts at step=1 (the natural L-BFGS-B Newton
            # step) and halves until the candidate either accepts or
            # the step shrinks below 1e-12. Replaces the previous
            # fixed-step trial set [0.001, 0.01, 0.1], which was the
            # main reason the rewrite was actually *worse* on Rosenbrock
            # than humpday's old FD+momentum at small budgets — most
            # iterations were wasted on undersized steps.
            c1 = 1e-4
            gd = sum(float(grad[j]) * direction[j] for j in range(n))
            step = 1.0
            new_x = x.copy()
            new_f = f
            if gd < -1e-30:
                while step > 1e-12:
                    if self.evaluations >= self.n_trials:
                        break
                    candidate = _A.clip(
                        _A.asarray(
                            [float(x[j]) + step * direction[j] for j in range(n)]
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
                    # Backtracking exhausted — keep current point.
                    pass
            else:
                # Non-descent direction (most often: zero-curvature
                # update or accumulated numerical drift). Forget the
                # memory and restart with plain steepest-descent next
                # iteration.
                s_list.clear()
                y_list.clear()

            new_grad = self._fd_gradient(new_x)

            # Convergence on gradient norm.
            if float(_A.norm(new_grad)) < 1e-6:
                break

            # Update memory with the (s, y) pair if curvature is positive.
            s = _A.asarray([float(new_x[j]) - float(x[j]) for j in range(n)])
            y = _A.asarray([float(new_grad[j]) - float(grad[j]) for j in range(n)])
            if float(_A.dot(s, y)) > 1e-12:
                s_list.append(s)
                y_list.append(y)
                if len(s_list) > memory:
                    s_list.pop(0)
                    y_list.pop(0)

            x, f, grad = new_x, new_f, new_grad

        return self.best_value, self.best_x

    def _fd_gradient(self, x):
        """Central-difference gradient, costing 2·n_dim evaluations.

        Returns a zero gradient (and stops costing budget) once the
        outer optimizer would overshoot `self.n_trials` — the caller's
        convergence check will see norm 0 and terminate cleanly."""
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
