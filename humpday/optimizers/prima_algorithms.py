"""
PRIMA algorithm implementations: UOBYQA, NEWUOA, and BOBYQA.

**LIBRARY INTENT - CRITICAL:**
- Pure Python implementations of established optimization algorithms
- NO external dependencies except numpy (no scipy, no 3rd party optimization libs)
- Algorithms must be algorithmically correct versions of reference implementations
- 3rd party packages used ONLY in testing for validation/comparison
- Goal: optimization that works anywhere Python runs, no compilation/dependencies

These are pure Python implementations of the PRIMA (Powell's Recent Interpolation Methods)
algorithms by M.J.D. Powell. PRIMA represents state-of-the-art derivative-free optimization
for small to medium-scale problems.

Reference implementations: https://www.pdfo.net/
Must match algorithmic behavior of reference, not use reference directly.
"""

import math
import random

from humpday import _array as _A

from .base import BaseOptimizer


class _PRIMALinAlgError(ValueError):
    """Raised when a PRIMA linear-algebra step fails (singular system,
    rank-deficient interpolation, etc.). Catches in PRIMA optimize() loops
    use the broad `Exception` clause, so this never escapes the algorithm."""


class PRIMA_UOBYQA(BaseOptimizer):
    """PRIMA UOBYQA — quadratic-interpolation trust-region method.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    XPT (interpolation points) is a list-of-vectors; the quadratic model
    coefficient matrix A is a list-of-rows for the SVD/pinv solve.
    """

    def optimize(self):
        n = self.n_dim
        npt = (n + 1) * (n + 2) // 2  # Full quadratic model points

        # Trust region parameters
        rhobeg = 0.5
        rhoend = 1e-8
        rho = rhobeg

        # Initial base point pulled into the interior of [0, 1]^n.
        xbase = _A.clip(0.5 * _A.ones(n), 0.1, 0.9)
        _ = self.evaluate(xbase)

        XPT, FVAL = self._initialize_interpolation_points(xbase, rho, npt, n)
        nused = min(len(FVAL), npt)
        # argmin via stdlib — works on lists, ndarrays, _Vec.
        kopt = min(range(nused), key=FVAL.__getitem__)

        iteration = 0
        max_iterations = min(100, self.n_trials // npt)

        while (
            self.evaluations < self.n_trials
            and rho > rhoend
            and iteration < max_iterations
        ):
            iteration += 1

            try:
                g, H = self._build_robust_quadratic_model(XPT, FVAL, nused, n, kopt)
            except Exception:
                rho *= 0.5
                continue

            try:
                d = self._solve_trust_region_advanced(g, H, rho, n)
            except Exception:
                d = self._fallback_step(g, rho, n)

            xnew = _A.clip(xbase + XPT[kopt] + d, 0, 1)

            if self.evaluations >= self.n_trials:
                break

            fnew = self.evaluate(xnew)

            # Predicted reduction: -(g·d + 0.5 d·H·d).
            Hd = _A.linalg.matvec(H, d)
            predicted_reduction = -(_A.dot(g, d) + 0.5 * _A.dot(d, Hd))
            actual_reduction = FVAL[kopt] - fnew

            # Update interpolation set.
            if nused < npt:
                XPT.append(xnew - xbase)
                FVAL.append(fnew)
                nused += 1
            else:
                # Replace worst point that isn't the current best.
                candidates = [i for i in range(nused) if i != kopt]
                if candidates and fnew < max(FVAL[i] for i in candidates):
                    worst_idx = max(candidates, key=lambda i: FVAL[i])
                    XPT[worst_idx] = xnew - xbase
                    FVAL[worst_idx] = fnew

            kopt = min(range(nused), key=FVAL.__getitem__)

            if abs(predicted_reduction) > 1e-12:
                ratio = actual_reduction / predicted_reduction
            else:
                ratio = 10 if actual_reduction > 0 else 0

            if ratio >= 0.75:
                rho = min(rho * 2.0, rhobeg)
            elif ratio >= 0.25:
                pass  # Keep rho
            else:
                rho = max(rho * 0.5, rhoend)

            # Base point shifting — recentre the model if the best point is
            # too far from the centre.
            if _A.norm(XPT[kopt]) > 0.5 * rho:
                shift = XPT[kopt].copy()
                xbase = _A.clip(xbase + shift, 0, 1)
                for i in range(nused):
                    XPT[i] = XPT[i] - shift

        return self.best_value, self.best_x

    def _build_robust_quadratic_model(self, XPT, FVAL, nused, n, kopt):
        """Build the quadratic model coefficients via SVD-based regression.

        Returns (g, H) where g is the gradient at the base point and H
        is the symmetric Hessian. Solves the over-/under-determined
        Vandermonde-style system Aᵀ c = b with rank-tolerant pinv.
        """
        if nused < n + 1:
            raise _PRIMALinAlgError("Insufficient points")

        ncoeffs = 1 + n + n * (n + 1) // 2

        # Build the design matrix A as list-of-rows.
        A = _A.linalg.matrix_zeros(nused, ncoeffs)
        b = [FVAL[i] - FVAL[kopt] for i in range(nused)]

        for i in range(nused):
            x = XPT[i]
            row = A[i]
            col = 0
            row[col] = 1.0
            col += 1
            for j in range(n):
                row[col] = float(x[j])
                col += 1
            for j in range(n):
                for k in range(j, n):
                    row[col] = (
                        0.5 * float(x[j]) * float(x[k])
                        if j == k
                        else float(x[j]) * float(x[k])
                    )
                    col += 1

        try:
            # Tikhonov-regularised SVD-based solve: pinv(A) @ b with floor
            # on tiny singular values.
            U, s, Vt = _A.linalg.svd(A, full_matrices=False)
            s_floor = s[0] * 1e-12 if len(s) > 0 else 1e-12
            s_safe = [max(float(si), s_floor) for si in s]
            # coeffs = V @ diag(1/s_safe) @ Uᵀ @ b
            UT = _A.linalg.transpose(U)
            UTb = _A.linalg.matvec(UT, b)
            scaled = [UTb[i] / s_safe[i] for i in range(len(s_safe))]
            V = _A.linalg.transpose(Vt)
            coeffs = list(_A.linalg.matvec(V, scaled))
        except Exception:
            coeffs = list(_A.linalg.matvec(_A.linalg.pinv(A), b))

        g = _A.asarray(coeffs[1 : n + 1])
        H = _A.linalg.matrix_zeros(n, n)
        col = n + 1
        for i in range(n):
            for j in range(i, n):
                if col < len(coeffs):
                    H[i][j] = coeffs[col]
                    if i != j:
                        H[j][i] = coeffs[col]
                    col += 1

        return g, H

    def _solve_trust_region_advanced(self, g, H, rho, n):
        """Trust-region subproblem solve. Tries the Newton direction if H is
        positive-definite; falls back to the Cauchy point otherwise."""
        try:
            eigenvals, _Q = _A.linalg.eigh(H)
            min_eigval = min(eigenvals)
            if min_eigval > 1e-8:
                # H is SPD — try the Newton step.
                d_newton = -_A.linalg.solve(H, g)
                if _A.norm(d_newton) <= rho:
                    return d_newton
        except Exception:
            pass
        return self._cauchy_point(g, H, rho)

    def _cauchy_point(self, g, H, rho):
        """Cauchy point along the steepest-descent direction."""
        g_norm = _A.norm(g)
        if g_norm < 1e-12:
            return _A.zeros(len(g))

        Hg = _A.linalg.matvec(H, g)
        gHg = _A.dot(g, Hg)
        if gHg > 0:
            tau = min(1, (g_norm**3) / (rho * gHg))
        else:
            tau = 1

        return -tau * rho * g / g_norm

    def _initialize_interpolation_points(self, xbase, rho, npt, n):
        """Lay out the initial interpolation set. Returns (XPT_list, FVAL_list)
        where XPT is a list of length-n vectors (centred at xbase = 0) and
        FVAL is a parallel list of objective values."""
        XPT = []
        FVAL = []

        # Base point at xbase (offset = 0).
        XPT.append(_A.zeros(n))
        FVAL.append(self.evaluate(xbase))
        if len(FVAL) >= npt:
            return XPT, FVAL

        # ±rho along each coordinate.
        for i in range(n):
            for sign in (+1, -1):
                if len(FVAL) >= npt:
                    return XPT, FVAL
                offset = _A.zeros(n)
                offset[i] = sign * rho
                XPT.append(offset)
                FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

        # Cross-term diagonals at rho/sqrt(2) per coordinate.
        diag_step = rho / math.sqrt(2)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if len(FVAL) >= npt:
                    return XPT, FVAL
                offset = _A.zeros(n)
                offset[i] = diag_step
                offset[j] = diag_step
                XPT.append(offset)
                FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

        return XPT, FVAL

    def _fallback_step(self, g, rho, n):
        """Fallback step when the trust-region solve fails: steepest descent
        scaled to the trust radius, or a small random kick if g vanishes."""
        g_norm = _A.norm(g)
        if g_norm > 1e-12:
            return -rho * g / g_norm
        return (rho / 3.0) * _A.random_normal(n)


class PRIMA_NEWUOA(BaseOptimizer):
    """PRIMA NEWUOA — 2n+1 underdetermined interpolation trust-region method.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    XPT is stored as a list of length-n vectors; the interpolation matrix
    A is a list-of-rows for the QR-based regression.
    """

    def optimize(self):
        n = self.n_dim
        npt = 2 * n + 1  # NEWUOA's signature interpolation count

        rhobeg = 0.5
        rhoend = 1e-8

        # First-pass seed: deterministic cube-centre, clipped a hair
        # off the bound so the coordinate-axis init points have headroom.
        # Restart passes (below) perturb self.best_x by ~rhobeg so the
        # caller's n_trials budget actually gets spent on non-smooth
        # surfaces where a single TR pass naturally terminates in ~50
        # evals when rho drops below rhoend.
        xseed = _A.clip(0.5 * _A.ones(n), 0.1, 0.9)

        while self.evaluations < self.n_trials:
            # ---------- one trust-region pass ----------
            rho = rhobeg
            xbase = _A.clip(xseed, 0, 1)
            _ = self.evaluate(xbase)
            if self.evaluations >= self.n_trials:
                break

            XPT, FVAL = self._initialize_newuoa_points(xbase, rho, npt, n)
            A, b = self._build_interpolation_system(XPT, FVAL, npt, n)

            kopt = min(range(len(FVAL)), key=FVAL.__getitem__)

            # Cap iteration count by budget directly. The previous
            # `min(100, n_trials // npt)` gave only floor(80/7) = 11
            # iterations on 3-D budget=80, terminating the TR loop long
            # before rho could converge. The outer
            # `evaluations < n_trials` guard is sufficient.
            max_iterations = self.n_trials
            iteration = 0

            while (
                self.evaluations < self.n_trials
                and rho > rhoend
                and iteration < max_iterations
            ):
                iteration += 1

                try:
                    g, H = self._build_newuoa_model(XPT, FVAL, A, b, kopt, n)
                except Exception:
                    rho *= 0.5
                    continue

                try:
                    d = self._solve_trust_region_newuoa(g, H, rho, n)
                except Exception:
                    d = self._fallback_step(g, rho, n)

                xnew = _A.clip(xbase + XPT[kopt] + d, 0, 1)

                if self.evaluations >= self.n_trials:
                    break

                fnew = self.evaluate(xnew)

                predicted_reduction = self._predict_reduction(g, H, d)
                actual_reduction = FVAL[kopt] - fnew

                point_updated = self._update_newuoa_interpolation(
                    XPT, FVAL, A, b, xnew - xbase, fnew, kopt, npt, n
                )

                if point_updated:
                    kopt_new = min(range(len(FVAL)), key=FVAL.__getitem__)
                    if FVAL[kopt_new] < FVAL[kopt]:
                        kopt = kopt_new

                rho = self._update_trust_region_radius(
                    predicted_reduction, actual_reduction, rho, rhobeg, rhoend
                )

                if _A.norm(XPT[kopt]) > 0.5 * rho:
                    xbase = self._shift_base_point(XPT, xbase, kopt, npt)
            # ---------- end one trust-region pass ----------

            # Jitter self.best_x by a uniform random vector of magnitude
            # ≈ rhobeg, clipped into [0, 1]^n. Large enough to escape the
            # basin that just trapped this pass; if best_x is already at
            # the global minimum the restart costs one extra pass without
            # worsening the result.
            if self.evaluations < self.n_trials:
                xseed = _A.clip(
                    [
                        float(self.best_x[i]) + (random.random() - 0.5) * 2.0 * rhobeg
                        for i in range(n)
                    ],
                    0,
                    1,
                )

        return self.best_value, self.best_x

    def _initialize_newuoa_points(self, xbase, rho, npt, n):
        """Lay out the 2n+1 NEWUOA interpolation points. Returns
        (XPT_list, FVAL_list) — XPT is a list of length-n offset vectors."""
        XPT = []
        FVAL = []

        # Base point at xbase (offset = 0).
        XPT.append(_A.zeros(n))
        FVAL.append(self.evaluate(xbase))

        # 2n coordinate directions (±rho along each axis).
        for i in range(n):
            for sign in (+1, -1):
                if len(FVAL) >= npt:
                    return XPT, FVAL
                offset = _A.zeros(n)
                offset[i] = sign * rho
                XPT.append(offset)
                FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

        # One extra point along the diagonal to bring count to 2n+1.
        if len(FVAL) < npt:
            diag_step = rho / math.sqrt(n)
            offset = _A.full(n, diag_step)
            XPT.append(offset)
            FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

        return XPT, FVAL

    def _build_interpolation_system(self, XPT, FVAL, npt, n):
        """Build the NEWUOA interpolation system: 2n+1 rows, (1 + 2n) cols
        (constant + linear + diagonal quadratic)."""
        nterms = 1 + n + n

        A = _A.linalg.matrix_zeros(npt, nterms)
        b = list(FVAL)

        for k in range(npt):
            x = XPT[k]
            row = A[k]
            col = 0
            row[col] = 1.0
            col += 1
            for i in range(n):
                row[col] = float(x[i])
                col += 1
            for i in range(n):
                row[col] = 0.5 * float(x[i]) * float(x[i])
                col += 1

        return A, b

    def _build_newuoa_model(self, XPT, FVAL, A, b, kopt, n):
        """Build the NEWUOA quadratic model via QR factorisation."""
        try:
            Q, R = _A.linalg.qr(A)
            QT = _A.linalg.transpose(Q)
            QTb = _A.linalg.matvec(QT, b)
            coeffs = list(_A.linalg.solve(R, QTb))

            g = _A.asarray(coeffs[1 : n + 1])
            # Diagonal Hessian from the diagonal-quadratic coefficients.
            diag_vals = list(coeffs[n + 1 : 2 * n + 1])
            min_eigval = min(diag_vals) if diag_vals else 0.0
            if min_eigval <= 0:
                shift = -min_eigval + 1e-6
                diag_vals = [v + shift for v in diag_vals]
            H = _A.linalg.diag(diag_vals)
        except Exception:
            g = self._finite_difference_gradient(XPT, FVAL, kopt, n)
            H = _A.linalg.eye(n)

        return g, H

    def _finite_difference_gradient(self, XPT, FVAL, kopt, n):
        """Best-effort central-difference gradient at XPT[kopt], using
        whichever coordinate-direction points are present in the set."""
        g_list = [0.0] * n

        for i in range(n):
            pos_val = neg_val = FVAL[kopt]
            step_size = 0.0

            for k in range(len(FVAL)):
                if k == kopt:
                    continue
                diff = XPT[k] - XPT[kopt]
                # Heuristic: this is "the ith coordinate direction" if diff[i]
                # is the dominant nonzero component.
                if abs(diff[i]) > 1e-6 and sum(abs(float(v)) for v in diff) < 2 * abs(
                    float(diff[i])
                ):
                    if float(diff[i]) > 0:
                        pos_val = FVAL[k]
                    else:
                        neg_val = FVAL[k]
                    step_size = max(step_size, abs(float(diff[i])))

            if pos_val != FVAL[kopt] and neg_val != FVAL[kopt] and step_size > 0:
                g_list[i] = (pos_val - neg_val) / (2 * step_size)

        return _A.asarray(g_list)

    def _solve_trust_region_newuoa(self, g, H, rho, n):
        """Newton step if H is SPD enough; dogleg otherwise."""
        try:
            eigvals, _ = _A.linalg.eigh(H)
            if all(v > 1e-8 for v in eigvals):
                d_newton = -_A.linalg.solve(H, g)
                if _A.norm(d_newton) <= rho:
                    return d_newton
        except Exception:
            pass
        return self._dogleg_method(g, H, rho, n)

    def _dogleg_method(self, g, H, rho, n):
        """Dogleg trust-region solver."""
        g_norm_sq = _A.dot(g, g)
        if g_norm_sq < 1e-12:
            return _A.zeros(n)

        Hg = _A.linalg.matvec(H, g)
        gHg = _A.dot(g, Hg)
        alpha_c = g_norm_sq / gHg if gHg > 1e-12 else 1.0

        d_cauchy = -alpha_c * g

        if _A.norm(d_cauchy) >= rho:
            return -rho * g / math.sqrt(g_norm_sq)

        try:
            d_newton = -_A.linalg.solve(H, g)
            if _A.norm(d_newton) <= rho:
                return d_newton

            # Dogleg path: solve for tau s.t. ||d_cauchy + tau (d_newton-d_cauchy)|| == rho.
            diff = d_newton - d_cauchy
            a = _A.dot(diff, diff)
            b_coef = 2 * _A.dot(d_cauchy, diff)
            c = _A.dot(d_cauchy, d_cauchy) - rho * rho

            discriminant = b_coef * b_coef - 4 * a * c
            if discriminant >= 0 and a > 1e-12:
                tau = (-b_coef + math.sqrt(discriminant)) / (2 * a)
                return d_cauchy + tau * diff
        except Exception:
            pass

        return d_cauchy

    def _update_newuoa_interpolation(self, XPT, FVAL, A, b, d, fnew, kopt, npt, n):
        """Replace the point furthest from the new position (and ≠ kopt)
        with `(XPT[kopt] + d, fnew)`, then rebuild the interpolation system."""
        candidates = [i for i in range(npt) if i != kopt]
        if not candidates:
            return False

        new_pos = XPT[kopt] + d
        # `distances[k]` paired with `k`; pick the largest.
        furthest_idx = max(candidates, key=lambda i: float(_A.norm(XPT[i] - new_pos)))

        XPT[furthest_idx] = new_pos
        FVAL[furthest_idx] = fnew

        # Rebuild A and b in place.
        new_A, new_b = self._build_interpolation_system(XPT, FVAL, npt, n)
        # `A` and `b` are the local references passed in; mutate them so
        # the caller's references stay valid.
        for row_i in range(len(A)):
            A[row_i] = new_A[row_i]
        for j in range(len(b)):
            b[j] = new_b[j]

        return True

    def _predict_reduction(self, g, H, d):
        Hd = _A.linalg.matvec(H, d)
        return -(_A.dot(g, d) + 0.5 * _A.dot(d, Hd))

    def _update_trust_region_radius(self, pred_red, actual_red, rho, rhobeg, rhoend):
        if abs(pred_red) < 1e-12:
            ratio = 0 if actual_red <= 0 else 10
        else:
            ratio = actual_red / pred_red

        if ratio >= 0.75:
            return min(rho * 2.0, rhobeg)
        elif ratio >= 0.25:
            return rho
        elif ratio >= 0.1:
            return rho * 0.8
        else:
            return max(rho * 0.5, rhoend)

    def _shift_base_point(self, XPT, xbase, kopt, npt):
        """Recentre XPT so the best point sits at the origin. Returns the
        new xbase (the input xbase is treated as immutable to keep the
        list-of-vectors semantics simple)."""
        shift = XPT[kopt].copy()
        new_xbase = _A.clip(xbase + shift, 0, 1)
        for i in range(npt):
            XPT[i] = XPT[i] - shift
        return new_xbase


class PRIMA_BOBYQA(BaseOptimizer):
    """PRIMA BOBYQA — Bound-constrained derivative-free trust-region method.

    Pure-Python via the `humpday._array` shim — no direct numpy use.
    Bounds for humpday's [0, 1]^n cube are tracked explicitly via xl, xu
    lists; XPT is a list of length-n offset vectors.
    """

    def optimize(self):
        n = self.n_dim
        npt = 2 * n + 1

        xl = _A.zeros(n)
        xu = _A.ones(n)

        rhobeg = 0.5
        rhoend = 1e-8

        # First-pass seed: cube-centre, clipped a hair off the bounds
        # so the coordinate-axis init points have headroom. Restart
        # passes (below) perturb self.best_x by ~rhobeg so the caller's
        # n_trials budget actually gets spent on non-smooth surfaces
        # where a single TR pass naturally terminates in ~50 evals.
        xseed = _A.clip(0.5 * _A.ones(n), 0.1, 0.9)

        while self.evaluations < self.n_trials:
            # ---------- one trust-region pass ----------
            rho = rhobeg
            xbase = _A.clip(xseed, 0.1, 0.9)
            _ = self.evaluate(xbase)
            if self.evaluations >= self.n_trials:
                break

            XPT, FVAL = self._initialize_bobyqa_points(xbase, rho, npt, n, xl, xu)
            kopt = min(range(len(FVAL)), key=FVAL.__getitem__)

            # Cap by budget directly (see NEWUOA comment).
            max_iterations = self.n_trials
            iteration = 0

            while (
                self.evaluations < self.n_trials
                and rho > rhoend
                and iteration < max_iterations
            ):
                iteration += 1

                try:
                    g, H = self._build_bobyqa_model(XPT, FVAL, kopt, n)
                except Exception:
                    rho *= 0.5
                    continue

                try:
                    d = self._solve_bound_constrained_tr(
                        g, H, rho, n, xbase + XPT[kopt], xl, xu
                    )
                except Exception:
                    d = self._fallback_bound_step(g, rho, n, xbase + XPT[kopt], xl, xu)

                xnew = _A.clip(xbase + XPT[kopt] + d, 0, 1)

                if self.evaluations >= self.n_trials:
                    break

                fnew = self.evaluate(xnew)

                predicted_reduction = self._predict_reduction(g, H, d)
                actual_reduction = FVAL[kopt] - fnew

                point_updated = self._update_bobyqa_interpolation(
                    XPT, FVAL, xnew - xbase, fnew, kopt, npt, n
                )

                if point_updated:
                    kopt_new = min(range(len(FVAL)), key=FVAL.__getitem__)
                    if FVAL[kopt_new] < FVAL[kopt]:
                        kopt = kopt_new

                rho = self._update_trust_region_radius(
                    predicted_reduction, actual_reduction, rho, rhobeg, rhoend
                )

                if _A.norm(XPT[kopt]) > 0.5 * rho:
                    xbase = self._shift_base_point_bounded(
                        XPT, xbase, kopt, npt, xl, xu
                    )
            # ---------- end one trust-region pass ----------

            # Jitter self.best_x for the next pass (clipped to [0.1, 0.9]
            # for the same headroom reason as the first-pass init).
            if self.evaluations < self.n_trials:
                xseed = _A.clip(
                    [
                        float(self.best_x[i]) + (random.random() - 0.5) * 2.0 * rhobeg
                        for i in range(n)
                    ],
                    0.1,
                    0.9,
                )

        return self.best_value, self.best_x

    def _initialize_bobyqa_points(self, xbase, rho, npt, n, xl, xu):
        """Lay out the initial interpolation set, respecting [xl, xu] bounds.
        Returns (XPT_list, FVAL_list)."""
        XPT = []
        FVAL = []

        XPT.append(_A.zeros(n))
        FVAL.append(self.evaluate(xbase))

        # Coordinate directions, clipped to bound-feasible step sizes.
        for i in range(n):
            if len(FVAL) >= npt:
                return XPT, FVAL
            step_pos = min(rho, float(xu[i]) - float(xbase[i]))
            if step_pos > 1e-10:
                offset = _A.zeros(n)
                offset[i] = step_pos
                XPT.append(offset)
                FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

            if len(FVAL) >= npt:
                return XPT, FVAL
            step_neg = max(-rho, float(xl[i]) - float(xbase[i]))
            if step_neg < -1e-10:
                offset = _A.zeros(n)
                offset[i] = step_neg
                XPT.append(offset)
                FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

        # Optional diagonal-direction point, clipped to bounds.
        if len(FVAL) < npt:
            diagonal_step = [rho / math.sqrt(n)] * n
            for i in range(n):
                xi_target = float(xbase[i]) + diagonal_step[i]
                if xi_target > float(xu[i]):
                    diagonal_step[i] = float(xu[i]) - float(xbase[i])
                elif xi_target < float(xl[i]):
                    diagonal_step[i] = float(xl[i]) - float(xbase[i])
            offset = _A.asarray(diagonal_step)
            XPT.append(offset)
            FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

        return XPT, FVAL

    def _build_bobyqa_model(self, XPT, FVAL, kopt, n):
        """Quadratic model via QR-based regression with diagonal Hessian."""
        try:
            nterms = 1 + n + n
            nrows = len(FVAL)
            A = _A.linalg.matrix_zeros(nrows, nterms)
            b = [FVAL[i] - FVAL[kopt] for i in range(nrows)]

            for k in range(nrows):
                x = XPT[k]
                row = A[k]
                col = 0
                row[col] = 1.0
                col += 1
                for i in range(n):
                    row[col] = float(x[i])
                    col += 1
                for i in range(n):
                    row[col] = 0.5 * float(x[i]) * float(x[i])
                    col += 1

            Q, R = _A.linalg.qr(A)
            QT = _A.linalg.transpose(Q)
            QTb = _A.linalg.matvec(QT, b)
            coeffs = list(_A.linalg.solve(R, QTb))

            g = _A.asarray(coeffs[1 : n + 1])
            diag_vals = list(coeffs[n + 1 : 2 * n + 1])
            min_eigval = min(diag_vals) if diag_vals else 0.0
            if min_eigval <= 0:
                shift = -min_eigval + 1e-6
                diag_vals = [v + shift for v in diag_vals]
            H = _A.linalg.diag(diag_vals)
        except Exception:
            g = self._finite_difference_gradient_bounded(XPT, FVAL, kopt, n)
            H = _A.linalg.eye(n)

        return g, H

    def _finite_difference_gradient_bounded(self, XPT, FVAL, kopt, n):
        """Coordinate-wise finite-difference gradient using whichever
        positive/negative axis points are present."""
        g_list = [0.0] * n

        for i in range(n):
            pos_val = neg_val = FVAL[kopt]
            pos_step = neg_step = 0.0

            for k in range(len(FVAL)):
                if k == kopt:
                    continue
                diff = XPT[k] - XPT[kopt]
                if abs(float(diff[i])) > 1e-6 and sum(
                    abs(float(v)) for v in diff
                ) < 2 * abs(float(diff[i])):
                    if float(diff[i]) > 0:
                        pos_val = FVAL[k]
                        pos_step = float(diff[i])
                    else:
                        neg_val = FVAL[k]
                        neg_step = abs(float(diff[i]))

            if pos_step > 0 and neg_step > 0:
                g_list[i] = (pos_val - neg_val) / (pos_step + neg_step)
            elif pos_step > 0:
                g_list[i] = (pos_val - FVAL[kopt]) / pos_step
            elif neg_step > 0:
                g_list[i] = (FVAL[kopt] - neg_val) / neg_step

        return _A.asarray(g_list)

    def _solve_bound_constrained_tr(self, g, H, rho, n, x_current, xl, xu):
        """Bound-constrained trust-region solve. Tries the Newton step
        first and falls back to projected-Cauchy if Newton is infeasible
        or H isn't SPD enough."""
        try:
            eigvals, _ = _A.linalg.eigh(H)
            if all(v > 1e-8 for v in eigvals):
                d_newton = -_A.linalg.solve(H, g)
                x_new = x_current + d_newton
                if (
                    all(float(x_new[i]) >= float(xl[i]) for i in range(n))
                    and all(float(x_new[i]) <= float(xu[i]) for i in range(n))
                    and _A.norm(d_newton) <= rho
                ):
                    return d_newton
        except Exception:
            pass
        return self._projected_cauchy_point(g, H, rho, n, x_current, xl, xu)

    def _projected_cauchy_point(self, g, H, rho, n, x_current, xl, xu):
        """Projected Cauchy point — direction is steepest descent with
        active-bound components zeroed, then projected onto the trust
        region and the bound box."""
        if _A.norm(g) < 1e-12:
            return _A.zeros(n)

        # p = -g, with components zeroed if they'd push outside the bounds.
        p = -g.copy()
        for i in range(n):
            if float(x_current[i]) <= float(xl[i]) + 1e-10 and float(p[i]) < 0:
                p[i] = 0.0
            elif float(x_current[i]) >= float(xu[i]) - 1e-10 and float(p[i]) > 0:
                p[i] = 0.0

        if _A.norm(p) < 1e-12:
            return _A.zeros(n)

        Hg = _A.linalg.matvec(H, g)
        gHg = _A.dot(g, Hg)
        alpha = _A.dot(g, g) / gHg if gHg > 1e-12 else 1.0

        d = alpha * p

        d_norm = _A.norm(d)
        if d_norm > rho:
            d = rho * d / d_norm

        # Final projection: pull any out-of-bounds component back to the box.
        for i in range(n):
            xi_new = float(x_current[i]) + float(d[i])
            if xi_new < float(xl[i]):
                d[i] = float(xl[i]) - float(x_current[i])
            elif xi_new > float(xu[i]):
                d[i] = float(xu[i]) - float(x_current[i])

        return d

    def _update_bobyqa_interpolation(self, XPT, FVAL, d, fnew, kopt, npt, n):
        """Replace the point furthest from the new position with `(new, fnew)`."""
        candidates = [i for i in range(npt) if i != kopt]
        if not candidates:
            return False
        new_pos = XPT[kopt] + d
        furthest_idx = max(candidates, key=lambda i: float(_A.norm(XPT[i] - new_pos)))
        XPT[furthest_idx] = new_pos
        FVAL[furthest_idx] = fnew
        return True

    def _shift_base_point_bounded(self, XPT, xbase, kopt, npt, xl, xu):
        """Recentre XPT so the best point sits at the origin, with the
        new base clipped to the bound box."""
        shift = XPT[kopt].copy()
        new_base = _A.clip(xbase + shift, 0, 1)
        # Use the realised shift (might differ from `shift` if clipping
        # truncated it) so XPT stays consistent.
        actual_shift = new_base - xbase
        for i in range(npt):
            XPT[i] = XPT[i] - actual_shift
        return new_base

    def _fallback_bound_step(self, g, rho, n, x_current, xl, xu):
        """Bound-aware fallback step: scaled steepest descent or random
        kick, then projected onto the bound box."""
        if _A.norm(g) > 1e-12:
            d = -rho * g / _A.norm(g)
        else:
            d = (rho / 3.0) * _A.random_normal(n)

        for i in range(n):
            xi_new = float(x_current[i]) + float(d[i])
            if xi_new < float(xl[i]):
                d[i] = float(xl[i]) - float(x_current[i])
            elif xi_new > float(xu[i]):
                d[i] = float(xu[i]) - float(x_current[i])

        return d

    def _predict_reduction(self, g, H, d):
        Hd = _A.linalg.matvec(H, d)
        return -(_A.dot(g, d) + 0.5 * _A.dot(d, Hd))

    def _update_trust_region_radius(self, pred_red, actual_red, rho, rhobeg, rhoend):
        if abs(pred_red) < 1e-12:
            ratio = 0 if actual_red <= 0 else 10
        else:
            ratio = actual_red / pred_red

        if ratio >= 0.75:
            return min(rho * 2.0, rhobeg)
        elif ratio >= 0.25:
            return rho
        elif ratio >= 0.1:
            return rho * 0.8
        else:
            return max(rho * 0.5, rhoend)
