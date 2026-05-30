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


def _build_min_frobenius_quadratic(XPT, FVAL, H_prev, n):
    """Powell's NEWUOA-style minimum-Frobenius-norm quadratic update.

    Builds a *full*-Hessian quadratic m(s) = c + g·s + ½ sᵀ H s that
    interpolates (XPT[k], FVAL[k]) for k = 0..npt-1, with H chosen to
    minimise ‖H − H_prev‖_F² over the remaining degrees of freedom.

    For npt = 2n+1 the interpolation system is underdetermined: it has
    (n+1)(n+2)/2 unknowns but only 2n+1 equations, leaving n(n−1)/2
    free directions in H. Powell resolves the under-determination by
    keeping H as close as possible to the previous iteration's
    Hessian — the "min-Frobenius update" that lets NEWUOA capture
    off-diagonal Hessian information (e.g. Rosenbrock's 100·(y − x²)²
    cross-coupling) despite using only 2n+1 points.

    Derivation. Let H = H_prev + ΔH (symmetric). The interpolation
    constraints become

        c + g·s_k + ½ s_kᵀ ΔH s_k = FVAL[k] − ½ s_kᵀ H_prev s_k

    where s_k = XPT[k]. Splitting the unknowns into linear
    [c; g] (n+1 entries, no objective weight) and quadratic vech(ΔH)
    (n(n+1)/2 entries, Frobenius-weighted), the KKT conditions reduce
    to a small (n_null × n_null) symmetric system where
    n_null = npt − (n+1) is the number of constraints "left over" for
    the quadratic part. We solve that system, recover ΔH, then back-
    solve A_l x_l = b − A_q vech(ΔH) for (c, g).

    Returns (c, g, H) on success; raises _PRIMALinAlgError if the
    linear-only design matrix A_l is rank-deficient — the caller then
    falls back to the diagonal-only model that was the previous
    implementation.
    """
    npt = len(FVAL)
    p_lin = 1 + n  # c, g_1, ..., g_n
    p_quad = n * (n + 1) // 2  # H_11..H_nn diagonals + H_12..H_(n-1)n off-diagonals
    n_null = npt - p_lin

    if n_null < 0:
        raise _PRIMALinAlgError("npt < p_lin")

    # Non-finite FVAL (objective returned inf/NaN at one of the init or TR
    # points, common with bound-touching evaluations) makes the linear
    # solve degenerate. Bail out early so the caller can use its FD-
    # gradient fallback instead of propagating NaN through the SVD.
    for v in FVAL:
        if not math.isfinite(float(v)):
            raise _PRIMALinAlgError("FVAL contains non-finite values")

    # Build design columns. A_l is (npt × p_lin), A_q is (npt × p_quad).
    # vech(H) ordering: first the n diagonals, then the n(n−1)/2 strict
    # upper-triangle entries in row-major order.
    A_l = _A.linalg.matrix_zeros(npt, p_lin)
    A_q = _A.linalg.matrix_zeros(npt, p_quad)
    b = [0.0] * npt

    for k in range(npt):
        x = XPT[k]
        A_l[k][0] = 1.0
        for i in range(n):
            A_l[k][i + 1] = float(x[i])

        col = 0
        for i in range(n):
            A_q[k][col] = 0.5 * float(x[i]) * float(x[i])
            col += 1
        for i in range(n):
            for j in range(i + 1, n):
                A_q[k][col] = float(x[i]) * float(x[j])
                col += 1

        # b[k] = FVAL[k] − ½ x_k^T H_prev x_k.
        Hp_x = _A.linalg.matvec(H_prev, x)
        quad_form = 0.5 * float(_A.dot(x, Hp_x))
        b[k] = float(FVAL[k]) - quad_form

    # Frobenius weights for vech(ΔH). Diagonals appear once in H, weight 1.
    # Off-diagonals appear twice (above + below the diagonal), weight 2.
    w_q = [1.0] * n + [2.0] * (p_quad - n)

    # Full SVD of A_l so its null space U_perp = U[:, p_lin:] is available.
    U_full, s_l, _Vt_l = _A.linalg.svd(A_l, full_matrices=True)
    if len(s_l) < p_lin or float(s_l[p_lin - 1]) <= 1e-12 * float(s_l[0]):
        raise _PRIMALinAlgError("A_l rank-deficient")

    # x_q = -W_q^{-1} A_q^T Z μ  (vech of ΔH).
    # x_l recovered from A_l x_l = b - A_q x_q via QR.
    if n_null > 0:
        # Z: orthonormal basis for null(A_l^T); shape npt × n_null.
        Z = _A.linalg.matrix_zeros(npt, n_null)
        for i in range(npt):
            for j in range(n_null):
                Z[i][j] = U_full[i][p_lin + j]

        # A_q_scaled[k][c] = A_q[k][c] / w_q[c]; M = (Z^T A_q)(A_q_scaled^T Z).
        ZT_b = [0.0] * n_null
        for j in range(n_null):
            acc = 0.0
            for k in range(npt):
                acc += Z[k][j] * b[k]
            ZT_b[j] = acc

        ZT_Aq = _A.linalg.matrix_zeros(n_null, p_quad)
        for j in range(n_null):
            for c in range(p_quad):
                acc = 0.0
                for k in range(npt):
                    acc += Z[k][j] * A_q[k][c]
                ZT_Aq[j][c] = acc

        Aq_scaled_T_Z = _A.linalg.matrix_zeros(p_quad, n_null)
        for c in range(p_quad):
            inv_w = 1.0 / w_q[c]
            for j in range(n_null):
                acc = 0.0
                for k in range(npt):
                    acc += A_q[k][c] * inv_w * Z[k][j]
                Aq_scaled_T_Z[c][j] = acc

        M = _A.linalg.matmul(ZT_Aq, Aq_scaled_T_Z)
        neg_ZT_b = [-v for v in ZT_b]
        try:
            mu = list(_A.linalg.solve(M, neg_ZT_b))
        except Exception:
            mu = list(_A.linalg.matvec(_A.linalg.pinv(M), neg_ZT_b))

        x_q = [0.0] * p_quad
        for c in range(p_quad):
            acc = 0.0
            for j in range(n_null):
                acc += Aq_scaled_T_Z[c][j] * mu[j]
            x_q[c] = -acc
    else:
        x_q = [0.0] * p_quad  # no underdetermination → ΔH = 0

    # x_l: solve A_l x_l = b - A_q x_q via QR (overdetermined least-squares,
    # but consistent by construction since x_q satisfied the null-space part).
    Aq_xq = [0.0] * npt
    for k in range(npt):
        acc = 0.0
        for c in range(p_quad):
            acc += A_q[k][c] * x_q[c]
        Aq_xq[k] = acc
    rhs = [b[k] - Aq_xq[k] for k in range(npt)]

    Q_l, R_l = _A.linalg.qr(A_l)
    QlT_rhs = _A.linalg.matvec(_A.linalg.transpose(Q_l), rhs)
    x_l = list(_A.linalg.solve(R_l, QlT_rhs))

    c = x_l[0]
    g_arr = _A.asarray(x_l[1:])

    # Reconstruct symmetric H = H_prev + ΔH from vech(ΔH).
    H = _A.linalg.matrix_zeros(n, n)
    col = 0
    for i in range(n):
        H[i][i] = float(H_prev[i][i]) + x_q[col]
        col += 1
    for i in range(n):
        for j in range(i + 1, n):
            val = float(H_prev[i][j]) + x_q[col]
            H[i][j] = val
            H[j][i] = val
            col += 1

    # Belt-and-braces: a barely-singular SVD can produce non-finite
    # coefficients that propagate into H_prev and poison subsequent
    # iterations. Raise so the caller's fallback path is taken and the
    # prior (good) H_prev is preserved.
    if not math.isfinite(float(c)):
        raise _PRIMALinAlgError("non-finite c")
    for v in g_arr:
        if not math.isfinite(float(v)):
            raise _PRIMALinAlgError("non-finite g")
    for i in range(n):
        for j in range(n):
            if not math.isfinite(float(H[i][j])):
                raise _PRIMALinAlgError("non-finite H")

    return c, g_arr, H


def _solve_trsbox(g, H, rho, x_current, xl, xu, n):
    """Powell's TRSBOX trust-region subproblem solver for BOBYQA.

    Solves
        min  Q(d) = g·d + ½ dᵀ H d
        s.t. ‖d‖₂ ≤ rho
             xl_i ≤ x_current_i + d_i ≤ xu_i   ∀ i

    Active-set Steihaug-Toint truncated CG: each outer pass runs CG in
    the current free subspace until either (a) CG converges in that
    subspace, (b) a new bound becomes active (add it, restart CG), or
    (c) the step hits the TR boundary. Negative curvature on the CG
    direction triggers a step to the TR boundary. At the start of each
    outer pass any active bound whose gradient points back into the
    feasible region is released, so the active set can shrink as well
    as grow.

    Phase 3 of Powell's BOBYQA TRSBOX — the "alternative iteration" that
    moves along the TR boundary after CG hits it — is *not* implemented
    here. The 2-phase version is enough to recover the curvature
    information the new full-Hessian model carries (e.g. closes the
    Rosenbrock gap to within a few ULPs); the alternative iteration
    can be added later if benchmarks justify it.

    Returns d as an `_A` vector. d satisfies the bound constraints
    exactly (active components pinned to the bound). The trust-region
    constraint is satisfied up to a small numerical tolerance.
    """
    tol = 1e-12

    # Reject non-finite inputs cleanly so the caller's fallback path
    # can run instead of NaN-propagating through the CG iteration.
    for v in g:
        if not math.isfinite(float(v)):
            raise _PRIMALinAlgError("non-finite g")
    for i in range(n):
        for j in range(n):
            if not math.isfinite(float(H[i][j])):
                raise _PRIMALinAlgError("non-finite H")

    # Recast bounds into d-space.
    lo_d = [float(xl[i]) - float(x_current[i]) for i in range(n)]
    hi_d = [float(xu[i]) - float(x_current[i]) for i in range(n)]

    d = [0.0] * n
    # active[i] ∈ {None, 'lo', 'hi'}.
    active = [None] * n

    # Generous caps; both inner and outer loops terminate well before
    # these in practice (CG converges in ≤ n iterations in a fixed
    # subspace; active set changes ≤ 2n times).
    outer_cap = 2 * n + 10
    inner_cap = 2 * n + 10

    for _outer in range(outer_cap):
        # Gradient of Q at the current d: grad = g + H d.
        Hd = list(_A.linalg.matvec(H, _A.asarray(d)))
        grad = [float(g[i]) + Hd[i] for i in range(n)]

        # Catch coordinates that the previous CG step landed exactly on
        # (ties in α_box, or numerical drift). If we don't mark them
        # active here, the next inner step can walk straight past the
        # bound — α_box would skip them (no strictly-positive step) and
        # α_cg can be large. Must run *before* the release check so that
        # freshly-detected bound activations still get a chance to be
        # released when the gradient points away from the constraint
        # (e.g. starting at a corner with a gradient pointing inward).
        for i in range(n):
            if active[i] is None:
                if d[i] <= lo_d[i] + tol:
                    active[i] = "lo"
                    d[i] = lo_d[i]
                elif d[i] >= hi_d[i] - tol:
                    active[i] = "hi"
                    d[i] = hi_d[i]

        # Release any active bound whose gradient component points back
        # into the feasible region (∂Q/∂d_i > 0 at lower bound means we
        # want to *decrease* d_i, which is infeasible — keep the bound;
        # ∂Q/∂d_i < 0 at lower bound means decreasing Q wants to
        # increase d_i, which IS feasible — release).
        for i in range(n):
            if active[i] == "lo" and grad[i] < -tol:
                active[i] = None
            elif active[i] == "hi" and grad[i] > tol:
                active[i] = None

        # Projected residual r = -grad on the free subspace.
        r = [-grad[i] if active[i] is None else 0.0 for i in range(n)]
        r_norm_sq = sum(ri * ri for ri in r)
        if r_norm_sq < tol * tol:
            break  # nothing left to do in the current free subspace

        # CG inner loop in the current subspace.
        p = list(r)
        d_norm_sq = sum(di * di for di in d)
        bound_added = False

        for _inner in range(inner_cap):
            # Compute Hp restricted to the free subspace.
            Hp_full = list(_A.linalg.matvec(H, _A.asarray(p)))
            Hp = [Hp_full[i] if active[i] is None else 0.0 for i in range(n)]
            pHp = sum(p[i] * Hp[i] for i in range(n))

            # alpha_box: smallest positive step before a new bound is hit.
            alpha_box = float("inf")
            new_idx = -1
            new_side = None
            for i in range(n):
                if active[i] is not None:
                    continue
                if p[i] > tol:
                    a = (hi_d[i] - d[i]) / p[i]
                    if 0 < a < alpha_box:
                        alpha_box = a
                        new_idx = i
                        new_side = "hi"
                elif p[i] < -tol:
                    a = (lo_d[i] - d[i]) / p[i]
                    if 0 < a < alpha_box:
                        alpha_box = a
                        new_idx = i
                        new_side = "lo"

            # alpha_tr: step to the TR boundary ‖d + α p‖ = rho.
            pp = sum(pi * pi for pi in p)
            if pp < tol * tol:
                break
            dp = sum(d[i] * p[i] for i in range(n))
            disc = dp * dp - pp * (d_norm_sq - rho * rho)
            if disc < 0:
                alpha_tr = 0.0  # already at/outside TR (numerical edge)
            else:
                alpha_tr = (-dp + math.sqrt(disc)) / pp

            # alpha_cg: CG step length (infinite on non-positive curvature
            # so the boundary terms govern the step).
            if pHp > tol:
                alpha_cg = r_norm_sq / pHp
            else:
                alpha_cg = float("inf")

            alpha = min(alpha_cg, alpha_box, alpha_tr)
            if alpha <= 0:
                break

            d = [d[i] + alpha * p[i] for i in range(n)]
            d_norm_sq = sum(di * di for di in d)

            # Termination at the TR boundary — Phase 3 omitted; we accept
            # the current d.
            if alpha >= alpha_tr - tol:
                return _A.asarray(d)

            # Bound hit: pin the new active coordinate exactly to the
            # bound (kills accumulated rounding) and restart CG in the
            # smaller subspace.
            if alpha >= alpha_box - tol and new_idx >= 0:
                active[new_idx] = new_side
                d[new_idx] = hi_d[new_idx] if new_side == "hi" else lo_d[new_idx]
                bound_added = True
                break

            # Otherwise alpha == alpha_cg — continue CG in this subspace.
            r_new = [r[i] - alpha * Hp[i] for i in range(n)]
            r_new_norm_sq = sum(ri * ri for ri in r_new)
            if r_new_norm_sq < tol * tol:
                return _A.asarray(d)

            beta = r_new_norm_sq / r_norm_sq
            p = [r_new[i] + beta * p[i] for i in range(n)]
            # Belt-and-braces re-projection (numerical drift on active
            # coordinates could otherwise pull d off the bound).
            for i in range(n):
                if active[i] is not None:
                    p[i] = 0.0
            r = r_new
            r_norm_sq = r_new_norm_sq

        if not bound_added:
            break  # CG terminated in the current subspace, no bound change.

    return _A.asarray(d)


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
        if not FVAL:
            return self.best_value, self.best_x
        nused = min(len(FVAL), npt)
        # argmin via stdlib — works on lists, ndarrays, _Vec.
        kopt = min(range(nused), key=FVAL.__getitem__)

        # Shift xbase so the best init point sits at the origin BEFORE
        # the first TR iteration — see NEWUOA #172 comment.
        if float(_A.norm(XPT[kopt])) > 1e-12:
            shift = XPT[kopt].copy()
            new_xbase = _A.clip(xbase + shift, 0, 1)
            actual_shift = new_xbase - xbase
            xbase = new_xbase
            for i in range(nused):
                XPT[i] = XPT[i] - actual_shift

        iteration = 0
        # Cap the trust-region loop by budget directly. The previous
        # `min(100, n_trials // npt)` was the same cap NEWUOA had before
        # #167 fixed it; on a 3-D problem with budget=80 and npt=10 it
        # gave only 8 iterations and the loop terminated long before
        # rho could converge. The outer `evaluations < n_trials` guard
        # is sufficient; this is just a guard against pathological
        # infinite loops.
        max_iterations = self.n_trials

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

            # Unconditional base-point shift — see NEWUOA #172 comment.
            # Plain `g` is the gradient at xbase; the TR step is taken
            # from xbase + XPT[kopt], so the subproblem only stays
            # consistent when XPT[kopt] ≈ 0 at the start of every
            # iteration. Uses actual_shift to handle the post-clip
            # case correctly.
            if float(_A.norm(XPT[kopt])) > 1e-12:
                shift = XPT[kopt].copy()
                new_xbase = _A.clip(xbase + shift, 0, 1)
                actual_shift = new_xbase - xbase
                xbase = new_xbase
                for i in range(nused):
                    XPT[i] = XPT[i] - actual_shift

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
        FVAL is a parallel list of objective values.

        Each evaluate() is budget-guarded so a restart triggered close to
        n_trials can't overshoot via the init-set (same pattern as the
        BOBYQA fix in #156)."""
        XPT = []
        FVAL = []

        # Base point at xbase (offset = 0).
        XPT.append(_A.zeros(n))
        if self.evaluations >= self.n_trials:
            return XPT, FVAL
        FVAL.append(self.evaluate(xbase))
        if len(FVAL) >= npt:
            return XPT, FVAL

        # ±rho along each coordinate.
        for i in range(n):
            for sign in (+1, -1):
                if len(FVAL) >= npt or self.evaluations >= self.n_trials:
                    return XPT, FVAL
                offset = _A.zeros(n)
                offset[i] = sign * rho
                XPT.append(offset)
                FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

        # Cross-term diagonals at rho/sqrt(2) per coordinate.
        diag_step = rho / math.sqrt(2)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if len(FVAL) >= npt or self.evaluations >= self.n_trials:
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
            # Init can return < npt points if the budget exhausted mid-
            # layout (the per-evaluate budget guard). Without a full
            # interpolation set there's nothing useful to fit, so end
            # this pass; the outer `while` will terminate naturally.
            if len(FVAL) < npt:
                break
            A, b = self._build_interpolation_system(XPT, FVAL, npt, n)

            kopt = min(range(len(FVAL)), key=FVAL.__getitem__)

            # Reset the min-Frobenius update's prior Hessian for this TR
            # pass. Powell's NEWUOA initialises H_prev = 0 at restart and
            # carries it forward through each interpolation-set update;
            # this is the state that lets the model accumulate off-
            # diagonal curvature from non-axial points.
            self._H_prev = _A.linalg.matrix_zeros(n, n)

            # Shift xbase so the best init point sits at the origin
            # BEFORE the first TR iteration. Powell's NEWUOA shifts
            # every iteration so that the model's gradient g is the
            # gradient at the trial origin (xbase + XPT[kopt]); the
            # first TR step uses plain `g`, so it must already be at
            # the right place by then. Without this initial shift,
            # the first iteration's TR step is geometrically mis-
            # aligned (#172).
            if float(_A.norm(XPT[kopt])) > 1e-12:
                xbase = self._shift_base_point(XPT, xbase, kopt, npt)

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

                # Unconditional base-point shift — see the per-iteration
                # shift comment at the top of this restart pass. Without
                # this, plain `g` is the gradient at xbase (not at
                # xbase + XPT[kopt]), so the TR subproblem is mis-aligned
                # and `predicted_reduction` is computed in the wrong
                # frame. Powell's NEWUOA does this shift every iteration
                # to make the per-iter model consistent.
                if float(_A.norm(XPT[kopt])) > 1e-12:
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
        (XPT_list, FVAL_list) — XPT is a list of length-n offset vectors.

        Each evaluate() is budget-guarded so a restart triggered close
        to n_trials can't overshoot via the init-set (same pattern as
        the BOBYQA fix in #156)."""
        XPT = []
        FVAL = []

        # Base point at xbase (offset = 0).
        XPT.append(_A.zeros(n))
        if self.evaluations >= self.n_trials:
            return XPT, FVAL
        FVAL.append(self.evaluate(xbase))

        # 2n coordinate directions (±rho along each axis).
        for i in range(n):
            for sign in (+1, -1):
                if len(FVAL) >= npt or self.evaluations >= self.n_trials:
                    return XPT, FVAL
                offset = _A.zeros(n)
                offset[i] = sign * rho
                XPT.append(offset)
                FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

        # One extra point along the diagonal to bring count to 2n+1.
        if len(FVAL) < npt and self.evaluations < self.n_trials:
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
        """Build the NEWUOA quadratic model via Powell's minimum-
        Frobenius-norm Hessian update — a full symmetric H, not a
        diagonal approximation.

        On rank-deficient interpolation sets (e.g. early iterations on
        a degenerate configuration), falls back to a finite-difference
        gradient with identity Hessian so the TR loop can still
        progress. The previous diagonal-QR path is gone: when min-
        Frobenius succeeds it strictly dominates diagonal-QR, and when
        it fails on rank deficiency, diagonal-QR fails for the same
        reason.

        The ignored `A`, `b` arguments are left in the signature only
        because they're still threaded through the optimize() loop;
        the helper builds its own design matrix from XPT.
        """
        H_prev = getattr(self, "_H_prev", None)
        if H_prev is None:
            H_prev = _A.linalg.matrix_zeros(n, n)
        try:
            _c, g, H = _build_min_frobenius_quadratic(XPT, FVAL, H_prev, n)
            self._H_prev = H
            return g, H
        except Exception:
            g = self._finite_difference_gradient(XPT, FVAL, kopt, n)
            H = _A.linalg.eye(n)
            # Don't update H_prev — keep the last successful Hessian.
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
        new xbase.

        Uses the realised (post-clip) shift to keep the model self-
        consistent — when XPT[kopt] would push xbase outside [0,1] the
        clip truncates the move, so XPT must be shifted by that same
        truncated amount or the model coefficients no longer interpolate
        the data points the optimiser thinks they do. Mirrors the
        existing pattern in `_shift_base_point_bounded` (BOBYQA).

        Iterates over `len(XPT)` rather than `npt` so partial init-sets
        (init bailed out on budget) don't IndexError."""
        shift = XPT[kopt].copy()
        new_xbase = _A.clip(xbase + shift, 0, 1)
        actual_shift = new_xbase - xbase
        for i in range(len(XPT)):
            XPT[i] = XPT[i] - actual_shift
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
            if not FVAL:
                break
            kopt = min(range(len(FVAL)), key=FVAL.__getitem__)

            # Reset the min-Frobenius update's prior Hessian for this TR
            # pass (see NEWUOA comment for rationale).
            self._H_prev = _A.linalg.matrix_zeros(n, n)

            # Shift xbase so the best init point sits at the origin
            # BEFORE the first TR iteration — see NEWUOA #172 comment.
            if float(_A.norm(XPT[kopt])) > 1e-12:
                xbase = self._shift_base_point_bounded(XPT, xbase, kopt, npt, xl, xu)

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

                # Unconditional base-point shift — see #172 / the
                # NEWUOA comment. Plain `g` is the gradient at xbase;
                # the TR step is taken from xbase + XPT[kopt], so the
                # subproblem only stays consistent when XPT[kopt] ≈ 0
                # at the start of every iteration.
                if float(_A.norm(XPT[kopt])) > 1e-12:
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
        if self.evaluations >= self.n_trials:
            return XPT, FVAL
        FVAL.append(self.evaluate(xbase))

        # Coordinate directions, clipped to bound-feasible step sizes.
        # Each evaluate() is budget-guarded so a restart triggered close
        # to n_trials can't overshoot via the init-set (caught by the
        # test_numpy_backend BOBYQA test — 206 vs 200 evals).
        for i in range(n):
            if len(FVAL) >= npt or self.evaluations >= self.n_trials:
                return XPT, FVAL
            step_pos = min(rho, float(xu[i]) - float(xbase[i]))
            if step_pos > 1e-10:
                offset = _A.zeros(n)
                offset[i] = step_pos
                XPT.append(offset)
                FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

            if len(FVAL) >= npt or self.evaluations >= self.n_trials:
                return XPT, FVAL
            step_neg = max(-rho, float(xl[i]) - float(xbase[i]))
            if step_neg < -1e-10:
                offset = _A.zeros(n)
                offset[i] = step_neg
                XPT.append(offset)
                FVAL.append(self.evaluate(_A.clip(xbase + offset, 0, 1)))

        # Optional diagonal-direction point, clipped to bounds.
        if len(FVAL) < npt and self.evaluations < self.n_trials:
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
        """BOBYQA quadratic model via Powell's minimum-Frobenius-norm
        Hessian update (full symmetric H — see NEWUOA's
        `_build_newuoa_model` and the module-level helper for the
        derivation). BOBYQA's only structural difference from NEWUOA is
        that its TR subproblem honours the [xl, xu] box bounds; the
        model build itself is identical.

        Falls back to a bound-aware finite-difference gradient with
        identity Hessian on rank-deficient interpolation sets.
        """
        H_prev = getattr(self, "_H_prev", None)
        if H_prev is None:
            H_prev = _A.linalg.matrix_zeros(n, n)
        try:
            _c, g, H = _build_min_frobenius_quadratic(XPT, FVAL, H_prev, n)
            self._H_prev = H
            return g, H
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
        """Bound-constrained trust-region solve via Powell's TRSBOX
        (see the module-level `_solve_trsbox` helper). Falls back to the
        old projected-Cauchy path if TRSBOX raises — shouldn't happen on
        well-formed inputs but it's the same fallback shape the rest of
        BOBYQA uses."""
        try:
            return _solve_trsbox(g, H, rho, x_current, xl, xu, n)
        except Exception:
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
        new base clipped to the bound box. Iterates over the actual
        list length rather than `npt` so partial init-sets (init
        bailed out on budget) don't IndexError."""
        shift = XPT[kopt].copy()
        new_base = _A.clip(xbase + shift, 0, 1)
        # Use the realised shift (might differ from `shift` if clipping
        # truncated it) so XPT stays consistent.
        actual_shift = new_base - xbase
        for i in range(len(XPT)):
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
