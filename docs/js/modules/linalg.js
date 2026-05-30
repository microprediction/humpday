/**
 * Pure-JS linear-algebra helpers for the PRIMA optimizers.
 *
 * The optimizers need:
 *   - Householder QR with the full m×m Q reconstructed, so the null space
 *     of Aᵀ is just Q[:, n:] (no SVD required).
 *   - Upper-triangular back-substitution.
 *   - QR-based linear solve for square or overdetermined systems.
 *   - Matrix / matrix-vector products and transpose.
 *
 * Matrices are arrays-of-rows (Array<Array<number>>). Vectors are Array<number>.
 * Everything is double-precision JS Number, no typed arrays — small dense
 * matrices are the only use case so the overhead is fine.
 */

const Linalg = {
    /** zeros(rows, cols) → rows × cols matrix of 0s. */
    zeros(rows, cols) {
        const A = new Array(rows);
        for (let i = 0; i < rows; i++) A[i] = new Array(cols).fill(0);
        return A;
    },

    /** eye(n) → n × n identity. */
    eye(n) {
        const I = Linalg.zeros(n, n);
        for (let i = 0; i < n; i++) I[i][i] = 1;
        return I;
    },

    /** transpose(A) → Aᵀ. */
    transpose(A) {
        const rows = A.length;
        const cols = A[0].length;
        const T = Linalg.zeros(cols, rows);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    },

    /** matmul(A, B) → A B. Naive triple-loop; only used for small matrices. */
    matmul(A, B) {
        const m = A.length;
        const k = A[0].length;
        const n = B[0].length;
        const C = Linalg.zeros(m, n);
        for (let i = 0; i < m; i++) {
            const Ai = A[i];
            const Ci = C[i];
            for (let p = 0; p < k; p++) {
                const Aip = Ai[p];
                const Bp = B[p];
                for (let j = 0; j < n; j++) Ci[j] += Aip * Bp[j];
            }
        }
        return C;
    },

    /** matvec(A, x) → A x. */
    matvec(A, x) {
        const m = A.length;
        const n = A[0].length;
        const y = new Array(m).fill(0);
        for (let i = 0; i < m; i++) {
            let s = 0;
            const Ai = A[i];
            for (let j = 0; j < n; j++) s += Ai[j] * x[j];
            y[i] = s;
        }
        return y;
    },

    /** dot(a, b) → scalar. */
    dot(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    },

    /**
     * Householder QR with full Q reconstruction.
     *
     * Given A (m × n), returns { Q, R, Qfull } where
     *   - Qfull is the m × m orthogonal matrix accumulated from the
     *     Householder reflectors,
     *   - Q is Qfull[:, :min(m,n)] (the standard "reduced" Q),
     *   - R is min(m,n) × n upper triangular.
     *
     * The full Qfull is what lets the min-Frobenius helper get the null
     * space of Aᵀ for free (Qfull[:, n:] spans it).
     */
    householderQR(A) {
        const m = A.length;
        const n = A[0].length;
        // Work on a copy of A — R will be left in the upper triangle.
        const R = A.map(row => row.slice());
        // Accumulate Qfull as we apply each reflector.
        const Qfull = Linalg.eye(m);

        const nCols = Math.min(m, n);
        for (let k = 0; k < nCols; k++) {
            // Build the Householder vector v that zeros R[k+1:m, k].
            // a = R[k:m, k]; alpha = -sign(a[0]) * ||a||; v = a - alpha e_1.
            let normSq = 0;
            for (let i = k; i < m; i++) normSq += R[i][k] * R[i][k];
            const norm = Math.sqrt(normSq);
            if (norm < 1e-15) continue;

            const sign = R[k][k] >= 0 ? 1 : -1;
            const alpha = -sign * norm;

            // v has length m - k.
            const v = new Array(m - k);
            v[0] = R[k][k] - alpha;
            for (let i = 1; i < m - k; i++) v[i] = R[k + i][k];

            let vNormSq = 0;
            for (let i = 0; i < v.length; i++) vNormSq += v[i] * v[i];
            if (vNormSq < 1e-30) continue;
            const beta = 2 / vNormSq;

            // Apply H = I - beta v vᵀ to R[k:m, k:n]:
            //   each column j of R[k:, j] becomes R[k:, j] - beta (v · R[k:, j]) v.
            for (let j = k; j < n; j++) {
                let d = 0;
                for (let i = 0; i < m - k; i++) d += v[i] * R[k + i][j];
                const s = beta * d;
                for (let i = 0; i < m - k; i++) R[k + i][j] -= s * v[i];
            }

            // Apply H from the right to Qfull (Qfull ← Qfull · H), which on
            // each row i is Qfull[i, k:] - beta (Qfull[i, k:] · v) v.
            for (let i = 0; i < m; i++) {
                let d = 0;
                const Qi = Qfull[i];
                for (let j = 0; j < m - k; j++) d += Qi[k + j] * v[j];
                const s = beta * d;
                for (let j = 0; j < m - k; j++) Qi[k + j] -= s * v[j];
            }
        }

        // Reduced Q and R.
        const Q = Qfull.map(row => row.slice(0, nCols));
        const Rreduced = new Array(nCols);
        for (let i = 0; i < nCols; i++) {
            Rreduced[i] = R[i].slice(0, n);
            // Zero out any residual sub-diagonal noise.
            for (let j = 0; j < i; j++) Rreduced[i][j] = 0;
        }

        return { Q, R: Rreduced, Qfull };
    },

    /**
     * Back-substitute an upper-triangular system R x = b. Throws on a
     * singular diagonal.
     */
    solveUpperTriangular(R, b) {
        const n = R.length;
        const x = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            let s = b[i];
            for (let j = i + 1; j < n; j++) s -= R[i][j] * x[j];
            if (Math.abs(R[i][i]) < 1e-15) {
                throw new Error("singular upper-triangular system");
            }
            x[i] = s / R[i][i];
        }
        return x;
    },

    /**
     * Solve A x = b (square or overdetermined) by QR. Returns the
     * least-squares solution. Throws if A is rank-deficient.
     */
    solveLinearSystem(A, b) {
        const { Q, R } = Linalg.householderQR(A);
        // QᵀA = R, so x = R^-1 Qᵀ b.
        const QTb = Linalg.matvec(Linalg.transpose(Q), b);
        return Linalg.solveUpperTriangular(R, QTb);
    },

    /**
     * Cholesky factorisation A = L Lᵀ for symmetric positive-definite A.
     * Returns the lower-triangular L. Throws if A is not SPD — the
     * standard "negative pivot" test, used as a cheap SPD check by the
     * NEWUOA TR solver.
     */
    cholesky(A) {
        const n = A.length;
        const L = Linalg.zeros(n, n);
        for (let j = 0; j < n; j++) {
            let s = A[j][j];
            for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
            if (s <= 1e-12) throw new Error("matrix not SPD");
            L[j][j] = Math.sqrt(s);
            for (let i = j + 1; i < n; i++) {
                let s2 = A[i][j];
                for (let k = 0; k < j; k++) s2 -= L[i][k] * L[j][k];
                L[i][j] = s2 / L[j][j];
            }
        }
        return L;
    },

    /** outer(u, v) → u vᵀ (m × n matrix). */
    outer(u, v) {
        const m = u.length;
        const n = v.length;
        const A = Linalg.zeros(m, n);
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) A[i][j] = u[i] * v[j];
        }
        return A;
    },

    /** diag(d) → n × n diagonal matrix with d on the diagonal. */
    diag(d) {
        const n = d.length;
        const A = Linalg.zeros(n, n);
        for (let i = 0; i < n; i++) A[i][i] = d[i];
        return A;
    },

    /**
     * Symmetric eigendecomposition via the classic Jacobi method.
     * Returns { eigvals, eigvecs } where `eigvecs[:, i]` is the eigenvector
     * for eigenvalue `eigvals[i]`. Eigenvalues are not sorted.
     *
     * Jacobi is O(n³) per sweep with cubic convergence for well-separated
     * eigenvalues — fine for the small (n ≤ ~20) symmetric matrices the
     * CMA-ES covariance update needs. The CyclicJacobi sweep schedule is
     * used: visit each off-diagonal pair (p, q) once per sweep.
     */
    eigh(A) {
        const n = A.length;
        // Work on a deep copy of A.
        const M = A.map(row => row.slice());
        const V = Linalg.eye(n);

        const maxSweeps = 50;
        const tol = 1e-14;

        for (let sweep = 0; sweep < maxSweeps; sweep++) {
            // Frobenius norm of off-diagonal part.
            let off = 0;
            for (let p = 0; p < n; p++) {
                for (let q = p + 1; q < n; q++) off += 2 * M[p][q] * M[p][q];
            }
            if (off < tol * tol) break;

            for (let p = 0; p < n - 1; p++) {
                for (let q = p + 1; q < n; q++) {
                    const Mpq = M[p][q];
                    if (Math.abs(Mpq) < tol) continue;

                    // Compute the Jacobi rotation that zeros M[p][q].
                    const theta = (M[q][q] - M[p][p]) / (2 * Mpq);
                    let t;
                    if (Math.abs(theta) > 1e150) {
                        t = 0.5 / theta;
                    } else {
                        const sign = theta >= 0 ? 1 : -1;
                        t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    }
                    const cVal = 1 / Math.sqrt(t * t + 1);
                    const sVal = t * cVal;

                    // Update M (only the affected rows/cols).
                    M[p][p] -= t * Mpq;
                    M[q][q] += t * Mpq;
                    M[p][q] = 0;
                    M[q][p] = 0;
                    for (let i = 0; i < n; i++) {
                        if (i === p || i === q) continue;
                        const Mip = M[i][p];
                        const Miq = M[i][q];
                        M[i][p] = cVal * Mip - sVal * Miq;
                        M[p][i] = M[i][p];
                        M[i][q] = sVal * Mip + cVal * Miq;
                        M[q][i] = M[i][q];
                    }
                    // Accumulate the rotation in V (V ← V · R).
                    for (let i = 0; i < n; i++) {
                        const Vip = V[i][p];
                        const Viq = V[i][q];
                        V[i][p] = cVal * Vip - sVal * Viq;
                        V[i][q] = sVal * Vip + cVal * Viq;
                    }
                }
            }
        }

        const eigvals = new Array(n);
        for (let i = 0; i < n; i++) eigvals[i] = M[i][i];
        return { eigvals, eigvecs: V };
    },

    /**
     * Solve a SPD system A x = b given the Cholesky factor L of A.
     */
    solveSPDFromCholesky(L, b) {
        const n = L.length;
        // Forward solve L y = b.
        const y = new Array(n);
        for (let i = 0; i < n; i++) {
            let s = b[i];
            for (let j = 0; j < i; j++) s -= L[i][j] * y[j];
            y[i] = s / L[i][i];
        }
        // Back solve Lᵀ x = y.
        const x = new Array(n);
        for (let i = n - 1; i >= 0; i--) {
            let s = y[i];
            for (let j = i + 1; j < n; j++) s -= L[j][i] * x[j];
            x[i] = s / L[i][i];
        }
        return x;
    },
};

// Export for both Node (tests) and browser globals.
if (typeof module !== "undefined" && module.exports) {
    module.exports = Linalg;
} else {
    window.Linalg = Linalg;
}
