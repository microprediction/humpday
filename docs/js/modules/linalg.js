/**
 * Pure-JS linear-algebra helpers for the PRIMA optimizers.
 *
 * The optimizers need:
 *   - Householder QR with the full m×m Q reconstructed, so the null space
 *     of Aᵀ is just Q[:, n:] (no SVD required).
 *   - Upper-triangular back-substitution.
 *   - QR-based linear solve for square or overdetermined systems.
 *   - Matrix / matrix-vector products and transpose.
 *   - Twins of the pure-Python backend's solve, QR, SVD and pinv
 *     (humpday/_array_pure_linalg.py), which the PRIMA model builders use
 *     so that a fit here takes the same arithmetic path as in Python.
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

    // ------------------------------------------------------------------ //
    // Twins of humpday/_array_pure_linalg.py, statement for statement.    //
    // The PRIMA model builders call these rather than the Householder     //
    // routines above, so that a fit in JavaScript takes the same          //
    // arithmetic path as the pure-Python fit that records the transition  //
    // vectors. Every tolerance is relative to the matrix it came from, as //
    // there: an absolute pivot threshold rejects a perfectly conditioned  //
    // system whose entries are merely small, and a shrinking trust region //
    // produces exactly those systems.                                     //
    // ------------------------------------------------------------------ //

    /** Left-to-right float sum (twin of `_fold_sum`). */
    _foldSum(values) {
        let total = 0.0;
        for (const v of values) total += v;
        return total;
    },

    /** Largest absolute entry (twin of `_scale`). */
    _scale(A) {
        let best = 0.0;
        for (const row of A) {
            for (const v of row) {
                const a = Math.abs(v);
                if (a > best) best = a;
            }
        }
        return best;
    },

    /** Extend orthonormal columns to a basis of R^m (twin of `_complete_orthonormal`). */
    _completeOrthonormal(cols, m) {
        const out = cols.map(c => c.slice());
        for (let i = 0; i < m; i++) {
            if (out.length >= m) break;
            const cand = new Array(m).fill(0.0);
            cand[i] = 1.0;
            for (let pass = 0; pass < 2; pass++) {
                for (const q of out) {
                    let p = 0.0;
                    for (let k = 0; k < m; k++) p += q[k] * cand[k];
                    for (let k = 0; k < m; k++) cand[k] -= p * q[k];
                }
            }
            const nrm = Math.sqrt(Linalg._foldSum(cand.map(v => v * v)));
            if (nrm > 1e-8) out.push(cand.map(v => v / nrm));
        }
        return out;
    },

    /** Gaussian elimination with partial pivoting (twin of `solve`). */
    solve(A, b) {
        const n = A.length;
        const M = A.map(row => row.slice());
        const rhs = b.slice();
        const pivotFloor = 1e-14 * Linalg._scale(M);
        for (let k = 0; k < n; k++) {
            let pivotRow = k;
            let pivotVal = Math.abs(M[k][k]);
            for (let i = k + 1; i < n; i++) {
                const v = Math.abs(M[i][k]);
                if (v > pivotVal) {
                    pivotVal = v;
                    pivotRow = i;
                }
            }
            if (pivotVal <= pivotFloor) {
                throw new Error('solve: singular matrix (pivot below tolerance)');
            }
            if (pivotRow !== k) {
                const tmp = M[k]; M[k] = M[pivotRow]; M[pivotRow] = tmp;
                const t2 = rhs[k]; rhs[k] = rhs[pivotRow]; rhs[pivotRow] = t2;
            }
            const pivot = M[k][k];
            for (let i = k + 1; i < n; i++) {
                const factor = M[i][k] / pivot;
                if (factor === 0.0) continue;
                const Mi = M[i];
                const Mk = M[k];
                Mi[k] = 0.0;
                for (let j = k + 1; j < n; j++) Mi[j] -= factor * Mk[j];
                rhs[i] -= factor * rhs[k];
            }
        }
        const x = new Array(n).fill(0.0);
        for (let i = n - 1; i >= 0; i--) {
            let s = rhs[i];
            const Mi = M[i];
            for (let j = i + 1; j < n; j++) s -= Mi[j] * x[j];
            x[i] = s / Mi[i];
        }
        return x;
    },

    /** In-place Jacobi rotation zeroing A[p][q] (twin of `_jacobi_rotate`). */
    _jacobiRotate(A, V, p, q) {
        const app = A[p][p];
        const aqq = A[q][q];
        const apq = A[p][q];
        if (apq === 0.0) return;
        const theta = (aqq - app) / (2.0 * apq);
        let t;
        if (theta >= 0) t = 1.0 / (theta + Math.sqrt(1.0 + theta * theta));
        else t = 1.0 / (theta - Math.sqrt(1.0 + theta * theta));
        const c = 1.0 / Math.sqrt(1.0 + t * t);
        const s = t * c;
        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;
        const n = A.length;
        for (let i = 0; i < n; i++) {
            if (i === p || i === q) continue;
            const aip = A[i][p];
            const aiq = A[i][q];
            A[i][p] = c * aip - s * aiq;
            A[p][i] = A[i][p];
            A[i][q] = s * aip + c * aiq;
            A[q][i] = A[i][q];
        }
        for (let i = 0; i < n; i++) {
            const vip = V[i][p];
            const viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
        }
    },

    /**
     * Cyclic-Jacobi symmetric eigendecomposition, eigenvalues ascending
     * (twin of `eigh` in the pure backend; the CMA-ES `eigh` above is a
     * different routine and is left alone). Returns { eigvals, eigvecs }.
     */
    eighJacobi(A, tol = 1e-12, maxSweeps = 100) {
        const n = A.length;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                if (Math.abs(A[i][j] - A[j][i]) > tol * (1.0 + Math.abs(A[i][j]) + Math.abs(A[j][i]))) {
                    throw new Error('eigh: A is not symmetric');
                }
            }
        }
        const M = A.map(row => row.slice());
        const V = Linalg.eye(n);
        const tolAbs = tol * Linalg._scale(M);
        for (let sweep = 0; sweep < maxSweeps; sweep++) {
            let off = 0.0;
            for (let p = 0; p < n; p++) {
                const Mp = M[p];
                for (let q = p + 1; q < n; q++) off += Mp[q] * Mp[q];
            }
            if (off <= tolAbs * tolAbs) break;
            for (let p = 0; p < n; p++) {
                for (let q = p + 1; q < n; q++) {
                    if (Math.abs(M[p][q]) > tolAbs) Linalg._jacobiRotate(M, V, p, q);
                }
            }
        }
        const eig = new Array(n);
        for (let i = 0; i < n; i++) eig[i] = M[i][i];
        // Python's sorted() is stable; so is Array.prototype.sort.
        const order = [...Array(n).keys()].sort((a, b) => (eig[a] < eig[b] ? -1 : (eig[b] < eig[a] ? 1 : 0)));
        const eigvals = order.map(i => eig[i]);
        const eigvecs = new Array(n);
        for (let i = 0; i < n; i++) {
            eigvecs[i] = new Array(n);
            for (let j = 0; j < n; j++) eigvecs[i][j] = V[i][order[j]];
        }
        return { eigvals, eigvecs };
    },

    /** Reduced QR by modified Gram-Schmidt (twin of `qr`). Returns { Q, R }. */
    qr(A) {
        const m = A.length;
        const n = A[0].length;
        if (m < n) throw new Error(`qr requires m >= n, got ${m}x${n}`);
        const cols = new Array(n);
        for (let j = 0; j < n; j++) {
            cols[j] = new Array(m);
            for (let i = 0; i < m; i++) cols[j][i] = A[i][j];
        }
        const R = Linalg.zeros(n, n);
        let colMax = 0.0;
        for (let j = 0; j < n; j++) {
            const nr = Math.sqrt(Linalg._foldSum(cols[j].map(v => v * v)));
            if (j === 0 || nr > colMax) colMax = nr;
        }
        const rankFloor = 1e-14 * colMax;
        for (let j = 0; j < n; j++) {
            for (let i = 0; i < j; i++) {
                const qi = cols[i];
                let rij = 0.0;
                for (let k = 0; k < m; k++) rij += qi[k] * cols[j][k];
                R[i][j] = rij;
                for (let k = 0; k < m; k++) cols[j][k] -= rij * qi[k];
            }
            const rjj = Math.sqrt(Linalg._foldSum(cols[j].map(v => v * v)));
            R[j][j] = rjj;
            if (rjj <= rankFloor) {
                cols[j] = Linalg._completeOrthonormal(cols.slice(0, j), m)[j];
            } else {
                cols[j] = cols[j].map(v => v / rjj);
            }
        }
        const Q = new Array(m);
        for (let i = 0; i < m; i++) {
            Q[i] = new Array(n);
            for (let j = 0; j < n; j++) Q[i][j] = cols[j][i];
        }
        return { Q, R };
    },

    /**
     * SVD via the eigendecomposition of AᵀA (twin of `svd`), numpy's
     * conventions: returns { U, s, Vt }, with U m × m when fullMatrices.
     */
    svd(A, fullMatrices = false) {
        const m = A.length;
        const n = A[0].length;
        const k = Math.min(m, n);
        const A2 = A.map(row => row.slice());
        const B = Linalg.matmul(Linalg.transpose(A2), A2);
        const { eigvals, eigvecs: V } = Linalg.eighJacobi(B);
        const order = [...Array(n).keys()].reverse();
        const sigmaSq = order.map(i => eigvals[i]);
        const Vs = new Array(n);
        for (let r = 0; r < n; r++) {
            Vs[r] = new Array(n);
            for (let c = 0; c < n; c++) Vs[r][c] = V[r][order[c]];
        }
        // Python's max(s, 0.0) keeps s unless 0.0 > s.
        const s = sigmaSq.slice(0, k).map(v => Math.sqrt(0.0 > v ? 0.0 : v));
        const sigmaFloor = 1e-14 * (s.length ? s[0] : 0.0);
        let Ucols = [];
        for (let j = 0; j < k; j++) {
            const sigma = s[j];
            if (sigma > sigmaFloor) {
                const vj = new Array(n);
                for (let r = 0; r < n; r++) vj[r] = Vs[r][j];
                const Av = Linalg.matvec(A2, vj);
                Ucols.push(Av.map(c => c / sigma));
            } else {
                Ucols.push(Linalg._completeOrthonormal(Ucols, m)[j]);
            }
        }
        const uWidth = fullMatrices ? m : k;
        if (fullMatrices && m > k) Ucols = Linalg._completeOrthonormal(Ucols, m);
        const U = new Array(m);
        for (let i = 0; i < m; i++) {
            U[i] = new Array(uWidth);
            for (let j = 0; j < uWidth; j++) U[i][j] = Ucols[j][i];
        }
        const vRows = fullMatrices ? n : k;
        const Vt = new Array(vRows);
        for (let c = 0; c < vRows; c++) {
            Vt[c] = new Array(n);
            for (let r = 0; r < n; r++) Vt[c][r] = Vs[r][c];
        }
        return { U, s, Vt };
    },

    /** Moore-Penrose pseudo-inverse through `svd` (twin of `pinv`). */
    pinv(A, rcond = 1e-15) {
        const { U, s, Vt } = Linalg.svd(A, false);
        if (s.length === 0) return Linalg.zeros(A[0].length, A.length);
        let sMax = s[0];
        for (const v of s) if (v > sMax) sMax = v;
        const threshold = sMax > 0 ? rcond * sMax : 0.0;
        const sInv = s.map(v => (v > threshold ? 1.0 / v : 0.0));
        const V = Linalg.transpose(Vt);
        const UT = Linalg.transpose(U);
        const M = sInv.map((si, i) => UT[i].map(v => si * v));
        return Linalg.matmul(V, M);
    },
};

// Export for both Node (tests) and browser globals.
if (typeof module !== "undefined" && module.exports) {
    module.exports = Linalg;
} else {
    window.Linalg = Linalg;
}
