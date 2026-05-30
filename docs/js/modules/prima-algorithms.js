/**
 * PRIMA algorithm implementations: UOBYQA, NEWUOA, and BOBYQA.
 *
 * These are JavaScript implementations of Powell's Recent Interpolation Methods (PRIMA)
 * algorithms. PRIMA represents state-of-the-art derivative-free optimization
 * for small to medium-scale problems.
 *
 * Reference: https://www.pdfo.net/
 */

// Make Optimizer and MathUtils available as globals so the class
// declarations below (class X extends Optimizer …) resolve in both
// environments. In the browser, base-optimizer.js — loaded as a
// <script> before this file — already sets window.Optimizer /
// window.MathUtils, so we just need to handle Node here. Using
// globalThis avoids the redeclaration error you get if every
// per-family module declares `const Optimizer` at script top level.
if (typeof module !== 'undefined' && module.exports) {
    const _base = require('./base-optimizer.js');
    globalThis.Optimizer = _base.Optimizer;
    globalThis.MathUtils = _base.MathUtils;
    globalThis.Linalg = require('./linalg.js');
}

/**
 * Powell's NEWUOA-style minimum-Frobenius-norm quadratic update —
 * mirrors `humpday/optimizers/prima_algorithms.py::
 * _build_min_frobenius_quadratic` step-for-step (same KKT
 * derivation, same vech ordering, same Frobenius weights). See the
 * Python helper's docstring for the math; this is the pure-JS port.
 *
 * Returns `{ c, g, H }` on success; throws on rank-deficient A_l or
 * non-finite FVAL so the caller can fall back to a simpler model.
 */
function buildMinFrobeniusQuadratic(XPT, FVAL, Hprev, n) {
    const npt = FVAL.length;
    const pLin = 1 + n;
    const pQuad = n * (n + 1) / 2;
    const nNull = npt - pLin;

    if (nNull < 0) throw new Error('PRIMA: npt < pLin');

    for (let k = 0; k < npt; k++) {
        if (!Number.isFinite(FVAL[k])) {
            throw new Error('PRIMA: non-finite FVAL');
        }
    }

    // Build A_l (npt × pLin), A_q (npt × pQuad), and b.
    // vech(H) ordering: n diagonals first, then n(n-1)/2 strict-upper
    // off-diagonals in row-major order.
    const Al = Linalg.zeros(npt, pLin);
    const Aq = Linalg.zeros(npt, pQuad);
    const b = new Array(npt);

    for (let k = 0; k < npt; k++) {
        const x = XPT[k];
        Al[k][0] = 1.0;
        for (let i = 0; i < n; i++) Al[k][i + 1] = x[i];

        let col = 0;
        for (let i = 0; i < n; i++) {
            Aq[k][col++] = 0.5 * x[i] * x[i];
        }
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                Aq[k][col++] = x[i] * x[j];
            }
        }

        // b[k] = FVAL[k] − ½ x_k^T H_prev x_k.
        const HpX = Linalg.matvec(Hprev, x);
        let quadForm = 0;
        for (let i = 0; i < n; i++) quadForm += x[i] * HpX[i];
        b[k] = FVAL[k] - 0.5 * quadForm;
    }

    // Frobenius weights: 1 on diagonal entries, 2 on off-diagonals.
    const wQ = new Array(pQuad);
    for (let i = 0; i < n; i++) wQ[i] = 1.0;
    for (let i = n; i < pQuad; i++) wQ[i] = 2.0;

    // Householder QR of A_l; Qfull is the m × m orthogonal matrix so
    // Qfull[:, pLin:] spans null(A_l^T) — the basis we need below.
    const { Q: Ql, R: Rl, Qfull: QlFull } = Linalg.householderQR(Al);

    // Rank check on R's diagonal (= singular-value proxy for the SVD-
    // based check the Python helper uses).
    let maxR = 0, minR = Infinity;
    for (let i = 0; i < pLin; i++) {
        const v = Math.abs(Rl[i][i]);
        if (v > maxR) maxR = v;
        if (v < minR) minR = v;
    }
    if (maxR === 0 || minR <= 1e-12 * maxR) {
        throw new Error('PRIMA: A_l rank-deficient');
    }

    // KKT-reduced solve for x_q (vech(ΔH)).
    let xQ;
    if (nNull > 0) {
        // Z = QlFull[:, pLin:] : npt × nNull.
        const Z = Linalg.zeros(npt, nNull);
        for (let i = 0; i < npt; i++) {
            for (let j = 0; j < nNull; j++) Z[i][j] = QlFull[i][pLin + j];
        }

        // Z^T b : length nNull.
        const ZTb = new Array(nNull);
        for (let j = 0; j < nNull; j++) {
            let acc = 0;
            for (let k = 0; k < npt; k++) acc += Z[k][j] * b[k];
            ZTb[j] = acc;
        }

        // Z^T A_q : nNull × pQuad.
        const ZTAq = Linalg.zeros(nNull, pQuad);
        for (let j = 0; j < nNull; j++) {
            for (let c = 0; c < pQuad; c++) {
                let acc = 0;
                for (let k = 0; k < npt; k++) acc += Z[k][j] * Aq[k][c];
                ZTAq[j][c] = acc;
            }
        }

        // (A_q / W_q)^T Z : pQuad × nNull.
        const AqScaledTZ = Linalg.zeros(pQuad, nNull);
        for (let c = 0; c < pQuad; c++) {
            const invW = 1.0 / wQ[c];
            for (let j = 0; j < nNull; j++) {
                let acc = 0;
                for (let k = 0; k < npt; k++) acc += Aq[k][c] * invW * Z[k][j];
                AqScaledTZ[c][j] = acc;
            }
        }

        // M = (Z^T A_q)(A_q/W_q)^T Z : nNull × nNull, symmetric PSD.
        const M = Linalg.matmul(ZTAq, AqScaledTZ);
        const negZTb = ZTb.map(v => -v);
        const mu = Linalg.solveLinearSystem(M, negZTb);

        xQ = new Array(pQuad);
        for (let c = 0; c < pQuad; c++) {
            let acc = 0;
            for (let j = 0; j < nNull; j++) acc += AqScaledTZ[c][j] * mu[j];
            xQ[c] = -acc;
        }
    } else {
        xQ = new Array(pQuad).fill(0);
    }

    // x_l: solve A_l x_l = b − A_q x_q via QR.
    const AqXq = new Array(npt);
    for (let k = 0; k < npt; k++) {
        let acc = 0;
        for (let c = 0; c < pQuad; c++) acc += Aq[k][c] * xQ[c];
        AqXq[k] = acc;
    }
    const rhs = new Array(npt);
    for (let k = 0; k < npt; k++) rhs[k] = b[k] - AqXq[k];

    const QlTrhs = Linalg.matvec(Linalg.transpose(Ql), rhs);
    const xL = Linalg.solveUpperTriangular(Rl, QlTrhs);

    const c = xL[0];
    const g = xL.slice(1, n + 1);

    // Reconstruct symmetric H = H_prev + ΔH.
    const H = Linalg.zeros(n, n);
    let col = 0;
    for (let i = 0; i < n; i++) H[i][i] = Hprev[i][i] + xQ[col++];
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const val = Hprev[i][j] + xQ[col++];
            H[i][j] = val;
            H[j][i] = val;
        }
    }

    // Belt-and-braces finite check — a barely-singular QR can produce
    // non-finite coefficients that poison Hprev for subsequent calls.
    if (!Number.isFinite(c)) throw new Error('PRIMA: non-finite c');
    for (let i = 0; i < n; i++) {
        if (!Number.isFinite(g[i])) throw new Error('PRIMA: non-finite g');
        for (let j = 0; j < n; j++) {
            if (!Number.isFinite(H[i][j])) throw new Error('PRIMA: non-finite H');
        }
    }

    return { c, g, H };
}

/**
 * Powell's TRSBOX bound-constrained TR subproblem solver — mirrors
 * `humpday/optimizers/prima_algorithms.py::_solve_trsbox` step-for-step.
 *
 * Solves min Q(d) = g·d + ½ dᵀ H d  subject to ‖d‖₂ ≤ rho and
 * xl ≤ x_current + d ≤ xu via active-set Steihaug-Toint CG. Phase 3
 * (Powell's alternative iteration along the TR boundary) is omitted;
 * see the Python helper's docstring for the rationale.
 */
function solveTrsbox(g, H, rho, xCurrent, xl, xu, n) {
    const tol = 1e-12;

    for (let i = 0; i < n; i++) {
        if (!Number.isFinite(g[i])) throw new Error('PRIMA: non-finite g');
        for (let j = 0; j < n; j++) {
            if (!Number.isFinite(H[i][j])) throw new Error('PRIMA: non-finite H');
        }
    }

    const loD = new Array(n);
    const hiD = new Array(n);
    for (let i = 0; i < n; i++) {
        loD[i] = xl[i] - xCurrent[i];
        hiD[i] = xu[i] - xCurrent[i];
    }

    const d = new Array(n).fill(0);
    const active = new Array(n).fill(null);  // null | 'lo' | 'hi'

    const outerCap = 2 * n + 10;
    const innerCap = 2 * n + 10;

    for (let outer = 0; outer < outerCap; outer++) {
        const Hd = Linalg.matvec(H, d);
        const grad = new Array(n);
        for (let i = 0; i < n; i++) grad[i] = g[i] + Hd[i];

        // Catch coordinates that landed exactly on a bound from the
        // prior step (ties in α_box, or numerical drift). Must run
        // *before* the release check so freshly-detected activations
        // can still be released when the gradient points inward.
        for (let i = 0; i < n; i++) {
            if (active[i] === null) {
                if (d[i] <= loD[i] + tol) {
                    active[i] = 'lo';
                    d[i] = loD[i];
                } else if (d[i] >= hiD[i] - tol) {
                    active[i] = 'hi';
                    d[i] = hiD[i];
                }
            }
        }

        // Release any active bound whose gradient points into the
        // feasible region.
        for (let i = 0; i < n; i++) {
            if (active[i] === 'lo' && grad[i] < -tol) active[i] = null;
            else if (active[i] === 'hi' && grad[i] > tol) active[i] = null;
        }

        // Projected residual r on the current free subspace.
        const r = new Array(n);
        let rNormSq = 0;
        for (let i = 0; i < n; i++) {
            r[i] = active[i] === null ? -grad[i] : 0;
            rNormSq += r[i] * r[i];
        }
        if (rNormSq < tol * tol) break;

        // Inner CG.
        let p = r.slice();
        let dNormSq = 0;
        for (let i = 0; i < n; i++) dNormSq += d[i] * d[i];
        let boundAdded = false;

        let rCur = r;
        let rCurNormSq = rNormSq;

        for (let inner = 0; inner < innerCap; inner++) {
            const HpFull = Linalg.matvec(H, p);
            const Hp = new Array(n);
            let pHp = 0;
            for (let i = 0; i < n; i++) {
                Hp[i] = active[i] === null ? HpFull[i] : 0;
                pHp += p[i] * Hp[i];
            }

            // α_box: smallest positive step before a new bound is hit.
            let alphaBox = Infinity;
            let newIdx = -1;
            let newSide = null;
            for (let i = 0; i < n; i++) {
                if (active[i] !== null) continue;
                if (p[i] > tol) {
                    const a = (hiD[i] - d[i]) / p[i];
                    if (a > 0 && a < alphaBox) {
                        alphaBox = a;
                        newIdx = i;
                        newSide = 'hi';
                    }
                } else if (p[i] < -tol) {
                    const a = (loD[i] - d[i]) / p[i];
                    if (a > 0 && a < alphaBox) {
                        alphaBox = a;
                        newIdx = i;
                        newSide = 'lo';
                    }
                }
            }

            // α_tr: step to TR boundary ‖d + α p‖ = rho.
            let pp = 0;
            for (let i = 0; i < n; i++) pp += p[i] * p[i];
            if (pp < tol * tol) break;
            let dp = 0;
            for (let i = 0; i < n; i++) dp += d[i] * p[i];
            const disc = dp * dp - pp * (dNormSq - rho * rho);
            const alphaTr = disc < 0 ? 0 : (-dp + Math.sqrt(disc)) / pp;

            // α_cg: CG step length on positive curvature.
            const alphaCg = pHp > tol ? rCurNormSq / pHp : Infinity;

            const alpha = Math.min(alphaCg, alphaBox, alphaTr);
            if (!(alpha > 0) || !Number.isFinite(alpha)) break;

            for (let i = 0; i < n; i++) d[i] += alpha * p[i];
            dNormSq = 0;
            for (let i = 0; i < n; i++) dNormSq += d[i] * d[i];

            if (alpha >= alphaTr - tol) {
                return d;  // TR boundary; Phase 3 omitted.
            }

            if (alpha >= alphaBox - tol && newIdx >= 0) {
                active[newIdx] = newSide;
                d[newIdx] = newSide === 'hi' ? hiD[newIdx] : loD[newIdx];
                boundAdded = true;
                break;
            }

            // α_cg branch: continue CG.
            const rNew = new Array(n);
            let rNewNormSq = 0;
            for (let i = 0; i < n; i++) {
                rNew[i] = rCur[i] - alpha * Hp[i];
                rNewNormSq += rNew[i] * rNew[i];
            }
            if (rNewNormSq < tol * tol) return d;

            const beta = rNewNormSq / rCurNormSq;
            const pNew = new Array(n);
            for (let i = 0; i < n; i++) {
                pNew[i] = active[i] === null ? rNew[i] + beta * p[i] : 0;
            }
            p = pNew;
            rCur = rNew;
            rCurNormSq = rNewNormSq;
        }

        if (!boundAdded) break;
    }

    return d;
}

/**
 * Unbounded trust-region subproblem solver — Newton step when H is SPD
 * and the step is inside the TR, otherwise dogleg. Mirrors
 * `_solve_trust_region_newuoa` + `_dogleg_method` in the Python module.
 *
 * NEWUOA's TR step has no bound handling (Python clips xnew at the
 * end). For BOBYQA the bound-constrained TRSBOX is used instead.
 */
function solveTrustRegionUnbounded(g, H, rho, n) {
    // Newton path if H is SPD enough and Newton step is inside TR.
    try {
        const L = Linalg.cholesky(H);
        const negG = g.map(v => -v);
        const dNewton = Linalg.solveSPDFromCholesky(L, negG);
        let nrm = 0;
        for (let i = 0; i < n; i++) nrm += dNewton[i] * dNewton[i];
        nrm = Math.sqrt(nrm);
        if (nrm <= rho) return dNewton;
        // Continue: Newton outside TR → fall through to dogleg with Newton hint.
    } catch (e) {
        // Not SPD → dogleg.
    }
    return dogleg(g, H, rho, n);
}

function dogleg(g, H, rho, n) {
    let gNormSq = 0;
    for (let i = 0; i < n; i++) gNormSq += g[i] * g[i];
    if (gNormSq < 1e-24) return new Array(n).fill(0);

    const Hg = Linalg.matvec(H, g);
    let gHg = 0;
    for (let i = 0; i < n; i++) gHg += g[i] * Hg[i];
    const alphaC = gHg > 1e-12 ? gNormSq / gHg : 1.0;

    const dCauchy = g.map(v => -alphaC * v);
    let cauchyNorm = 0;
    for (let i = 0; i < n; i++) cauchyNorm += dCauchy[i] * dCauchy[i];
    cauchyNorm = Math.sqrt(cauchyNorm);

    if (cauchyNorm >= rho) {
        // Cauchy is already outside TR → step rho along -g.
        const gNorm = Math.sqrt(gNormSq);
        return g.map(v => (-rho / gNorm) * v);
    }

    try {
        const L = Linalg.cholesky(H);
        const negG = g.map(v => -v);
        const dNewton = Linalg.solveSPDFromCholesky(L, negG);
        let nNorm = 0;
        for (let i = 0; i < n; i++) nNorm += dNewton[i] * dNewton[i];
        nNorm = Math.sqrt(nNorm);
        if (nNorm <= rho) return dNewton;

        // Dogleg: find τ ∈ [0,1] s.t. ‖dC + τ(dN − dC)‖ = rho.
        const diff = new Array(n);
        for (let i = 0; i < n; i++) diff[i] = dNewton[i] - dCauchy[i];
        let a = 0, b = 0, c = 0;
        for (let i = 0; i < n; i++) {
            a += diff[i] * diff[i];
            b += 2 * dCauchy[i] * diff[i];
            c += dCauchy[i] * dCauchy[i];
        }
        c -= rho * rho;
        const disc = b * b - 4 * a * c;
        if (disc >= 0 && a > 1e-12) {
            const tau = (-b + Math.sqrt(disc)) / (2 * a);
            const out = new Array(n);
            for (let i = 0; i < n; i++) out[i] = dCauchy[i] + tau * diff[i];
            return out;
        }
    } catch (e) {
        // Not SPD → just return Cauchy.
    }
    return dCauchy;
}

class PRIMA_UOBYQA extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'PRIMA_UOBYQA';
        // UOBYQA uses up to (n+1)(n+2)/2 interpolation points for full quadratic model
        this.npt = Math.min((nDim + 1) * (nDim + 2) / 2, Math.max(2 * nDim + 1, nTrials / 4));
    }

    optimize() {
        const n = this.nDim;
        const npt = this.npt;

        // Initialize starting point (away from boundaries for stability)
        let xbase = Array(n).fill(0).map(() => 0.3 + 0.4 * Math.random());
        let fbase = this.evaluate(xbase);

        // Trust region parameters EXACTLY matching PDFO's aggressive behavior
        let rho = 0.5;  // LARGER initial trust region radius like PDFO
        // Final trust-region radius. Was 1e-3 ("relaxed for visualization")
        // which terminated UOBYQA at a coarse precision — well before the
        // user's evaluation budget was exhausted on most problems.
        // Matches NEWUOA and BOBYQA's 1e-8.
        const rhoend = 1e-8;
        const eta1 = 0.01; // MORE AGGRESSIVE step acceptance (accept more steps)
        const eta2 = 0.25; // MORE AGGRESSIVE trust region expansion
        const gamma1 = 0.5; // Contraction factor
        const gamma2 = 4.0; // MORE AGGRESSIVE expansion factor

        // Initialize interpolation set - REAL Lagrange interpolation points
        let XPT = Array(npt).fill().map(() => Array(n).fill(0)); // Interpolation points relative to xbase
        let FVAL = Array(npt).fill(0); // Function values at interpolation points

        XPT[0] = Array(n).fill(0); // First point is xbase (origin in shifted coordinates)
        FVAL[0] = fbase;

        // Build PROPER initial interpolation set using coordinate directions
        this.buildInitialLagrangeSet(XPT, FVAL, xbase, rho);

        let kopt = 0; // Index of best point
        for (let k = 1; k < XPT.length; k++) {
            if (FVAL[k] < FVAL[kopt]) {
                kopt = k;
            }
        }

        let xopt = MathUtils.add(xbase, XPT[kopt]); // Best point in original coordinates
        let fopt = FVAL[kopt];

        // Main UOBYQA loop with PDFO-like aggressive behavior
        let iterations = 0;
        const maxIterations = this.nTrials;

        while (this.evaluations < this.nTrials && rho > rhoend && iterations < maxIterations) {
            iterations++;

            // Build LAGRANGE quadratic model
            const model = this.buildLagrangeQuadraticModel(XPT, FVAL, kopt);

            // Solve trust region subproblem with proper optimization
            const step = this.solveTrustRegionSubproblem(model, rho);

            if (this.evaluations >= this.nTrials) break;

            // Compute trial point
            let d = step; // Step from current best point
            let xnew = MathUtils.add(xopt, d);
            xnew = MathUtils.clipArray(xnew, 0, 1); // Enforce bounds

            const fnew = this.evaluate(xnew);

            // Compute predicted reduction using PROPER quadratic model
            const predred = this.computeQuadraticPrediction(model, d);
            const actred = fopt - fnew;
            const ratio = predred > 1e-15 ? actred / predred : -1;

            // AGGRESSIVE trust region update like PDFO
            let rho_new = rho;
            const stepnorm = MathUtils.norm(d);

            if (ratio <= eta1) {
                rho_new = gamma1 * rho;
            } else if (ratio >= eta2 && stepnorm >= 0.8 * rho) {
                rho_new = Math.min(gamma2 * rho, 2.0); // Cap expansion
            }

            // AGGRESSIVE accept/reject criteria like PDFO
            if (ratio > eta1 || fnew < fopt) {
                // Accept the step - update best point
                xopt = xnew.slice();
                fopt = fnew;

                // Update interpolation set with PROPER geometry management
                this.updateLagrangeInterpolationSet(XPT, FVAL, xbase, MathUtils.subtract(xnew, xbase), fnew);

                // Find new best point index
                kopt = 0;
                for (let k = 1; k < XPT.length; k++) {
                    if (FVAL[k] < FVAL[kopt]) {
                        kopt = k;
                    }
                }

                xopt = MathUtils.add(xbase, XPT[kopt]);
                fopt = FVAL[kopt];
            }

            // Update trust region radius
            rho = Math.max(rho_new, rhoend);
        }

        return {
            bestValue: fopt,
            bestX: xopt,
            evaluations: this.evaluations,
            success: rho <= rhoend,
            path: this.trackPath ? this.path : null
        };
    }

    buildInitialLagrangeSet(XPT, FVAL, xbase, rho) {
        // Build UOBYQA interpolation set EXACTLY like PDFO does
        const n = this.nDim;
        const npt = this.npt;

        // First point is already set to origin (xbase)
        let kptNum = 1;

        // EXACTLY replicate PDFO's coordinate direction strategy
        for (let j = 0; j < n && kptNum < npt; j++) {
            // Positive direction - AGGRESSIVE step like PDFO
            let stepa = Math.min(1.0, 1.0 - xbase[j]); // Full step to boundary or 1.0
            XPT[kptNum] = Array(n).fill(0);
            XPT[kptNum][j] = stepa;

            let xnew = MathUtils.add(xbase, XPT[kptNum]);
            FVAL[kptNum] = this.evaluate(xnew);
            kptNum++;

            if (kptNum >= npt) break;

            // Negative direction - AGGRESSIVE step like PDFO
            let stepb = -Math.min(xbase[j], 1.0); // Full step to boundary or -1.0
            XPT[kptNum] = Array(n).fill(0);
            XPT[kptNum][j] = stepb;

            xnew = MathUtils.add(xbase, XPT[kptNum]);
            FVAL[kptNum] = this.evaluate(xnew);
            kptNum++;
        }

        // Add corner points like PDFO does for better coverage
        while (kptNum < npt && this.evaluations < this.nTrials) {
            let xpt_new = Array(n).fill(0);

            // Add systematic corner/diagonal points
            for (let i = 0; i < n; i++) {
                // Alternate between moving toward origin and away from origin
                if ((kptNum % 2) === 0) {
                    xpt_new[i] = -xbase[i]; // Move toward origin (0,0)
                } else {
                    xpt_new[i] = (0.5 - xbase[i]) * (Math.random() - 0.5) * 2;
                }
            }

            XPT[kptNum] = xpt_new;
            let xnew = MathUtils.add(xbase, XPT[kptNum]);
            // Clip to bounds
            for (let i = 0; i < n; i++) {
                xnew[i] = Math.max(0, Math.min(1, xnew[i]));
            }
            FVAL[kptNum] = this.evaluate(xnew);
            kptNum++;
        }

        // Fill remaining slots if needed
        while (kptNum < npt) {
            XPT[kptNum] = Array(n).fill(0);
            FVAL[kptNum] = FVAL[0]; // Use base value
            kptNum++;
        }
    }

    buildLagrangeQuadraticModel(XPT, FVAL, kopt) {
        // Build PROPER Lagrange interpolation quadratic model
        // Mathematical foundation: m(s) = Σᵢ FVAL[i] * lᵢ(s)
        // where lᵢ(s) are Lagrange basis polynomials and s = x - xopt

        const n = this.nDim;
        const npt = XPT.length;
        const xopt = XPT[kopt].slice(); // Optimal point in shifted coordinates

        // Initialize quadratic model: m(s) = c + gᵀs + ½sᵀHs
        const model = {
            c: FVAL[kopt], // Constant term at optimal point
            g: Array(n).fill(0), // Gradient vector
            H: Array(n).fill().map(() => Array(n).fill(0)) // Hessian matrix
        };

        // Build coefficient matrix for quadratic polynomial interpolation
        // Each row represents the polynomial evaluated at one interpolation point
        const numCoeffs = 1 + n + n*(n+1)/2; // constant + linear + quadratic terms
        const A = Array(npt).fill().map(() => Array(numCoeffs).fill(0));
        const b = FVAL.slice();

        for (let k = 0; k < npt; k++) {
            const s = MathUtils.subtract(XPT[k], xopt); // Shift to optimal point
            let col = 0;

            // Constant term
            A[k][col++] = 1.0;

            // Linear terms: sⱼ
            for (let j = 0; j < n; j++) {
                A[k][col++] = s[j];
            }

            // Quadratic terms: sᵢsⱼ (symmetric, so only upper triangle)
            for (let i = 0; i < n; i++) {
                for (let j = i; j < n; j++) {
                    A[k][col++] = s[i] * s[j];
                }
            }
        }

        // Solve least squares system: A * coeff = b
        // Since we may have more points than polynomial terms, use overdetermined system
        const coeff = this.solveLeastSquares(A, b);

        if (coeff) {
            let col = 1; // Skip constant term (already set)

            // Extract gradient coefficients
            for (let j = 0; j < n; j++) {
                model.g[j] = coeff[col++];
            }

            // Extract Hessian coefficients
            for (let i = 0; i < n; i++) {
                for (let j = i; j < n; j++) {
                    const hess_coeff = coeff[col++];
                    if (i === j) {
                        model.H[i][j] = 2.0 * hess_coeff; // Diagonal terms: 2 * coefficient
                    } else {
                        model.H[i][j] = model.H[j][i] = hess_coeff; // Off-diagonal symmetric
                    }
                }
            }
        } else {
            // Fallback: use finite differences if Lagrange interpolation fails
            console.warn("Lagrange interpolation failed, falling back to finite differences");
            this.buildFiniteDifferenceModel(XPT, FVAL, kopt, model);
        }

        return model;
    }

    solveLeastSquares(A, b) {
        // Solve overdetermined system using normal equations: (AᵀA)x = Aᵀb
        const m = A.length;
        const n = A[0].length;

        // Compute AᵀA
        const AtA = Array(n).fill().map(() => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                for (let k = 0; k < m; k++) {
                    AtA[i][j] += A[k][i] * A[k][j];
                }
            }
        }

        // Compute Aᵀb
        const Atb = Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            for (let k = 0; k < m; k++) {
                Atb[i] += A[k][i] * b[k];
            }
        }

        // Solve AtA * x = Atb using simple Gaussian elimination with pivoting
        return this.gaussianElimination(AtA, Atb);
    }

    gaussianElimination(A, b) {
        const n = A.length;
        const augmented = A.map((row, i) => [...row, b[i]]);

        // Forward elimination with partial pivoting
        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }

            // Swap rows
            [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

            // Check for singular matrix
            if (Math.abs(augmented[i][i]) < 1e-12) {
                return null; // Singular matrix
            }

            // Eliminate column
            for (let k = i + 1; k < n; k++) {
                const factor = augmented[k][i] / augmented[i][i];
                for (let j = i; j <= n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }

        // Back substitution
        const x = Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            x[i] = augmented[i][n];
            for (let j = i + 1; j < n; j++) {
                x[i] -= augmented[i][j] * x[j];
            }
            x[i] /= augmented[i][i];
        }

        return x;
    }

    buildFiniteDifferenceModel(XPT, FVAL, kopt, model) {
        // Fallback finite difference approximation
        const n = this.nDim;
        const xopt = XPT[kopt];

        // Simple gradient estimation using available points
        for (let i = 0; i < n; i++) {
            let forward = null, backward = null;
            let min_forward_dist = Infinity, min_backward_dist = Infinity;

            for (let k = 0; k < XPT.length; k++) {
                if (k === kopt) continue;

                const diff = MathUtils.subtract(XPT[k], xopt);
                const coord_diff = diff[i];
                const other_norm = Math.sqrt(diff.reduce((sum, x, j) => j !== i ? sum + x*x : sum, 0));

                if (other_norm < 0.1) { // Approximately along coordinate i
                    const dist = Math.abs(coord_diff);
                    if (coord_diff > 0 && dist < min_forward_dist) {
                        forward = k;
                        min_forward_dist = dist;
                    } else if (coord_diff < 0 && dist < min_backward_dist) {
                        backward = k;
                        min_backward_dist = dist;
                    }
                }
            }

            // Compute gradient using available points
            if (forward !== null && backward !== null) {
                const h = XPT[forward][i] - XPT[backward][i];
                model.g[i] = (FVAL[forward] - FVAL[backward]) / h;
                model.H[i][i] = (FVAL[forward] - 2*FVAL[kopt] + FVAL[backward]) / (h*h/4);
            } else if (forward !== null) {
                const h = XPT[forward][i] - xopt[i];
                model.g[i] = (FVAL[forward] - FVAL[kopt]) / h;
            } else if (backward !== null) {
                const h = xopt[i] - XPT[backward][i];
                model.g[i] = (FVAL[kopt] - FVAL[backward]) / h;
            }
        }
    }

    solveTrustRegionSubproblem(model, rho) {
        // Solve: min_s m(s) = c + g^T s + 0.5 s^T H s  subject to ||s|| <= rho
        // Using dogleg method for proper trust region optimization

        const n = this.nDim;
        const g = model.g;
        const H = model.H;

        // Compute Cauchy point (steepest descent step)
        const gNorm = MathUtils.norm(g);
        if (gNorm < 1e-4) { // Much more relaxed gradient tolerance for visualization
            return Array(n).fill(0); // Zero gradient, no step
        }

        // Cauchy step: s_c = -(tau * rho / ||g||) * g
        // where tau is chosen to minimize the quadratic along steepest descent

        const gHg = this.quadraticForm(g, H, g); // g^T H g
        let tau;
        if (gHg <= 0) {
            tau = 1.0; // Indefinite Hessian, go to trust region boundary
        } else {
            tau = Math.min(1.0, (gNorm * gNorm * gNorm) / (rho * gHg));
        }

        const sCauchy = MathUtils.scale(g, -tau * rho / gNorm);

        // If Cauchy point is on boundary or Hessian is indefinite, return Cauchy point
        if (tau >= 0.99 || gHg <= 0) {
            return sCauchy;
        }

        // Try Newton step: s_n = -H^{-1} g
        const sNewton = this.solveLinearSystem(H, MathUtils.scale(g, -1));
        if (!sNewton) {
            return sCauchy; // Singular Hessian, use Cauchy point
        }

        const newtonNorm = MathUtils.norm(sNewton);

        // If Newton step is within trust region, use it
        if (newtonNorm <= rho) {
            return sNewton;
        }

        // Dogleg path: combine Cauchy and Newton directions
        // Find intersection of line from Cauchy to Newton with trust region boundary
        const diff = MathUtils.subtract(sNewton, sCauchy);
        const a = MathUtils.dot(diff, diff);
        const b = 2 * MathUtils.dot(sCauchy, diff);
        const c = MathUtils.dot(sCauchy, sCauchy) - rho * rho;

        const discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            return sCauchy; // No intersection, shouldn't happen
        }

        const alpha = (-b + Math.sqrt(discriminant)) / (2 * a);
        const sDogleg = MathUtils.add(sCauchy, MathUtils.scale(diff, alpha));

        return sDogleg;
    }

    quadraticForm(x, A, y) {
        // Compute x^T A y
        let result = 0;
        for (let i = 0; i < x.length; i++) {
            for (let j = 0; j < y.length; j++) {
                result += x[i] * A[i][j] * y[j];
            }
        }
        return result;
    }

    solveLinearSystem(A, b) {
        // Solve Ax = b using Gaussian elimination with pivoting
        const n = A.length;
        const augmented = A.map((row, i) => [...row, b[i]]);

        // Forward elimination with partial pivoting
        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }

            // Swap rows
            [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

            // Check for singular matrix
            if (Math.abs(augmented[i][i]) < 1e-12) {
                return null; // Singular matrix
            }

            // Eliminate column
            for (let k = i + 1; k < n; k++) {
                const factor = augmented[k][i] / augmented[i][i];
                for (let j = i; j <= n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }

        // Back substitution
        const x = Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            x[i] = augmented[i][n];
            for (let j = i + 1; j < n; j++) {
                x[i] -= augmented[i][j] * x[j];
            }
            x[i] /= augmented[i][i];
        }

        return x;
    }

    computeQuadraticPrediction(model, step) {
        // Compute predicted reduction = -(g^T*step + 0.5*step^T*H*step)
        // This is the decrease predicted by the quadratic model
        const n = this.nDim;
        let pred = 0;

        // Linear term: g^T * step
        for (let i = 0; i < n; i++) {
            pred -= model.g[i] * step[i];
        }

        // Quadratic term: 0.5 * step^T * H * step
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                pred -= 0.5 * step[i] * model.H[i][j] * step[j];
            }
        }

        return Math.max(pred, 1e-15); // Ensure positive predicted reduction
    }

    updateLagrangeInterpolationSet(XPT, FVAL, xbase, snew, fnew) {
        // Proper geometry management for Lagrange interpolation
        // Following UOBYQA's interpolation set updating strategy

        const n = this.nDim;
        const npt = XPT.length;

        // Find the interpolation point to replace based on geometry considerations
        let kReplace = -1;
        let maxDistance = -1;

        // Strategy 1: Replace point that is closest to the new point (avoid clustering)
        for (let k = 0; k < npt; k++) {
            const distance = MathUtils.norm(MathUtils.subtract(XPT[k], snew));

            // Only consider replacing if this would improve geometry
            if (distance < 0.1) { // Very close points - consider replacement
                if (FVAL[k] > fnew || distance < 1e-3) { // Relaxed distance tolerance for visualization
                    kReplace = k;
                    break;
                }
            }
        }

        // Strategy 2: If no close point found, replace the worst point
        if (kReplace === -1) {
            let worstValue = -Infinity;
            for (let k = 0; k < npt; k++) {
                if (FVAL[k] > worstValue) {
                    worstValue = FVAL[k];
                    kReplace = k;
                }
            }

            // Only replace if new point is significantly better
            if (fnew >= worstValue - 1e-12) {
                return; // Don't update if new point isn't better
            }
        }

        // Replace the selected point
        if (kReplace >= 0) {
            XPT[kReplace] = snew.slice();
            FVAL[kReplace] = fnew;
        }

        // Maintain geometric quality by ensuring points are not too close
        this.maintainInterpolationSetGeometry(XPT, FVAL);
    }

    maintainInterpolationSetGeometry(XPT, FVAL) {
        // Ensure interpolation points maintain good geometric properties
        const n = this.nDim;
        const npt = XPT.length;
        const minDistance = 1e-8; // Minimum distance between points

        // Remove points that are too close to each other
        for (let i = 0; i < npt; i++) {
            for (let j = i + 1; j < npt; j++) {
                const distance = MathUtils.norm(MathUtils.subtract(XPT[i], XPT[j]));

                if (distance < minDistance) {
                    // Keep the point with better function value
                    if (FVAL[i] > FVAL[j]) {
                        // Replace point i with a perturbed version
                        for (let k = 0; k < n; k++) {
                            XPT[i][k] += (Math.random() - 0.5) * minDistance * 2;
                        }
                    } else {
                        // Replace point j with a perturbed version
                        for (let k = 0; k < n; k++) {
                            XPT[j][k] += (Math.random() - 0.5) * minDistance * 2;
                        }
                    }
                }
            }
        }
    }

    doSystematicRefinement(XPT, FVAL, kopt, xbase) {
        // PDFO-like systematic refinement around best point
        // This mimics PDFO's pattern of going f=0.01 -> f=0.0001 -> f=0.0
        const n = this.nDim;
        const xbest = MathUtils.add(xbase, XPT[kopt]);
        let fbest = FVAL[kopt];

        // Series of increasingly small trust regions for refinement
        const refinementRadii = [0.1, 0.01, 0.001];

        for (let radius of refinementRadii) {
            if (this.evaluations >= this.nTrials) break;

            // Try systematic directions around current best
            for (let i = 0; i < n && this.evaluations < this.nTrials; i++) {
                // Positive direction
                let xtest = xbest.slice();
                xtest[i] = Math.min(1, xtest[i] + radius);
                let ftest = this.evaluate(xtest);

                if (ftest < fbest) {
                    fbest = ftest;
                    // Update interpolation set
                    const snew = MathUtils.subtract(xtest, xbase);
                    this.updateLagrangeInterpolationSet(XPT, FVAL, xbase, snew, ftest);
                }

                if (this.evaluations >= this.nTrials) break;

                // Negative direction
                xtest = xbest.slice();
                xtest[i] = Math.max(0, xtest[i] - radius);
                ftest = this.evaluate(xtest);

                if (ftest < fbest) {
                    fbest = ftest;
                    // Update interpolation set
                    const snew = MathUtils.subtract(xtest, xbase);
                    this.updateLagrangeInterpolationSet(XPT, FVAL, xbase, snew, ftest);
                }
            }

            // If we found a much better point, stop refining
            if (fbest < 1e-8) break; // Relaxed for visualization
        }
    }

    isAlmostCoordinateDirection(diff, coordIndex, tolerance = 0.1) {
        const coordValue = Math.abs(diff[coordIndex]);
        if (coordValue < 1e-8) return false;

        for (let i = 0; i < diff.length; i++) {
            if (i !== coordIndex && Math.abs(diff[i]) > tolerance * coordValue) {
                return false;
            }
        }
        return true;
    }
}

// PROPER PRIMA NEWUOA implementation using Lagrange interpolation
class PRIMA_NEWUOA extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'PRIMA_NEWUOA';
        // NEWUOA uses 2n+1 interpolation points for underdetermined quadratic model
        this.npt = Math.min(2 * nDim + 1, Math.max(nDim + 2, Math.floor(nTrials / 3)));
    }

    optimize() {
        const n = this.nDim;
        const npt = this.npt;

        // Trust region parameters mirroring the Python reference. The
        // old eta1=0.01/eta2=0.25 was very permissive (any step that
        // reduced the predicted-reduction-ratio above 1% counted as
        // success). Python uses a 0.75/0.25/0.1 cascade: ratio<0.1
        // shrinks 0.5×; 0.1–0.25 shrinks 0.8×; 0.25–0.75 keeps;
        // ≥0.75 expands 2× capped at rhobeg.
        const rhobeg = 0.5;
        const rho_end = 1e-8;

        // First-pass seed: deterministic cube centre, matching the
        // Python reference impl. Subsequent restart passes perturb
        // this.bestX so the user's budget actually gets spent on a
        // non-smooth surface where a single TR run terminates early.
        let xseed = Array(n).fill(0.5);

        while (this.evaluations < this.nTrials) {
        // ---------- one trust-region pass ----------
        let xbase = xseed.slice();
        let fbase = this.evaluate(xbase);
        if (this.evaluations >= this.nTrials) break;

        let rho = rhobeg;

        let XPT = Array(npt).fill().map(() => Array(n).fill(0));
        let FVAL = Array(npt).fill(0);
        XPT[0] = Array(n).fill(0);
        FVAL[0] = fbase;
        this.buildNEWUOAInterpolationSet(XPT, FVAL, xbase, rho);

        let kopt = 0;
        for (let k = 1; k < XPT.length; k++) {
            if (FVAL[k] < FVAL[kopt]) kopt = k;
        }

        let xopt = MathUtils.add(xbase, XPT[kopt]);
        let fopt = FVAL[kopt];

        // Reset the min-Frobenius update's prior Hessian for this TR
        // pass (mirrors the Python port — Powell's NEWUOA initialises
        // H_prev = 0 at restart and carries it forward).
        this._Hprev = Linalg.zeros(n, n);

        while (this.evaluations < this.nTrials && rho > rho_end) {
            const model = this.buildNEWUOAQuadraticModel(XPT, FVAL, xopt);
            // Effective gradient at xopt (= xbase + XPT[kopt]). The model
            // m(s) = c + g·s + ½ sᵀ H s is centred at xbase, so the
            // gradient at the trial origin xopt is g + H · XPT[kopt].
            // Using g directly would solve a TR problem geometrically
            // centred at xbase — fine when base-point shifting keeps
            // XPT[kopt] ≈ 0 (Powell's NEWUOA does this every iteration)
            // but otherwise mis-aligned with the TR step we then add to
            // xopt.
            const gEff = MathUtils.add(model.g, Linalg.matvec(model.H, XPT[kopt]));
            const step = solveTrustRegionUnbounded(gEff, model.H, rho, n);

            if (this.evaluations >= this.nTrials) break;

            const xTrial = MathUtils.add(xopt, step);
            const xTrialBounded = xTrial.map(x => Math.min(1, Math.max(0, x)));
            const fTrial = this.evaluate(xTrialBounded);

            // Predicted reduction at xopt: −(gEff · step + ½ stepᵀ H step).
            const HStep = Linalg.matvec(model.H, step);
            let predRed = -MathUtils.dot(gEff, step);
            for (let i = 0; i < n; i++) predRed -= 0.5 * step[i] * HStep[i];
            const actualRed = fopt - fTrial;
            const ratio = predRed > 0 ? actualRed / predRed : -1;

            // Python's trust-region update cascade (verbatim from
            // _update_trust_region_radius in prima_algorithms.py).
            let rhoNew;
            if (ratio >= 0.75) {
                rhoNew = Math.min(rho * 2.0, rhobeg);
            } else if (ratio >= 0.25) {
                rhoNew = rho;
            } else if (ratio >= 0.1) {
                rhoNew = rho * 0.8;
            } else {
                rhoNew = Math.max(rho * 0.5, rho_end);
            }

            // Accept the step whenever it actually reduces the
            // objective. updateNEWUOAInterpolationSet already gates
            // on its own "is this better than the worst point" check
            // so this is sufficient.
            let stepAccepted = false;
            if (fTrial < fopt) {
                xopt = xTrialBounded.slice();
                fopt = fTrial;
                kopt = this.updateNEWUOAInterpolationSet(XPT, FVAL, xopt, fopt, xbase);
                stepAccepted = true;
            }

            rho = rhoNew;

            // Geometry step on REJECTED iterations only — the model
            // is demonstrably wrong (predicted reduction not realised)
            // so a probe to refresh interpolation conditioning is
            // exactly what we want. Was previously gated at 10% chance
            // on EVERY iteration, which both wasted evals on accepted
            // steps and missed the cases where it actually mattered.
            if (!stepAccepted && this.evaluations < this.nTrials - 1) {
                this.improveGeometry(XPT, FVAL, xopt, rho, xbase);
            }
        }
        // ---------- end one trust-region pass ----------

        // Prepare next-pass seed: jitter the global best by ~rhobeg
        // in a uniformly-random direction, clipped into [0, 1]^n. The
        // perturbation magnitude is large enough to escape the basin
        // that just trapped us; if the global best is already at the
        // true minimum, the restart costs the budget of one more pass
        // but doesn't worsen the result.
        if (this.evaluations < this.nTrials) {
            xseed = this.bestX.map(x => {
                const jitter = (Math.random() - 0.5) * 2 * rhobeg;
                return Math.min(1, Math.max(0, x + jitter));
            });
        }
        }   // end restart loop

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    buildNEWUOAQuadraticModel(XPT, FVAL, _xopt) {
        // Powell's full-Hessian min-Frobenius-norm update (matches the
        // Python port). The previous diagonal-Hessian FD heuristic could
        // not capture Rosenbrock's off-diagonal 100·(−2xy) cross-term;
        // this can. Carries `this._Hprev` across iterations within a TR
        // pass; the outer optimize() resets it per restart pass.
        //
        // Falls back to a finite-difference gradient with identity
        // Hessian on rank-deficient interpolation sets or non-finite
        // FVAL — same fallback shape the Python port uses.
        const n = this.nDim;
        const Hprev = this._Hprev || Linalg.zeros(n, n);
        try {
            const { c, g, H } = buildMinFrobeniusQuadratic(XPT, FVAL, Hprev, n);
            this._Hprev = H;
            return { c, g, H };
        } catch (e) {
            const g = this._fdGradient(XPT, FVAL, n);
            const H = Linalg.eye(n);
            // Don't update _Hprev — keep the last successful Hessian.
            return { c: 0, g, H };
        }
    }

    _fdGradient(XPT, FVAL, n) {
        // Central-difference gradient at the best point of the
        // interpolation set, using whichever ± coordinate-axis points
        // are present. Used only as a last-resort fallback when the
        // min-Frobenius solve fails.
        let kopt = 0;
        for (let k = 1; k < FVAL.length; k++) {
            if (FVAL[k] < FVAL[kopt]) kopt = k;
        }
        const g = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            let posV = FVAL[kopt], negV = FVAL[kopt], step = 0;
            for (let k = 0; k < FVAL.length; k++) {
                if (k === kopt) continue;
                let dominant = true, absI = Math.abs(XPT[k][i] - XPT[kopt][i]);
                if (absI < 1e-8) continue;
                let other = 0;
                for (let j = 0; j < n; j++) if (j !== i) other += Math.abs(XPT[k][j] - XPT[kopt][j]);
                if (other > absI) dominant = false;
                if (!dominant) continue;
                if (XPT[k][i] > XPT[kopt][i]) posV = FVAL[k]; else negV = FVAL[k];
                if (absI > step) step = absI;
            }
            if (posV !== FVAL[kopt] && negV !== FVAL[kopt] && step > 0) {
                g[i] = (posV - negV) / (2 * step);
            }
        }
        return g;
    }

    updateNEWUOAInterpolationSet(XPT, FVAL, xnew, fnew, xbase) {
        // Replace the point furthest from `xnew` (excluding the
        // current best) with `(xnew - xbase, fnew)`. This is closer to
        // Python's "furthest from new position" rule than the previous
        // "replace the worst f-value" rule — the latter could degrade
        // the geometry of the interpolation set even when accepting an
        // improving step.
        const npt = FVAL.length;
        let koptIdx = 0;
        for (let k = 1; k < npt; k++) {
            if (FVAL[k] < FVAL[koptIdx]) koptIdx = k;
        }
        const xnewOffset = MathUtils.subtract(xnew, xbase);
        let furthestK = -1, furthestDist = -1;
        for (let k = 0; k < npt; k++) {
            if (k === koptIdx) continue;
            const d = MathUtils.norm(MathUtils.subtract(XPT[k], xnewOffset));
            if (d > furthestDist) { furthestDist = d; furthestK = k; }
        }
        if (furthestK < 0) return koptIdx;
        XPT[furthestK] = xnewOffset;
        FVAL[furthestK] = fnew;
        // Recompute kopt after the swap (the new point may itself
        // become best, since the caller just accepted it as fopt).
        let newKopt = 0;
        for (let k = 1; k < npt; k++) {
            if (FVAL[k] < FVAL[newKopt]) newKopt = k;
        }
        return newKopt;
    }

    improveGeometry(XPT, FVAL, xopt, rho, xbase) {
        // Geometry-improvement step: probe a direction that least
        // resembles any existing interpolation offset, at distance rho.
        // Replaces a random direction (was Math.random() - 0.5) with a
        // direction chosen to *maximise the minimum cosine distance* to
        // existing XPT offsets — which is a cheap proxy for improving
        // interpolation matrix conditioning.
        const n = this.nDim;
        if (this.evaluations >= this.nTrials) return;

        // Try each unit coordinate direction (and its negative) and
        // pick the one most orthogonal to the current interpolation
        // offsets. Cheap and deterministic. For n=2 this is 4 probes.
        let bestDir = null;
        let bestMaxOverlap = Infinity;
        for (let i = 0; i < n; i++) {
            for (const sgn of [1, -1]) {
                const dir = Array(n).fill(0);
                dir[i] = sgn;
                let maxOverlap = 0;
                for (let k = 0; k < XPT.length; k++) {
                    const s = XPT[k];
                    const sNorm = MathUtils.norm(s);
                    if (sNorm < 1e-12) continue;
                    const cos = Math.abs(MathUtils.dot(s, dir) / sNorm);
                    if (cos > maxOverlap) maxOverlap = cos;
                }
                if (maxOverlap < bestMaxOverlap) {
                    bestMaxOverlap = maxOverlap;
                    bestDir = dir;
                }
            }
        }
        if (!bestDir) bestDir = Array(n).fill(0).map(() => 0.5);

        const stepVec = MathUtils.scale(bestDir, rho);
        const xTest = MathUtils.add(xopt, stepVec);
        const xBounded = xTest.map(x => Math.min(1, Math.max(0, x)));
        const fTest = this.evaluate(xBounded);

        // Drop into the worst point's slot if the new point is at
        // least competitive. Use the REAL xbase passed in so the
        // shifted-coord offset is correct (previous version used
        // zeros(n), which broke alignment).
        let worstK = 0;
        for (let k = 1; k < FVAL.length; k++) {
            if (FVAL[k] > FVAL[worstK]) worstK = k;
        }
        if (fTest < FVAL[worstK]) {
            XPT[worstK] = MathUtils.subtract(xBounded, xbase);
            FVAL[worstK] = fTest;
        }
    }

    buildNEWUOAInterpolationSet(XPT, FVAL, xbase, rho) {
        // Build NEWUOA interpolation set: 2n+1 points for underdetermined quadratic model
        const n = this.nDim;
        const npt = this.npt;

        let kptNum = 1; // First point already set to origin

        // Add coordinate directions (like UOBYQA but only 2n points, not 2n+1)
        for (let j = 0; j < n && kptNum < npt; j++) {
            // Positive direction
            let stepa = Math.min(1.0, 1.0 - xbase[j]);
            XPT[kptNum] = Array(n).fill(0);
            XPT[kptNum][j] = stepa;

            let xnew = MathUtils.add(xbase, XPT[kptNum]);
            FVAL[kptNum] = this.evaluate(xnew);
            kptNum++;

            if (kptNum >= npt) break;

            // Negative direction (if we have room)
            if (kptNum < npt) {
                let stepb = -Math.min(xbase[j], 1.0);
                XPT[kptNum] = Array(n).fill(0);
                XPT[kptNum][j] = stepb;

                xnew = MathUtils.add(xbase, XPT[kptNum]);
                FVAL[kptNum] = this.evaluate(xnew);
                kptNum++;
            }
        }

        // Fill remaining points if needed
        while (kptNum < npt) {
            XPT[kptNum] = Array(n).fill(0);
            FVAL[kptNum] = FVAL[0];
            kptNum++;
        }
    }

}

// PRIMA BOBYQA - Bound-constrained Optimization BY Quadratic Approximation.
// Pure-JS port mirroring humpday/optimizers/prima_algorithms.py::PRIMA_BOBYQA
// step-for-step, so the two ports give the same answer up to RNG drift.
// All values come from this.evaluate(x) — function values only,
// no analytical gradients (BOBYQA fits a quadratic *surrogate*
// m(s) = c + g.s + 0.5 * sum(diag_i * s_i^2) to recorded FVAL values
// and trust-region-minimises that, never the user's f).
class PRIMA_BOBYQA extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'PRIMA_BOBYQA';
    }

    optimize() {
        const n = this.nDim;
        const npt = 2 * n + 1;
        const xl = new Array(n).fill(0);
        const xu = new Array(n).fill(1);

        const rhobeg = 0.5;
        const rhoend = 1e-8;

        // First-pass seed: cube centre, clipped a hair away from the
        // bounds so the coordinate-axis init points have headroom.
        // Subsequent restart passes perturb this.bestX, so the user's
        // budget actually gets spent on non-smooth surfaces where a
        // single TR pass converges in ~50 evals.
        let xseed = new Array(n).fill(0.5).map(v => Math.min(0.9, Math.max(0.1, v)));

        while (this.evaluations < this.nTrials) {
        // ---------- one trust-region pass ----------
        let rho = rhobeg;
        let xbase = xseed.slice();
        this.evaluate(xbase);
        if (this.evaluations >= this.nTrials) break;

        let { XPT, FVAL } = this._initBOBYQAPoints(xbase, rho, npt, n, xl, xu);
        let kopt = this._argmin(FVAL);

        // Reset min-Frobenius prior Hessian for this TR pass.
        this._Hprev = Linalg.zeros(n, n);

        let iteration = 0;
        // Cap iteration count by budget directly rather than by
        // floor(budget / npt) — the previous formula gave only
        // floor(80/7) = 11 iterations on a 3-D problem with budget 80,
        // so the trust-region loop terminated long before it should have.
        // Each iteration uses ~1 evaluation, so the while-loop's own
        // `this.evaluations < this.nTrials` check is sufficient; this
        // cap just guards against pathological infinite loops.
        const maxIter = this.nTrials;

        while (this.evaluations < this.nTrials && rho > rhoend && iteration < maxIter) {
            iteration += 1;

            let g, H;
            try {
                ({ g, H } = this._buildBOBYQAModel(XPT, FVAL, kopt, n));
            } catch (e) {
                rho *= 0.5;
                continue;
            }

            const xCurr = this._addVec(xbase, XPT[kopt]);
            // Effective gradient at xCurr: g + H · XPT[kopt] (the model
            // is centred at xbase; the TR subproblem is solved from
            // xCurr = xbase + XPT[kopt]).
            const gEff = MathUtils.add(g, Linalg.matvec(H, XPT[kopt]));
            let d;
            try {
                d = solveTrsbox(gEff, H, rho, xCurr, xl, xu, n);
            } catch (e) {
                d = this._fallbackBoundStep(gEff, rho, n, xCurr, xl, xu);
            }

            const xnew = this._clip01(this._addVec(xCurr, d));

            if (this.evaluations >= this.nTrials) break;
            const fnew = this.evaluate(xnew);

            // Predicted reduction at xCurr: −(gEff · d + ½ dᵀ H d).
            const Hd = Linalg.matvec(H, d);
            let predRed = -MathUtils.dot(gEff, d);
            for (let i = 0; i < n; i++) predRed -= 0.5 * d[i] * Hd[i];
            const actualRed = FVAL[kopt] - fnew;

            const updated = this._updateBOBYQAInterpolation(
                XPT, FVAL, this._subVec(xnew, xbase), fnew, kopt, npt
            );
            if (updated) {
                const koptNew = this._argmin(FVAL);
                if (FVAL[koptNew] < FVAL[kopt]) kopt = koptNew;
            }

            rho = this._updateTrustRegionRadius(predRed, actualRed, rho, rhobeg, rhoend);

            if (this._norm(XPT[kopt]) > 0.5 * rho) {
                xbase = this._shiftBasePointBounded(XPT, xbase, kopt, npt, xl, xu);
            }
        }
        // ---------- end one trust-region pass ----------

        // Restart seed: jitter the global best by ~rhobeg in a random
        // direction (clipped to [0.1, 0.9] for the same headroom reason
        // as the first-pass init). One TR pass on a non-smooth surface
        // costs ~50 evals; without this, the user's 5000-budget request
        // returns in 50 evals stuck at the first local minimum.
        if (this.evaluations < this.nTrials) {
            xseed = this.bestX.map(x => {
                const jitter = (Math.random() - 0.5) * 2 * rhobeg;
                return Math.min(0.9, Math.max(0.1, x + jitter));
            });
        }
        }   // end restart loop

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null,
        };
    }

    // ----- helpers (named to mirror the Python implementation) -----

    _argmin(arr) {
        let k = 0;
        for (let i = 1; i < arr.length; i++) if (arr[i] < arr[k]) k = i;
        return k;
    }

    _addVec(a, b) { return a.map((v, i) => v + b[i]); }
    _subVec(a, b) { return a.map((v, i) => v - b[i]); }
    _clip01(a)    { return a.map(v => Math.max(0, Math.min(1, v))); }
    _norm(a)      { return Math.sqrt(a.reduce((s, v) => s + v * v, 0)); }

    _initBOBYQAPoints(xbase, rho, npt, n, xl, xu) {
        // Initial interpolation set, 2n+1 coordinate-axis points respecting bounds.
        // Each evaluate() is budget-guarded so a restart triggered close to
        // nTrials can't overshoot via the init-set (mirrors Python fix #141).
        const XPT = [];
        const FVAL = [];

        XPT.push(new Array(n).fill(0));
        if (this.evaluations >= this.nTrials) return { XPT, FVAL };
        FVAL.push(this.evaluate(xbase));

        for (let i = 0; i < n; i++) {
            if (FVAL.length >= npt || this.evaluations >= this.nTrials) return { XPT, FVAL };
            const step_pos = Math.min(rho, xu[i] - xbase[i]);
            if (step_pos > 1e-10) {
                const offset = new Array(n).fill(0);
                offset[i] = step_pos;
                XPT.push(offset);
                FVAL.push(this.evaluate(this._clip01(this._addVec(xbase, offset))));
            }

            if (FVAL.length >= npt || this.evaluations >= this.nTrials) return { XPT, FVAL };
            const step_neg = Math.max(-rho, xl[i] - xbase[i]);
            if (step_neg < -1e-10) {
                const offset = new Array(n).fill(0);
                offset[i] = step_neg;
                XPT.push(offset);
                FVAL.push(this.evaluate(this._clip01(this._addVec(xbase, offset))));
            }
        }

        // Optional diagonal-direction point.
        if (FVAL.length < npt && this.evaluations < this.nTrials) {
            const diag = new Array(n).fill(rho / Math.sqrt(n));
            for (let i = 0; i < n; i++) {
                const xi = xbase[i] + diag[i];
                if (xi > xu[i]) diag[i] = xu[i] - xbase[i];
                else if (xi < xl[i]) diag[i] = xl[i] - xbase[i];
            }
            XPT.push(diag);
            FVAL.push(this.evaluate(this._clip01(this._addVec(xbase, diag))));
        }

        return { XPT, FVAL };
    }

    _buildBOBYQAModel(XPT, FVAL, kopt, n) {
        // Powell's full-Hessian min-Frobenius-norm update (matches the
        // Python port from #167). The previous diagonal-Hessian
        // least-squares fit could not capture cross-term curvature
        // like Rosenbrock's 100·(−2xy); this can. Carries
        // `this._Hprev` across iterations within a TR pass; the outer
        // optimize() resets it per restart pass.
        //
        // Falls back to a finite-difference gradient + identity Hessian
        // on rank-deficient interpolation sets or non-finite FVAL.
        const Hprev = this._Hprev || Linalg.zeros(n, n);
        try {
            const { g, H } = buildMinFrobeniusQuadratic(XPT, FVAL, Hprev, n);
            this._Hprev = H;
            return { g, H };
        } catch (e) {
            const g = this._finiteDifferenceGradientBounded(XPT, FVAL, kopt, n);
            const H = Linalg.eye(n);
            return { g, H };
        }
    }

    _finiteDifferenceGradientBounded(XPT, FVAL, kopt, n) {
        // Coordinate-wise FD using whichever positive/negative axis
        // points are present. Mirrors Python's
        // `_finite_difference_gradient_bounded` line-for-line.
        const g = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            let posVal = FVAL[kopt];
            let negVal = FVAL[kopt];
            let posStep = 0;
            let negStep = 0;
            for (let k = 0; k < FVAL.length; k++) {
                if (k === kopt) continue;
                const diff = this._subVec(XPT[k], XPT[kopt]);
                const absI = Math.abs(diff[i]);
                if (absI > 1e-6) {
                    let sumAbs = 0;
                    for (let j = 0; j < n; j++) sumAbs += Math.abs(diff[j]);
                    if (sumAbs < 2 * absI) {
                        if (diff[i] > 0) { posVal = FVAL[k]; posStep = diff[i]; }
                        else              { negVal = FVAL[k]; negStep = -diff[i]; }
                    }
                }
            }
            if (posStep > 0 && negStep > 0) g[i] = (posVal - negVal) / (posStep + negStep);
            else if (posStep > 0)            g[i] = (posVal - FVAL[kopt]) / posStep;
            else if (negStep > 0)            g[i] = (FVAL[kopt] - negVal) / negStep;
        }
        return g;
    }

    _updateBOBYQAInterpolation(XPT, FVAL, dFromBase, fnew, kopt, npt) {
        // Mirror Python's `new_pos = XPT[kopt] + d` where the caller passes
        // d = xnew - xbase. The result is XPT[kopt] + (xnew - xbase), which
        // is what gets stored. (Whether this is a Python bug or a feature
        // we don't know — Powell's reference would store xnew - xbase
        // directly — but matching it keeps the ports equivalent.)
        const candidates = [];
        for (let i = 0; i < npt; i++) if (i !== kopt) candidates.push(i);
        if (!candidates.length) return false;

        const newPos = XPT[kopt].map((v, i) => v + dFromBase[i]);

        let furthest = candidates[0];
        let furthestDist = this._norm(this._subVec(XPT[furthest], newPos));
        for (const i of candidates) {
            const dist = this._norm(this._subVec(XPT[i], newPos));
            if (dist > furthestDist) { furthest = i; furthestDist = dist; }
        }
        XPT[furthest] = newPos;
        FVAL[furthest] = fnew;
        return true;
    }

    _shiftBasePointBounded(XPT, xbase, kopt, npt, xl, xu) {
        // Recentre XPT so the best point sits at the origin (Python verbatim).
        const shift = XPT[kopt].slice();
        const newBase = this._clip01(this._addVec(xbase, shift));
        const actualShift = this._subVec(newBase, xbase);
        for (let i = 0; i < npt; i++) {
            XPT[i] = this._subVec(XPT[i], actualShift);
        }
        return newBase;
    }

    _fallbackBoundStep(g, rho, n, xCurrent, xl, xu) {
        let d;
        const gn = this._norm(g);
        if (gn > 1e-12) {
            d = g.map(v => -rho * v / gn);
        } else {
            // No model gradient — small random kick. Note this is the
            // *algorithm's* tie-breaker, not a derivative of the user's f.
            d = new Array(n).fill(0).map(() => (rho / 3.0) * (2 * Math.random() - 1));
        }
        for (let i = 0; i < n; i++) {
            const xi = xCurrent[i] + d[i];
            if (xi < xl[i]) d[i] = xl[i] - xCurrent[i];
            else if (xi > xu[i]) d[i] = xu[i] - xCurrent[i];
        }
        return d;
    }

    _updateTrustRegionRadius(predRed, actualRed, rho, rhobeg, rhoend) {
        let ratio;
        if (Math.abs(predRed) < 1e-12) {
            ratio = actualRed <= 0 ? 0 : 10;
        } else {
            ratio = actualRed / predRed;
        }
        if (ratio >= 0.75) return Math.min(rho * 2.0, rhobeg);
        if (ratio >= 0.25) return rho;
        if (ratio >= 0.1)  return rho * 0.8;
        return Math.max(rho * 0.5, rhoend);
    }
}

// Export PRIMA algorithms
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = { PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA };
} else {
    // Browser environment
    window.PRIMA_UOBYQA = PRIMA_UOBYQA;
    window.PRIMA_NEWUOA = PRIMA_NEWUOA;
    window.PRIMA_BOBYQA = PRIMA_BOBYQA;
}
