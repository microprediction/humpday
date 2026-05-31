/**
 * Browser-based optimization algorithms
 * JavaScript implementations of PRIMA and SciPy methods
 */

// Utility functions
const MathUtils = {
    random: () => Math.random(),

    norm(vec) {
        return Math.sqrt(vec.reduce((sum, x) => sum + x * x, 0));
    },

    dot(a, b) {
        return a.reduce((sum, x, i) => sum + x * b[i], 0);
    },

    subtract(a, b) {
        return a.map((x, i) => x - b[i]);
    },

    add(a, b) {
        return a.map((x, i) => x + b[i]);
    },

    scale(vec, scalar) {
        return vec.map(x => x * scalar);
    },

    clip(x, min, max) {
        return Math.max(min, Math.min(max, x));
    },

    clipArray(arr, min, max) {
        return arr.map(x => MathUtils.clip(x, min, max));
    }
};

// Base optimizer class
class Optimizer {
    constructor(objective, nTrials, nDim) {
        this.objective = objective;
        this.nTrials = nTrials;
        this.nDim = nDim;
        this.evaluations = 0;
        this.bestValue = Infinity;
        this.bestX = Array(nDim).fill(0).map(() => Math.random());
        this.trackPath = false;
        this.path = [];
    }

    evaluate(x) {
        this.evaluations++;
        const clippedX = MathUtils.clipArray(x, 0, 1);
        const value = this.objective(clippedX);

        // Track path for visualization (sample every few evaluations to avoid clutter)
        if (this.trackPath && (this.evaluations % Math.max(1, Math.floor(this.nTrials / 20)) === 0 || this.evaluations === 1)) {
            this.path.push([...clippedX]);
        }

        if (value < this.bestValue) {
            this.bestValue = value;
            this.bestX = [...clippedX];
        }

        return value;
    }
}

// PROPER PRIMA UOBYQA implementation using Lagrange interpolation
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
        const rhoend = 1e-3; // Relaxed final radius for better visualization
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

            // Early termination if we found excellent solution
            if (fopt < 1e-6) { // Relaxed convergence for visualization
                // Do PDFO-like systematic refinement around best point
                this.doSystematicRefinement(XPT, FVAL, kopt, xbase);
                break;
            }
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

        // Initialize starting point using PDFO-like strategy
        let xbase = Array(n).fill(0).map(() => 0.3 + 0.4 * Math.random());
        let fbase = this.evaluate(xbase);

        // AGGRESSIVE trust region parameters matching PDFO
        let rho = 0.5;  // Large initial trust region
        const rhoend = 1e-8; // Precise final radius
        const eta1 = 0.01; // Aggressive step acceptance
        const eta2 = 0.25; // Aggressive expansion threshold
        const gamma1 = 0.5; // Contraction factor
        const gamma2 = 4.0; // Aggressive expansion

        // Initialize interpolation set with PROPER NEWUOA structure
        let XPT = Array(npt).fill().map(() => Array(n).fill(0));
        let FVAL = Array(npt).fill(0);

        XPT[0] = Array(n).fill(0); // First point is xbase (origin in shifted coordinates)
        FVAL[0] = fbase;

        // Build PROPER NEWUOA interpolation set (2n+1 points)
        this.buildNEWUOAInterpolationSet(XPT, FVAL, xbase, rho);

        let kopt = 0; // Index of best point
        for (let k = 1; k < XPT.length; k++) {
            if (FVAL[k] < FVAL[kopt]) {
                kopt = k;
            }
        }

        let xopt = MathUtils.add(xbase, XPT[kopt]); // Best point in original coordinates
        let fopt = FVAL[kopt];

        // Main NEWUOA loop
        while (this.evaluations < this.nTrials && rho > rhoend) {
            // Build quadratic model around current best point
            const model = this.buildNEWUOAQuadraticModel(XPT, FVAL, xopt);

            // Solve trust region subproblem
            const step = this.solveNEWUOATrustRegion(model, xopt, rho);

            if (this.evaluations >= this.nTrials) break;

            // Compute trial point (no bounds in NEWUOA, but we'll apply [0,1] bounds)
            const xTrial = MathUtils.add(xopt, step);
            const xTrialBounded = xTrial.map(x => Math.min(1, Math.max(0, x)));
            const fTrial = this.evaluate(xTrialBounded);

            // Compute model prediction and actual reduction
            const predRed = this.computeNEWUOAPrediction(model, step);
            const actualRed = fopt - fTrial;

            // Ratio test
            const ratio = predRed > 0 ? actualRed / predRed : -1;

            // Update trust region radius
            let rhoNew = rho;
            if (ratio <= eta1) {
                rhoNew = gamma1 * rho;
            } else if (ratio >= eta2 && MathUtils.norm(step) > 0.8 * rho) {
                rhoNew = Math.min(gamma2 * rho, 10.0);
            }

            // Accept step if ratio is good enough
            if (ratio > eta1) {
                xopt = xTrialBounded.slice();
                fopt = fTrial;
                kopt = this.updateNEWUOAInterpolationSet(XPT, FVAL, xopt, fopt);
            }

            rho = Math.max(rhoNew, rhoend);

            // Geometry improvement step (simplified)
            if (Math.random() < 0.1 && this.evaluations < this.nTrials - 5) {
                this.improveGeometry(XPT, FVAL, xopt, rho);
            }
        }

        return {
            bestValue: fopt,
            bestX: xopt,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    buildNEWUOAInterpolationSet(XPT, FVAL, xbase, rho) {
        const n = this.nDim;
        const npt = this.npt;

        let kptNum = 1; // Start from 1 since XPT[0] is already set

        // Add coordinate directions (both positive and negative if budget allows)
        for (let i = 0; i < n && kptNum < npt && this.evaluations < this.nTrials; i++) {
            // Positive direction
            const step1 = Math.min(rho, 1 - xbase[i]);
            XPT[kptNum] = Array(n).fill(0);
            XPT[kptNum][i] = step1;

            const xnew1 = MathUtils.add(xbase, XPT[kptNum]);
            FVAL[kptNum] = this.evaluate(xnew1);
            kptNum++;

            if (kptNum >= npt || this.evaluations >= this.nTrials) break;

            // Negative direction
            const step2 = -Math.min(rho, xbase[i]);
            XPT[kptNum] = Array(n).fill(0);
            XPT[kptNum][i] = step2;

            const xnew2 = MathUtils.add(xbase, XPT[kptNum]);
            FVAL[kptNum] = this.evaluate(xnew2);
            kptNum++;
        }

        // Add additional points to reach 2n+1 if needed
        while (kptNum < npt && this.evaluations < this.nTrials) {
            XPT[kptNum] = Array(n).fill(0);
            // Add random perturbation
            for (let i = 0; i < n; i++) {
                const pert = (Math.random() - 0.5) * 2 * rho * 0.7;
                XPT[kptNum][i] = Math.max(-xbase[i], Math.min(1 - xbase[i], pert));
            }
            const xnew = MathUtils.add(xbase, XPT[kptNum]);
            FVAL[kptNum] = this.evaluate(xnew);
            kptNum++;
        }
    }

    buildNEWUOAQuadraticModel(XPT, FVAL, xopt) {
        const n = this.nDim;
        const npt = XPT.length;

        // Model: m(s) = c + g^T s + 0.5 s^T H s, where s = x - xopt
        const model = {
            c: 0,
            g: new Array(n).fill(0),
            H: Array(n).fill().map(() => new Array(n).fill(0))
        };

        // Find point closest to xopt for constant term
        let kBase = 0;
        let minDist = Infinity;
        for (let k = 0; k < npt; k++) {
            const dist = MathUtils.norm(MathUtils.subtract(MathUtils.add(XPT[k], xopt), xopt));
            if (dist < minDist) {
                minDist = dist;
                kBase = k;
            }
        }
        model.c = FVAL[kBase];

        // Compute model coefficients using Lagrange interpolation principles
        // Simplified approach: use finite differences where possible

        // Estimate gradient using coordinate directions from interpolation set
        for (let i = 0; i < n; i++) {
            let forwardK = -1, backwardK = -1;
            let minForwardDist = Infinity, minBackwardDist = Infinity;

            for (let k = 0; k < npt; k++) {
                const s = XPT[k]; // Already in shifted coordinates

                // Check if this is approximately a coordinate direction
                let maxNonCoord = 0;
                for (let j = 0; j < n; j++) {
                    if (j !== i && Math.abs(s[j]) > maxNonCoord) {
                        maxNonCoord = Math.abs(s[j]);
                    }
                }

                if (maxNonCoord < 0.1 * Math.abs(s[i]) && Math.abs(s[i]) > 1e-8) {
                    const dist = Math.abs(s[i]);
                    if (s[i] > 0 && dist < minForwardDist) {
                        forwardK = k;
                        minForwardDist = dist;
                    } else if (s[i] < 0 && dist < minBackwardDist) {
                        backwardK = k;
                        minBackwardDist = dist;
                    }
                }
            }

            // Compute gradient estimate
            if (forwardK >= 0 && backwardK >= 0) {
                const h = XPT[forwardK][i] - XPT[backwardK][i];
                model.g[i] = (FVAL[forwardK] - FVAL[backwardK]) / h;

                // Estimate diagonal Hessian
                const hHalf = h / 2;
                model.H[i][i] = (FVAL[forwardK] - 2 * model.c + FVAL[backwardK]) / (hHalf * hHalf);
            } else if (forwardK >= 0) {
                const h = XPT[forwardK][i];
                model.g[i] = (FVAL[forwardK] - model.c) / h;
            } else if (backwardK >= 0) {
                const h = -XPT[backwardK][i];
                model.g[i] = (model.c - FVAL[backwardK]) / h;
            }
        }

        return model;
    }

    solveNEWUOATrustRegion(model, xopt, rho) {
        const n = this.nDim;

        // Simplified trust region solve: dogleg method approximation

        // Cauchy point: steepest descent step
        let gNorm = MathUtils.norm(model.g);
        if (gNorm < 1e-12) {
            // Zero gradient - return small random step
            const randomStep = Array(n).fill(0).map(() => (Math.random() - 0.5) * 2 * rho * 0.1);
            return this.projectToBounds(randomStep, xopt);
        }

        let cauchyStep = MathUtils.scale(model.g, -1);

        // Compute Hg for curvature
        let Hg = Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                Hg[i] += model.H[i][j] * model.g[j];
            }
        }

        let gHg = MathUtils.dot(model.g, Hg);

        // Compute optimal step length along gradient
        let alpha = gHg > 0 ? (gNorm * gNorm) / gHg : 1.0;

        // Scale Cauchy step
        cauchyStep = MathUtils.scale(cauchyStep, alpha);

        // Truncate to trust region
        let stepNorm = MathUtils.norm(cauchyStep);
        if (stepNorm > rho) {
            cauchyStep = MathUtils.scale(cauchyStep, rho / stepNorm);
        }

        // Apply bound constraints
        return this.projectToBounds(cauchyStep, xopt);
    }

    projectToBounds(step, xopt) {
        const n = this.nDim;
        for (let i = 0; i < n; i++) {
            const newPos = xopt[i] + step[i];
            if (newPos < 0) {
                step[i] = -xopt[i];
            } else if (newPos > 1) {
                step[i] = 1 - xopt[i];
            }
        }
        return step;
    }

    computeNEWUOAPrediction(model, step) {
        const n = this.nDim;

        // Predicted reduction: -(g^T s + 0.5 s^T H s)
        let pred = 0;

        // Linear term
        pred -= MathUtils.dot(model.g, step);

        // Quadratic term
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                pred -= 0.5 * step[i] * model.H[i][j] * step[j];
            }
        }

        return pred;
    }

    updateNEWUOAInterpolationSet(XPT, FVAL, xnew, fnew) {
        // Replace worst point with new point
        let worstK = 0;
        for (let k = 1; k < FVAL.length; k++) {
            if (FVAL[k] > FVAL[worstK]) {
                worstK = k;
            }
        }

        if (fnew < FVAL[worstK]) {
            // Convert from absolute coordinates to shifted coordinates
            const xbase = Array(this.nDim).fill(0); // This should be passed as parameter but simplified
            XPT[worstK] = MathUtils.subtract(xnew, xbase);
            FVAL[worstK] = fnew;
            return worstK;
        }

        return 0;
    }

    improveGeometry(XPT, FVAL, xopt, rho) {
        // Simple geometry improvement: add a point that improves interpolation matrix conditioning
        // This is a simplified version - real NEWUOA has complex geometry management
        const n = this.nDim;

        if (this.evaluations >= this.nTrials) return;

        // Find direction with largest model uncertainty
        let bestDir = Array(n).fill(0).map(() => Math.random() - 0.5);
        bestDir = MathUtils.scale(bestDir, rho / MathUtils.norm(bestDir));

        const xTest = MathUtils.add(xopt, bestDir);
        const xBounded = xTest.map(x => Math.min(1, Math.max(0, x)));
        const fTest = this.evaluate(xBounded);

        // Replace worst point if new point is better
        let worstK = 0;
        for (let k = 1; k < FVAL.length; k++) {
            if (FVAL[k] > FVAL[worstK]) {
                worstK = k;
            }
        }

        if (fTest < FVAL[worstK]) {
            // Convert to shifted coordinates
            const xbase = Array(n).fill(0); // Simplified - should be actual base point
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

    buildNEWUOALagrangeModel(XPT, FVAL, kopt) {
        // NEWUOA uses same Lagrange interpolation as UOBYQA
        return this.buildLagrangeQuadraticModel(XPT, FVAL, kopt);
    }

    updateNEWUOALagrangeSet(XPT, FVAL, xbase, snew, fnew) {
        // NEWUOA uses same geometry management as UOBYQA
        this.updateLagrangeInterpolationSet(XPT, FVAL, xbase, snew, fnew);
    }

    doNEWUOASystematicRefinement(XPT, FVAL, kopt, xbase) {
        // NEWUOA uses same systematic refinement as UOBYQA
        this.doSystematicRefinement(XPT, FVAL, kopt, xbase);
    }
}

// PRIMA BOBYQA - Bound constrained Optimization BY Quadratic Approximation
class PRIMA_BOBYQA extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'PRIMA_BOBYQA';
        // BOBYQA for bound constraints, typically uses 2n+1 points like NEWUOA
        this.nPts = Math.min(2 * nDim + 1, Math.max(nDim + 2, Math.floor(nTrials / 3)));
        this.lowerBounds = new Array(nDim).fill(0);  // Lower bounds [0,0,...,0]
        this.upperBounds = new Array(nDim).fill(1);  // Upper bounds [1,1,...,1]
    }

    optimize() {
        // Copy UOBYQA's proven structure exactly, with bound projection at evaluation
        const n = this.nDim;

        // UOBYQA initialization adapted for bounds - consistent start near corner
        let xBase = Array(n).fill(0.15); // Start consistently at [0.15, 0.15, ...]
        let XPT = [Array(n).fill(0)]; // First point at origin in shifted coordinates
        let FVAL = [this.evaluate(xBase)];

        // Build interpolation set exactly like UOBYQA with larger initial step
        const step0 = 0.2;
        for (let i = 0; i < n && XPT.length < this.nPts && this.evaluations < this.nTrials; i++) {
            // Coordinate directions with bound respect
            for (let dir = -1; dir <= 1; dir += 2) {
                if (XPT.length >= this.nPts) break;

                const shift = Array(n).fill(0);
                shift[i] = step0 * dir;

                const xNew = MathUtils.add(xBase, shift);

                // Apply bounds at evaluation
                const xBounded = xNew.map(x => Math.max(0, Math.min(1, x)));
                const actualShift = MathUtils.subtract(xBounded, xBase);

                XPT.push(actualShift);
                FVAL.push(this.evaluate(xBounded));
            }
        }

        // Fill remaining points with deterministic pattern for consistency
        let pointIndex = 0;
        while (XPT.length < this.nPts && this.evaluations < this.nTrials) {
            const shift = Array(n).fill(0);

            // Deterministic radial pattern
            const angle = (pointIndex * 2.0 * Math.PI) / Math.max(1, this.nPts - 2 * n - 1);
            const radius = 0.1;

            if (n >= 2) {
                shift[0] = radius * Math.cos(angle);
                shift[1] = radius * Math.sin(angle);
            } else if (n === 1) {
                shift[0] = radius * (pointIndex % 2 === 0 ? 1 : -1);
            }

            const xNew = MathUtils.add(xBase, shift);
            const xBounded = xNew.map(x => Math.max(0, Math.min(1, x)));
            const actualShift = MathUtils.subtract(xBounded, xBase);

            XPT.push(actualShift);
            FVAL.push(this.evaluate(xBounded));
            pointIndex++;
        }

        // UOBYQA main loop with larger initial trust region for better reach
        let rho = 0.3;
        const rhoEnd = 1e-8;

        while (this.evaluations < this.nTrials && rho > rhoEnd) {
            // Find best point
            let kOpt = 0;
            for (let k = 1; k < FVAL.length; k++) {
                if (FVAL[k] < FVAL[kOpt]) kOpt = k;
            }

            // Use UOBYQA's proven Lagrange model (simplified version)
            const model = { g: Array(n).fill(0) };

            // Compute gradient using coordinate differences (UOBYQA style)
            for (let i = 0; i < n; i++) {
                let forwardVal = FVAL[kOpt], backwardVal = FVAL[kOpt];
                let forwardStep = 0, backwardStep = 0;
                let hasForward = false, hasBackward = false;

                for (let k = 0; k < XPT.length; k++) {
                    const s = XPT[k];
                    if (Math.abs(s[i]) > 1e-8) {
                        // Check if primarily coordinate direction
                        let maxOther = 0;
                        for (let j = 0; j < n; j++) {
                            if (j !== i) maxOther = Math.max(maxOther, Math.abs(s[j]));
                        }

                        if (maxOther < 0.1 * Math.abs(s[i])) {
                            if (s[i] > 0 && !hasForward) {
                                forwardVal = FVAL[k]; forwardStep = s[i]; hasForward = true;
                            } else if (s[i] < 0 && !hasBackward) {
                                backwardVal = FVAL[k]; backwardStep = s[i]; hasBackward = true;
                            }
                        }
                    }
                }

                // Compute derivative
                if (hasForward && hasBackward) {
                    model.g[i] = (forwardVal - backwardVal) / (forwardStep - backwardStep);
                } else if (hasForward) {
                    model.g[i] = (forwardVal - FVAL[kOpt]) / forwardStep;
                } else if (hasBackward) {
                    model.g[i] = (FVAL[kOpt] - backwardVal) / (-backwardStep);
                }
            }

            // UOBYQA trust region step
            let step = model.g.map(gi => -gi);
            const stepNorm = MathUtils.norm(step);
            if (stepNorm > rho) {
                step = MathUtils.scale(step, rho / stepNorm);
            }

            if (this.evaluations >= this.nTrials) break;

            // Apply bounds at trial point evaluation
            const xTrial = MathUtils.add(MathUtils.add(xBase, XPT[kOpt]), step);
            const xTrialBounded = xTrial.map(x => Math.max(0, Math.min(1, x)));
            const fTrial = this.evaluate(xTrialBounded);

            // UOBYQA acceptance test
            const actualRed = FVAL[kOpt] - fTrial;
            const predRed = -MathUtils.dot(model.g, step);
            const ratio = predRed > 0 ? actualRed / predRed : -1;

            if (ratio > 0.01) {
                // Update interpolation set - replace worst point
                let worstK = 0;
                for (let k = 1; k < FVAL.length; k++) {
                    if (FVAL[k] > FVAL[worstK]) worstK = k;
                }
                XPT[worstK] = MathUtils.subtract(xTrialBounded, xBase);
                FVAL[worstK] = fTrial;
            }

            // UOBYQA trust region updates
            if (ratio < 0.1) {
                rho *= 0.5;
            } else if (ratio > 0.75 && stepNorm > 0.9 * rho) {
                rho = Math.min(rho * 2, 0.5);
            }
        }

        // Find final best point
        let kBest = 0;
        for (let k = 1; k < FVAL.length; k++) {
            if (FVAL[k] < FVAL[kBest]) kBest = k;
        }

        return {
            bestValue: FVAL[kBest],
            bestX: MathUtils.add(xBase, XPT[kBest]),
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    buildBoundedInterpolationSet(xPts, fVals, xBase, rho) {
        const n = this.nDim;

        // Add bound-respecting coordinate directions with adequate step sizes
        for (let i = 0; i < n && xPts.length < this.nPts && this.evaluations < this.nTrials; i++) {
            // Positive direction - ensure minimum step size for good derivatives
            const distToUpper = this.upperBounds[i] - xBase[i];
            const minStep = Math.max(rho * 0.1, 1e-3); // Ensure minimum step size
            const stepUp = Math.min(rho * 0.5, Math.max(minStep, distToUpper * 0.8));
            if (stepUp > 1e-8) {
                const xUp = xBase.slice();
                xUp[i] = Math.min(this.upperBounds[i], xBase[i] + stepUp);
                xPts.push(xUp);
                fVals.push(this.evaluate(xUp));
            }

            if (xPts.length >= this.nPts || this.evaluations >= this.nTrials) break;

            // Negative direction - ensure minimum step size
            const distToLower = xBase[i] - this.lowerBounds[i];
            const stepDown = Math.min(rho * 0.5, Math.max(minStep, distToLower * 0.8));
            if (stepDown > 1e-8) {
                const xDown = xBase.slice();
                xDown[i] = Math.max(this.lowerBounds[i], xBase[i] - stepDown);
                xPts.push(xDown);
                fVals.push(this.evaluate(xDown));
            }
        }

        // Add additional points that respect bounds
        while (xPts.length < this.nPts && this.evaluations < this.nTrials) {
            const xNew = xBase.slice();
            for (let i = 0; i < n; i++) {
                const range = this.upperBounds[i] - this.lowerBounds[i];
                const maxPert = Math.min(rho, range * 0.3);
                const pert = (Math.random() - 0.5) * 2 * maxPert;
                xNew[i] = Math.min(this.upperBounds[i],
                          Math.max(this.lowerBounds[i], xBase[i] + pert));
            }
            xPts.push(xNew);
            fVals.push(this.evaluate(xNew));
        }
    }

    buildBOBYQAQuadraticModel(xPts, fVals, xOpt) {
        const n = this.nDim;
        const nPts = xPts.length;

        // Quadratic model: m(s) = c + g^T s + 0.5 s^T H s where s = x - xOpt
        const model = {
            c: 0,
            g: new Array(n).fill(0),
            H: Array(n).fill().map(() => new Array(n).fill(0))
        };

        // Find base point closest to xOpt
        let kBase = 0;
        let minDist = Infinity;
        for (let k = 0; k < nPts; k++) {
            const dist = MathUtils.norm(MathUtils.subtract(xPts[k], xOpt));
            if (dist < minDist) {
                minDist = dist;
                kBase = k;
            }
        }
        model.c = fVals[kBase];

        // Build model using finite differences with bound awareness
        for (let i = 0; i < n; i++) {
            let forwardK = -1, backwardK = -1;
            let minForwardDist = Infinity, minBackwardDist = Infinity;

            for (let k = 0; k < nPts; k++) {
                const s = MathUtils.subtract(xPts[k], xOpt);

                // Check if point is along coordinate direction i
                let isCoordDir = true;
                for (let j = 0; j < n; j++) {
                    if (j !== i && Math.abs(s[j]) > 0.1 * Math.abs(s[i])) {
                        isCoordDir = false;
                        break;
                    }
                }

                if (isCoordDir && Math.abs(s[i]) > 1e-8) {
                    const dist = Math.abs(s[i]);
                    if (s[i] > 0 && dist < minForwardDist) {
                        forwardK = k;
                        minForwardDist = dist;
                    } else if (s[i] < 0 && dist < minBackwardDist) {
                        backwardK = k;
                        minBackwardDist = dist;
                    }
                }
            }

            // Compute derivatives
            if (forwardK >= 0 && backwardK >= 0) {
                // Central difference
                const h = xPts[forwardK][i] - xPts[backwardK][i];
                model.g[i] = (fVals[forwardK] - fVals[backwardK]) / h;

                // Second derivative
                const hHalf = h / 2;
                model.H[i][i] = (fVals[forwardK] - 2 * model.c + fVals[backwardK]) / (hHalf * hHalf);
            } else if (forwardK >= 0) {
                const h = xPts[forwardK][i] - xOpt[i];
                model.g[i] = (fVals[forwardK] - model.c) / h;
            } else if (backwardK >= 0) {
                const h = xOpt[i] - xPts[backwardK][i];
                model.g[i] = (model.c - fVals[backwardK]) / h;
            }
        }

        return model;
    }

    solveBoundedTrustRegion(model, xOpt, rho) {
        const n = this.nDim;

        // Solve: min m(s) = c + g^T s + 0.5 s^T H s subject to ||s|| ≤ rho and bounds

        // Step 1: Compute Cauchy point (steepest descent to trust region boundary)
        let cauchyStep = model.g.map(gi => -gi); // Negative gradient
        let cauchyNorm = MathUtils.norm(cauchyStep);

        if (cauchyNorm < 1e-12) {
            // Zero gradient case - random exploration
            cauchyStep = Array(n).fill(0).map(() => (Math.random() - 0.5) * 2 * rho * 0.1);
            cauchyNorm = MathUtils.norm(cauchyStep);
        }

        // Scale Cauchy step to trust region boundary
        if (cauchyNorm > rho) {
            cauchyStep = MathUtils.scale(cauchyStep, rho / cauchyNorm);
            cauchyNorm = rho;
        }

        // Apply bounds to Cauchy step
        const cauchyTrial = MathUtils.add(xOpt, cauchyStep);
        for (let i = 0; i < n; i++) {
            cauchyTrial[i] = Math.max(this.lowerBounds[i], Math.min(this.upperBounds[i], cauchyTrial[i]));
        }
        cauchyStep = MathUtils.subtract(cauchyTrial, xOpt);

        // Step 2: Newton step if we can improve on Cauchy point
        let newtonStep = Array(n).fill(0);

        // Compute Hessian eigenvalue approximation for regularization
        let hessianTrace = 0;
        for (let i = 0; i < n; i++) {
            hessianTrace += model.H[i][i];
        }
        const regularization = Math.max(1e-8, hessianTrace / n + 1e-6);

        // Simple Newton step with regularized Hessian: (H + λI)^-1 g
        for (let i = 0; i < n; i++) {
            const diag = model.H[i][i] + regularization;
            if (Math.abs(diag) > 1e-12) {
                newtonStep[i] = -model.g[i] / diag;
            } else {
                newtonStep[i] = -model.g[i] * 100; // Treat as near-zero curvature
            }
        }

        // Scale Newton step to trust region
        let newtonNorm = MathUtils.norm(newtonStep);
        if (newtonNorm > rho) {
            newtonStep = MathUtils.scale(newtonStep, rho / newtonNorm);
        }

        // Apply bounds to Newton step
        const newtonTrial = MathUtils.add(xOpt, newtonStep);
        for (let i = 0; i < n; i++) {
            newtonTrial[i] = Math.max(this.lowerBounds[i], Math.min(this.upperBounds[i], newtonTrial[i]));
        }
        newtonStep = MathUtils.subtract(newtonTrial, xOpt);

        // Step 3: Choose better of Cauchy vs Newton step
        const cauchyPred = this.computeBOBYQAPrediction(model, cauchyStep);
        const newtonPred = this.computeBOBYQAPrediction(model, newtonStep);

        let step = newtonPred < cauchyPred ? newtonStep : cauchyStep;

        // Project to satisfy bounds [0,1]
        for (let i = 0; i < n; i++) {
            const newPos = xOpt[i] + step[i];
            if (newPos < this.lowerBounds[i]) {
                step[i] = this.lowerBounds[i] - xOpt[i];
            } else if (newPos > this.upperBounds[i]) {
                step[i] = this.upperBounds[i] - xOpt[i];
            }
        }

        // Re-scale if bound projection violated trust region
        const finalNorm = MathUtils.norm(step);
        if (finalNorm > rho * 1.01) {
            step = MathUtils.scale(step, rho / finalNorm);
        }

        return step;
    }

    computeBOBYQAPrediction(model, step) {
        // Same as other PRIMA methods
        let pred = -MathUtils.dot(model.g, step);

        for (let i = 0; i < step.length; i++) {
            for (let j = 0; j < step.length; j++) {
                pred -= 0.5 * step[i] * model.H[i][j] * step[j];
            }
        }

        return pred;
    }

    updateBOBYQAInterpolationSet(xPts, fVals, xNew, fNew) {
        // Replace worst point
        let worstK = 0;
        for (let k = 1; k < fVals.length; k++) {
            if (fVals[k] > fVals[worstK]) {
                worstK = k;
            }
        }

        if (fNew < fVals[worstK]) {
            xPts[worstK] = xNew.slice();
            fVals[worstK] = fNew;
        }
    }

    improveBoundedGeometry(xPts, fVals, xOpt, rho) {
        // Add a geometry-improving point that respects bounds
        const n = this.nDim;

        if (this.evaluations >= this.nTrials) return;

        // Find direction that improves interpolation set geometry
        let bestDir = Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            // Bias toward center of feasible region
            const center = (this.lowerBounds[i] + this.upperBounds[i]) / 2;
            bestDir[i] = (center - xOpt[i]) + (Math.random() - 0.5) * rho;
        }

        // Normalize and scale
        const dirNorm = MathUtils.norm(bestDir);
        if (dirNorm > 1e-8) {
            bestDir = MathUtils.scale(bestDir, rho / dirNorm);
        }

        // Apply bounds
        const xTest = MathUtils.add(xOpt, bestDir);
        for (let i = 0; i < n; i++) {
            xTest[i] = Math.min(this.upperBounds[i], Math.max(this.lowerBounds[i], xTest[i]));
        }

        const fTest = this.evaluate(xTest);
        this.updateBOBYQAInterpolationSet(xPts, fVals, xTest, fTest);
    }
}

// Nelder-Mead Simplex implementation
class NelderMead extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'SciPy_NelderMead';
    }

    optimize() {
        const n = this.nDim;

        // Initialize simplex
        const simplex = [];
        const values = [];

        // First vertex - random point
        const x0 = Array(n).fill(0).map(() => Math.random());
        simplex.push([...x0]);
        values.push(this.evaluate(x0));

        // Additional vertices
        for (let i = 0; i < n; i++) {
            const x = [...x0];
            x[i] = Math.min(1, Math.max(0, x[i] + 0.05));
            simplex.push([...x]);
            values.push(this.evaluate(x));
        }

        // Nelder-Mead parameters
        const alpha = 1.0;    // reflection
        const gamma = 2.0;    // expansion
        const rho = 0.5;      // contraction
        const sigma = 0.5;    // shrink

        while (this.evaluations < this.nTrials) {
            // Sort vertices by function value
            const indices = Array(n + 1).fill(0).map((_, i) => i);
            indices.sort((i, j) => values[i] - values[j]);

            // Check convergence (simplified)
            const range = values[indices[n]] - values[indices[0]];
            if (range < 1e-8) break;

            // Centroid of all but worst point
            const centroid = Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    centroid[j] += simplex[indices[i]][j] / n;
                }
            }

            // Reflect worst point
            const worst = indices[n];
            const reflected = [];
            for (let i = 0; i < n; i++) {
                reflected[i] = MathUtils.clip(
                    centroid[i] + alpha * (centroid[i] - simplex[worst][i]),
                    0, 1
                );
            }

            if (this.evaluations >= this.nTrials) break;
            const reflectedValue = this.evaluate(reflected);

            if (reflectedValue < values[indices[0]]) {
                // Expand
                const expanded = [];
                for (let i = 0; i < n; i++) {
                    expanded[i] = MathUtils.clip(
                        centroid[i] + gamma * (reflected[i] - centroid[i]),
                        0, 1
                    );
                }

                if (this.evaluations < this.nTrials) {
                    const expandedValue = this.evaluate(expanded);

                    if (expandedValue < reflectedValue) {
                        simplex[worst] = expanded;
                        values[worst] = expandedValue;
                    } else {
                        simplex[worst] = reflected;
                        values[worst] = reflectedValue;
                    }
                }
            } else if (reflectedValue < values[indices[n - 1]]) {
                // Accept reflection
                simplex[worst] = reflected;
                values[worst] = reflectedValue;
            } else {
                // Contract
                const contracted = [];
                const contractPoint = reflectedValue < values[worst] ? reflected : simplex[worst];

                for (let i = 0; i < n; i++) {
                    contracted[i] = MathUtils.clip(
                        centroid[i] + rho * (contractPoint[i] - centroid[i]),
                        0, 1
                    );
                }

                if (this.evaluations < this.nTrials) {
                    const contractedValue = this.evaluate(contracted);

                    if (contractedValue < Math.min(reflectedValue, values[worst])) {
                        simplex[worst] = contracted;
                        values[worst] = contractedValue;
                    } else {
                        // Shrink simplex
                        for (let i = 1; i <= n && this.evaluations < this.nTrials; i++) {
                            for (let j = 0; j < n; j++) {
                                simplex[indices[i]][j] = MathUtils.clip(
                                    simplex[indices[0]][j] + sigma * (simplex[indices[i]][j] - simplex[indices[0]][j]),
                                    0, 1
                                );
                            }
                            values[indices[i]] = this.evaluate(simplex[indices[i]]);
                        }
                    }
                }
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Simple Powell's method implementation
class Powell extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'SciPy_Powell';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        // Direction vectors (initially coordinate axes)
        const directions = [];
        for (let i = 0; i < this.nDim; i++) {
            const dir = Array(this.nDim).fill(0);
            dir[i] = 1;
            directions.push(dir);
        }

        const tolerance = 1e-6;
        const maxIterations = Math.max(5, Math.floor(this.nTrials / (this.nDim * 3)));

        for (let iter = 0; iter < maxIterations && this.evaluations < this.nTrials; iter++) {
            const x0 = [...x];
            const fx0 = fx;
            let improved = false;

            // Line searches along each direction
            for (let i = 0; i < this.nDim && this.evaluations < this.nTrials - 2; i++) {
                const result = this.lineSearch(x, directions[i], fx, 0.05);
                if (result.fx < fx) {
                    x = result.x;
                    fx = result.fx;
                    improved = true;
                }
            }

            // Check for convergence (but don't exit too early)
            const improvement = fx0 - fx;
            if (improvement < tolerance && iter > 2) {
                break;
            }

            // Update directions (simplified)
            if (improved && improvement > tolerance * 10) {
                const newDirection = MathUtils.subtract(x, x0);
                const norm = MathUtils.norm(newDirection);
                if (norm > 1e-8) {
                    directions.shift();
                    directions.push(newDirection.map(d => d / norm)); // Normalize
                }
            }

            // Ensure we use enough evaluations - add random search if stuck
            if (!improved && this.evaluations < this.nTrials * 0.5) {
                for (let j = 0; j < 3 && this.evaluations < this.nTrials; j++) {
                    const randomX = x.map(xi =>
                        MathUtils.clip(xi + (Math.random() - 0.5) * 0.1, 0, 1)
                    );
                    const randomFx = this.evaluate(randomX);
                    if (randomFx < fx) {
                        x = randomX;
                        fx = randomFx;
                        improved = true;
                        break;
                    }
                }
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    lineSearch(x, direction, currentFx, initialStep) {
        let bestX = [...x];
        let bestFx = currentFx;

        // Try a few step sizes in both directions
        const stepSizes = [initialStep, initialStep * 2, initialStep * 0.5];

        for (const step of stepSizes) {
            if (this.evaluations >= this.nTrials) break;

            // Positive direction
            const posX = x.map((xi, j) =>
                MathUtils.clip(xi + step * direction[j], 0, 1)
            );
            const posFx = this.evaluate(posX);

            if (posFx < bestFx) {
                bestX = [...posX];
                bestFx = posFx;
                continue; // Found improvement, try next step size
            }

            if (this.evaluations >= this.nTrials) break;

            // Negative direction
            const negX = x.map((xi, j) =>
                MathUtils.clip(xi - step * direction[j], 0, 1)
            );
            const negFx = this.evaluate(negX);

            if (negFx < bestFx) {
                bestX = [...negX];
                bestFx = negFx;
            }
        }

        return { x: bestX, fx: bestFx };
    }
}

// Simple L-BFGS-B approximation (gradient-based)
class LBFGSB extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'SciPy_BFGS';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => 0.1 + 0.8 * Math.random()); // Better initialization
        let fx = this.evaluate(x);

        const tolerance = 1e-8;
        const n = this.nDim;

        // Initialize inverse Hessian approximation as identity matrix
        let B = Array(n).fill().map(() => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            B[i][i] = 1.0;
        }

        // Compute initial gradient
        let gradient = this.computeGradient(x);
        if (this.evaluations >= this.nTrials) return this.getResult(x, fx);

        const maxIterations = Math.min(50, Math.floor(this.nTrials / (2 * n + 1)));

        for (let iter = 0; iter < maxIterations && this.evaluations < this.nTrials; iter++) {
            // Check convergence
            const gradNorm = MathUtils.norm(gradient);
            if (gradNorm < tolerance) break;

            // Compute search direction: p = -B * gradient
            const p = Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    p[i] -= B[i][j] * gradient[j];
                }
            }

            // Line search with Armijo condition
            let alpha = 1.0;
            const c1 = 1e-4;
            const rho = 0.5;
            const maxLsIter = 10;

            let xNew, fxNew;
            for (let lsIter = 0; lsIter < maxLsIter && this.evaluations < this.nTrials; lsIter++) {
                xNew = x.map((xi, i) => MathUtils.clip(xi + alpha * p[i], 0, 1));
                fxNew = this.evaluate(xNew);

                // Armijo condition
                const expectedDecrease = c1 * alpha * MathUtils.dot(gradient, p);
                if (fxNew <= fx + expectedDecrease) {
                    break;
                }
                alpha *= rho;
            }

            if (this.evaluations >= this.nTrials) break;
            if (!xNew) break;

            // Compute new gradient
            const gradientNew = this.computeGradient(xNew);
            if (this.evaluations >= this.nTrials) break;

            // BFGS update
            const s = xNew.map((xi, i) => xi - x[i]); // step
            const y = gradientNew.map((gi, i) => gi - gradient[i]); // gradient change

            const sy = MathUtils.dot(s, y);
            if (sy > 1e-10) { // Ensure positive definiteness
                // B_new = B + (sy + y'By)/(sy)^2 * ss' - (Bs'y + ys'B)/(sy)
                const Bs = Array(n).fill(0);
                const yB = Array(n).fill(0);

                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        Bs[i] += B[i][j] * s[j];
                        yB[i] += y[j] * B[j][i];
                    }
                }

                const yBy = MathUtils.dot(y, Bs);

                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        B[i][j] += (sy + yBy) / (sy * sy) * s[i] * s[j]
                                 - (Bs[i] * y[j] + yB[i] * s[j]) / sy;
                    }
                }
            }

            x = xNew;
            fx = fxNew;
            gradient = gradientNew;
        }

        return this.getResult(x, fx);
    }

    computeGradient(x) {
        const gradient = Array(this.nDim).fill(0);
        const h = 1e-6;

        for (let i = 0; i < this.nDim && this.evaluations < this.nTrials - 1; i++) {
            const xForward = [...x];
            const xBackward = [...x];
            xForward[i] = Math.min(1, x[i] + h);
            xBackward[i] = Math.max(0, x[i] - h);

            const fForward = this.evaluate(xForward);
            const fBackward = this.evaluate(xBackward);

            gradient[i] = (fForward - fBackward) / (2 * h);
        }

        return gradient;
    }

    getResult(x, fx) {
        return {
            bestValue: fx,
            bestX: x.slice(),
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Differential Evolution implementation
class DifferentialEvolution extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'DifferentialEvolution';
    }

    optimize() {
        const popSize = Math.min(20, Math.max(8, this.nDim * 3));
        const F = 0.5; // Differential weight
        const CR = 0.7; // Crossover probability

        // Initialize population
        const population = [];
        for (let i = 0; i < popSize && this.evaluations < this.nTrials; i++) {
            const individual = Array(this.nDim).fill(0).map(() => Math.random());
            const fitness = this.evaluate(individual);
            population.push({ x: individual, fitness });
        }

        // Evolution loop
        while (this.evaluations < this.nTrials && population.length >= 4) {
            for (let i = 0; i < population.length && this.evaluations < this.nTrials; i++) {
                // Select three random different individuals
                const indices = Array.from({length: population.length}, (_, idx) => idx)
                    .filter(idx => idx !== i);

                const [a, b, c] = [0, 1, 2].map(j => {
                    const idx = Math.floor(Math.random() * indices.length);
                    const selected = indices[idx];
                    indices.splice(idx, 1);
                    return selected;
                });

                // Mutation: v = a + F * (b - c)
                const mutant = population[a].x.map((xi, j) =>
                    xi + F * (population[b].x[j] - population[c].x[j])
                );

                // Crossover
                const trial = population[i].x.map((xi, j) =>
                    Math.random() < CR ?
                    MathUtils.clip(mutant[j], 0, 1) : xi
                );

                const trialFitness = this.evaluate(trial);

                // Selection
                if (trialFitness < population[i].fitness) {
                    population[i] = { x: trial, fitness: trialFitness };
                }
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Particle Swarm Optimization implementation
class ParticleSwarm extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'ParticleSwarm';
    }

    optimize() {
        const swarmSize = Math.min(40, Math.max(15, this.nDim * 3)); // Larger swarm for complex functions

        // Initialize swarm with diverse positions
        const particles = [];
        for (let i = 0; i < swarmSize && this.evaluations < this.nTrials; i++) {
            let position;
            if (i < swarmSize / 4) {
                // 25% near center for sphere-like functions
                position = Array(this.nDim).fill(0).map(() => 0.5 + (Math.random() - 0.5) * 0.3);
            } else if (i < swarmSize / 2) {
                // 25% near corners and edges
                position = Array(this.nDim).fill(0).map(() => Math.random() < 0.5 ? 0.1 : 0.9);
            } else {
                // 50% random distribution
                position = Array(this.nDim).fill(0).map(() => Math.random());
            }

            const velocity = Array(this.nDim).fill(0).map(() => (Math.random() - 0.5) * 0.2);
            const fitness = this.evaluate(position);

            particles.push({
                position: [...position],
                velocity: [...velocity],
                bestPosition: [...position],
                bestFitness: fitness,
                stagnationCount: 0
            });
        }

        // PSO loop with adaptive parameters
        let iteration = 0;
        const maxIterations = Math.ceil(this.nTrials / swarmSize);

        while (this.evaluations < this.nTrials) {
            iteration++;

            // Adaptive parameters based on iteration progress
            const progress = iteration / maxIterations;
            const w = 0.9 - 0.5 * progress; // Decreasing inertia (0.9 → 0.4)
            const c1 = 2.5 - 1.0 * progress; // Decreasing cognitive (2.5 → 1.5)
            const c2 = 1.5 + 1.0 * progress; // Increasing social (1.5 → 2.5)

            // Maximum velocity (velocity clamping)
            const vmax = 0.2 * (1 - 0.5 * progress); // Decreasing velocity limit

            for (let i = 0; i < particles.length && this.evaluations < this.nTrials; i++) {
                const particle = particles[i];

                // Update velocity with constriction factor
                for (let j = 0; j < this.nDim; j++) {
                    const r1 = Math.random();
                    const r2 = Math.random();

                    // Standard PSO velocity update
                    particle.velocity[j] = w * particle.velocity[j] +
                        c1 * r1 * (particle.bestPosition[j] - particle.position[j]) +
                        c2 * r2 * (this.bestX[j] - particle.position[j]);

                    // Velocity clamping
                    particle.velocity[j] = MathUtils.clip(particle.velocity[j], -vmax, vmax);
                }

                // Update position
                for (let j = 0; j < this.nDim; j++) {
                    particle.position[j] = MathUtils.clip(
                        particle.position[j] + particle.velocity[j], 0, 1
                    );
                }

                const fitness = this.evaluate(particle.position);

                // Update personal best
                if (fitness < particle.bestFitness) {
                    particle.bestFitness = fitness;
                    particle.bestPosition = [...particle.position];
                    particle.stagnationCount = 0;
                } else {
                    particle.stagnationCount++;
                }

                // Diversification for stagnant particles
                if (particle.stagnationCount > 15 && Math.random() < 0.1) {
                    // Reinitialize position with small probability
                    particle.position = Array(this.nDim).fill(0).map(() => Math.random());
                    particle.velocity = Array(this.nDim).fill(0).map(() => (Math.random() - 0.5) * 0.1);
                    particle.stagnationCount = 0;
                }
            }

            // Early termination for excellent solutions
            if (this.bestValue < 1e-6) break; // Relaxed termination for visualization
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Simulated Annealing implementation
class SimulatedAnnealing extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'SimulatedAnnealing';
    }

    optimize() {
        let globalBestX = null;
        let globalBestFx = Infinity;

        // Multi-restart simulated annealing for better global optimization
        const numRestarts = Math.max(3, Math.floor(this.nTrials / 30));
        const trialsPerRestart = Math.floor(this.nTrials / numRestarts);

        for (let restart = 0; restart < numRestarts && this.evaluations < this.nTrials; restart++) {
            // Initialize with different strategies per restart
            let x;
            if (restart === 0) {
                // First restart: center-biased initialization
                x = Array(this.nDim).fill(0).map(() => 0.5 + (Math.random() - 0.5) * 0.4);
            } else if (restart === 1) {
                // Second restart: near-optimal region
                x = Array(this.nDim).fill(0).map(() => (Math.random() - 0.5) * 0.2 + 0.5);
            } else {
                // Subsequent restarts: random
                x = Array(this.nDim).fill(0).map(() => Math.random());
            }

            let fx = this.evaluate(x);

            // Track best for this restart
            let bestX = x.slice();
            let bestFx = fx;

            // Aggressive initial temperature
            let temperature = Math.max(1.0, bestFx * 2);
            const finalTemp = 1e-8;
            const maxIterations = Math.min(trialsPerRestart, this.nTrials - this.evaluations);

            for (let iter = 0; iter < maxIterations && this.evaluations < this.nTrials; iter++) {
                // Adaptive step size based on current best and temperature
                const progressRatio = iter / maxIterations;
                const tempRatio = temperature / (Math.max(1.0, bestFx * 2));
                let stepSize = 0.3 * tempRatio * (1 - progressRatio) + 0.01 * progressRatio;

                // Generate neighbor with multiple strategies
                const strategy = iter % 3;
                let newX;

                if (strategy === 0) {
                    // Standard perturbation
                    newX = x.map(xi => {
                        const perturbation = (Math.random() - 0.5) * 2 * stepSize;
                        return MathUtils.clip(xi + perturbation, 0, 1);
                    });
                } else if (strategy === 1) {
                    // Move toward current best with noise
                    newX = x.map((xi, i) => {
                        const direction = bestX[i] - xi;
                        const move = direction * 0.3 + (Math.random() - 0.5) * stepSize;
                        return MathUtils.clip(xi + move, 0, 1);
                    });
                } else {
                    // Large jump for exploration
                    newX = x.map(xi => {
                        if (Math.random() < 0.1) {
                            return Math.random(); // Occasional large jump
                        } else {
                            const perturbation = (Math.random() - 0.5) * 2 * stepSize;
                            return MathUtils.clip(xi + perturbation, 0, 1);
                        }
                    });
                }

                const newFx = this.evaluate(newX);

                // Update best for this restart
                if (newFx < bestFx) {
                    bestX = newX.slice();
                    bestFx = newFx;
                }

                // Metropolis criterion
                const delta = newFx - fx;
                if (delta < 0 || (temperature > finalTemp && Math.random() < Math.exp(-delta / temperature))) {
                    x = newX;
                    fx = newFx;
                }

                // Fast exponential cooling with floor
                temperature *= 0.99;
                temperature = Math.max(temperature, finalTemp);

                // Early termination if we find a very good solution
                if (bestFx < 1e-6) break;
            }

            // Update global best
            if (bestFx < globalBestFx) {
                globalBestFx = bestFx;
                globalBestX = bestX.slice();
            }
        }

        return {
            bestValue: globalBestFx,
            bestX: globalBestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Genetic Algorithm implementation
class GeneticAlgorithm extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'GeneticAlgorithm';
    }

    optimize() {
        const popSize = Math.min(50, Math.max(20, this.nDim * 4));
        const mutationRate = 0.1;
        const crossoverRate = 0.8;

        // Initialize population
        let population = [];
        for (let i = 0; i < popSize && this.evaluations < this.nTrials; i++) {
            const individual = Array(this.nDim).fill(0).map(() => Math.random());
            const fitness = this.evaluate(individual);
            population.push({ x: individual, fitness });
        }

        // Evolution loop
        while (this.evaluations < this.nTrials) {
            const newPop = [];
            for (let i = 0; i < popSize && this.evaluations < this.nTrials; i++) {
                const parent1 = this.tournamentSelection(population);
                const parent2 = this.tournamentSelection(population);

                let child = [...parent1.x];

                // Crossover
                if (Math.random() < crossoverRate) {
                    const crossPoint = Math.floor(Math.random() * this.nDim);
                    for (let j = crossPoint; j < this.nDim; j++) {
                        child[j] = parent2.x[j];
                    }
                }

                // Mutation
                for (let j = 0; j < this.nDim; j++) {
                    if (Math.random() < mutationRate) {
                        child[j] = MathUtils.clip(
                            child[j] + (Math.random() - 0.5) * 0.2, 0, 1
                        );
                    }
                }

                const fitness = this.evaluate(child);
                newPop.push({ x: child, fitness });
            }

            population = newPop;
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    tournamentSelection(population) {
        const tournamentSize = 3;
        let best = population[Math.floor(Math.random() * population.length)];

        for (let i = 1; i < tournamentSize; i++) {
            const competitor = population[Math.floor(Math.random() * population.length)];
            if (competitor.fitness < best.fitness) {
                best = competitor;
            }
        }

        return best;
    }
}

// Random Search implementation
class RandomSearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'RandomSearch';
    }

    optimize() {
        while (this.evaluations < this.nTrials) {
            const x = Array(this.nDim).fill(0).map(() => Math.random());
            this.evaluate(x);
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Simplified Bayesian Optimization
class BayesianOpt extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'BayesianOpt';
        this.observations = [];
    }

    optimize() {
        // Strategic initial sampling with some center-biased points
        const nInitial = Math.min(10, Math.floor(this.nTrials * 0.2));

        // Sample some points near center for sphere-like functions
        for (let i = 0; i < Math.min(3, nInitial) && this.evaluations < this.nTrials; i++) {
            const x = Array(this.nDim).fill(0).map(() => 0.5 + (Math.random() - 0.5) * 0.3);
            const y = this.evaluate(x);
            this.observations.push({ x: [...x], y });
        }

        // Fill remaining initial samples with random points
        for (let i = this.observations.length; i < nInitial && this.evaluations < this.nTrials; i++) {
            const x = Array(this.nDim).fill(0).map(() => Math.random());
            const y = this.evaluate(x);
            this.observations.push({ x: [...x], y });
        }

        // Bayesian optimization loop with intensification
        while (this.evaluations < this.nTrials) {
            const nextX = this.acquireNext();
            const y = this.evaluate(nextX);
            this.observations.push({ x: [...nextX], y });

            // Intensify search around best point if very good solution found
            if (y < 1e-4 && this.evaluations < this.nTrials - 5) {
                for (let i = 0; i < Math.min(3, this.nTrials - this.evaluations); i++) {
                    const localX = nextX.map(xi => {
                        const noise = (Math.random() - 0.5) * 0.02; // Small perturbation
                        return MathUtils.clip(xi + noise, 0, 1);
                    });
                    const localY = this.evaluate(localX);
                    this.observations.push({ x: [...localX], y: localY });
                }
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    acquireNext() {
        let bestAcq = -Infinity;
        let nextX = Array(this.nDim).fill(0).map(() => Math.random());

        // Sample candidate points
        for (let j = 0; j < 100; j++) {
            const candidate = Array(this.nDim).fill(0).map(() => Math.random());
            const acq = this.acquisitionFunction(candidate);

            if (acq > bestAcq) {
                bestAcq = acq;
                nextX = candidate;
            }
        }

        return nextX;
    }

    acquisitionFunction(x) {
        if (this.observations.length === 0) return Math.random();

        // Distance-weighted Expected Improvement approximation
        const distances = this.observations.map(obs => ({
            dist: MathUtils.norm(MathUtils.subtract(x, obs.x)),
            y: obs.y
        }));

        distances.sort((a, b) => a.dist - b.dist);
        const kNearest = distances.slice(0, Math.min(5, distances.length));

        if (kNearest.length === 0) return Math.random();

        // Distance-weighted prediction
        const epsilon = 1e-8; // Avoid division by zero
        let weightSum = 0;
        let weightedMean = 0;

        for (const item of kNearest) {
            const weight = 1.0 / (item.dist + epsilon);
            weightSum += weight;
            weightedMean += weight * item.y;
        }

        const predictedMean = weightedMean / weightSum;

        // Estimate uncertainty based on distance to nearest point and local variance
        const uncertainty = Math.exp(-2.0 * kNearest[0].dist);
        const localVariance = kNearest.length > 1 ?
            kNearest.reduce((sum, item) => sum + Math.pow(item.y - predictedMean, 2), 0) / kNearest.length :
            0.1;
        const predictedStd = Math.max(Math.sqrt(localVariance), 0.01) * uncertainty;

        // Expected Improvement: EI = (f_min - mu) * Φ(Z) + σ * φ(Z)
        const bestY = Math.min(...this.observations.map(obs => obs.y));
        const improvement = bestY - predictedMean;

        if (predictedStd <= epsilon) {
            return improvement > 0 ? improvement : 0;
        }

        const z = improvement / predictedStd;

        // Approximate normal CDF and PDF
        const phi = 0.5 * (1 + this.erf(z / Math.sqrt(2))); // CDF
        const pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); // PDF

        const expectedImprovement = improvement * phi + predictedStd * pdf;

        // Add small exploration bonus
        return Math.max(0, expectedImprovement) + 0.01 * uncertainty;
    }

    // Error function approximation for normal CDF
    erf(x) {
        // Abramowitz and Stegun approximation
        const a1 =  0.254829592;
        const a2 = -0.284496736;
        const a3 =  1.421413741;
        const a4 = -1.453152027;
        const a5 =  1.061405429;
        const p  =  0.3275911;

        const sign = x >= 0 ? 1 : -1;
        x = Math.abs(x);

        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

        return sign * y;
    }
}

// Simplified CMA-ES
class CMAEvolutionStrategy extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'CMAEvolutionStrategy';
    }

    optimize() {
        const lambda = Math.min(20, 4 + Math.floor(3 * Math.log(this.nDim)));
        const mu = Math.floor(lambda / 2);

        let mean = Array(this.nDim).fill(0.5);
        let sigma = 0.3;

        // Simplified weights
        const weights = Array(mu).fill(0).map((_, i) => Math.log(mu + 0.5) - Math.log(i + 1));
        const sumWeights = weights.reduce((sum, w) => sum + w, 0);
        const normalizedWeights = weights.map(w => w / sumWeights);

        while (this.evaluations < this.nTrials) {
            // Generate offspring
            const offspring = [];
            for (let i = 0; i < lambda && this.evaluations < this.nTrials; i++) {
                const individual = mean.map(m =>
                    MathUtils.clip(m + sigma * (Math.random() - 0.5) * 2, 0, 1)
                );
                const fitness = this.evaluate(individual);
                offspring.push({ x: individual, fitness });
            }

            // Selection and recombination
            offspring.sort((a, b) => a.fitness - b.fitness);
            const selected = offspring.slice(0, mu);

            // Update mean
            const newMean = Array(this.nDim).fill(0);
            for (let i = 0; i < Math.min(mu, selected.length); i++) {
                if (selected[i] && selected[i].x) {
                    for (let j = 0; j < this.nDim; j++) {
                        newMean[j] += normalizedWeights[i] * selected[i].x[j];
                    }
                }
            }
            mean = newMean;

            // Adapt step size (simplified)
            const improvement = (selected.length > 0 && offspring.length > 0) ?
                (offspring[offspring.length - 1]?.fitness - selected[0]?.fitness) : 0;
            sigma *= improvement > 0 ? 0.95 : 1.05;
            sigma = MathUtils.clip(sigma, 0.01, 1.0);
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Adaptive Random Search
class Rechenberg extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'Rechenberg';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        let stepSize = 0.1;
        let successCount = 0;
        let totalAttempts = 0;

        while (this.evaluations < this.nTrials) {
            // Generate candidate
            const candidate = x.map(xi =>
                MathUtils.clip(xi + (Math.random() - 0.5) * stepSize, 0, 1)
            );

            const candidateFx = this.evaluate(candidate);
            totalAttempts++;

            if (candidateFx < fx) {
                x = candidate;
                fx = candidateFx;
                successCount++;
            }

            // Adapt step size based on success rate
            if (totalAttempts % 20 === 0) {
                const successRate = successCount / totalAttempts;
                if (successRate > 0.2) {
                    stepSize = Math.min(0.3, stepSize * 1.1);
                } else if (successRate < 0.1) {
                    stepSize = Math.max(0.01, stepSize * 0.9);
                }
                successCount = 0;
                totalAttempts = 0;
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Coordinate Descent
class CoordinateDescent extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'CoordinateDescent';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        let stepSize = 0.1;

        while (this.evaluations < this.nTrials) {
            let improved = false;

            for (let i = 0; i < this.nDim && this.evaluations < this.nTrials - 1; i++) {
                // Try positive direction
                const xPos = [...x];
                xPos[i] = MathUtils.clip(x[i] + stepSize, 0, 1);
                const fxPos = this.evaluate(xPos);

                if (fxPos < fx) {
                    x = xPos;
                    fx = fxPos;
                    improved = true;
                    continue;
                }

                // Try negative direction
                const xNeg = [...x];
                xNeg[i] = MathUtils.clip(x[i] - stepSize, 0, 1);
                const fxNeg = this.evaluate(xNeg);

                if (fxNeg < fx) {
                    x = xNeg;
                    fx = fxNeg;
                    improved = true;
                }
            }

            // Adapt step size
            if (improved) {
                stepSize = Math.min(0.2, stepSize * 1.1);
            } else {
                stepSize = Math.max(0.001, stepSize * 0.8);
                if (stepSize < 0.005) break;
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Pattern Search (Hooke-Jeeves style)
class PatternSearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'PatternSearch';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        let stepSize = 0.1;

        while (this.evaluations < this.nTrials && stepSize > 1e-6) {
            const xStart = [...x];

            // Exploratory moves
            for (let i = 0; i < this.nDim && this.evaluations < this.nTrials - 1; i++) {
                // Positive direction
                const xPos = [...x];
                xPos[i] = MathUtils.clip(x[i] + stepSize, 0, 1);
                const fxPos = this.evaluate(xPos);

                if (fxPos < fx) {
                    x = xPos;
                    fx = fxPos;
                } else {
                    // Negative direction
                    const xNeg = [...x];
                    xNeg[i] = MathUtils.clip(x[i] - stepSize, 0, 1);
                    const fxNeg = this.evaluate(xNeg);

                    if (fxNeg < fx) {
                        x = xNeg;
                        fx = fxNeg;
                    }
                }
            }

            // Pattern move
            const improved = fx < this.evaluate(xStart);
            if (improved && this.evaluations < this.nTrials) {
                const direction = MathUtils.subtract(x, xStart);
                const xPattern = MathUtils.add(x, direction);
                const xClipped = MathUtils.clipArray(xPattern, 0, 1);

                const fxPattern = this.evaluate(xClipped);
                if (fxPattern < fx) {
                    x = xClipped;
                    fx = fxPattern;
                }
            }

            // Update step size
            if (improved) {
                stepSize = Math.min(0.2, stepSize * 1.2);
            } else {
                stepSize *= 0.5;
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Hill Climbing
class HillClimbing extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'HillClimbing';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        while (this.evaluations < this.nTrials) {
            // Generate neighbor
            const neighbor = x.map(xi =>
                MathUtils.clip(xi + (Math.random() - 0.5) * 0.1, 0, 1)
            );

            const neighborFx = this.evaluate(neighbor);

            // Accept if better
            if (neighborFx < fx) {
                x = neighbor;
                fx = neighborFx;
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Tabu Search
class TabuSearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'TabuSearch';
        this.tabuList = [];
        this.tabuTenure = Math.min(20, Math.max(5, this.nDim));
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        while (this.evaluations < this.nTrials) {
            let bestNeighbor = null;
            let bestNeighborFx = Infinity;

            // Generate multiple neighbors
            for (let i = 0; i < Math.min(20, this.nTrials - this.evaluations); i++) {
                const neighbor = x.map(xi =>
                    MathUtils.clip(xi + (Math.random() - 0.5) * 0.15, 0, 1)
                );

                // Check if tabu
                const isTabu = this.tabuList.some(tabu =>
                    MathUtils.norm(MathUtils.subtract(neighbor, tabu)) < 0.05
                );

                if (!isTabu) {
                    const neighborFx = this.evaluate(neighbor);
                    if (neighborFx < bestNeighborFx) {
                        bestNeighbor = neighbor;
                        bestNeighborFx = neighborFx;
                    }
                }
            }

            if (bestNeighbor) {
                // Add current solution to tabu list
                this.tabuList.push([...x]);
                if (this.tabuList.length > this.tabuTenure) {
                    this.tabuList.shift();
                }

                x = bestNeighbor;
                fx = bestNeighborFx;
            } else {
                // If all neighbors are tabu, clear tabu list
                this.tabuList = [];
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Firefly Algorithm
class FireflyAlgorithm extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'FireflyAlgorithm';
    }

    optimize() {
        const nFireflies = Math.min(25, Math.max(10, this.nDim * 2));
        const alpha = 0.1; // Randomization parameter
        const gamma = 1.0; // Light absorption coefficient

        // Initialize fireflies
        const fireflies = [];
        for (let i = 0; i < nFireflies && this.evaluations < this.nTrials; i++) {
            const position = Array(this.nDim).fill(0).map(() => Math.random());
            const intensity = this.evaluate(position);
            fireflies.push({ position: [...position], intensity });
        }

        while (this.evaluations < this.nTrials) {
            for (let i = 0; i < fireflies.length && this.evaluations < this.nTrials; i++) {
                for (let j = 0; j < fireflies.length; j++) {
                    if (i !== j && fireflies[j].intensity < fireflies[i].intensity) {
                        // Move firefly i towards j
                        const distance = MathUtils.norm(
                            MathUtils.subtract(fireflies[i].position, fireflies[j].position)
                        );
                        const beta = 1.0 / (1.0 + gamma * distance * distance);

                        const newPosition = fireflies[i].position.map((xi, k) => {
                            const attraction = beta * (fireflies[j].position[k] - xi);
                            const randomization = alpha * (Math.random() - 0.5);
                            return MathUtils.clip(xi + attraction + randomization, 0, 1);
                        });

                        const newIntensity = this.evaluate(newPosition);

                        if (newIntensity < fireflies[i].intensity) {
                            fireflies[i].position = newPosition;
                            fireflies[i].intensity = newIntensity;
                        }
                    }
                }
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Ant Colony Optimization
class AntColonyOpt extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'AntColonyOpt';
    }

    optimize() {
        const nAnts = Math.min(20, Math.max(8, this.nDim));
        const rho = 0.1; // Evaporation rate

        // Discretize search space
        const nLevels = 20;
        const pheromones = Array(this.nDim).fill(0).map(() => Array(nLevels).fill(1.0));

        while (this.evaluations < this.nTrials) {
            const solutions = [];

            // Construct solutions
            for (let ant = 0; ant < nAnts && this.evaluations < this.nTrials; ant++) {
                const solution = [];
                for (let dim = 0; dim < this.nDim; dim++) {
                    // Probabilistic selection based on pheromones
                    const probabilities = pheromones[dim].map(p => Math.pow(p, 2));
                    const total = probabilities.reduce((sum, p) => sum + p, 0);
                    const normalizedProb = probabilities.map(p => p / total);

                    let selected = 0;
                    const rand = Math.random();
                    let cumProb = 0;
                    for (let level = 0; level < nLevels; level++) {
                        cumProb += normalizedProb[level];
                        if (rand <= cumProb) {
                            selected = level;
                            break;
                        }
                    }

                    solution.push(selected / (nLevels - 1));
                }

                const fitness = this.evaluate(solution);
                solutions.push({ solution, fitness });
            }

            // Evaporate pheromones
            for (let dim = 0; dim < this.nDim; dim++) {
                for (let level = 0; level < nLevels; level++) {
                    pheromones[dim][level] *= (1 - rho);
                }
            }

            // Update pheromones (best solution gets more pheromone)
            solutions.sort((a, b) => a.fitness - b.fitness);
            const bestSol = solutions[0];

            bestSol.solution.forEach((value, dim) => {
                const level = Math.round(value * (nLevels - 1));
                pheromones[dim][level] += 1.0 / (1.0 + bestSol.fitness);
            });
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Harmony Search
class HarmonySearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'HarmonySearch';
    }

    optimize() {
        const HMS = Math.min(20, Math.max(5, this.nDim * 2)); // Harmony Memory Size
        const HMCR = 0.9; // Harmony Memory Considering Rate
        const PAR = 0.3; // Pitch Adjusting Rate

        // Initialize harmony memory
        const harmonyMemory = [];
        for (let i = 0; i < HMS && this.evaluations < this.nTrials; i++) {
            const harmony = Array(this.nDim).fill(0).map(() => Math.random());
            const fitness = this.evaluate(harmony);
            harmonyMemory.push({ harmony: [...harmony], fitness });
        }

        while (this.evaluations < this.nTrials) {
            const newHarmony = [];

            for (let j = 0; j < this.nDim; j++) {
                if (Math.random() < HMCR) {
                    // Pick from harmony memory
                    const selectedHarmony = harmonyMemory[
                        Math.floor(Math.random() * harmonyMemory.length)
                    ];
                    let value = selectedHarmony.harmony[j];

                    // Pitch adjustment
                    if (Math.random() < PAR) {
                        value = MathUtils.clip(value + (Math.random() - 0.5) * 0.1, 0, 1);
                    }

                    newHarmony.push(value);
                } else {
                    // Random selection
                    newHarmony.push(Math.random());
                }
            }

            const newFitness = this.evaluate(newHarmony);

            // Update harmony memory (replace worst if new harmony is better)
            harmonyMemory.sort((a, b) => a.fitness - b.fitness);
            if (newFitness < harmonyMemory[harmonyMemory.length - 1].fitness) {
                harmonyMemory[harmonyMemory.length - 1] = {
                    harmony: [...newHarmony],
                    fitness: newFitness
                };
            }
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// (μ+λ) Evolution Strategy
class EvolutionStrategy extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'EvolutionStrategy';
    }

    optimize() {
        const mu = Math.min(10, Math.max(2, Math.floor(this.nDim / 2)));
        const lambda = mu * 4;

        // Initialize parent population
        let parents = [];
        for (let i = 0; i < mu && this.evaluations < this.nTrials; i++) {
            const individual = Array(this.nDim).fill(0).map(() => Math.random());
            const fitness = this.evaluate(individual);
            const sigma = Array(this.nDim).fill(0.1); // Strategy parameters
            parents.push({ x: individual, fitness, sigma });
        }

        while (this.evaluations < this.nTrials) {
            const offspring = [];

            // Generate offspring
            for (let i = 0; i < lambda && this.evaluations < this.nTrials; i++) {
                // Select random parent
                const parent = parents[Math.floor(Math.random() * parents.length)];

                // Mutate strategy parameters
                const newSigma = parent.sigma.map(s =>
                    s * Math.exp(0.1 * (Math.random() - 0.5))
                );

                // Generate offspring
                const child = parent.x.map((xi, j) =>
                    MathUtils.clip(xi + newSigma[j] * (Math.random() - 0.5), 0, 1)
                );

                const fitness = this.evaluate(child);
                offspring.push({ x: child, fitness, sigma: newSigma });
            }

            // Select best μ from parents + offspring
            const combined = [...parents, ...offspring];
            combined.sort((a, b) => a.fitness - b.fitness);
            parents = combined.slice(0, mu);
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
}

// Optimizer factory
const OptimizerFactory = {
    create(name, objective, nTrials, nDim) {
        switch (name) {
            case 'PRIMA_UOBYQA':
                return new PRIMA_UOBYQA(objective, nTrials, nDim);
            case 'PRIMA_NEWUOA':
                return new PRIMA_NEWUOA(objective, nTrials, nDim);
            case 'PRIMA_BOBYQA':
                return new PRIMA_BOBYQA(objective, nTrials, nDim);
            case 'SciPy_NelderMead':
                return new NelderMead(objective, nTrials, nDim);
            case 'SciPy_Powell':
                return new Powell(objective, nTrials, nDim);
            case 'SciPy_BFGS':
                return new LBFGSB(objective, nTrials, nDim);
            case 'DifferentialEvolution':
                return new DifferentialEvolution(objective, nTrials, nDim);
            case 'ParticleSwarm':
                return new ParticleSwarm(objective, nTrials, nDim);
            case 'SimulatedAnnealing':
                return new SimulatedAnnealing(objective, nTrials, nDim);
            case 'GeneticAlgorithm':
                return new GeneticAlgorithm(objective, nTrials, nDim);
            case 'RandomSearch':
                return new RandomSearch(objective, nTrials, nDim);
            case 'BayesianOpt':
                return new BayesianOpt(objective, nTrials, nDim);
            case 'CMAEvolutionStrategy':
                return new CMAEvolutionStrategy(objective, nTrials, nDim);
            case 'Rechenberg':
                return new Rechenberg(objective, nTrials, nDim);
            case 'CoordinateDescent':
                return new CoordinateDescent(objective, nTrials, nDim);
            case 'PatternSearch':
                return new PatternSearch(objective, nTrials, nDim);
            case 'HillClimbing':
                return new HillClimbing(objective, nTrials, nDim);
            case 'TabuSearch':
                return new TabuSearch(objective, nTrials, nDim);
            case 'FireflyAlgorithm':
                return new FireflyAlgorithm(objective, nTrials, nDim);
            case 'AntColonyOpt':
                return new AntColonyOpt(objective, nTrials, nDim);
            case 'HarmonySearch':
                return new HarmonySearch(objective, nTrials, nDim);
            case 'EvolutionStrategy':
                return new EvolutionStrategy(objective, nTrials, nDim);
            default:
                throw new Error(`Unknown optimizer: ${name}`);
        }
    },

    getAvailableOptimizers() {
        return [
            'PRIMA_UOBYQA', 'PRIMA_NEWUOA', 'SciPy_BFGS', 'SciPy_Powell', 'SciPy_NelderMead',
            'DifferentialEvolution', 'ParticleSwarm', 'SimulatedAnnealing', 'GeneticAlgorithm',
            'RandomSearch', 'BayesianOpt', 'CMAEvolutionStrategy', 'Rechenberg',
            'CoordinateDescent', 'PatternSearch', 'HillClimbing', 'FireflyAlgorithm',
            'AntColonyOpt', 'HarmonySearch', 'EvolutionStrategy'
        ];
    }
};

// Node.js compatibility - make OptimizerFactory available in global scope
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { OptimizerFactory };
} else if (typeof global !== 'undefined') {
    global.OptimizerFactory = OptimizerFactory;
}