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

// Simplified PRIMA UOBYQA implementation
class PRIMA_UOBYQA extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'PRIMA_UOBYQA';
        // UOBYQA uses up to (n+1)(n+2)/2 interpolation points for full quadratic model
        this.maxInterpolationPoints = Math.min((nDim + 1) * (nDim + 2) / 2, Math.max(2 * nDim + 1, nTrials / 4));
    }

    optimize() {
        const n = this.nDim;

        // Initialize starting point (away from boundaries)
        let xBase = Array(n).fill(0).map(() => 0.2 + 0.6 * Math.random());
        let fBase = this.evaluate(xBase);

        // Trust region parameters (matching PDFO defaults)
        let rho = Math.min(0.5, 0.1 * Math.sqrt(n)); // Initial trust region radius
        const rhoEnd = 1e-6; // Final trust region radius
        const eta1 = 0.1; // Threshold for rejecting steps
        const eta2 = 0.7; // Threshold for expanding trust region
        const gamma1 = 0.5; // Trust region contraction factor
        const gamma2 = 2.0; // Trust region expansion factor

        // Interpolation set: points and function values
        let interpolationPoints = [xBase.slice()];
        let interpolationValues = [fBase];

        // Build initial interpolation set
        this.buildInitialInterpolationSet(interpolationPoints, interpolationValues, xBase, rho);

        let xOpt = xBase.slice();
        let fOpt = fBase;
        let kOpt = 0; // Index of best point

        // Find current best point
        for (let k = 0; k < interpolationValues.length; k++) {
            if (interpolationValues[k] < fOpt) {
                fOpt = interpolationValues[k];
                xOpt = interpolationPoints[k].slice();
                kOpt = k;
            }
        }

        // Main UOBYQA loop
        while (this.evaluations < this.nTrials && rho > rhoEnd) {
            // Build quadratic model around xOpt
            const model = this.buildQuadraticModel(interpolationPoints, interpolationValues, xOpt);

            // Solve trust region subproblem to get step
            const step = this.solveTrustRegionSubproblem(model, xOpt, rho);

            if (this.evaluations >= this.nTrials) break;

            // Compute trial point
            const xTrial = MathUtils.add(xOpt, step);
            // Enforce bounds [0,1]
            const xTrialBounded = xTrial.map(x => Math.min(1, Math.max(0, x)));
            const fTrial = this.evaluate(xTrialBounded);

            // Compute predicted reduction and actual reduction
            const predReduction = this.computePredictedReduction(model, step);
            const actualReduction = fOpt - fTrial;

            // Ratio of actual to predicted reduction
            const ratio = predReduction > 0 ? actualReduction / predReduction : -1;

            // Update trust region radius
            let rho_new = rho;
            if (ratio <= eta1) {
                // Poor model agreement - shrink trust region
                rho_new = gamma1 * rho;
            } else if (ratio >= eta2 && MathUtils.norm(step) > 0.8 * rho) {
                // Good model agreement and step at boundary - expand
                rho_new = Math.min(gamma2 * rho, 10.0);
            }

            // Accept or reject the step
            if (ratio > eta1) {
                // Accept step
                xOpt = xTrialBounded.slice();
                fOpt = fTrial;

                // Add new point to interpolation set
                this.updateInterpolationSet(interpolationPoints, interpolationValues, xOpt, fOpt);
            }

            rho = Math.max(rho_new, rhoEnd);
        }

        return {
            bestValue: fOpt,
            bestX: xOpt,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    buildInitialInterpolationSet(points, values, xBase, rho) {
        const n = this.nDim;

        // Add coordinate direction points
        for (let i = 0; i < n && points.length < this.maxInterpolationPoints && this.evaluations < this.nTrials; i++) {
            // Positive direction
            const xPlus = xBase.slice();
            const step = Math.min(rho, 1 - xBase[i]);
            xPlus[i] += step;
            points.push(xPlus);
            values.push(this.evaluate(xPlus));

            if (points.length >= this.maxInterpolationPoints || this.evaluations >= this.nTrials) break;

            // Negative direction
            const xMinus = xBase.slice();
            const stepMinus = Math.min(rho, xBase[i]);
            xMinus[i] -= stepMinus;
            points.push(xMinus);
            values.push(this.evaluate(xMinus));
        }

        // Add some diagonal points if budget allows
        while (points.length < this.maxInterpolationPoints && this.evaluations < this.nTrials) {
            const xDiag = xBase.slice();
            for (let i = 0; i < n; i++) {
                const perturbation = (Math.random() - 0.5) * 2 * rho * 0.5;
                xDiag[i] = Math.min(1, Math.max(0, xBase[i] + perturbation));
            }
            points.push(xDiag);
            values.push(this.evaluate(xDiag));
        }
    }

    buildQuadraticModel(points, values, xOpt) {
        const n = this.nDim;
        const nPts = points.length;

        // Simple quadratic model: f(s) = c + g'*s + 0.5*s'*H*s where s = x - xOpt
        const model = {
            c: 0,
            g: new Array(n).fill(0),
            H: Array(n).fill().map(() => new Array(n).fill(0))
        };

        // Find base point closest to xOpt
        let baseIdx = 0;
        let minDist = Infinity;
        for (let i = 0; i < nPts; i++) {
            const dist = MathUtils.norm(MathUtils.subtract(points[i], xOpt));
            if (dist < minDist) {
                minDist = dist;
                baseIdx = i;
            }
        }

        model.c = values[baseIdx];

        // Build model using finite differences
        // Estimate gradient
        for (let i = 0; i < n; i++) {
            let forwardIdx = -1, backwardIdx = -1;
            let forwardDist = Infinity, backwardDist = Infinity;

            for (let j = 0; j < nPts; j++) {
                const diff = MathUtils.subtract(points[j], xOpt);

                // Look for points along coordinate i
                if (this.isAlmostCoordinateDirection(diff, i, 0.1)) {
                    const coordDiff = diff[i];
                    const dist = Math.abs(coordDiff);

                    if (coordDiff > 0 && dist < forwardDist) {
                        forwardIdx = j;
                        forwardDist = dist;
                    } else if (coordDiff < 0 && dist < backwardDist) {
                        backwardIdx = j;
                        backwardDist = dist;
                    }
                }
            }

            // Compute finite difference approximations
            if (forwardIdx >= 0 && backwardIdx >= 0) {
                // Central difference
                const h = points[forwardIdx][i] - points[backwardIdx][i];
                model.g[i] = (values[forwardIdx] - values[backwardIdx]) / h;

                // Second derivative approximation
                const h_half = h / 2;
                model.H[i][i] = (values[forwardIdx] - 2 * model.c + values[backwardIdx]) / (h_half * h_half);
            } else if (forwardIdx >= 0) {
                // Forward difference
                const h = points[forwardIdx][i] - xOpt[i];
                model.g[i] = (values[forwardIdx] - model.c) / h;
            } else if (backwardIdx >= 0) {
                // Backward difference
                const h = xOpt[i] - points[backwardIdx][i];
                model.g[i] = (model.c - values[backwardIdx]) / h;
            }
        }

        return model;
    }

    solveTrustRegionSubproblem(model, xOpt, rho) {
        const n = this.nDim;

        // Simplified trust region solve: Cauchy point method with bound constraints

        // Compute steepest descent direction
        let step = model.g.map(gi => -gi);
        let stepNorm = MathUtils.norm(step);

        if (stepNorm < 1e-12) {
            // Zero gradient - try random direction
            step = Array(n).fill(0).map(() => (Math.random() - 0.5) * 2);
            stepNorm = MathUtils.norm(step);
        }

        // Scale to trust region boundary if necessary
        if (stepNorm > rho) {
            step = MathUtils.scale(step, rho / stepNorm);
        }

        // Apply bound constraints
        for (let i = 0; i < n; i++) {
            const newPos = xOpt[i] + step[i];
            if (newPos < 0) {
                step[i] = -xOpt[i];
            } else if (newPos > 1) {
                step[i] = 1 - xOpt[i];
            }
        }

        // If step violates trust region after bound projection, rescale
        const finalNorm = MathUtils.norm(step);
        if (finalNorm > rho * 1.01) { // Small tolerance
            step = MathUtils.scale(step, rho / finalNorm);
        }

        return step;
    }

    computePredictedReduction(model, step) {
        const n = this.nDim;

        // Predicted reduction = -(g'*step + 0.5*step'*H*step)
        let pred = 0;

        // Linear term
        for (let i = 0; i < n; i++) {
            pred -= model.g[i] * step[i];
        }

        // Quadratic term
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                pred -= 0.5 * step[i] * model.H[i][j] * step[j];
            }
        }

        return pred;
    }

    updateInterpolationSet(points, values, xNew, fNew) {
        // Simple replacement strategy: replace worst point if new point is better
        let worstIdx = 0;
        let worstValue = values[0];

        for (let i = 1; i < values.length; i++) {
            if (values[i] > worstValue) {
                worstIdx = i;
                worstValue = values[i];
            }
        }

        if (fNew < worstValue && points.length < this.maxInterpolationPoints) {
            points.push(xNew.slice());
            values.push(fNew);
        } else if (fNew < worstValue) {
            points[worstIdx] = xNew.slice();
            values[worstIdx] = fNew;
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

// Simplified PRIMA NEWUOA implementation
class PRIMA_NEWUOA extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'PRIMA_NEWUOA';
        // NEWUOA typically uses 2n+1 interpolation points (underdetermined quadratic model)
        this.nPts = Math.min(2 * nDim + 1, Math.max(nDim + 2, Math.floor(nTrials / 3)));
    }

    optimize() {
        const n = this.nDim;

        // Initialize starting point (unconstrained, so we use full [0,1] range)
        let xBase = Array(n).fill(0).map(() => 0.1 + 0.8 * Math.random());
        let fBase = this.evaluate(xBase);

        // NEWUOA trust region parameters
        let rho = Math.min(0.5, 1.0 / Math.sqrt(n)); // Initial trust region radius
        const rhoEnd = 1e-6;
        const eta1 = 0.1;  // Threshold for step acceptance
        const eta2 = 0.7;  // Threshold for trust region expansion
        const gamma1 = 0.5; // Trust region contraction
        const gamma2 = 2.0; // Trust region expansion

        // Initialize interpolation set
        let xPts = [xBase.slice()];  // Interpolation points
        let fVals = [fBase];         // Function values at interpolation points

        // Build initial interpolation set with 2n+1 points
        this.buildInitialInterpolationSetNEWUOA(xPts, fVals, xBase, rho);

        // Find best point
        let kOpt = 0;
        let fOpt = fVals[0];
        for (let k = 0; k < fVals.length; k++) {
            if (fVals[k] < fOpt) {
                fOpt = fVals[k];
                kOpt = k;
            }
        }
        let xOpt = xPts[kOpt].slice();

        // Main NEWUOA loop
        while (this.evaluations < this.nTrials && rho > rhoEnd) {
            // Build quadratic model around current best point
            const model = this.buildNEWUOAQuadraticModel(xPts, fVals, xOpt);

            // Solve trust region subproblem
            const step = this.solveNEWUOATrustRegion(model, xOpt, rho);

            if (this.evaluations >= this.nTrials) break;

            // Compute trial point (no bounds in NEWUOA, but we'll apply [0,1] bounds)
            const xTrial = MathUtils.add(xOpt, step);
            const xTrialBounded = xTrial.map(x => Math.min(1, Math.max(0, x)));
            const fTrial = this.evaluate(xTrialBounded);

            // Compute model prediction and actual reduction
            const predRed = this.computeNEWUOAPrediction(model, step);
            const actualRed = fOpt - fTrial;

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
                xOpt = xTrialBounded.slice();
                fOpt = fTrial;
                kOpt = this.updateNEWUOAInterpolationSet(xPts, fVals, xOpt, fOpt);
            }

            rho = Math.max(rhoNew, rhoEnd);

            // Geometry improvement step (simplified)
            if (Math.random() < 0.1 && this.evaluations < this.nTrials - 5) {
                this.improveGeometry(xPts, fVals, xOpt, rho);
            }
        }

        return {
            bestValue: fOpt,
            bestX: xOpt,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    buildInitialInterpolationSetNEWUOA(xPts, fVals, xBase, rho) {
        const n = this.nDim;

        // Add coordinate directions (both positive and negative if budget allows)
        for (let i = 0; i < n && xPts.length < this.nPts && this.evaluations < this.nTrials; i++) {
            // Positive direction
            const xNew = xBase.slice();
            const stepSize = Math.min(rho, Math.min(1 - xBase[i], rho));
            xNew[i] = Math.min(1, xBase[i] + stepSize);
            xPts.push(xNew);
            fVals.push(this.evaluate(xNew));

            if (xPts.length >= this.nPts || this.evaluations >= this.nTrials) break;

            // Negative direction
            const xNew2 = xBase.slice();
            const stepSize2 = Math.min(rho, Math.min(xBase[i], rho));
            xNew2[i] = Math.max(0, xBase[i] - stepSize2);
            xPts.push(xNew2);
            fVals.push(this.evaluate(xNew2));
        }

        // Add additional points to reach 2n+1 if needed
        while (xPts.length < this.nPts && this.evaluations < this.nTrials) {
            const xNew = xBase.slice();
            // Add random perturbation
            for (let i = 0; i < n; i++) {
                const pert = (Math.random() - 0.5) * 2 * rho * 0.7;
                xNew[i] = Math.min(1, Math.max(0, xBase[i] + pert));
            }
            xPts.push(xNew);
            fVals.push(this.evaluate(xNew));
        }
    }

    buildNEWUOAQuadraticModel(xPts, fVals, xOpt) {
        const n = this.nDim;
        const nPts = xPts.length;

        // Model: m(s) = c + g^T s + 0.5 s^T H s, where s = x - xOpt
        const model = {
            c: 0,
            g: new Array(n).fill(0),
            H: Array(n).fill().map(() => new Array(n).fill(0))
        };

        // Find point closest to xOpt for constant term
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

        // Compute model coefficients using Lagrange interpolation principles
        // Simplified approach: use finite differences where possible

        // Estimate gradient
        for (let i = 0; i < n; i++) {
            let forwardK = -1, backwardK = -1;
            let minForwardDist = Infinity, minBackwardDist = Infinity;

            for (let k = 0; k < nPts; k++) {
                const s = MathUtils.subtract(xPts[k], xOpt);

                // Check if this is approximately a coordinate direction
                let isCoordDir = true;
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
                const h = xPts[forwardK][i] - xPts[backwardK][i];
                model.g[i] = (fVals[forwardK] - fVals[backwardK]) / h;

                // Estimate diagonal Hessian
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

    solveNEWUOATrustRegion(model, xOpt, rho) {
        const n = this.nDim;

        // Simplified trust region solve: dogleg method approximation

        // Cauchy point: steepest descent step
        let gNorm = MathUtils.norm(model.g);
        if (gNorm < 1e-12) {
            // Zero gradient - return small random step
            const randomStep = Array(n).fill(0).map(() => (Math.random() - 0.5) * 2 * rho * 0.1);
            return this.projectToBounds(randomStep, xOpt);
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
        return this.projectToBounds(cauchyStep, xOpt);
    }

    projectToBounds(step, xOpt) {
        const n = this.nDim;
        for (let i = 0; i < n; i++) {
            const newPos = xOpt[i] + step[i];
            if (newPos < 0) {
                step[i] = -xOpt[i];
            } else if (newPos > 1) {
                step[i] = 1 - xOpt[i];
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

    updateNEWUOAInterpolationSet(xPts, fVals, xNew, fNew) {
        // Replace worst point with new point
        let worstK = 0;
        for (let k = 1; k < fVals.length; k++) {
            if (fVals[k] > fVals[worstK]) {
                worstK = k;
            }
        }

        if (fNew < fVals[worstK]) {
            xPts[worstK] = xNew.slice();
            fVals[worstK] = fNew;
            return worstK;
        }

        return 0;
    }

    improveGeometry(xPts, fVals, xOpt, rho) {
        // Simple geometry improvement: add a point that improves interpolation matrix conditioning
        // This is a simplified version - real NEWUOA has complex geometry management
        const n = this.nDim;

        if (this.evaluations >= this.nTrials) return;

        // Find direction with largest model uncertainty
        let bestDir = Array(n).fill(0).map(() => Math.random() - 0.5);
        bestDir = MathUtils.scale(bestDir, rho / MathUtils.norm(bestDir));

        const xTest = MathUtils.add(xOpt, bestDir);
        const xBounded = xTest.map(x => Math.min(1, Math.max(0, x)));
        const fTest = this.evaluate(xBounded);

        // Replace worst point if new point is better
        let worstK = 0;
        for (let k = 1; k < fVals.length; k++) {
            if (fVals[k] > fVals[worstK]) {
                worstK = k;
            }
        }

        if (fTest < fVals[worstK]) {
            xPts[worstK] = xBounded.slice();
            fVals[worstK] = fTest;
        }
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
        const n = this.nDim;

        // Initialize starting point away from boundaries
        let xBase = Array(n).fill(0).map(() => 0.2 + 0.6 * Math.random());
        let fBase = this.evaluate(xBase);

        // BOBYQA trust region parameters
        let rho = 0.1; // Initial trust region radius
        const rhoEnd = 1e-6;
        const eta1 = 0.1;  // Threshold for step rejection
        const eta2 = 0.7;  // Threshold for trust region expansion
        const gamma1 = 0.5; // Trust region contraction factor
        const gamma2 = 2.0; // Trust region expansion factor

        // Initialize interpolation set with bound-respecting points
        let xPts = [xBase.slice()];
        let fVals = [fBase];

        this.buildBoundedInterpolationSet(xPts, fVals, xBase, rho);

        // Find best point
        let kOpt = 0;
        let fOpt = fVals[0];
        for (let k = 0; k < fVals.length; k++) {
            if (fVals[k] < fOpt) {
                fOpt = fVals[k];
                kOpt = k;
            }
        }
        let xOpt = xPts[kOpt].slice();

        // Main BOBYQA loop
        while (this.evaluations < this.nTrials && rho > rhoEnd) {
            // Build quadratic model around current best point
            const model = this.buildBOBYQAQuadraticModel(xPts, fVals, xOpt);

            // Solve bound-constrained trust region subproblem
            const step = this.solveBoundedTrustRegion(model, xOpt, rho);

            if (this.evaluations >= this.nTrials) break;

            // Trial point automatically satisfies bounds due to trust region solve
            const xTrial = MathUtils.add(xOpt, step);
            const fTrial = this.evaluate(xTrial);

            // Compute predicted and actual reduction
            const predRed = this.computeBOBYQAPrediction(model, step);
            const actualRed = fOpt - fTrial;

            // Ratio test for trust region management
            const ratio = predRed > 0 ? actualRed / predRed : -1;

            // Update trust region radius
            let rhoNew = rho;
            if (ratio <= eta1) {
                rhoNew = gamma1 * rho;
            } else if (ratio >= eta2 && MathUtils.norm(step) > 0.8 * rho) {
                rhoNew = Math.min(gamma2 * rho, 1.0); // Cap at problem scale
            }

            // Accept step if good enough
            if (ratio > eta1) {
                xOpt = xTrial.slice();
                fOpt = fTrial;
                this.updateBOBYQAInterpolationSet(xPts, fVals, xOpt, fOpt);
            }

            rho = Math.max(rhoNew, rhoEnd);

            // Periodic geometry improvement for bound-constrained problems
            if (Math.random() < 0.1 && this.evaluations < this.nTrials - 3) {
                this.improveBoundedGeometry(xPts, fVals, xOpt, rho);
            }
        }

        return {
            bestValue: fOpt,
            bestX: xOpt,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    buildBoundedInterpolationSet(xPts, fVals, xBase, rho) {
        const n = this.nDim;

        // Add bound-respecting coordinate directions
        for (let i = 0; i < n && xPts.length < this.nPts && this.evaluations < this.nTrials; i++) {
            // Positive direction - respect upper bound
            const distToUpper = this.upperBounds[i] - xBase[i];
            const stepUp = Math.min(rho, distToUpper * 0.9);
            if (stepUp > 1e-8) {
                const xUp = xBase.slice();
                xUp[i] = xBase[i] + stepUp;
                xPts.push(xUp);
                fVals.push(this.evaluate(xUp));
            }

            if (xPts.length >= this.nPts || this.evaluations >= this.nTrials) break;

            // Negative direction - respect lower bound
            const distToLower = xBase[i] - this.lowerBounds[i];
            const stepDown = Math.min(rho, distToLower * 0.9);
            if (stepDown > 1e-8) {
                const xDown = xBase.slice();
                xDown[i] = xBase[i] - stepDown;
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

        // Bound-constrained trust region solve using projected gradient method
        let step = model.g.map(gi => -gi); // Steepest descent direction
        let stepNorm = MathUtils.norm(step);

        // Handle zero gradient
        if (stepNorm < 1e-12) {
            // Try to move away from any nearby bounds
            step = Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                const distToLower = xOpt[i] - this.lowerBounds[i];
                const distToUpper = this.upperBounds[i] - xOpt[i];
                const minDist = Math.min(distToLower, distToUpper);

                if (minDist < rho * 0.5) {
                    // Move away from closer bound
                    step[i] = distToLower < distToUpper ? rho * 0.3 : -rho * 0.3;
                } else {
                    step[i] = (Math.random() - 0.5) * 2 * rho * 0.1;
                }
            }
            stepNorm = MathUtils.norm(step);
        }

        // Scale to trust region
        if (stepNorm > rho) {
            step = MathUtils.scale(step, rho / stepNorm);
        }

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
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        const tolerance = 1e-6;
        const maxIterations = Math.floor(this.nTrials / (this.nDim * 2 + 5));

        for (let iter = 0; iter < maxIterations && this.evaluations < this.nTrials; iter++) {
            // Estimate gradient using finite differences
            const gradient = [];
            const h = 1e-5;

            for (let i = 0; i < this.nDim && this.evaluations < this.nTrials - 1; i++) {
                const xForward = [...x];
                const xBackward = [...x];
                xForward[i] = Math.min(1, x[i] + h);
                xBackward[i] = Math.max(0, x[i] - h);

                const fForward = this.evaluate(xForward);
                const fBackward = this.evaluate(xBackward);

                gradient[i] = (fForward - fBackward) / (2 * h);
            }

            if (this.evaluations >= this.nTrials) break;

            // Check gradient norm for convergence
            const gradNorm = MathUtils.norm(gradient);
            if (gradNorm < tolerance) break;

            // Simple gradient descent step with line search
            let stepSize = 0.01;
            let improved = false;

            for (let ls = 0; ls < 5 && this.evaluations < this.nTrials; ls++) {
                const xNew = x.map((xi, i) =>
                    MathUtils.clip(xi - stepSize * gradient[i], 0, 1)
                );
                const fxNew = this.evaluate(xNew);

                if (fxNew < fx) {
                    x = xNew;
                    fx = fxNew;
                    improved = true;
                    break;
                } else {
                    stepSize *= 0.5;
                }
            }

            if (!improved) {
                stepSize = 0.001; // Very small step as last resort
                const xNew = x.map((xi, i) =>
                    MathUtils.clip(xi - stepSize * gradient[i], 0, 1)
                );
                if (this.evaluations < this.nTrials) {
                    const fxNew = this.evaluate(xNew);
                    if (fxNew < fx) {
                        x = xNew;
                        fx = fxNew;
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
        const swarmSize = Math.min(30, Math.max(10, this.nDim * 2));
        const w = 0.7; // Inertia weight
        const c1 = 1.5; // Cognitive coefficient
        const c2 = 1.5; // Social coefficient

        // Initialize swarm
        const particles = [];
        for (let i = 0; i < swarmSize && this.evaluations < this.nTrials; i++) {
            const position = Array(this.nDim).fill(0).map(() => Math.random());
            const velocity = Array(this.nDim).fill(0).map(() => (Math.random() - 0.5) * 0.1);
            const fitness = this.evaluate(position);

            particles.push({
                position: [...position],
                velocity: [...velocity],
                bestPosition: [...position],
                bestFitness: fitness
            });
        }

        // PSO loop
        while (this.evaluations < this.nTrials) {
            for (let i = 0; i < particles.length && this.evaluations < this.nTrials; i++) {
                const particle = particles[i];

                // Update velocity and position
                for (let j = 0; j < this.nDim; j++) {
                    const r1 = Math.random();
                    const r2 = Math.random();

                    particle.velocity[j] = w * particle.velocity[j] +
                        c1 * r1 * (particle.bestPosition[j] - particle.position[j]) +
                        c2 * r2 * (this.bestX[j] - particle.position[j]);

                    particle.position[j] = MathUtils.clip(
                        particle.position[j] + particle.velocity[j], 0, 1
                    );
                }

                const fitness = this.evaluate(particle.position);

                // Update personal best
                if (fitness < particle.bestFitness) {
                    particle.bestFitness = fitness;
                    particle.bestPosition = [...particle.position];
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

// Simulated Annealing implementation
class SimulatedAnnealing extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'SimulatedAnnealing';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        const initialTemp = 10.0;
        const finalTemp = 0.01;
        const coolingRate = Math.pow(finalTemp / initialTemp, 1 / this.nTrials);
        let temperature = initialTemp;

        while (this.evaluations < this.nTrials) {
            // Generate neighbor
            const newX = x.map(xi => {
                const perturbation = (Math.random() - 0.5) * 0.1 * temperature / initialTemp;
                return MathUtils.clip(xi + perturbation, 0, 1);
            });

            const newFx = this.evaluate(newX);

            // Acceptance criterion
            const delta = newFx - fx;
            if (delta < 0 || Math.random() < Math.exp(-delta / temperature)) {
                x = newX;
                fx = newFx;
            }

            // Cool down
            temperature *= coolingRate;
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
        // Initial random samples
        const nInitial = Math.min(10, Math.floor(this.nTrials * 0.2));
        for (let i = 0; i < nInitial && this.evaluations < this.nTrials; i++) {
            const x = Array(this.nDim).fill(0).map(() => Math.random());
            const y = this.evaluate(x);
            this.observations.push({ x: [...x], y });
        }

        // Bayesian optimization loop
        while (this.evaluations < this.nTrials) {
            const nextX = this.acquireNext();
            const y = this.evaluate(nextX);
            this.observations.push({ x: [...nextX], y });
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

        // Simple expected improvement approximation
        const distances = this.observations.map(obs => ({
            dist: MathUtils.norm(MathUtils.subtract(x, obs.x)),
            y: obs.y
        }));

        distances.sort((a, b) => a.dist - b.dist);
        const kNearest = distances.slice(0, Math.min(5, distances.length));

        if (kNearest.length === 0) return Math.random();

        const meanY = kNearest.reduce((sum, item) => sum + item.y, 0) / kNearest.length;
        const uncertainty = 1.0 / (1.0 + kNearest[0].dist);
        const bestY = Math.min(...this.observations.map(obs => obs.y));
        const improvement = Math.max(0, bestY - meanY);

        return improvement + 0.1 * uncertainty;
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
            for (let i = 0; i < mu; i++) {
                for (let j = 0; j < this.nDim; j++) {
                    newMean[j] += normalizedWeights[i] * selected[i].x[j];
                }
            }
            mean = newMean;

            // Adapt step size (simplified)
            const improvement = selected.length > 0 ?
                (offspring[lambda - 1]?.fitness - selected[0].fitness) : 0;
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
class AdaptiveRandomSearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'AdaptiveRandomSearch';
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
            case 'AdaptiveRandomSearch':
                return new AdaptiveRandomSearch(objective, nTrials, nDim);
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
            'RandomSearch', 'BayesianOpt', 'CMAEvolutionStrategy', 'AdaptiveRandomSearch',
            'CoordinateDescent', 'PatternSearch', 'HillClimbing', 'TabuSearch', 'FireflyAlgorithm',
            'AntColonyOpt', 'HarmonySearch', 'EvolutionStrategy'
        ];
    }
};