/**
 * SciPy-based algorithm implementations: Nelder-Mead, Powell, and L-BFGS-B.
 *
 * These are JavaScript implementations of classical optimization algorithms
 * that are commonly found in SciPy. Well-established and reliable methods
 * for derivative-free and gradient-based optimization.
 *
 * Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html
 */

// Import base classes and utilities
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    const { Optimizer, MathUtils } = require('./base-optimizer.js');
} else {
    // Browser environment - base classes already loaded
}
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

// Export SciPy algorithms
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = { NelderMead, Powell, LBFGSB };
} else {
    // Browser environment
    window.NelderMead = NelderMead;
    window.Powell = Powell;
    window.LBFGSB = LBFGSB;
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
