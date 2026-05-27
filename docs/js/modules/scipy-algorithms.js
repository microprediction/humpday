/**
 * SciPy-based algorithm implementations: Nelder-Mead, Powell, and L-BFGS-B.
 *
 * These are JavaScript implementations of classical optimization algorithms
 * that are commonly found in SciPy. Well-established and reliable methods
 * for derivative-free and gradient-based optimization.
 *
 * Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html
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
}

class NelderMead extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'NelderMead';
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
            // Sort simplex by function values
            const indices = Array.from({length: n + 1}, (_, i) => i);
            indices.sort((i, j) => values[i] - values[j]);

            // Check convergence
            const best = values[indices[0]];
            const worst = values[indices[n]];
            if (worst - best < 1e-8) break;

            // Centroid of all points except worst
            const centroid = Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    centroid[j] += simplex[indices[i]][j];
                }
            }
            centroid.forEach((_, j) => centroid[j] /= n);

            // Reflection
            const worstPoint = simplex[indices[n]];
            const reflected = centroid.map((c, j) =>
                MathUtils.clip(c + alpha * (c - worstPoint[j]), 0, 1)
            );

            if (this.evaluations >= this.nTrials) break;
            const reflectedValue = this.evaluate(reflected);

            if (reflectedValue >= values[indices[0]] && reflectedValue < values[indices[n - 1]]) {
                // Accept reflection
                simplex[indices[n]] = reflected;
                values[indices[n]] = reflectedValue;
            } else if (reflectedValue < values[indices[0]]) {
                // Try expansion
                const expanded = centroid.map((c, j) =>
                    MathUtils.clip(c + gamma * (reflected[j] - c), 0, 1)
                );

                if (this.evaluations >= this.nTrials) {
                    simplex[indices[n]] = reflected;
                    values[indices[n]] = reflectedValue;
                    break;
                }

                const expandedValue = this.evaluate(expanded);

                if (expandedValue < reflectedValue) {
                    simplex[indices[n]] = expanded;
                    values[indices[n]] = expandedValue;
                } else {
                    simplex[indices[n]] = reflected;
                    values[indices[n]] = reflectedValue;
                }
            } else {
                // Contraction
                const contracted = centroid.map((c, j) =>
                    MathUtils.clip(c + rho * (worstPoint[j] - c), 0, 1)
                );

                if (this.evaluations >= this.nTrials) break;
                const contractedValue = this.evaluate(contracted);

                if (contractedValue < values[indices[n]]) {
                    simplex[indices[n]] = contracted;
                    values[indices[n]] = contractedValue;
                } else {
                    // Shrink
                    const best_point = simplex[indices[0]];
                    for (let i = 1; i <= n && this.evaluations < this.nTrials; i++) {
                        for (let j = 0; j < n; j++) {
                            simplex[indices[i]][j] = best_point[j] + sigma * (simplex[indices[i]][j] - best_point[j]);
                            simplex[indices[i]][j] = MathUtils.clip(simplex[indices[i]][j], 0, 1);
                        }
                        values[indices[i]] = this.evaluate(simplex[indices[i]]);
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

class Powell extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'Powell';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        // Initialize direction set (coordinate directions)
        let directions = Array(this.nDim).fill(0).map((_, i) => {
            const dir = Array(this.nDim).fill(0);
            dir[i] = 1;
            return dir;
        });

        while (this.evaluations < this.nTrials) {
            const x_start = [...x];
            const fx_start = fx;

            // Minimize along each direction
            for (let i = 0; i < this.nDim && this.evaluations < this.nTrials - 10; i++) {
                const result = this.lineSearch(x, directions[i]);
                x = result.x;
                fx = result.fx;
            }

            // Check for improvement
            if (Math.abs(fx - fx_start) < 1e-8) break;

            // Update direction set
            const newDirection = MathUtils.subtract(x, x_start);
            const norm = MathUtils.norm(newDirection);

            if (norm > 1e-12) {
                // Replace first direction with new direction
                for (let i = 0; i < this.nDim - 1; i++) {
                    directions[i] = [...directions[i + 1]];
                }
                directions[this.nDim - 1] = newDirection.map(d => d / norm);
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

    lineSearch(x0, direction) {
        let x = [...x0];
        let fx = this.objective(x);

        const stepSize = 0.1;
        let bestStep = 0;
        let bestX = [...x];
        let bestFx = fx;

        // Try different step sizes
        for (let step of [-stepSize * 2, -stepSize, stepSize, stepSize * 2]) {
            if (this.evaluations >= this.nTrials) break;

            const candidateX = x0.map((xi, i) =>
                MathUtils.clip(xi + step * direction[i], 0, 1)
            );

            const candidateFx = this.evaluate(candidateX);

            if (candidateFx < bestFx) {
                bestStep = step;
                bestX = candidateX;
                bestFx = candidateFx;
            }
        }

        return { x: bestX, fx: bestFx };
    }
}

class LBFGSB extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'LBFGSB';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);
        let grad = this.numericGradient(x);

        const m = Math.min(5, this.nDim); // Memory size
        const s_list = []; // Step vectors
        const y_list = []; // Gradient difference vectors

        while (this.evaluations < this.nTrials - this.nDim * 2) {
            // L-BFGS direction calculation
            let direction = [...grad].map(g => -g);

            // Two-loop recursion
            const alpha = [];
            for (let i = s_list.length - 1; i >= 0; i--) {
                const rho = 1 / MathUtils.dot(s_list[i], y_list[i]);
                alpha[i] = rho * MathUtils.dot(s_list[i], direction);
                direction = MathUtils.subtract(direction, MathUtils.scale(y_list[i], alpha[i]));
            }

            for (let i = 0; i < s_list.length; i++) {
                const rho = 1 / MathUtils.dot(s_list[i], y_list[i]);
                const beta = rho * MathUtils.dot(y_list[i], direction);
                direction = MathUtils.add(direction, MathUtils.scale(s_list[i], alpha[i] - beta));
            }

            // Line search with bounds
            const stepResult = this.boundedLineSearch(x, direction);
            const newX = stepResult.x;
            const newFx = stepResult.fx;
            const newGrad = this.numericGradient(newX);

            // Check convergence
            const gradNorm = MathUtils.norm(newGrad);
            if (gradNorm < 1e-6) break;

            // Update L-BFGS memory
            const s = MathUtils.subtract(newX, x);
            const y = MathUtils.subtract(newGrad, grad);

            if (MathUtils.dot(s, y) > 1e-12) {
                s_list.push(s);
                y_list.push(y);

                if (s_list.length > m) {
                    s_list.shift();
                    y_list.shift();
                }
            }

            x = newX;
            fx = newFx;
            grad = newGrad;
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    boundedLineSearch(x, direction) {
        let bestX = [...x];
        let bestFx = this.evaluate(x);  // FIXED: Use evaluate() not objective()

        const stepSizes = [0.001, 0.01, 0.1];

        for (const step of stepSizes) {
            if (this.evaluations >= this.nTrials) break;

            const candidateX = x.map((xi, i) =>
                MathUtils.clip(xi + step * direction[i], 0, 1)
            );

            const candidateFx = this.evaluate(candidateX);

            if (candidateFx < bestFx) {
                bestX = candidateX;
                bestFx = candidateFx;
            }
        }

        return { x: bestX, fx: bestFx };
    }

    numericGradient(x) {
        const gradient = Array(this.nDim).fill(0);
        const h = 1e-6;

        for (let i = 0; i < this.nDim; i++) {
            if (this.evaluations >= this.nTrials - 1) break;

            // Forward difference
            const xForward = [...x];
            xForward[i] = Math.min(1, x[i] + h);
            const fForward = this.evaluate(xForward);

            const xBackward = [...x];
            xBackward[i] = Math.max(0, x[i] - h);
            const fBackward = this.evaluate(xBackward);

            gradient[i] = (fForward - fBackward) / (2 * h);
        }

        return gradient;
    }
}

// Export SciPy algorithms - placed at end after all class definitions
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = { NelderMead, Powell, LBFGSB };
} else {
    // Browser environment
    window.NelderMead = NelderMead;
    window.Powell = Powell;
    window.LBFGSB = LBFGSB;
}