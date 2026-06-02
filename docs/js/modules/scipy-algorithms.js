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

        // Nelder-Mead parameters (scipy defaults).
        const alpha = 1.0;    // reflection
        const gamma = 2.0;    // expansion
        const rho = 0.5;      // contraction
        const sigma = 0.5;    // shrink

        // Tighter convergence than scipy's 1e-4 default so the algorithm
        // uses its budget on easy landscapes instead of terminating early
        // (matches Python; was 1e-8 in the prior JS port).
        const xatol = 1e-12;
        const fatol = 1e-12;
        const zdelt = 0.00025;

        // Kelley (1999) "Detection and Remediation of Stagnation in the
        // Nelder-Mead Algorithm" showed vanilla NM can converge to a
        // non-stationary point on a collapsed simplex. We wrap the
        // classical loop in a restart layer: when the convergence test
        // fires, we reseed the simplex and continue until the budget is
        // exhausted. Different `nonzdelt` per restart so the new simplex
        // isn't a scaled copy of the collapsed one; even restarts reseed
        // around the current best (intensification), odd restarts reseed
        // from a fresh uniform draw (diversification).
        const nonzdeltSchedule = [0.05, 0.15, 0.30, 0.10, 0.50, 0.20];

        let seedPoint = Array(n).fill(0).map(() => 0.3 + 0.4 * Math.random());
        let restartCount = 0;

        while (this.evaluations < this.nTrials) {
            const nonzdelt = nonzdeltSchedule[restartCount % nonzdeltSchedule.length];

            // (Re)build the simplex around seedPoint with this restart's
            // perturbation magnitude.
            const simplex = [];
            const values = [];

            simplex.push([...seedPoint]);
            for (let k = 0; k < n; k++) {
                const y = [...seedPoint];
                if (y[k] !== 0) {
                    y[k] = (1 + nonzdelt) * y[k];
                } else {
                    y[k] = zdelt;
                }
                simplex.push(y);
            }
            for (let i = 0; i < simplex.length; i++) {
                for (let j = 0; j < n; j++) {
                    simplex[i][j] = MathUtils.clip(simplex[i][j], 0, 1);
                }
            }

            for (let k = 0; k < n + 1; k++) {
                if (this.evaluations >= this.nTrials) break;
                values.push(this.evaluate(simplex[k]));
            }
            if (this.evaluations >= this.nTrials) break;

            // Inner NM loop — runs until simplex collapses (then breaks
            // out so the outer loop reseeds), or the budget is exhausted.
            let collapsed = false;
            while (this.evaluations < this.nTrials && !collapsed) {
                // Sort simplex by fitness ascending (best first).
                const indices = Array.from({length: n + 1}, (_, i) => i);
                indices.sort((i, j) => values[i] - values[j]);

                // Convergence: max coordinate spread and max f-spread
                // among non-best vertices are both below tolerance.
                let xMax = 0;
                for (let i = 1; i < n + 1; i++) {
                    for (let k = 0; k < n; k++) {
                        const d = Math.abs(simplex[indices[i]][k] - simplex[indices[0]][k]);
                        if (d > xMax) xMax = d;
                    }
                }
                let fMax = 0;
                for (let i = 1; i < n + 1; i++) {
                    const d = Math.abs(values[indices[0]] - values[indices[i]]);
                    if (d > fMax) fMax = d;
                }
                if (xMax <= xatol && fMax <= fatol) {
                    collapsed = true;
                    break;
                }

                // Centroid of the best n vertices.
                const centroid = Array(n).fill(0);
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        centroid[j] += simplex[indices[i]][j];
                    }
                }
                for (let j = 0; j < n; j++) centroid[j] /= n;

                const worstPoint = simplex[indices[n]];
                const reflected = centroid.map((c, j) =>
                    MathUtils.clip(c + alpha * (c - worstPoint[j]), 0, 1)
                );

                if (this.evaluations >= this.nTrials) break;
                const reflectedValue = this.evaluate(reflected);

                if (reflectedValue >= values[indices[0]] && reflectedValue < values[indices[n - 1]]) {
                    simplex[indices[n]] = reflected;
                    values[indices[n]] = reflectedValue;
                } else if (reflectedValue < values[indices[0]]) {
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
                    const contracted = centroid.map((c, j) =>
                        MathUtils.clip(c + rho * (worstPoint[j] - c), 0, 1)
                    );
                    if (this.evaluations >= this.nTrials) break;
                    const contractedValue = this.evaluate(contracted);
                    if (contractedValue < values[indices[n]]) {
                        simplex[indices[n]] = contracted;
                        values[indices[n]] = contractedValue;
                    } else {
                        // Shrink: every non-best vertex moves toward best.
                        const bestPoint = simplex[indices[0]];
                        for (let i = 1; i <= n && this.evaluations < this.nTrials; i++) {
                            for (let j = 0; j < n; j++) {
                                simplex[indices[i]][j] = MathUtils.clip(
                                    bestPoint[j] + sigma * (simplex[indices[i]][j] - bestPoint[j]),
                                    0, 1
                                );
                            }
                            values[indices[i]] = this.evaluate(simplex[indices[i]]);
                        }
                    }
                }
            }

            // Inner loop ended — pick next restart's seed.
            restartCount++;
            if (this.evaluations >= this.nTrials) break;

            const sortedFinal = Array.from({length: n + 1}, (_, i) => i)
                .sort((i, j) => values[i] - values[j]);
            if (restartCount % 2 === 1) {
                seedPoint = [...simplex[sortedFinal[0]]];  // intensification
            } else {
                seedPoint = Array(n).fill(0).map(() => Math.random());  // diversification
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
        let fx = this.evaluate(x);

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
        // Seed bestX with a random starting point and delegate to the
        // proper L-BFGS-B port (shared with DE/SA polish on the base
        // Optimizer class): two-loop recursion + bound-aware direction
        // projection + projected-gradient pgtol + factr·eps_mach
        // termination + feasibility-capped Armijo line search.
        this.bestX = Array(this.nDim).fill(0).map(() => Math.random());
        this.bestValue = this.evaluate(this.bestX);
        this._lbfgsPolish();
        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
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