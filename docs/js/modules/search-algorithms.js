/**
 * Search algorithm implementations.
 *
 * These algorithms include various search and local optimization methods
 * that systematically explore the search space using different strategies.
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

// Rechenberg's (1+1)-ES with the 1/5-success-rule. Was previously
// named `AdaptiveRandomSearch`; the new name matches the canonical
// literature reference and the win-rate parity comparison. The legacy
// name is exported below as a backwards-compatibility alias.
class Rechenberg extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'Rechenberg';
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        // Step bounds: 1e-12 floor lets the algorithm refine to machine
        // precision on smooth basins (the previous 0.01 floor was the
        // entire reason this port was ~5.7e+08× off the reference on
        // the sphere benchmark — the algorithm worked, the floor just
        // capped its precision at ~1e-4).
        let stepSize = 0.1;
        const stepMin = 1e-12;
        const stepMax = 1.0;
        const window = [];
        const windowSize = 10;

        while (this.evaluations < this.nTrials) {
            const candidate = x.map(xi =>
                MathUtils.clip(xi + (Math.random() - 0.5) * stepSize, 0, 1)
            );

            const candidateFx = this.evaluate(candidate);
            const accepted = candidateFx < fx;
            if (accepted) {
                x = candidate;
                fx = candidateFx;
            }
            window.push(accepted ? 1 : 0);
            if (window.length > windowSize) window.shift();

            // Strict 1/5-rule on the rolling window.
            if (window.length >= windowSize) {
                const rate = window.reduce((s, v) => s + v, 0) / windowSize;
                if (rate > 1 / 5) {
                    stepSize = Math.min(stepMax, stepSize * 1.5);
                } else if (rate < 1 / 5) {
                    stepSize = Math.max(stepMin, stepSize / 1.5);
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

// Backwards-compatibility alias — the class was renamed in this commit.
const AdaptiveRandomSearch = Rechenberg;

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

// Export search algorithms
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = {
        Rechenberg,
        AdaptiveRandomSearch,  // backwards-compat alias
        CoordinateDescent,
        PatternSearch,
        HillClimbing,
    };
} else {
    // Browser environment
    window.Rechenberg = Rechenberg;
    window.AdaptiveRandomSearch = AdaptiveRandomSearch;  // backwards-compat alias
    window.CoordinateDescent = CoordinateDescent;
    window.PatternSearch = PatternSearch;
    window.HillClimbing = HillClimbing;
}