/**
 * Search algorithm implementations.
 *
 * These algorithms include various search and local optimization methods
 * that systematically explore the search space using different strategies.
 */

// Import base classes and utilities
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    const { Optimizer, MathUtils } = require('./base-optimizer.js');
} else {
    // Browser environment - base classes already loaded
}

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

// Export search algorithms
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = { AdaptiveRandomSearch, CoordinateDescent, PatternSearch, HillClimbing };
} else {
    // Browser environment
    window.AdaptiveRandomSearch = AdaptiveRandomSearch;
    window.CoordinateDescent = CoordinateDescent;
    window.PatternSearch = PatternSearch;
    window.HillClimbing = HillClimbing;
}