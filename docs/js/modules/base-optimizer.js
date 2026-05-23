/**
 * Base optimizer class and utilities for browser-based optimization algorithms.
 *
 * Provides common functionality for objective evaluation, best value tracking,
 * and path visualization support shared by all optimization algorithms.
 */

// Mathematical utility functions
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

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = { Optimizer, MathUtils };
} else {
    // Browser environment
    window.Optimizer = Optimizer;
    window.MathUtils = MathUtils;
}