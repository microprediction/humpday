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

    *_run() {
        // Statement-for-statement twin of Rechenberg._run in
        // humpday/optimizers/search_algorithms.py (same draw order:
        // one uniform vector, then per-iteration one normal vector).
        const stepMax = 1.0;
        const stepMin = 1e-12;
        const windowSize = 10;

        let x = MathUtils.randomUniform(this.nDim);
        let f = yield x;
        let sigma = 0.1;
        const window = [];

        while (this.evaluations < this.nTrials) {
            const z = MathUtils.randomNormal(this.nDim);
            const xNew = MathUtils.clipArray(x.map((xi, i) => xi + sigma * z[i]), 0, 1);

            if (this.evaluations >= this.nTrials) break;
            const fNew = yield xNew;

            const accepted = fNew < f;
            if (accepted) {
                x = xNew;
                f = fNew;
            }
            window.push(accepted);
            if (window.length > windowSize) window.shift();

            if (window.length >= windowSize) {
                let rate = 0.0;
                for (const w of window) rate += w ? 1.0 : 0.0;
                rate /= windowSize;
                if (rate > 1 / 5) {
                    sigma = Math.min(stepMax, sigma * 1.5);
                } else if (rate < 1 / 5) {
                    sigma = Math.max(stepMin, sigma / 1.5);
                }
            }
        }
    }
}

// Backwards-compatibility alias — the class was renamed in this commit.
const AdaptiveRandomSearch = Rechenberg;

// Coordinate Descent
// Coordinate descent with an adaptive expanding line search per axis.
// For each coordinate i, take a step ± step; if it improves, keep
// stepping in the same direction until it stops improving. After a
// full sweep over all coordinates with no improvement, halve the
// step. Mirrors the Python port; replaces the previous fixed-shrink
// (× 0.8, floor 0.001) variant that could never refine below ~1e-3.
class CoordinateDescent extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'CoordinateDescent';
    }

    *_run() {
        // Statement-for-statement twin of CoordinateDescent._run in
        // humpday/optimizers/search_algorithms.py.
        const n = this.nDim;
        let x = MathUtils.randomUniform(n);
        let f = yield x;

        let step = 0.1;
        const restartStepThreshold = 1e-6;
        const convergedThreshold = 1e-8;

        while (this.evaluations < this.nTrials) {
            if (step <= restartStepThreshold) {
                if (f > convergedThreshold) {
                    x = MathUtils.randomUniform(n);
                    f = yield x;
                    step = 0.1;
                    continue;
                }
                break;
            }

            let improvedAnywhere = false;

            for (let i = 0; i < n; i++) {
                if (this.evaluations >= this.nTrials) break;

                for (const sign of [1, -1]) {
                    if (this.evaluations >= this.nTrials) break;

                    const xiNew = Math.max(0.0, Math.min(1.0, x[i] + sign * step));
                    if (Math.abs(xiNew - x[i]) < 1e-15) continue;
                    const xTrial = x.slice();
                    xTrial[i] = xiNew;
                    const fTrial = yield xTrial;
                    if (fTrial >= f) continue;

                    x = xTrial;
                    f = fTrial;
                    improvedAnywhere = true;

                    while (this.evaluations < this.nTrials) {
                        const xiNext = Math.max(0.0, Math.min(1.0, x[i] + sign * step));
                        if (Math.abs(xiNext - x[i]) < 1e-15) break;
                        const xTrial2 = x.slice();
                        xTrial2[i] = xiNext;
                        const fTrial2 = yield xTrial2;
                        if (fTrial2 >= f) break;
                        x = xTrial2;
                        f = fTrial2;
                    }

                    break;
                }
            }

            if (!improvedAnywhere) step *= 0.5;
        }
    }
}

// Classic Hooke-Jeeves pattern search (1961): exploratory sweep
// from a base point, then a *pattern move* (extrapolation through
// the new point) followed by another exploratory from the
// extrapolated point. Step halves on failed sweep; floor 1e-12.
// Mirrors the Python port and the textbook reference.
class PatternSearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'PatternSearch';
    }

    *_run() {
        // Twin of PatternSearch._run (Hooke-Jeeves) in
        // humpday/optimizers/search_algorithms.py.
        let base = MathUtils.randomUniform(this.nDim);
        let fBase = yield base;
        let step = 0.1;
        const restartStepThreshold = 1e-6;
        const convergedThreshold = 1e-8;

        while (this.evaluations < this.nTrials) {
            if (step <= restartStepThreshold) {
                if (fBase > convergedThreshold) {
                    base = MathUtils.randomUniform(this.nDim);
                    fBase = yield base;
                    step = 0.1;
                    continue;
                }
                break;
            }

            let [x, f] = yield* this._exploreGen(base.slice(), fBase, step);

            if (f < fBase) {
                if (this.evaluations < this.nTrials) {
                    const newBase = MathUtils.clipArray(
                        x.map((xi, i) => xi + (xi - base[i])), 0, 1
                    );
                    const fNewBase = yield newBase;
                    const [x2, f2] = yield* this._exploreGen(newBase.slice(), fNewBase, step);
                    if (f2 < f) {
                        base = x2; fBase = f2;
                    } else {
                        base = x; fBase = f;
                    }
                } else {
                    base = x; fBase = f;
                }
            } else {
                step *= 0.5;
            }
        }
    }

    *_exploreGen(x, f, step) {
        for (let i = 0; i < this.nDim; i++) {
            if (this.evaluations >= this.nTrials) break;
            for (const sign of [1, -1]) {
                if (this.evaluations >= this.nTrials) break;
                const xiNew = Math.max(0.0, Math.min(1.0, x[i] + sign * step));
                if (Math.abs(xiNew - x[i]) < 1e-15) continue;
                const xTrial = x.slice();
                xTrial[i] = xiNew;
                const fTrial = yield xTrial;
                if (fTrial < f) {
                    x = xTrial;
                    f = fTrial;
                    break;
                }
            }
        }
        return [x, f];
    }
}

// to a (1+1)-Evolution Strategy with a deterministic step-size
// schedule. Step size decays from sigma_init = 0.1 to sigma_final =
// 1e-3 over the budget. Mirrors the Python port and the textbook
// reference; the previous fixed-step uniform-perturbation variant
// could not refine below ~1e-2.
class HillClimbing extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'HillClimbing';
    }

    *_run() {
        // Twin of HillClimbing._run in
        // humpday/optimizers/evolutionary_algorithms.py.
        const n = this.nDim;
        let x = MathUtils.randomUniform(n);
        let fx = yield x;

        const sigmaInit = 0.1;
        const sigmaFinal = 1e-3;
        // portableExp/portableLog on BOTH sides — Math.pow diverged from
        // CPython's ** by one ulp on Linux (caught by the vector replay).
        const decay = MathUtils.portableExp(
            (1.0 / Math.max(1, this.nTrials - 1)) * MathUtils.portableLog(sigmaFinal / sigmaInit)
        );
        let sigma = sigmaInit;

        while (this.evaluations < this.nTrials) {
            const z = MathUtils.randomNormal(n);
            const xNew = MathUtils.clipArray(x.map((xi, i) => xi + sigma * z[i]), 0, 1);
            const fxNew = yield xNew;
            if (fxNew < fx) {
                x = xNew;
                fx = fxNew;
            }
            sigma *= decay;
        }
    }
}

// Regular-grid baseline. Like RandomSearch, this is included as a
// baseline (regression check, contest sanity floor), not as a SOTA
// algorithm. Grid size scales as `nPerAxis^nDim`; practically useful
// for nDim <= 3.
class GridSearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'GridSearch';
    }

    *_run() {
        // Twin of GridSearch._run in humpday/optimizers/search_algorithms.py.
        const n = this.nDim;
        const nPerAxis = Math.max(2, Math.round(Math.pow(this.nTrials, 1.0 / n)));
        const indices = new Array(n).fill(0);
        while (this.evaluations < this.nTrials) {
            yield indices.map(i => (i + 0.5) / nPerAxis);
            let d = n - 1;
            while (d >= 0) {
                indices[d]++;
                if (indices[d] < nPerAxis) break;
                indices[d] = 0;
                d--;
            }
            if (d < 0) break;
        }
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
        GridSearch,
    };
} else {
    // Browser environment
    window.Rechenberg = Rechenberg;
    window.AdaptiveRandomSearch = AdaptiveRandomSearch;  // backwards-compat alias
    window.CoordinateDescent = CoordinateDescent;
    window.PatternSearch = PatternSearch;
    window.HillClimbing = HillClimbing;
    window.GridSearch = GridSearch;
}