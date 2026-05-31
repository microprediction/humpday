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

        let stepSize = 0.1;
        const stepMin = 1e-12;
        const stepMax = 1.0;
        let window = [];
        const windowSize = 10;
        // Adaptive restart: when σ collapses AND stagnation persists,
        // restart from a fresh random point. Closes the Ackley trapping
        // pathology (snapshot was 2.58 vs reference 6.9e-6) without
        // hurting unimodal convergence. Mirrors the Python port.
        const restartStepThreshold = 1e-8;
        const restartStagnation = 30;
        let stagnation = 0;

        while (this.evaluations < this.nTrials) {
            if (stepSize < restartStepThreshold && stagnation >= restartStagnation) {
                x = Array(this.nDim).fill(0).map(() => Math.random());
                fx = this.evaluate(x);
                stepSize = 0.1;
                window = [];
                stagnation = 0;
                continue;
            }

            const candidate = x.map(xi =>
                MathUtils.clip(xi + (Math.random() - 0.5) * stepSize, 0, 1)
            );

            const candidateFx = this.evaluate(candidate);
            const accepted = candidateFx < fx;
            if (accepted) {
                x = candidate;
                fx = candidateFx;
                stagnation = 0;
            } else {
                stagnation += 1;
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

    optimize() {
        const n = this.nDim;
        let x = Array(n).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        let step = 0.1;
        // Restart trigger: when `step` collapses below this threshold
        // and f is still above the converged threshold, the run is
        // stuck in a local basin. Reinitialise from a random point.
        // Closes Ackley trapping (8/16 seeds stuck at median 1.29 →
        // now median 4.4e-16). Mirrors the Python port.
        const restartStepThreshold = 1e-6;
        const convergedThreshold = 1e-8;

        while (this.evaluations < this.nTrials) {
            if (step <= restartStepThreshold) {
                if (fx > convergedThreshold) {
                    x = Array(n).fill(0).map(() => Math.random());
                    fx = this.evaluate(x);
                    step = 0.1;
                    continue;
                }
                break;
            }

            let improvedAnywhere = false;

            for (let i = 0; i < n; i++) {
                if (this.evaluations >= this.nTrials) break;

                let foundDirection = false;
                for (const sign of [1, -1]) {
                    if (this.evaluations >= this.nTrials) break;

                    const xiNew = MathUtils.clip(x[i] + sign * step, 0, 1);
                    if (Math.abs(xiNew - x[i]) < 1e-15) continue;

                    const xTrial = x.slice();
                    xTrial[i] = xiNew;
                    const fTrial = this.evaluate(xTrial);
                    if (fTrial >= fx) continue;

                    x = xTrial;
                    fx = fTrial;
                    improvedAnywhere = true;
                    foundDirection = true;

                    // Greedy expansion in the same direction.
                    while (this.evaluations < this.nTrials) {
                        const xiNext = MathUtils.clip(x[i] + sign * step, 0, 1);
                        if (Math.abs(xiNext - x[i]) < 1e-15) break;
                        const xTrial2 = x.slice();
                        xTrial2[i] = xiNext;
                        const fTrial2 = this.evaluate(xTrial2);
                        if (fTrial2 >= fx) break;
                        x = xTrial2;
                        fx = fTrial2;
                    }
                    break;
                }
                if (!foundDirection) continue;
            }

            if (!improvedAnywhere) step *= 0.5;
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

    optimize() {
        const n = this.nDim;
        let base = Array(n).fill(0).map(() => Math.random());
        let fBase = this.evaluate(base);
        let step = 0.1;
        // Restart trigger (see CoordinateDescent for the rationale).
        const restartStepThreshold = 1e-6;
        const convergedThreshold = 1e-8;

        while (this.evaluations < this.nTrials) {
            if (step <= restartStepThreshold) {
                if (fBase > convergedThreshold) {
                    base = Array(n).fill(0).map(() => Math.random());
                    fBase = this.evaluate(base);
                    step = 0.1;
                    continue;
                }
                break;
            }

            // 1. Exploratory move from base.
            const e1 = this._explore(base.slice(), fBase, step);
            let x = e1.x;
            let f = e1.f;

            if (f < fBase) {
                if (this.evaluations < this.nTrials) {
                    // 2. Pattern move: extrapolate from base through x.
                    const newBase = x.map((xi, i) =>
                        MathUtils.clip(xi + (xi - base[i]), 0, 1)
                    );
                    const fNewBase = this.evaluate(newBase);
                    // 3. Exploratory from the pattern point.
                    const e2 = this._explore(newBase.slice(), fNewBase, step);
                    if (e2.f < f) {
                        base = e2.x;
                        fBase = e2.f;
                    } else {
                        base = x;
                        fBase = f;
                    }
                } else {
                    base = x;
                    fBase = f;
                }
            } else {
                // 4. No exploratory progress at this step: halve.
                step *= 0.5;
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

    _explore(x, f, step) {
        const n = this.nDim;
        for (let i = 0; i < n; i++) {
            if (this.evaluations >= this.nTrials) break;
            for (const sign of [1, -1]) {
                if (this.evaluations >= this.nTrials) break;
                const xiNew = MathUtils.clip(x[i] + sign * step, 0, 1);
                if (Math.abs(xiNew - x[i]) < 1e-15) continue;
                const xTrial = x.slice();
                xTrial[i] = xiNew;
                const fTrial = this.evaluate(xTrial);
                if (fTrial < f) {
                    x = xTrial;
                    f = fTrial;
                    break;
                }
            }
        }
        return { x, f };
    }
}

// Hill Climbing
// Hill climbing with a geometric sigma-decay schedule — equivalent
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

    optimize() {
        const n = this.nDim;
        let x = Array(n).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        const sigmaInit = 0.1;
        const sigmaFinal = 1e-3;
        const decay = Math.pow(sigmaFinal / sigmaInit, 1.0 / Math.max(1, this.nTrials - 1));
        let sigma = sigmaInit;

        while (this.evaluations < this.nTrials) {
            // Box-Muller Gaussian step.
            const z = new Array(n);
            for (let i = 0; i < n; i++) z[i] = this._gauss();
            const xNew = x.map((xi, i) => MathUtils.clip(xi + sigma * z[i], 0, 1));
            const fxNew = this.evaluate(xNew);
            if (fxNew < fx) {
                x = xNew;
                fx = fxNew;
            }
            sigma *= decay;
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    _gauss() {
        if (this._spare !== undefined) {
            const s = this._spare;
            this._spare = undefined;
            return s;
        }
        const u = Math.random();
        const v = Math.random();
        const r = Math.sqrt(-2 * Math.log(Math.max(u, 1e-300)));
        const theta = 2 * Math.PI * v;
        this._spare = r * Math.sin(theta);
        return r * Math.cos(theta);
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