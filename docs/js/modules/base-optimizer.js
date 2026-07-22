/**
 * Base optimizer class and utilities for browser-based optimization algorithms.
 *
 * Provides common functionality for objective evaluation, best value tracking,
 * and path visualization support shared by all optimization algorithms.
 */

// Portable RNG plumbing. Legacy default is Math.random (unseedable,
// fine for demos); usePortableRng(seed, seq) switches every converted
// optimizer onto the cross-language PCG32 stream (prng.js), which is
// what the transition-vector replay uses. Mirrors humpday._array's
// use_portable_rng / use_legacy_rng.
const _PCG32 = (typeof module !== 'undefined' && module.exports)
    ? require('./prng.js').PCG32
    : (typeof PCG32 !== 'undefined' ? PCG32 : null);

let _portableRng = null;

function usePortableRng(seed, seq = 0) {
    _portableRng = new _PCG32(seed, seq);
    return _portableRng;
}

function useLegacyRng() {
    _portableRng = null;
}

// Mathematical utility functions
const MathUtils = {
    random: () => Math.random(),

    randomScalar() {
        return _portableRng !== null ? _portableRng.random() : Math.random();
    },

    randomUniform(n) {
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = MathUtils.randomScalar();
        return out;
    },

    gaussScalar() {
        if (_portableRng !== null) return _portableRng.gauss();
        // Legacy fallback: Box-Muller on Math.random (statistical use only).
        const u1 = Math.max(Math.random(), 1e-300);
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    },

    randomNormal(n) {
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = MathUtils.gaussScalar();
        return out;
    },

    randInt(n) {
        return _portableRng !== null
            ? _portableRng.randbelow(n)
            : Math.floor(Math.random() * n);
    },

    choice(seq) {
        return seq[MathUtils.randInt(seq.length)];
    },

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
        // Same stream position as Python: BaseOptimizer.__init__ draws
        // best_x from the RNG (n draws) before the run starts.
        this.bestX = MathUtils.randomUniform(nDim);
        this.trackPath = false;
        this.path = [];
    }

    _bookkeep(clippedX, value) {
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

    evaluate(x) {
        this.evaluations++;
        const clippedX = MathUtils.clipArray(x, 0, 1);
        const value = this.objective(clippedX);
        return this._bookkeep(clippedX, value);
    }

    // ------------------------------------------------------------------ //
    // Online (generator) protocol — the JS twin of BaseOptimizer._run    //
    // driving in humpday/optimizers/base.py. A converted class defines   //
    // *_run() and drops its optimize() override; the driver owns the     //
    // increment/clip/objective/bookkeep order so trajectories replay the //
    // Python transition vectors bit-for-bit under usePortableRng().      //
    // ------------------------------------------------------------------ //
    optimize() {
        if (typeof this._run !== 'function') {
            throw new Error(`${this.constructor.name} defines neither optimize() nor _run()`);
        }
        const gen = this._run();
        let res = gen.next();
        while (!res.done) {
            this.evaluations++;
            const clippedX = MathUtils.clipArray(res.value, 0, 1);
            const value = this.objective(clippedX);
            this._bookkeep(clippedX, value);
            res = gen.next(value);
        }
        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    // Threadless scalar ask/tell over _run (subset of the Python surface;
    // batch view arrives with the CMA-ES conversion).
    suggestNext() {
        if (typeof this._run !== 'function') {
            throw new Error(`${this.constructor.name} has no _run(); ask/tell unavailable`);
        }
        if (!this._gd) {
            const gen = this._run();
            const res = gen.next();
            this._gd = {
                gen,
                done: res.done,
                pending: res.done ? null : MathUtils.clipArray(res.value, 0, 1),
                awaiting: false,
            };
        }
        const gd = this._gd;
        if (gd.done) return null;
        if (gd.awaiting) throw new Error('suggestNext() called again before receiveUpdate()');
        gd.awaiting = true;
        return gd.pending;
    }

    receiveUpdate(value) {
        const gd = this._gd;
        if (!gd || !gd.awaiting) throw new Error('receiveUpdate() without a matching suggestNext()');
        gd.awaiting = false;
        this.evaluations++;
        this._bookkeep(gd.pending, value);
        const res = gd.gen.next(value);
        if (res.done) {
            gd.done = true;
            gd.pending = null;
        } else {
            gd.pending = MathUtils.clipArray(res.value, 0, 1);
        }
    }

    // ------------------------------------------------------------------ //
    // Shared L-BFGS-B (Byrd-Lu-Nocedal-Zhu 1995, simple-bounds form).   //
    // ------------------------------------------------------------------ //
    // Used by DifferentialEvolution (matches scipy DE `polish=True`),
    // SimulatedAnnealing (matches scipy.dual_annealing's local-search
    // stage), and the standalone LBFGSB optimizer.
    //
    // Faithful pure-JS port of scipy's _minimize_lbfgsb with four
    // elements the previous "two-loop + Armijo" did not have:
    //   1. Bound-aware direction projection at active bounds.
    //   2. Projected-gradient sup-norm for the pgtol convergence test.
    //   3. f-tolerance termination via factr · eps_mach.
    //   4. Feasibility-capped initial step so the line search doesn't
    //      waste iterations on clipped candidates.
    // Mirrors `BaseOptimizer._lbfgs_polish` in
    // `humpday/optimizers/base.py`.
    _lbfgsPolish() {
        const FACTR = 1e7;
        const PGTOL = 1e-5;
        const EPS_MACH = 2.220446049250313e-16;
        const MEMORY = Math.min(10, Math.max(1, this.nDim));
        const n = this.nDim;

        let x = this.bestX.slice();
        let f = this.bestValue;
        let grad = this._fdGradientForPolish(x);

        const sList = [];
        const yList = [];

        while (this.evaluations < this.nTrials - 2 * n) {
            // (1) Projected-gradient convergence test.
            if (this._projGradSupNorm(x, grad) < PGTOL) break;

            // (2) Two-loop recursion.
            let direction = this._lbfgsTwoLoop(grad, sList, yList);

            // (3) Project direction at active bounds.
            for (let k = 0; k < n; k++) {
                if (x[k] <= 0.0 && direction[k] < 0.0) direction[k] = 0.0;
                else if (x[k] >= 1.0 && direction[k] > 0.0) direction[k] = 0.0;
            }

            let gd = 0;
            for (let k = 0; k < n; k++) gd += grad[k] * direction[k];
            if (gd > -1e-30) {
                // Fall back to projected steepest descent.
                sList.length = 0;
                yList.length = 0;
                direction = grad.map(g => -g);
                for (let k = 0; k < n; k++) {
                    if (x[k] <= 0.0 && direction[k] < 0.0) direction[k] = 0.0;
                    else if (x[k] >= 1.0 && direction[k] > 0.0) direction[k] = 0.0;
                }
                gd = 0;
                for (let k = 0; k < n; k++) gd += grad[k] * direction[k];
                if (gd > -1e-30) break;
            }

            // (4) Cap step length to feasibility.
            let stepMax = Infinity;
            for (let k = 0; k < n; k++) {
                const dk = direction[k];
                if (dk > 0.0) stepMax = Math.min(stepMax, (1.0 - x[k]) / dk);
                else if (dk < 0.0) stepMax = Math.min(stepMax, (0.0 - x[k]) / dk);
            }
            let step = stepMax > 0.0 ? Math.min(1.0, stepMax) : 1.0;

            // (5) Armijo backtracking.
            const c1 = 1e-4;
            let newX = x.slice();
            let newF = f;
            let accepted = false;
            while (step > 1e-12) {
                if (this.evaluations >= this.nTrials) break;
                const candidate = new Array(n);
                for (let k = 0; k < n; k++) {
                    candidate[k] = Math.max(0, Math.min(1, x[k] + step * direction[k]));
                }
                const candF = this.evaluate(candidate);
                if (candF <= f + c1 * step * gd) {
                    newX = candidate;
                    newF = candF;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) break;

            // (6) f-tolerance termination.
            const fScale = Math.max(Math.abs(f), Math.abs(newF), 1.0);
            if ((f - newF) < FACTR * EPS_MACH * fScale) {
                x = newX; f = newF;
                break;
            }

            const newGrad = this._fdGradientForPolish(newX);

            // (7) Memory update.
            const s = new Array(n);
            const y = new Array(n);
            for (let k = 0; k < n; k++) {
                s[k] = newX[k] - x[k];
                y[k] = newGrad[k] - grad[k];
            }
            let sy = 0;
            for (let k = 0; k < n; k++) sy += s[k] * y[k];
            if (sy > 1e-12) {
                sList.push(s);
                yList.push(y);
                if (sList.length > MEMORY) { sList.shift(); yList.shift(); }
            }
            x = newX; f = newF; grad = newGrad;
        }
    }

    _lbfgsTwoLoop(grad, sList, yList) {
        const n = grad.length;
        let direction = grad.map(g => -g);
        const alpha = new Array(sList.length).fill(0);
        for (let i = sList.length - 1; i >= 0; i--) {
            let sy = 0;
            for (let k = 0; k < n; k++) sy += sList[i][k] * yList[i][k];
            if (Math.abs(sy) < 1e-30) continue;
            const rho = 1.0 / sy;
            let dot = 0;
            for (let k = 0; k < n; k++) dot += sList[i][k] * direction[k];
            alpha[i] = rho * dot;
            for (let k = 0; k < n; k++) direction[k] -= alpha[i] * yList[i][k];
        }
        for (let i = 0; i < sList.length; i++) {
            let sy = 0;
            for (let k = 0; k < n; k++) sy += sList[i][k] * yList[i][k];
            if (Math.abs(sy) < 1e-30) continue;
            const rho = 1.0 / sy;
            let dot = 0;
            for (let k = 0; k < n; k++) dot += yList[i][k] * direction[k];
            const beta = rho * dot;
            for (let k = 0; k < n; k++) direction[k] += (alpha[i] - beta) * sList[i][k];
        }
        return direction;
    }

    _projGradSupNorm(x, grad) {
        let m = 0.0;
        for (let k = 0; k < grad.length; k++) {
            const gk = grad[k], xk = x[k];
            if (xk <= 0.0 && gk > 0.0) continue;
            if (xk >= 1.0 && gk < 0.0) continue;
            if (Math.abs(gk) > m) m = Math.abs(gk);
        }
        return m;
    }

    _fdGradientForPolish(x) {
        const n = this.nDim;
        const h = 1e-6;
        const grad = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            if (this.evaluations >= this.nTrials) break;
            const xPlus = x.slice();
            xPlus[i] = Math.min(1.0, x[i] + h);
            const fPlus = this.evaluate(xPlus);
            if (this.evaluations >= this.nTrials) break;
            const xMinus = x.slice();
            xMinus[i] = Math.max(0.0, x[i] - h);
            const fMinus = this.evaluate(xMinus);
            const denom = xPlus[i] - xMinus[i];
            if (denom > 0) grad[i] = (fPlus - fMinus) / denom;
        }
        return grad;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = { Optimizer, MathUtils, usePortableRng, useLegacyRng };
} else {
    // Browser environment
    window.Optimizer = Optimizer;
    window.MathUtils = MathUtils;
    window.usePortableRng = usePortableRng;
    window.useLegacyRng = useLegacyRng;
}