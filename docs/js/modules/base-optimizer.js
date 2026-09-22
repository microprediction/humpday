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
const _prngmod = (typeof module !== 'undefined' && module.exports)
    ? require('./prng.js')
    : { PCG32: (typeof PCG32 !== 'undefined' ? PCG32 : null),
        portableLog: (typeof portableLog !== 'undefined' ? portableLog : null),
        portableExp: (typeof portableExp !== 'undefined' ? portableExp : null) };
const _PCG32 = _prngmod.PCG32;

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

    // k distinct elements by partial front Fisher-Yates — twin of
    // PCG32.sample in humpday/_prng.py (same draw sequence, so
    // replace=False choices replay bit-for-bit).
    sample(seq, k) {
        const pool = seq.slice();
        const n = pool.length;
        for (let i = 0; i < k; i++) {
            const j = i + MathUtils.randInt(n - i);
            const tmp = pool[i];
            pool[i] = pool[j];
            pool[j] = tmp;
        }
        return pool.slice(0, k);
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

// Deterministic transcendentals shared with the Python side (see
// humpday/_prng.py): trajectory code must use these, never Math.pow /
// Math.exp / Math.log, whose last-ulp behaviour is runtime-specific.
//
// A page that does not load prng.js first leaves these null, and the first
// optimizer to need one fails with "MathUtils.portableLog is not a function"
// somewhere in the middle of a run -- which is how every page on the site came
// to be missing the script without anyone noticing. Say what is actually
// wrong, at the point the name is used.
function _missingPrng(name) {
    return function () {
        throw new Error(
            'humpday: MathUtils.' + name + ' needs js/modules/prng.js, which this page loads ' +
            'after base-optimizer.js or not at all. Load prng.js first: this module captures ' +
            'the portable math when it runs.'
        );
    };
}

MathUtils.portableLog = _prngmod.portableLog || _missingPrng('portableLog');
MathUtils.portableExp = _prngmod.portableExp || _missingPrng('portableExp');

// Thrown by evaluate() when nTrials objective calls have been made. Caught by
// the wrapper the Optimizer constructor puts around legacy optimize()
// overrides, and by the generator driver; never seen by callers.
class BudgetExhausted extends Error {
    constructor(nTrials) {
        super(`budget of ${nTrials} objective evaluations exhausted`);
        this.name = 'BudgetExhausted';
    }
}

// Base optimizer class
class Optimizer {
    constructor(objective, nTrials, nDim) {
        this.objective = objective;
        if (!Number.isInteger(nTrials) || nTrials < 0) {
            throw new Error(`nTrials must be a non-negative integer, got ${nTrials}`);
        }
        this.nTrials = nTrials;
        this.nDim = nDim;
        this.evaluations = 0;
        this.bestValue = Infinity;
        // Same stream position as Python: BaseOptimizer.__init__ draws
        // best_x from the RNG (n draws) before the run starts.
        this.bestX = MathUtils.randomUniform(nDim);
        this.trackPath = false;
        this.path = [];

        // An optimizer that overrides optimize() owns its own loop, so the
        // driver below cannot cap it; evaluate() throws BudgetExhausted at the
        // cap instead, and this wrapper turns that into a normal return.
        if (this.optimize !== Optimizer.prototype.optimize) {
            const legacy = this.optimize.bind(this);
            this.optimize = () => {
                if (this.nTrials <= 0) return this._result();
                try {
                    return legacy();
                } catch (e) {
                    if (e instanceof BudgetExhausted) return this._result();
                    throw e;
                }
            };
        }
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
        // The cap for optimizers that still own their loop (the ones that
        // override optimize() instead of defining _run): the constructor
        // wraps their optimize() so this signal ends the run with the best
        // point banked so far. The objective is never called past nTrials.
        if (this.evaluations >= this.nTrials) {
            throw new BudgetExhausted(this.nTrials);
        }
        this.evaluations++;
        const clippedX = MathUtils.clipArray(x, 0, 1);
        const value = this.objective(clippedX);
        return this._bookkeep(clippedX, value);
    }

    _result() {
        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
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
        // nTrials is a hard cap on objective calls, enforced by the driver
        // rather than trusted to every algorithm (twin of BaseOptimizer.optimize).
        if (this.nTrials <= 0) return this._result();
        const gen = this._run();
        try {
            let res = gen.next();
            while (!res.done) {
                if (this.evaluations >= this.nTrials) {
                    gen.return();
                    break;
                }
                this.evaluations++;
                const clippedX = MathUtils.clipArray(res.value, 0, 1);
                const value = this.objective(clippedX);
                this._bookkeep(clippedX, value);
                res = gen.next(value);
            }
        } catch (e) {
            // A _run that calls evaluate() itself (e.g. for a polish stage)
            // hits the cap there.
            if (!(e instanceof BudgetExhausted)) throw e;
        }
        return this._result();
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
        if (gd.done || this.evaluations >= this.nTrials) return null;
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
        if (this.evaluations >= this.nTrials) {
            // Budget spent: the run is over (twin of BaseOptimizer._gen_feed).
            gd.done = true;
            gd.pending = null;
            gd.gen.return();
            return;
        }
        let res;
        try {
            res = gd.gen.next(value);
        } catch (e) {
            if (!(e instanceof BudgetExhausted)) throw e;
            res = { done: true };
        }
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
    // `humpday/optimizers/base.py`. The generator form is the twin of
    // `_lbfgs_polish_gen`; converted optimizers `yield*` it from their
    // _run(), while legacy optimize()-style callers use _lbfgsPolish(),
    // which drives the same generator (the twin of `_drive_gen`).
    _driveGen(gen) {
        let res = gen.next();
        while (!res.done) res = gen.next(this.evaluate(res.value));
        return res.value;
    }

    _lbfgsPolish() {
        return this._driveGen(this._lbfgsPolishGen());
    }

    // Polish from `start`, or from the best point seen when it is undefined. A polish that
    // always begins at the running best cannot restart: once converged, every further pass
    // re-derives the same point. Twin of _lbfgs_polish_gen in humpday/optimizers/base.py.
    *_lbfgsPolishGen(start, startValue) {
        const FACTR = 1e7;
        const PGTOL = 1e-5;
        const EPS_MACH = 2.220446049250313e-16;
        const MEMORY = Math.min(10, Math.max(1, this.nDim));
        const n = this.nDim;

        let x = start === undefined ? this.bestX.slice() : start.slice();
        let f = start === undefined ? this.bestValue : startValue;
        let grad = yield* this._fdGradientPolishGen(x);

        const sList = [];
        const yList = [];

        const lo = new Array(n).fill(0.0);
        const hi = new Array(n).fill(1.0);

        while (this.evaluations < this.nTrials - 2 * n) {
            // (1) Convergence on the projected gradient, scipy's `projgr`.
            if (this._projGradSupNorm(x, grad) < PGTOL) break;

            // (2) The limited-memory model, in compact form.
            const { theta, W, Minv } = Optimizer._lbfgsbCompact(sList, yList);

            // (3) The generalised Cauchy point chooses the active set.
            const { xcp, free, c } = Optimizer._lbfgsbCauchy(x, grad, theta, W, Minv, lo, hi);

            // (4) Subspace minimisation over whatever is still free.
            const xbar = Optimizer._lbfgsbSubspace(x, xcp, grad, theta, W, Minv, c, free, lo, hi);

            let direction = [];
            for (let k = 0; k < n; k++) direction.push(xbar[k] - x[k]);
            if (direction.every(v => Math.abs(v) <= 1e-300)) break;

            let gd = 0;
            for (let k = 0; k < n; k++) gd += grad[k] * direction[k];
            if (gd >= 0.0) {
                direction = grad.map(v => -v);
                for (let k = 0; k < n; k++) {
                    if ((x[k] <= lo[k] && direction[k] < 0.0) ||
                        (x[k] >= hi[k] && direction[k] > 0.0)) direction[k] = 0.0;
                }
                gd = 0;
                for (let k = 0; k < n; k++) gd += grad[k] * direction[k];
                if (gd >= 0.0) break;
                sList.length = 0;
                yList.length = 0;
            }

            // (5) Backtracking line search; every trial point is feasible by construction.
            let step = 1.0;
            let newX = x;
            let newF = f;
            let accepted = false;
            while (step > 1e-14) {
                if (this.evaluations >= this.nTrials) break;
                const candidate = [];
                for (let k = 0; k < n; k++) {
                    candidate.push(Math.min(hi[k], Math.max(lo[k], x[k] + step * direction[k])));
                }
                const candF = yield candidate;
                if (candF <= f + 1e-4 * step * gd) {
                    newX = candidate;
                    newF = candF;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) break;

            // (6) scipy's factr * eps_mach test on the relative decrease.
            const fScale = Math.max(Math.abs(f), Math.abs(newF), 1.0);
            if (f - newF < FACTR * EPS_MACH * fScale) {
                x = newX;
                f = newF;
                break;
            }

            const newGrad = yield* this._fdGradientPolishGen(newX);

            // (7) Curvature pair, kept only when it is one.
            const sVec = [];
            const yVec = [];
            for (let k = 0; k < n; k++) {
                sVec.push(newX[k] - x[k]);
                yVec.push(newGrad[k] - grad[k]);
            }
            const sy = Optimizer._lbfgsbDot(sVec, yVec);
            const ss = Optimizer._lbfgsbDot(sVec, sVec);
            const yy = Optimizer._lbfgsbDot(yVec, yVec);
            if (sy > 1e-12 * Math.sqrt(ss * yy + 1e-300)) {
                sList.push(sVec);
                yList.push(yVec);
                if (sList.length > MEMORY) {
                    sList.shift();
                    yList.shift();
                }
            }
            x = newX;
            f = newF;
            grad = newGrad;
        }
    }

    // ---- Byrd-Lu-Nocedal-Zhu machinery ---------------------------------------------------
    //
    // Twins of _lbfgsb_compact / _lbfgsb_cauchy / _lbfgsb_subspace in
    // humpday/optimizers/base.py. The three pieces that make L-BFGS-B what it is: the compact
    // limited-memory representation, the generalised Cauchy point that chooses the active set
    // by minimising the model along the piecewise projected-gradient path, and subspace
    // minimisation over whatever is still free. This port previously had none of them (#407).

    static _lbfgsbDot(a, b) {
        let t = 0.0;
        for (let i = 0; i < a.length; i++) t += a[i] * b[i];
        return t;
    }

    static _lbfgsbSolve(A, b) {
        const k = b.length;
        const M = [];
        for (let i = 0; i < k; i++) M.push(A[i].slice().concat([b[i]]));
        for (let col = 0; col < k; col++) {
            let pivot = col;
            for (let r = col + 1; r < k; r++) {
                if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
            }
            if (Math.abs(M[pivot][col]) < 1e-300) return null;
            const tmp = M[col]; M[col] = M[pivot]; M[pivot] = tmp;
            const inv = 1.0 / M[col][col];
            for (let r = col + 1; r < k; r++) {
                const factor = M[r][col] * inv;
                if (factor) {
                    for (let c = col; c <= k; c++) M[r][c] -= factor * M[col][c];
                }
            }
        }
        const out = new Array(k).fill(0.0);
        for (let r = k - 1; r >= 0; r--) {
            let total = M[r][k];
            for (let c = r + 1; c < k; c++) total -= M[r][c] * out[c];
            out[r] = total / M[r][r];
        }
        return out;
    }

    static _lbfgsbCompact(sList, yList) {
        const m = sList.length;
        if (m === 0) return { theta: 1.0, W: [], Minv: [] };
        const syLast = Optimizer._lbfgsbDot(sList[m - 1], yList[m - 1]);
        const yyLast = Optimizer._lbfgsbDot(yList[m - 1], yList[m - 1]);
        const theta = syLast > 1e-300 ? yyLast / syLast : 1.0;

        const W = yList.map(y => y.slice());
        for (const sv of sList) W.push(sv.map(v => theta * v));

        const D = [];
        for (let i = 0; i < m; i++) D.push(Optimizer._lbfgsbDot(sList[i], yList[i]));
        const size = 2 * m;
        const Minv = [];
        for (let i = 0; i < size; i++) Minv.push(new Array(size).fill(0.0));
        for (let i = 0; i < m; i++) {
            Minv[i][i] = -D[i];
            for (let j = 0; j < m; j++) {
                const Lij = i > j ? Optimizer._lbfgsbDot(sList[i], yList[j]) : 0.0;
                const Lji = j > i ? Optimizer._lbfgsbDot(sList[j], yList[i]) : 0.0;
                Minv[i][m + j] = Lji;
                Minv[m + i][j] = Lij;
                Minv[m + i][m + j] = theta * Optimizer._lbfgsbDot(sList[i], sList[j]);
            }
        }
        return { theta, W, Minv };
    }

    static _lbfgsbCauchy(x, g, theta, W, Minv, lo, hi) {
        const n = x.length;
        const m2 = W.length;
        const t = new Array(n).fill(0.0);
        const d = new Array(n).fill(0.0);
        for (let i = 0; i < n; i++) {
            const gi = g[i];
            if (gi < 0.0) t[i] = (x[i] - hi[i]) / gi;
            else if (gi > 0.0) t[i] = (x[i] - lo[i]) / gi;
            else t[i] = Infinity;
            d[i] = t[i] === 0.0 ? 0.0 : -gi;
        }

        const xcp = x.slice();
        let free = [];
        for (let i = 0; i < n; i++) if (t[i] > 0.0) free.push(i);
        if (!free.length) return { xcp, free: [], c: new Array(m2).fill(0.0) };

        const p = m2 ? W.map(col => Optimizer._lbfgsbDot(col, d)) : [];
        const c = new Array(m2).fill(0.0);
        let fp = -Optimizer._lbfgsbDot(d, d);
        let fpp;
        if (m2) {
            const Mp = Optimizer._lbfgsbSolve(Minv, p);
            fpp = -theta * fp - (Mp ? Optimizer._lbfgsbDot(p, Mp) : 0.0);
        } else {
            fpp = -theta * fp;
        }
        let dtMin = fpp > 1e-300 ? -fp / fpp : Infinity;

        // Only variables that can move are breakpoints; one already on the bound the gradient
        // pushes it against joins the active set at the start.
        const order = free.filter(i => t[i] < Infinity).sort((a, b) => t[a] - t[b]);
        let tOld = 0.0;
        for (const b of order) {
            const dt = t[b] - tOld;
            if (dtMin < dt) break;
            for (let i = 0; i < n; i++) if (d[i] !== 0.0) xcp[i] += dt * d[i];
            xcp[b] = d[b] > 0.0 ? hi[b] : lo[b];
            const zb = xcp[b] - x[b];
            const gb = g[b];
            if (m2) {
                const wb = W.map(col => col[b]);
                for (let j = 0; j < m2; j++) c[j] += dt * p[j];
                const Mc = Optimizer._lbfgsbSolve(Minv, c);
                const Mw = Optimizer._lbfgsbSolve(Minv, wb);
                const Mp = Optimizer._lbfgsbSolve(Minv, p);
                fp += dt * fpp + gb * gb + theta * gb * zb;
                if (Mc) fp -= gb * Optimizer._lbfgsbDot(wb, Mc);
                fpp -= theta * gb * gb;
                if (Mp) fpp -= 2.0 * gb * Optimizer._lbfgsbDot(wb, Mp);
                if (Mw) fpp -= gb * gb * Optimizer._lbfgsbDot(wb, Mw);
                for (let j = 0; j < m2; j++) p[j] += gb * wb[j];
            } else {
                fp += dt * fpp + gb * gb + theta * gb * zb;
                fpp -= theta * gb * gb;
            }
            d[b] = 0.0;
            tOld = t[b];
            dtMin = fpp > 1e-300 ? -fp / fpp : Infinity;
            if (fp >= 0.0) { dtMin = 0.0; break; }
        }

        dtMin = Math.max(dtMin, 0.0);
        for (let i = 0; i < n; i++) if (d[i] !== 0.0) xcp[i] += dtMin * d[i];
        for (let i = 0; i < n; i++) xcp[i] = Math.min(hi[i], Math.max(lo[i], xcp[i]));
        if (m2) for (let j = 0; j < m2; j++) c[j] += dtMin * p[j];

        free = [];
        for (let i = 0; i < n; i++) if (lo[i] < xcp[i] && xcp[i] < hi[i]) free.push(i);
        return { xcp, free, c };
    }

    static _lbfgsbSubspace(x, xcp, g, theta, W, Minv, c, free, lo, hi) {
        if (!free.length) return xcp.slice();
        const m2 = W.length;
        const Mc = m2 ? Optimizer._lbfgsbSolve(Minv, c) : null;

        const r = [];
        for (const i of free) {
            let ri = g[i] + theta * (xcp[i] - x[i]);
            if (Mc) for (let j = 0; j < m2; j++) ri -= W[j][i] * Mc[j];
            r.push(ri);
        }

        const k = free.length;
        const B = [];
        for (let a = 0; a < k; a++) {
            B.push(new Array(k).fill(0.0));
            B[a][a] = theta;
        }
        if (m2) {
            const MW = free.map(i => {
                const wi = W.map(col => col[i]);
                return Optimizer._lbfgsbSolve(Minv, wi) || new Array(m2).fill(0.0);
            });
            for (let a = 0; a < k; a++) {
                const wa = W.map(col => col[free[a]]);
                for (let b = 0; b < k; b++) B[a][b] -= Optimizer._lbfgsbDot(wa, MW[b]);
            }
        }

        const step = Optimizer._lbfgsbSolve(B, r.map(v => -v));
        if (!step) return xcp.slice();

        let alpha = 1.0;
        for (let a = 0; a < k; a++) {
            const i = free[a];
            if (step[a] > 1e-300) alpha = Math.min(alpha, (hi[i] - xcp[i]) / step[a]);
            else if (step[a] < -1e-300) alpha = Math.min(alpha, (lo[i] - xcp[i]) / step[a]);
        }
        alpha = Math.max(0.0, Math.min(1.0, alpha));

        const out = xcp.slice();
        for (let a = 0; a < k; a++) {
            const i = free[a];
            out[i] = Math.min(hi[i], Math.max(lo[i], xcp[i] + alpha * step[a]));
        }
        return out;
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

        // H0 = gamma * I with gamma = s.y / y.y from the newest pair, as in scipy. Using the
        // identity here leaves the first trial step wrong by whatever the curvature is (#407).
        if (sList.length) {
            const newestS = sList[sList.length - 1];
            const newestY = yList[yList.length - 1];
            let sy = 0;
            let yy = 0;
            for (let k = 0; k < n; k++) {
                sy += newestS[k] * newestY[k];
                yy += newestY[k] * newestY[k];
            }
            if (yy > 1e-30 && sy > 0.0) {
                const gamma = sy / yy;
                for (let k = 0; k < n; k++) direction[k] *= gamma;
            }
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

    // ||P(x - g) - x||_inf on [0,1]^n: scipy's `projgr`. A positive gradient component is
    // bounded by how far the variable can travel down to its lower bound and a negative one by
    // how far up to its upper bound, at every point rather than only on the boundary -- a step
    // of 100 from x = 0.01 moves 0.01. This used to clip only on a bound and so returned the
    // raw gradient everywhere inside the cube, four orders too large in that example, against a
    // tolerance the polish stops on (#407). Twin of _proj_grad_sup_norm in base.py.
    _projGradSupNorm(x, grad) {
        let m = 0.0;
        for (let k = 0; k < grad.length; k++) {
            let gk = grad[k];
            const xk = x[k];
            if (gk < 0.0) gk = Math.max(gk, xk - 1.0);
            else gk = Math.min(gk, xk);
            if (Math.abs(gk) > m) m = Math.abs(gk);
        }
        return m;
    }

    _fdGradientForPolish(x) {
        return this._driveGen(this._fdGradientPolishGen(x));
    }

    *_fdGradientPolishGen(x) {
        const n = this.nDim;
        const h = 1e-6;
        const grad = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            if (this.evaluations >= this.nTrials) break;
            const xPlus = x.slice();
            xPlus[i] = Math.min(1.0, x[i] + h);
            const fPlus = yield xPlus;
            if (this.evaluations >= this.nTrials) break;
            const xMinus = x.slice();
            xMinus[i] = Math.max(0.0, x[i] - h);
            const fMinus = yield xMinus;
            const denom = xPlus[i] - xMinus[i];
            if (denom > 0) grad[i] = (fPlus - fMinus) / denom;
        }
        return grad;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = { Optimizer, MathUtils, usePortableRng, useLegacyRng, BudgetExhausted };
} else {
    // Browser environment
    window.Optimizer = Optimizer;
    window.MathUtils = MathUtils;
    window.usePortableRng = usePortableRng;
    window.useLegacyRng = useLegacyRng;
}