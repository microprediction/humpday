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

    // Statement-for-statement twin of Powell in humpday/optimizers/scipy_algorithms.py, which
    // is itself a port of scipy.optimize._minimize_powell plus the Brent line search underneath
    // it.
    //
    // What was here before searched each direction by trying four fixed step sizes (+-0.1,
    // +-0.2) and keeping the best. On a smooth problem that is sampling rather than converging:
    // the JS-against-scipy gate added in #164 measured 8.06e-05 on the sphere where scipy
    // reaches exact zero, a ratio of 8e+10, and 0.457 against 1.48e-10 on Ackley. Parity with
    // the Python port never caught it, because parity compares the two ports against each other
    // within a tolerance and both were free to be equally blunt (#78).
    *_run() {
        yield* this._powellPassGen();
        while (this.evaluations + 2 + 2 * this.nDim <= this.nTrials) {
            yield* this._powellPassGen();
        }
    }

    *_powellPassGen() {
        const n = this.nDim;
        let x = MathUtils.randomUniform(n).map(v => 0.3 + 0.4 * v);  // Interior start

        // Initial direction set: identity matrix (coordinate directions).
        const direc = [];
        for (let i = 0; i < n; i++) {
            const row = new Array(n).fill(0.0);
            row[i] = 1.0;
            direc.push(row);
        }

        // scipy's default ftol=1e-4 stops far before the budget is spent on smooth problems;
        // 1e-12 lets Powell use what it was given.
        const ftol = 1e-12;
        const maxiter = n * 20;

        let fval = yield x;
        let x1 = x.slice();
        let iteration = 0;

        while (iteration < maxiter && this.evaluations < this.nTrials) {
            const fx = fval;
            let bigind = 0;
            let delta = 0.0;

            for (let i = 0; i < n; i++) {
                if (this.evaluations >= this.nTrials) break;
                const fx2 = fval;
                const res = yield* this._linesearchPowellGen(x, direc[i].slice(), fval);
                fval = res.fval;
                x = res.x;
                if (fx2 - fval > delta) {
                    delta = fx2 - fval;
                    bigind = i;
                }
            }

            iteration += 1;

            const bnd = ftol * (Math.abs(fx) + Math.abs(fval)) + 1e-20;
            if (2.0 * (fx - fval) <= bnd) break;
            if (this.evaluations >= this.nTrials) break;

            // The extrapolated point.
            let direc1 = x.map((v, i) => v - x1[i]);
            x1 = x.slice();
            const x2 = MathUtils.clipArray(x.map((v, i) => v + direc1[i]), 0, 1);

            if (this.evaluations >= this.nTrials) break;
            const fx2 = yield x2;

            if (fx > fx2) {
                let t = 2.0 * (fx + fx2 - 2.0 * fval);
                let temp = fx - fval - delta;
                t *= temp * temp;
                temp = fx - fx2;
                t -= delta * temp * temp;

                if (t < 0.0) {
                    const res = yield* this._linesearchPowellGen(x, direc1, fval);
                    fval = res.fval;
                    x = res.x;
                    direc1 = res.direction;
                    if (direc1.some(v => v !== 0)) {
                        direc[bigind] = direc[n - 1].slice();
                        direc[n - 1] = direc1.slice();
                    }
                }
            }
        }
    }

    // Budget-gated evaluation of p + alpha * xi. `state = [bestAlpha, bestF]`, updated in place,
    // so that a search cut off by the budget still returns its best point.
    *_lsEvalGen(p, xi, alpha, state) {
        if (this.evaluations >= this.nTrials) return null;
        const candidate = MathUtils.clipArray(p.map((v, i) => v + alpha * xi[i]), 0, 1);
        const f = yield candidate;
        if (f < state[1]) {
            state[0] = alpha;
            state[1] = f;
        }
        return f;
    }

    _linesearchResult(p, xi, fval, bestAlpha, bestF) {
        if (bestF < fval) {
            return {
                fval: bestF,
                x: MathUtils.clipArray(p.map((v, i) => v + bestAlpha * xi[i]), 0, 1),
                direction: xi.map(v => bestAlpha * v),
            };
        }
        return { fval, x: p, direction: new Array(xi.length).fill(0.0) };
    }

    // Bounded Brent line search along `xi` from `p`: bracket by downhill walk, then interleave
    // inverse parabolic interpolation with golden-section fallbacks. Twin of
    // _linesearch_powell_gen in scipy_algorithms.py.
    *_linesearchPowellGen(p, xi, fval) {
        if (!xi.some(v => v !== 0)) return { fval, x: p, direction: xi };

        // Clamp alpha so p + alpha * xi stays inside the cube.
        let alphaLo = -Infinity;
        let alphaHi = Infinity;
        for (let i = 0; i < xi.length; i++) {
            const xiI = xi[i];
            if (Math.abs(xiI) <= 1e-12) continue;
            const b1 = -p[i] / xiI;
            const b2 = (1.0 - p[i]) / xiI;
            const lo = Math.min(b1, b2);
            const hi = Math.max(b1, b2);
            if (lo > alphaLo) alphaLo = lo;
            if (hi < alphaHi) alphaHi = hi;
        }
        if (alphaHi <= alphaLo) return { fval, x: p, direction: xi };
        alphaLo = Math.max(alphaLo, -1.0);
        alphaHi = Math.min(alphaHi, 1.0);
        if (alphaHi <= alphaLo) return { fval, x: p, direction: xi };

        const state = [0.0, fval];

        // ---- Bracket the minimum ----
        const span = alphaHi - alphaLo;
        let xa = alphaLo < 0 && 0 < alphaHi ? 0.0 : alphaLo;
        let xb = xa + Math.min(span * 0.1, 1e-1);
        if (xb >= alphaHi) xb = alphaLo + 0.5 * (alphaHi - alphaLo);

        let fa = yield* this._lsEvalGen(p, xi, xa, state);
        if (fa === null) return this._linesearchResult(p, xi, fval, state[0], state[1]);
        let fb = yield* this._lsEvalGen(p, xi, xb, state);
        if (fb === null) return this._linesearchResult(p, xi, fval, state[0], state[1]);

        if (fa < fb) {
            [xa, xb] = [xb, xa];
            [fa, fb] = [fb, fa];
        }

        const gold = 1.618033988749895;
        let xc = xb + gold * (xb - xa);
        if (xc > alphaHi) xc = alphaHi;
        if (xc < alphaLo) xc = alphaLo;

        let fc = yield* this._lsEvalGen(p, xi, xc, state);
        if (fc === null) return this._linesearchResult(p, xi, fval, state[0], state[1]);

        for (let k = 0; k < 20; k++) {
            if (fc >= fb) break;
            let newXc = xc + gold * (xc - xb);
            if (newXc > alphaHi || newXc < alphaLo) {
                newXc = xc - xb > 0 ? alphaHi : alphaLo;
                if (newXc === xc) break;
            }
            xa = xb;
            xb = xc;
            fa = fb;
            fb = fc;
            xc = newXc;
            fc = yield* this._lsEvalGen(p, xi, xc, state);
            if (fc === null) return this._linesearchResult(p, xi, fval, state[0], state[1]);
        }

        const bracketed = (xa < xb && xb < xc) || (xc < xb && xb < xa);
        if (!bracketed || !(fb <= fa && fb <= fc)) {
            return this._linesearchResult(p, xi, fval, state[0], state[1]);
        }

        // ---- Brent inside the bracket ----
        let aBound = Math.min(xa, xc);
        let bBound = Math.max(xa, xc);

        let xBest = xb;
        let w = xb;
        let v = xb;
        let fx = fb;
        let fw = fb;
        let fv = fb;
        let deltax = 0.0;
        let rat = 0.0;
        const cg = 0.3819660;
        const brentTol = 1.48e-3;
        const mintol = 1.0e-11;

        for (let k = 0; k < 50; k++) {
            const tol1 = brentTol * Math.abs(xBest) + mintol;
            const tol2 = 2.0 * tol1;
            const xmid = 0.5 * (aBound + bBound);
            if (Math.abs(xBest - xmid) < tol2 - 0.5 * (bBound - aBound)) break;

            let u;
            if (Math.abs(deltax) <= tol1) {
                deltax = xBest >= xmid ? aBound - xBest : bBound - xBest;
                rat = cg * deltax;
            } else {
                let tmp1 = (xBest - w) * (fx - fv);
                let tmp2 = (xBest - v) * (fx - fw);
                let pNum = (xBest - v) * tmp2 - (xBest - w) * tmp1;
                tmp2 = 2.0 * (tmp2 - tmp1);
                if (tmp2 > 0.0) pNum = -pNum;
                tmp2 = Math.abs(tmp2);
                const dxTemp = deltax;
                deltax = rat;
                if (
                    pNum > tmp2 * (aBound - xBest) &&
                    pNum < tmp2 * (bBound - xBest) &&
                    Math.abs(pNum) < Math.abs(0.5 * tmp2 * dxTemp)
                ) {
                    rat = pNum / tmp2;
                    u = xBest + rat;
                    if (u - aBound < tol2 || bBound - u < tol2) {
                        rat = xmid - xBest >= 0 ? tol1 : -tol1;
                    }
                } else {
                    deltax = xBest >= xmid ? aBound - xBest : bBound - xBest;
                    rat = cg * deltax;
                }
            }

            u = Math.abs(rat) < tol1 ? xBest + (rat >= 0 ? tol1 : -tol1) : xBest + rat;

            const fu = yield* this._lsEvalGen(p, xi, u, state);
            if (fu === null) break;

            if (fu > fx) {
                if (u < xBest) {
                    aBound = u;
                } else {
                    bBound = u;
                }
                if (fu <= fw || w === xBest) {
                    v = w;
                    fv = fw;
                    w = u;
                    fw = fu;
                } else if (fu <= fv || v === xBest || v === w) {
                    v = u;
                    fv = fu;
                }
            } else {
                if (u >= xBest) {
                    aBound = xBest;
                } else {
                    bBound = xBest;
                }
                v = w;
                fv = fw;
                w = xBest;
                fw = fx;
                xBest = u;
                fx = fu;
            }
        }

        return this._linesearchResult(p, xi, fval, state[0], state[1]);
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
        this.bestX = MathUtils.randomUniform(this.nDim);
        this.bestValue = this.evaluate(this.bestX);
        this._lbfgsPolish();

        // And then again from somewhere else, for as long as the budget lasts. A descent
        // converges and stops, which on the sphere at nTrials=5000 meant eleven evaluations
        // used and 4,989 handed back. Multi-start is the standard remedy: keep descending
        // from fresh points, keeping the best seen. Twin of LBFGSB._run in
        // humpday/optimizers/scipy_algorithms.py.
        while (this.evaluations + 2 * this.nDim + 2 <= this.nTrials) {
            const start = MathUtils.randomUniform(this.nDim);
            const startValue = this.evaluate(start);
            this._driveGen(this._lbfgsPolishGen(start, startValue));
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