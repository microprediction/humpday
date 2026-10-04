/**
 * Evolutionary algorithm implementations.
 *
 * These algorithms are inspired by natural evolution processes and include
 * methods like Differential Evolution, Genetic Algorithms, Particle Swarm, etc.
 * They excel at global optimization and handling multimodal landscapes.
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
    globalThis.Linalg = require('./linalg.js');
}class DifferentialEvolution extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'DifferentialEvolution';
    }

    // Matches scipy.optimize.differential_evolution defaults:
    //   - best/1/bin mutation (base = current population best);
    //   - dithered F drawn per-generation in [0.5, 1.0];
    //   - CR = 0.7;
    //   - L-BFGS-B polish on half the budget (scipy `polish=True`).
    *_run() {
        // Twin of DifferentialEvolution._run in
        // humpday/optimizers/evolutionary_algorithms.py.
        const n = this.nDim;
        const polishBudget = Math.max(15, Math.floor(this.nTrials / 2));
        let deBudget = this.nTrials - polishBudget;

        const popSize = Math.max(10, Math.min(20, Math.floor(deBudget / 5)));
        const CR = 0.7;

        // All initial draws happen before the first yield (Python builds
        // the population in a list comprehension, then evaluates).
        const population = [];
        for (let i = 0; i < popSize; i++) population.push(MathUtils.randomUniform(n));
        const fitness = [];
        for (const ind of population) fitness.push(yield ind);

        while (true) {
            const before = this.evaluations;

            while (this.evaluations < deBudget) {
            // Dither: pick F uniformly in [0.5, 1.0] each generation.
            const F = 0.5 + 0.5 * MathUtils.randomScalar();

            for (let i = 0; i < popSize; i++) {
                if (this.evaluations >= deBudget) break;

                // best/1: base = current population best (first minimal
                // index, like Python's min(range, key=...)).
                let bestIdx = 0;
                for (let k = 1; k < popSize; k++) {
                    if (fitness[k] < fitness[bestIdx]) bestIdx = k;
                }

                // Two donors distinct from i and bestIdx.
                let candidates = [];
                for (let k = 0; k < popSize; k++) {
                    if (k !== i && k !== bestIdx) candidates.push(k);
                }
                if (candidates.length < 2) {
                    candidates = [];
                    for (let k = 0; k < popSize; k++) {
                        if (k !== i) candidates.push(k);
                    }
                }
                let b, c;
                if (candidates.length < 2) {
                    b = MathUtils.choice(candidates);
                    c = MathUtils.choice(candidates);
                } else {
                    [b, c] = MathUtils.sample(candidates, 2);
                }

                // Mutation: v = x_best + F * (x_b - x_c), clipped to [0, 1].
                const mutant = new Array(n);
                for (let j = 0; j < n; j++) {
                    mutant[j] = MathUtils.clip(
                        population[bestIdx][j] + F * (population[b][j] - population[c][j]),
                        0, 1
                    );
                }

                // Binomial crossover with one guaranteed coord. The CR
                // draw happens for every j (Python's `random() < CR or
                // j == j_guaranteed` — left operand always evaluated).
                const trial = population[i].slice();
                const jGuaranteed = MathUtils.randInt(n);
                for (let j = 0; j < n; j++) {
                    if (MathUtils.randomScalar() < CR || j === jGuaranteed) {
                        trial[j] = mutant[j];
                    }
                }

                // (1+1) selection.
                const trialFitness = yield trial;
                if (trialFitness < fitness[i]) {
                    population[i] = trial;
                    fitness[i] = trialFitness;
                }
            }
        }

            // --- Polish stage: L-BFGS from best DE point ------------
            // Matches scipy.differential_evolution `polish=True`.
            yield* this._lbfgsPolishGen();

            // Re-split whatever the polish did not spend and go round again; twin of the Python
            // change. The reserve is sized for the worst case, but the polish converges in eight
            // to eighteen evaluations and the remainder used to be forfeited -- about half the
            // budget, at every budget. The first round is unchanged, so the split tuned on 2-D
            // Rosenbrock at nTrials=200 is reproduced exactly.
            if (this.evaluations >= this.nTrials || this.evaluations === before) break;
            deBudget =
                this.nTrials - Math.max(15, Math.floor((this.nTrials - this.evaluations) / 2));
            if (this.evaluations >= deBudget) break;
        }
    }
}

// Particle Swarm Optimization implementation
class ParticleSwarm extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'ParticleSwarm';
    }

    *_run() {
        // Twin of ParticleSwarm._run in
        // humpday/optimizers/evolutionary_algorithms.py.
        const n = this.nDim;
        const polishReserve = Math.min(20 * n, Math.floor(this.nTrials / 2));
        const psoBudget = Math.max(this.evaluations, this.nTrials - polishReserve);

        const swarmSize = Math.min(40, Math.max(15, n * 3));

        // Initialize swarm. Python builds positions and velocities in
        // two list comprehensions (all position draws, then all velocity
        // draws) before the evaluation loop.
        const positions = [];
        for (let i = 0; i < swarmSize; i++) positions.push(MathUtils.randomUniform(n));
        const velocities = [];
        for (let i = 0; i < swarmSize; i++) {
            const u = MathUtils.randomUniform(n);
            velocities.push(u.map(v => (v - 0.5) * 0.2));
        }
        const personalBestPos = positions.map(p => p.slice());
        const personalBestFit = [];
        for (const p of positions) personalBestFit.push(yield p);

        const maxIterations = Math.max(1, Math.floor(psoBudget / swarmSize));

        // SPSO-2011-style stagnation detection: when the global best has
        // stalled for `stagnationWindow` iterations, reseed the worst
        // half of the swarm; the kept half retains its memory.
        const stagnationWindow = Math.max(10, Math.floor(maxIterations / 5));
        let stagnationCounter = 0;
        let lastGlobalBest = this.bestValue;
        const improvementAtol = 1e-12;

        for (let iteration = 0; iteration < maxIterations; iteration++) {
            if (this.evaluations >= psoBudget) break;

            // Adaptive coefficients (anneal inertia / explore-exploit balance).
            const w = 0.9 - 0.5 * (iteration / maxIterations);
            const c1 = 2.5 - 1.0 * (iteration / maxIterations);
            const c2 = 1.5 + 1.0 * (iteration / maxIterations);

            for (let i = 0; i < swarmSize; i++) {
                if (this.evaluations >= psoBudget) break;

                // r1, r2 are length-n uniform VECTORS drawn up front
                // (not per-coordinate scalars) — stream order matters.
                const r1 = MathUtils.randomUniform(n);
                const r2 = MathUtils.randomUniform(n);
                const v = velocities[i];
                const p = positions[i];
                const pb = personalBestPos[i];
                const newV = new Array(n);
                for (let k = 0; k < n; k++) {
                    newV[k] = w * v[k]
                        + (c1 * r1[k]) * (pb[k] - p[k])
                        + (c2 * r2[k]) * (this.bestX[k] - p[k]);
                }

                // Velocity clamping to [-vmax, +vmax].
                const vmax = 0.2 * (1 - 0.5 * iteration / maxIterations);
                for (let k = 0; k < n; k++) newV[k] = MathUtils.clip(newV[k], -vmax, vmax);
                velocities[i] = newV;

                // Update position with bounds clipping.
                const newP = new Array(n);
                for (let k = 0; k < n; k++) newP[k] = MathUtils.clip(p[k] + newV[k], 0, 1);
                positions[i] = newP;

                const fitness = yield positions[i];

                // Personal-best bookkeeping.
                if (fitness < personalBestFit[i]) {
                    personalBestFit[i] = fitness;
                    personalBestPos[i] = positions[i].slice();
                }
            }

            // Stagnation check against this.bestValue (the driver keeps
            // it current across all evals).
            if (lastGlobalBest - this.bestValue > improvementAtol) {
                stagnationCounter = 0;
                lastGlobalBest = this.bestValue;
            } else {
                stagnationCounter++;
            }

            if (stagnationCounter >= stagnationWindow) {
                // Reseed the worst half: rank by personal-best fitness
                // ascending (stable sort, like Python's sorted(key=...)).
                const ranked = new Array(swarmSize);
                for (let k = 0; k < swarmSize; k++) ranked[k] = k;
                ranked.sort((a, b) => personalBestFit[a] - personalBestFit[b]);
                const worst = ranked.slice(Math.floor(swarmSize / 2));
                for (const j of worst) {
                    positions[j] = MathUtils.randomUniform(n);
                    const u = MathUtils.randomUniform(n);
                    velocities[j] = u.map(v => (v - 0.5) * 0.2);
                    if (this.evaluations >= psoBudget) break;
                    const fNew = yield positions[j];
                    personalBestPos[j] = positions[j].slice();
                    personalBestFit[j] = fNew;
                }
                stagnationCounter = 0;
                lastGlobalBest = this.bestValue;
            }
        }

        // Polish stage: L-BFGS-B from the swarm best.
        yield* this._lbfgsPolishGen();
    }
}

// Simulated Annealing implementation
// Generalized Simulated Annealing constants, matching scipy.optimize.dual_annealing and the
// Python twin in humpday/optimizers/evolutionary_algorithms.py. The two factors are computed
// once as literals there and here: one of them needs a log-gamma, which JavaScript has no
// standard implementation of, and SimulatedAnnealing is in JS_EXACT so both sides must hold the
// identical number.
const GSA_QV = 2.62;
const GSA_QA = -5.0;
const GSA_T0 = 5230.0;
const GSA_RESTART_T = 0.1;
const GSA_TAIL_LIMIT = 1.0e8;
const GSA_MIN_VISIT_BOUND = 1.0e-10;
const GSA_FACTOR4P = 11.833986526687411;
const GSA_FACTOR6 = 8.054035404548971;
const GSA_T1 = 2.0737503625760247;


class SimulatedAnnealing extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'SimulatedAnnealing';
    }

    // Generalized Simulated Annealing with an L-BFGS-B local search: the algorithm
    // scipy.optimize.dual_annealing runs, not the spirit of it. A heavy-tailed Tsallis visiting
    // distribution scaled by the temperature, generalised Metropolis acceptance, the
    // T(i) = T0 (2^(qv-1) - 1) / ((i+2)^(qv-1) - 1) schedule, and re-annealing at the floor.
    //
    // What was here before was classic Metropolis with uniform proposals and geometric cooling.
    // Its proposals cannot make the long jumps a heavy tail gives, so it explored a
    // neighbourhood rather than a space: 3,195,457 times behind scipy on Rosenbrock.
    //
    // Twin of SimulatedAnnealing._run in evolutionary_algorithms.py.
    *_run() {
        const n = this.nDim;
        const lo = new Array(n).fill(0.0);
        const hi = new Array(n).fill(1.0);
        const span = [];
        for (let i = 0; i < n; i++) span.push(hi[i] - lo[i]);

        while (this.evaluations < this.nTrials) {
            const before = this.evaluations;
            const chainBudget = this.nTrials - Math.max(
                20, Math.floor((this.nTrials - this.evaluations) / 2)
            );

            let x = MathUtils.randomUniform(n);
            let e = yield x;
            let iteration = 0;

            while (this.evaluations < chainBudget) {
                const sStep = iteration + 2.0;
                const t2 = MathUtils.portableExp((GSA_QV - 1.0) * MathUtils.portableLog(sStep)) - 1.0;
                const temperature = GSA_T0 * GSA_T1 / t2;
                iteration += 1;

                if (temperature < GSA_RESTART_T) {
                    x = MathUtils.randomUniform(n);
                    e = yield x;
                    iteration = 0;
                    continue;
                }

                const temperatureStep = temperature / iteration;
                const bestBeforeChain = this.bestValue;

                for (let j = 0; j < 2 * n; j++) {
                    if (this.evaluations >= chainBudget) break;
                    const xVisit = this._gsaVisit(x, j, temperature, lo, hi, span);
                    const eNew = yield xVisit;
                    if (eNew < e) {
                        x = xVisit;
                        e = eNew;
                    } else {
                        const r = MathUtils.randomScalar();
                        const pqvTemp = 1.0 - ((1.0 - GSA_QA) * (eNew - e) / temperatureStep);
                        let pqv = 0.0;
                        if (pqvTemp > 0.0) {
                            pqv = MathUtils.portableExp(
                                MathUtils.portableLog(pqvTemp) / (1.0 - GSA_QA)
                            );
                        }
                        if (r <= pqv) {
                            x = xVisit;
                            e = eNew;
                        }
                    }
                }

                // The local search runs after a chain that improved on the best point: the
                // "dual" in dual_annealing. Annealing proposes a basin, the local method walks
                // down it, every chain rather than once at the end.
                if (this.bestValue < bestBeforeChain && this.evaluations < chainBudget) {
                    yield* this._lbfgsPolishGen();
                }
            }

            yield* this._lbfgsPolishGen();

            if (this.evaluations === before) break;
        }
    }

    // One draw from the Tsallis visiting distribution (Visita, reference [2] p. 405).
    _gsaVisit(x, step, temperature, lo, hi, span) {
        const n = x.length;
        const factor1 = MathUtils.portableExp(
            MathUtils.portableLog(temperature) / (GSA_QV - 1.0)
        );
        const factor4 = GSA_FACTOR4P * factor1;
        const sigmax = MathUtils.portableExp(
            -(GSA_QV - 1.0) * MathUtils.portableLog(GSA_FACTOR6 / factor4) / (3.0 - GSA_QV)
        );

        const oneVisit = () => {
            const a = MathUtils.gaussScalar();
            const b = MathUtils.gaussScalar();
            const den = MathUtils.portableExp(
                (GSA_QV - 1.0) * MathUtils.portableLog(Math.abs(b)) / (3.0 - GSA_QV)
            );
            return sigmax * a / den;
        };

        const wrap = (value, i) => {
            const a = value - lo[i];
            const b = (a % span[i]) + span[i];
            let out = (b % span[i]) + lo[i];
            if (Math.abs(out - lo[i]) < GSA_MIN_VISIT_BOUND) out += GSA_MIN_VISIT_BOUND;
            return out;
        };

        if (step < n) {
            const visits = [];
            for (let i = 0; i < n; i++) visits.push(oneVisit());
            const upperSample = MathUtils.randomScalar();
            const lowerSample = MathUtils.randomScalar();
            const out = [];
            for (let i = 0; i < n; i++) {
                let v = visits[i];
                if (v > GSA_TAIL_LIMIT) v = GSA_TAIL_LIMIT * upperSample;
                else if (v < -GSA_TAIL_LIMIT) v = -GSA_TAIL_LIMIT * lowerSample;
                out.push(wrap(v + x[i], i));
            }
            return out;
        }

        const out = x.slice();
        let visit = oneVisit();
        if (visit > GSA_TAIL_LIMIT) visit = GSA_TAIL_LIMIT * MathUtils.randomScalar();
        else if (visit < -GSA_TAIL_LIMIT) visit = -GSA_TAIL_LIMIT * MathUtils.randomScalar();
        const index = step - n;
        out[index] = wrap(visit + out[index], index);
        return out;
    }
}

// Genetic Algorithm implementation
class GeneticAlgorithm extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'GeneticAlgorithm';
    }

    *_run() {
        // Twin of GeneticAlgorithm._run in
        // humpday/optimizers/evolutionary_algorithms.py. Selection is
        // tournament-of-3 (distinct competitors via sample), crossover
        // is one-point, mutation is per-coordinate Bernoulli.
        const n = this.nDim;
        const popSize = Math.min(50, Math.max(20, n * 4));
        const mutationRate = 0.1;
        const crossoverRate = 0.8;

        // All initial draws happen before the first yield.
        let population = [];
        for (let i = 0; i < popSize; i++) population.push(MathUtils.randomUniform(n));
        let fitness = [];
        for (const ind of population) fitness.push(yield ind);

        const generations = Math.floor(this.nTrials / popSize);

        for (let gen = 0; gen < generations; gen++) {
            if (this.evaluations >= this.nTrials) break;

            const newPopulation = [];
            const newFitness = [];

            for (let i = 0; i < popSize; i++) {
                if (this.evaluations >= this.nTrials) break;

                const parent1 = this.tournamentSelection(population, fitness);
                const parent2 = this.tournamentSelection(population, fitness);

                const child = parent1.slice();

                // One-point crossover.
                if (MathUtils.randomScalar() < crossoverRate) {
                    const crossPoint = MathUtils.randInt(n);
                    for (let j = crossPoint; j < n; j++) child[j] = parent2[j];
                }

                // Per-coordinate mutation with uniform [-0.1, 0.1] noise.
                for (let j = 0; j < n; j++) {
                    if (MathUtils.randomScalar() < mutationRate) {
                        child[j] = Math.max(
                            0.0,
                            Math.min(1.0, child[j] + (MathUtils.randomScalar() - 0.5) * 0.2)
                        );
                    }
                }

                const fitnessVal = yield child;
                newPopulation.push(child);
                newFitness.push(fitnessVal);
            }

            population = newPopulation;
            fitness = newFitness;
        }
    }

    tournamentSelection(population, fitness) {
        // Tournament-of-3: three DISTINCT indices (partial Fisher-Yates
        // sample, matching _A.random_choice(..., replace=False)); return
        // a copy of the lowest-fitness competitor (first minimum in draw
        // order, like Python's min(competitors, key=...)).
        const indices = new Array(population.length);
        for (let k = 0; k < population.length; k++) indices[k] = k;
        const competitors = MathUtils.sample(indices, 3);
        let bestIdx = competitors[0];
        for (let k = 1; k < competitors.length; k++) {
            if (fitness[competitors[k]] < fitness[bestIdx]) bestIdx = competitors[k];
        }
        return population[bestIdx].slice();
    }
}

// Random Search implementation
class RandomSearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'RandomSearch';
    }

    *_run() {
        // Statement-for-statement twin of RandomSearch._run in
        // humpday/optimizers/evolutionary_algorithms.py.
        while (this.evaluations < this.nTrials) {
            yield MathUtils.randomUniform(this.nDim);
        }
    }
}

// Bayesian optimization: a Gaussian-process surrogate with the expected-improvement acquisition.
//
// Twin of BayesianOpt in humpday/optimizers/evolutionary_algorithms.py, statement for statement
// (#408): the same RBF kernel (length scale 0.2, signal variance 1, noise 1e-6) written the way
// the pure backend writes it, the same 128-point conditioning set, the same Cholesky with the
// same jitter retries, the factor cached once per observation set, the same linear solve, EI
// with xi = 0.01 over the exact normal CDF, and the same acquisition search -- 64 uniform
// candidates, then six rounds of coordinate refinement on the surrogate. It is a *_run()
// generator like its siblings, drawing through MathUtils.randomUniform, so usePortableRng()
// reaches it and ask/tell works. tests/test_js_bayesopt_gp.py holds the posterior and the
// acquisition to Python's at 1e-10.
//
// Not in JS_EXACT: the kernel calls Math.exp and the CDF erf, and Python's math.exp / math.erf
// are the platform libm, which is not promised to round the same way in the last bit.
//
// What stood here before #408 took the five nearest observations, predicted with
// inverse-distance weights, and called exp(-2 * nearestDistance) the uncertainty -- largest at
// the points already sampled, so the acquisition preferred to resample what it knew.
class BayesianOpt extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'BayesianOpt';
        this.XObserved = [];   // list of points
        this.yObserved = [];   // list of values
        this.lengthScale = 0.2;
        this.signalVariance = 1.0;
        this.noiseVariance = 1e-6;
        this._posteriorStamp = null;
        this._posterior = null;
    }

    // How many observations the GP conditions on: the best half, where the optimum is, and the
    // newest half, what the acquisition was just told. Uncapped, the cubic solve makes a run
    // quartic in the budget (#330). Twin of _GP_MAX_OBSERVATIONS.
    static GP_MAX_OBSERVATIONS = 128;

    // How hard the acquisition is searched: the measured knee recorded beside
    // _ACQ_CANDIDATES / _ACQ_REFINEMENTS in the Python port.
    static ACQ_CANDIDATES = 64;
    static ACQ_REFINEMENTS = 6;

    _conditioningSet() {
        const n = this.XObserved.length;
        const cap = BayesianOpt.GP_MAX_OBSERVATIONS;
        if (n <= cap) return [this.XObserved, this.yObserved];
        const half = Math.floor(cap / 2);
        // Python's sorted() is stable; so is Array.prototype.sort (ES2019).
        const byValue = [...Array(n).keys()]
            .sort((i, j) => this.yObserved[i] - this.yObserved[j])
            .slice(0, half);
        const keep = new Set(byValue);
        for (let i = n - half; i < n; i++) keep.add(i);
        const order = [...keep].sort((a, b) => a - b);
        return [order.map(i => this.XObserved[i]), order.map(i => this.yObserved[i])];
    }

    *_run() {
        const nInitial = Math.min(5, Math.max(2, this.nDim));

        for (let i = 0; i < nInitial; i++) {
            if (this.evaluations >= this.nTrials) break;
            const x = MathUtils.randomUniform(this.nDim);
            const y = yield x;
            this.XObserved.push(x);
            this.yObserved.push(y);
        }

        // Reserve budget for a final L-BFGS-B descent on the objective from the best point
        // found. A hybrid design choice, not a port of scikit-optimize: gp_minimize's L-BFGS-B
        // minimises the acquisition function, on the surrogate, and spends no objective
        // evaluations (#408). Ours spends them because the GP alone plateaus at its RBF
        // smoothing floor. 20·nDim evaluations, capped at half the budget, as in Python.
        const polishReserve = Math.min(20 * this.nDim, Math.floor(this.nTrials / 2));
        const loopBudget = Math.max(this.evaluations, this.nTrials - polishReserve);

        while (this.evaluations < loopBudget) {
            let xNext;
            try {
                xNext = this._optimizeAcquisition();
            } catch (e) {
                // Any failure in the GP machinery falls back to a random sample, as in Python.
                xNext = MathUtils.randomUniform(this.nDim);
            }
            const yNext = yield xNext;
            this.XObserved.push(xNext);
            this.yObserved.push(yNext);
        }

        yield* this._lbfgsPolishGen();
    }

    // ---- GP machinery ----

    // RBF kernel for every pair (X1[i], X2[j]). The squared distance is |a|^2 + |b|^2 - 2 a.b
    // with every sum a left fold from 0.0, which is how the pure backend computes it.
    _kernelMatrix(X1, X2) {
        const scaleSq = this.lengthScale * this.lengthScale;
        const sig = this.signalVariance;
        const sqNorm = (row) => {
            let s = 0.0;
            for (let k = 0; k < row.length; k++) s = s + row[k] * row[k];
            return s;
        };
        const norms2 = X2.map(sqNorm);
        const out = new Array(X1.length);
        for (let i = 0; i < X1.length; i++) {
            const a = X1[i];
            const n1 = sqNorm(a);
            const row = new Array(X2.length);
            for (let j = 0; j < X2.length; j++) {
                const b = X2[j];
                let cross = 0.0;
                for (let k = 0; k < a.length; k++) cross += a[k] * b[k];
                const sq = n1 + norms2[j] - 2.0 * cross;
                row[j] = sig * Math.exp(-0.5 * sq / scaleSq);
            }
            out[i] = row;
        }
        return out;
    }

    // Cholesky-Banachiewicz in the loop order of humpday._array_pure_linalg.cholesky.
    // Throws if A is not positive definite.
    static _cholesky(A) {
        const n = A.length;
        const L = Array.from({ length: n }, () => new Array(n).fill(0.0));
        for (let i = 0; i < n; i++) {
            const Li = L[i];
            for (let j = 0; j <= i; j++) {
                const Lj = L[j];
                let s = 0.0;
                for (let k = 0; k < j; k++) s += Li[k] * Lj[k];
                if (i === j) {
                    const d = A[i][i] - s;
                    if (d <= 0.0) throw new Error('cholesky: matrix is not positive definite');
                    Li[j] = Math.sqrt(d);
                } else {
                    Li[j] = (A[i][j] - s) / Lj[j];
                }
            }
        }
        return L;
    }

    // Solve A x = b by Gaussian elimination with partial pivoting: the twin of
    // humpday._array_pure_linalg.solve, which is what the Python port calls on L and L^T.
    static _solve(A, b) {
        const n = A.length;
        const M = A.map(row => [...row]);
        const rhs = [...b];
        let scale = 0.0;
        for (const row of M) for (const v of row) if (Math.abs(v) > scale) scale = Math.abs(v);
        const pivotFloor = 1e-14 * scale;

        for (let k = 0; k < n; k++) {
            let pivotRow = k;
            let pivotVal = Math.abs(M[k][k]);
            for (let i = k + 1; i < n; i++) {
                const v = Math.abs(M[i][k]);
                if (v > pivotVal) {
                    pivotVal = v;
                    pivotRow = i;
                }
            }
            if (pivotVal <= pivotFloor) throw new Error('solve: singular matrix (pivot below tolerance)');
            if (pivotRow !== k) {
                [M[k], M[pivotRow]] = [M[pivotRow], M[k]];
                [rhs[k], rhs[pivotRow]] = [rhs[pivotRow], rhs[k]];
            }
            const pivot = M[k][k];
            for (let i = k + 1; i < n; i++) {
                const factor = M[i][k] / pivot;
                if (factor === 0.0) continue;
                const Mi = M[i];
                const Mk = M[k];
                Mi[k] = 0.0;
                for (let j = k + 1; j < n; j++) Mi[j] -= factor * Mk[j];
                rhs[i] -= factor * rhs[k];
            }
        }

        const x = new Array(n).fill(0.0);
        for (let i = n - 1; i >= 0; i--) {
            let s = rhs[i];
            const Mi = M[i];
            for (let j = i + 1; j < n; j++) s -= Mi[j] * x[j];
            x[i] = s / Mi[i];
        }
        return x;
    }

    static _transpose(A) {
        return A[0].map((_, j) => A.map(row => row[j]));
    }

    // Factorise the kernel once for the current observation set, and cache it. Returns
    // [XObs, L, alpha], or null when the kernel will not factorise. Twin of _gp_posterior.
    _gpPosterior() {
        const [XObs, yObs] = this._conditioningSet();
        const nObs = XObs.length;
        const stamp = `${nObs}:${this.XObserved.length}:${this.lengthScale}`;
        if (this._posteriorStamp === stamp) return this._posterior;

        const K = this._kernelMatrix(XObs, XObs);
        for (let i = 0; i < nObs; i++) K[i][i] += this.noiseVariance;

        // Four attempts, adding 1e-8, then 1e-7, then 1e-6 to the diagonal between them.
        let jitter = 0.0;
        let L = null;
        for (let attempt = 0; attempt < 4; attempt++) {
            try {
                L = BayesianOpt._cholesky(K);
                break;
            } catch (e) {
                jitter = jitter > 0 ? Math.max(1e-8, jitter * 10) : 1e-8;
                for (let i = 0; i < nObs; i++) K[i][i] += jitter;
            }
        }

        let posterior = null;
        if (L !== null) {
            // alpha = K^-1 y as L^-T (L^-1 y).
            let alpha = BayesianOpt._solve(L, yObs);
            alpha = BayesianOpt._solve(BayesianOpt._transpose(L), alpha);
            posterior = [XObs, L, alpha];
        }

        this._posteriorStamp = stamp;
        this._posterior = posterior;
        return posterior;
    }

    // Posterior mean and standard deviation at one query point. Twin of _gp_predict.
    _gpPredict(xQuery) {
        const posterior = this._gpPosterior();
        if (posterior === null) {
            // Pathological kernel: a flat prior over the conditioning set.
            const [, yObs] = this._conditioningSet();
            const nObs = Math.max(1, yObs.length);
            let mu = 0.0;
            for (const y of yObs) mu = mu + y;
            mu = mu / nObs;
            let v = 0.0;
            for (const y of yObs) v = v + (y - mu) ** 2;
            v = v / nObs;
            return [mu, Math.sqrt(Math.max(v, 1e-8))];
        }

        const [XObs, L, alpha] = posterior;
        const nObs = XObs.length;

        // k(X, xQuery); k(xQuery, xQuery) is the signal variance, since exp(0) = 1.
        const KsCol = this._kernelMatrix(XObs, [xQuery]).map(row => row[0]);
        const Kss = this.signalVariance;

        let mu = 0.0;
        for (let i = 0; i < nObs; i++) mu = mu + KsCol[i] * alpha[i];

        // var = K_ss - |L^-1 K_s|^2
        const v = BayesianOpt._solve(L, KsCol);
        let vDotV = 0.0;
        for (let i = 0; i < nObs; i++) vDotV = vDotV + v[i] * v[i];
        const variance = Math.max(Kss - vDotV, 1e-8);

        return [mu, Math.sqrt(variance)];
    }

    // ---- Acquisition ----

    _expectedImprovement(x) {
        const [mu, sigma] = this._gpPredict(x);
        let fBest = Infinity;
        for (const y of this.yObserved) if (y < fBest) fBest = y;
        const improvement = fBest - mu - 0.01;
        if (sigma <= 0) return 0.0;
        const Z = improvement / sigma;
        return improvement * BayesianOpt._normalCdf(Z) + sigma * BayesianOpt._normalPdf(Z);
    }

    // The acquisition, under the name callers of the earlier ports used.
    acquisitionFunction(x) {
        return this._expectedImprovement(x);
    }

    // Maximise EI: a broad sample, then a coordinate pattern search on the surrogate, halving
    // the step when a round finds nothing. Twin of _optimize_acquisition; none of it is charged
    // to the budget.
    _optimizeAcquisition() {
        let bestX = null;
        let bestEi = -Infinity;
        for (let c = 0; c < BayesianOpt.ACQ_CANDIDATES; c++) {
            const x = MathUtils.randomUniform(this.nDim);
            const ei = this._expectedImprovement(x);
            if (ei > bestEi) {
                bestEi = ei;
                bestX = x;
            }
        }

        if (bestX === null) return MathUtils.randomUniform(this.nDim);

        let step = 0.25;
        for (let r = 0; r < BayesianOpt.ACQ_REFINEMENTS; r++) {
            let improved = false;
            for (let i = 0; i < this.nDim; i++) {
                for (const sign of [1.0, -1.0]) {
                    const trial = [...bestX];
                    trial[i] = Math.min(1.0, Math.max(0.0, trial[i] + sign * step));
                    const ei = this._expectedImprovement(trial);
                    if (ei > bestEi) {
                        bestEi = ei;
                        bestX = trial;
                        improved = true;
                    }
                }
            }
            if (!improved) step *= 0.5;
        }

        return MathUtils.clipArray(bestX, 0, 1);
    }

    // Standard-normal CDF through erf, as Python's _normal_cdf does with math.erf.
    static _normalCdf(x) {
        return 0.5 * (1.0 + BayesianOpt._erf(x / Math.sqrt(2.0)));
    }

    static _normalPdf(x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2.0 * Math.PI);
    }

    // erf, ported from fdlibm's s_erf.c (the rational approximations behind most C libraries'
    // erf, error under one ulp). JavaScript has no Math.erf. The Abramowitz-Stegun 7.1.26 used
    // before carries 1.5e-7 of error into the CDF, which put JS expected improvement 3e-8 away
    // from Python's on a seven-point design; against math.erf this agrees to 2.8e-16.
    static _erf(x) {
        if (Number.isNaN(x)) return x;
        const sign = x < 0 ? -1.0 : 1.0;
        const ax = Math.abs(x);
        if (ax < 0.84375) {
            if (ax < 3.725290298461914e-9) return x + 1.28379167095512586316e-01 * x;
            const z = x * x;
            const r = 1.28379167095512558561e-01 + z * (-3.25042107247001499370e-01 + z * (-2.84817495755985104766e-02 + z * (-5.77027029648944159157e-03 + z * -2.37630166566501626084e-05)));
            const s = 1.0 + z * (3.97917223959155352819e-01 + z * (6.50222499887672944485e-02 + z * (5.08130628187576562776e-03 + z * (1.32494738004321644526e-04 + z * -3.96022827877536812320e-06))));
            return x + x * (r / s);
        }
        if (ax < 1.25) {
            const s = ax - 1.0;
            const P = -2.36211856075265944077e-03 + s * (4.14856118683748331666e-01 + s * (-3.72207876035701323847e-01 + s * (3.18346619901161753674e-01 + s * (-1.10894694282396677476e-01 + s * (3.54783043256182359371e-02 + s * -2.16637559486879084300e-03)))));
            const Q = 1.0 + s * (1.06420880400844228286e-01 + s * (5.40397917702171048937e-01 + s * (7.18286544141962662868e-02 + s * (1.26171219808761642112e-01 + s * (1.36370839120290507362e-02 + s * 1.19844998467991074170e-02)))));
            return sign * (8.45062911510467529297e-01 + P / Q);
        }
        if (ax >= 6.0) return sign * 1.0;
        const s = 1.0 / (ax * ax);
        let R, S;
        if (ax < 1.0 / 0.35) {
            R = -9.86494403484714822705e-03 + s * (-6.93858572707181764372e-01 + s * (-1.05586262253232909814e+01 + s * (-6.23753324503260060396e+01 + s * (-1.62396669462573470355e+02 + s * (-1.84605092906711035994e+02 + s * (-8.12874355063065934246e+01 + s * -9.81432934416914548592e+00))))));
            S = 1.0 + s * (1.96512716674392571292e+01 + s * (1.37657754143519042600e+02 + s * (4.34565877475229228821e+02 + s * (6.45387271733267880336e+02 + s * (4.29008140027567833386e+02 + s * (1.08635005541779435134e+02 + s * (6.57024977031928170135e+00 + s * -6.04244152148580987438e-02)))))));
        } else {
            R = -9.86494292470009928597e-03 + s * (-7.99283237680523006574e-01 + s * (-1.77579549177547519889e+01 + s * (-1.60636384855821916062e+02 + s * (-6.37566443368389627722e+02 + s * (-1.02509513161107724954e+03 + s * -4.83519191608651397019e+02)))));
            S = 1.0 + s * (3.03380607434824582924e+01 + s * (3.25792512996573918826e+02 + s * (1.53672958608443695994e+03 + s * (3.19985821950859553908e+03 + s * (2.55305040643316442583e+03 + s * (4.74528541206955367215e+02 + s * -2.24409524465858183362e+01))))));
        }
        // z is |x| with the low 32 bits of its mantissa cleared, so z*z is exact.
        const buf = new DataView(new ArrayBuffer(8));
        buf.setFloat64(0, ax);
        buf.setUint32(4, 0);
        const z = buf.getFloat64(0);
        const r = Math.exp(-z * z - 0.5625) * Math.exp((z - ax) * (z + ax) + R / S);
        return sign * (1.0 - r / ax);
    }
}

class CMAEvolutionStrategy extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'CMAEvolutionStrategy';
    }

    // IPOP-CMA-ES (Auger & Hansen 2005, "A Restart CMA Evolution
    // Strategy with Increasing Population Size", CEC 2005). The
    // Hansen-standard inner CMA loop is wrapped in a restart layer:
    // when any of TolFun / TolX / ConditionCov fires, all state is
    // reset and λ is doubled. Mirrors humpday/optimizers/
    // evolutionary_algorithms.py::CMAEvolutionStrategy step-for-step.
    optimize() {
        const n = this.nDim;

        // Reserve budget for L-BFGS-B polish (mirrors Python port).
        const polishReserve = Math.min(20 * n, Math.floor(this.nTrials / 2));
        const cmaesBudget = Math.max(this.evaluations, this.nTrials - polishReserve);

        // IPOP termination constants.
        const IPOP_INCPOPSIZE = 2.0;
        const IPOP_TOLFUN = 1e-12;
        const IPOP_TOLX_FACTOR = 1e-12;
        const IPOP_CONDITION_COV = 1e14;
        const IPOP_TOLFUN_HISTORY = 10;

        const baseLambda = Math.min(50, 4 + Math.floor(3 * Math.log(n)));
        let restartCount = 0;

        // Outer IPOP loop: keep restarting (with growing λ) until budget
        // is exhausted. self.bestX / self.bestValue persist across
        // restarts via the base Optimizer's evaluate(), so the best
        // point found in any prior run is preserved.
        while (this.evaluations < cmaesBudget) {
            // Hansen-recommended parameters at the current population size.
            let lambda_ = Math.floor(baseLambda * Math.pow(IPOP_INCPOPSIZE, restartCount));
            lambda_ = Math.min(lambda_, cmaesBudget - this.evaluations);
            lambda_ = Math.max(lambda_, 4);
            const mu = Math.floor(lambda_ / 2);
            if (mu < 1) break;

            // Recombination weights w_i = log(μ + 0.5) − log(i + 1).
            const wRaw = new Array(mu);
            for (let i = 0; i < mu; i++) wRaw[i] = Math.log(mu + 0.5) - Math.log(i + 1);
            const sumW = wRaw.reduce((s, w) => s + w, 0);
            const weights = wRaw.map(w => w / sumW);
            const sumWsq = weights.reduce((s, w) => s + w * w, 0);
            const mueff = 1.0 / sumWsq;

            // Adaptation constants.
            const cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
            const cs = (mueff + 2) / (n + mueff + 5);
            const c1 = 2 / ((n + 1.3) ** 2 + mueff);
            const cmu = Math.min(
                1 - c1,
                2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)
            );
            const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs;
            // E||N(0, I_n)||: the scale the hsig gate and the step-size update
            // compare the evolution path against (twin of the Python port).
            const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));

            // Fresh state per restart.
            let mean = new Array(n);
            for (let i = 0; i < n; i++) mean[i] = 0.3 + 0.4 * MathUtils.randomScalar();
            let sigma = 0.2;
            let C = Linalg.eye(n);
            let pc = new Array(n).fill(0);
            let ps = new Array(n).fill(0);
            let invsqrtC = Linalg.eye(n);

            // TolFun window: rolling history of best-of-generation values.
            const tolfunWindow = Math.max(IPOP_TOLFUN_HISTORY, Math.floor(30 * n / lambda_));
            const fbestHistory = [];

            let generation = 0;
            const maxGenerations = this.nTrials;
            let converged = false;

            while (this.evaluations < cmaesBudget && generation < maxGenerations && !converged) {
                generation += 1;

                // Sample λ offspring from N(mean, σ² C) via Cholesky.
                let L_C;
                try {
                    L_C = Linalg.cholesky(C);
                } catch (e) {
                    L_C = Linalg.eye(n);
                }

                const population = [];
                for (let k = 0; k < lambda_; k++) {
                    if (this.evaluations >= cmaesBudget) break;
                    const stdZ = new Array(n);
                    for (let i = 0; i < n; i++) stdZ[i] = this._gaussian();
                    const z = Linalg.matvec(L_C, stdZ);
                    const x = new Array(n);
                    for (let i = 0; i < n; i++) {
                        x[i] = MathUtils.clip(mean[i] + sigma * z[i], 0, 1);
                    }
                    const f = this.evaluate(x);
                    population.push({ x, z, f });
                }

                if (!population.length) break;
                if (population.length < mu) break;

                population.sort((a, b) => a.f - b.f);

                // Recombination.
                const oldMean = mean.slice();
                mean = new Array(n).fill(0);
                for (let i = 0; i < mu; i++) {
                    for (let j = 0; j < n; j++) mean[j] += weights[i] * population[i].x[j];
                }

                // Evolution paths.
                const y = new Array(n);
                for (let i = 0; i < n; i++) y[i] = (mean[i] - oldMean[i]) / sigma;

                const psFactor = Math.sqrt(cs * (2 - cs) * mueff);
                const invsqrtY = Linalg.matvec(invsqrtC, y);
                for (let i = 0; i < n; i++) {
                    ps[i] = (1 - cs) * ps[i] + psFactor * invsqrtY[i];
                }

                let psNorm = 0;
                for (let i = 0; i < n; i++) psNorm += ps[i] * ps[i];
                psNorm = Math.sqrt(psNorm);

                const hsigDenom = Math.sqrt(1 - Math.pow(1 - cs, 2 * generation));
                const hsig = psNorm / hsigDenom < (1.4 + 2 / (n + 1)) * chiN ? 1 : 0;

                const pcFactor = hsig * Math.sqrt(cc * (2 - cc) * mueff);
                for (let i = 0; i < n; i++) {
                    pc[i] = (1 - cc) * pc[i] + pcFactor * y[i];
                }

                // Rank-μ update.
                const weightedDiffs = Linalg.zeros(n, n);
                for (let i = 0; i < mu; i++) {
                    const diff = new Array(n);
                    for (let j = 0; j < n; j++) {
                        diff[j] = (population[i].x[j] - oldMean[j]) / sigma;
                    }
                    for (let r = 0; r < n; r++) {
                        for (let c = 0; c < n; c++) {
                            weightedDiffs[r][c] += weights[i] * diff[r] * diff[c];
                        }
                    }
                }

                // hsig = 0 leaves the rank-one term short of its variance; the
                // standard recurrence keeps that much of the old C instead.
                const base = 1 - c1 - cmu + c1 * (1 - hsig) * cc * (2 - cc);
                const newC = Linalg.zeros(n, n);
                for (let r = 0; r < n; r++) {
                    for (let c = 0; c < n; c++) {
                        newC[r][c] =
                            base * C[r][c] +
                            c1 * pc[r] * pc[c] +
                            cmu * weightedDiffs[r][c];
                    }
                }
                C = newC;

                // Ensure C stays positive definite.
                try {
                    const { eigvals } = Linalg.eigh(C);
                    let minEig = Infinity;
                    for (let i = 0; i < n; i++) {
                        if (eigvals[i] < minEig) minEig = eigvals[i];
                    }
                    if (minEig < 1e-14) {
                        const shift = 1e-14 - minEig;
                        for (let k = 0; k < n; k++) C[k][k] += shift;
                    }
                } catch (e) {
                    /* leave C as-is */
                }

                // Refresh invsqrtC and capture eig_max + cond(C) for IPOP.
                let eigMax = 1.0;
                let condC = 1.0;
                try {
                    const { eigvals: D, eigvecs: B } = Linalg.eigh(C);
                    const Dinvsqrt = D.map(d => 1.0 / Math.sqrt(Math.max(d, 1e-14)));
                    const Ddiag = Linalg.diag(Dinvsqrt);
                    const tmp = Linalg.matmul(B, Ddiag);
                    invsqrtC = Linalg.matmul(tmp, Linalg.transpose(B));
                    eigMax = Math.max(...D);
                    const eigMin = Math.max(Math.min(...D), 1e-30);
                    condC = eigMax / eigMin;
                } catch (e) {
                    invsqrtC = Linalg.eye(n);
                }

                // Step-size update.
                sigma = sigma * Math.exp((cs / damps) * (psNorm / chiN - 1));

                // ---- IPOP termination checks ----
                fbestHistory.push(population[0].f);
                if (fbestHistory.length > tolfunWindow) fbestHistory.shift();

                if (fbestHistory.length >= tolfunWindow) {
                    const fMax = Math.max(...fbestHistory);
                    const fMin = Math.min(...fbestHistory);
                    if (fMax - fMin < IPOP_TOLFUN) {
                        converged = true;
                        continue;
                    }
                }

                if (sigma * Math.sqrt(eigMax) < IPOP_TOLX_FACTOR) {
                    converged = true;
                    continue;
                }

                if (condC > IPOP_CONDITION_COV) {
                    converged = true;
                    continue;
                }
            }

            // Inner loop ended. If it ended on convergence, restart with
            // a larger population; if it ran out of budget, fall through
            // to the polish stage.
            restartCount++;
            if (!converged) break;
        }

        // Polish stage: L-BFGS-B from the CMA-ES best (shared on base).
        this._lbfgsPolish();

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    // Box-Muller transform for Gaussian sampling — caches the spare
    // sample (each Box-Muller iteration produces two independent
    // normals).
    _gaussian() {
        if (this._spareGaussian !== undefined) {
            const s = this._spareGaussian;
            this._spareGaussian = undefined;
            return s;
        }
        const u = MathUtils.randomScalar();
        const v = MathUtils.randomScalar();
        const r = Math.sqrt(-2 * Math.log(Math.max(u, 1e-300)));
        const theta = 2 * Math.PI * v;
        this._spareGaussian = r * Math.sin(theta);
        return r * Math.cos(theta);
    }
}

// Firefly Algorithm
class FireflyAlgorithm extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'FireflyAlgorithm';
    }

    *_run() {
        // Twin of FireflyAlgorithm._run in
        // humpday/optimizers/evolutionary_algorithms.py.
        const n = this.nDim;
        const polishReserve = Math.min(20 * n, Math.floor(this.nTrials / 2));
        const fireflyBudget = Math.max(this.evaluations, this.nTrials - polishReserve);

        const nFireflies = Math.min(15, Math.max(2, Math.floor(fireflyBudget / 5)));
        const alpha0 = 0.2;  // Initial randomness coefficient.
        const beta0 = 1.0;   // Attractiveness at zero distance.
        const gamma = 1.0;   // Light-absorption coefficient.
        const alphaDamp = 0.99;  // mealpy FFA's alpha_damp.
        let alpha = alpha0;

        // Initialize fireflies — all draws before the first yield.
        const fireflies = [];
        for (let i = 0; i < nFireflies; i++) fireflies.push(MathUtils.randomUniform(n));
        const intensities = [];
        for (const fly of fireflies) intensities.push(yield fly);

        while (this.evaluations < fireflyBudget) {
            const evalsAtSweepStart = this.evaluations;
            for (let i = 0; i < nFireflies; i++) {
                for (let j = 0; j < nFireflies; j++) {
                    if (this.evaluations >= fireflyBudget) break;

                    if (intensities[j] < intensities[i]) {  // j is brighter
                        const r = MathUtils.norm(
                            MathUtils.subtract(fireflies[i], fireflies[j])
                        );
                        // portableExp, not Math.exp: V8's exp is
                        // fdlibm-derived and diverges from libm in the
                        // last ulp.
                        const beta = beta0 * MathUtils.portableExp(-gamma * r * r);

                        // Move firefly i toward the brighter firefly j,
                        // with a small random jitter.
                        const g = MathUtils.randomNormal(n);
                        const fi = fireflies[i];
                        const fj = fireflies[j];
                        const moved = new Array(n);
                        for (let k = 0; k < n; k++) {
                            moved[k] = MathUtils.clip(
                                (fi[k] + beta * (fj[k] - fi[k])) + alpha * g[k],
                                0, 1
                            );
                        }
                        fireflies[i] = moved;

                        if (this.evaluations < fireflyBudget) {
                            intensities[i] = yield fireflies[i];
                        }
                    }
                }
            }
            // Anneal α at the end of each outer (i, j) sweep.
            alpha *= alphaDamp;

            // Termination guard: a sweep with no evaluations means the
            // swarm has collapsed — no future sweep can differ.
            if (this.evaluations === evalsAtSweepStart) break;
        }

        // Polish stage: L-BFGS-B from the firefly best.
        yield* this._lbfgsPolishGen();
    }
}

// Ant Colony Optimization
// ACOR — Ant Colony Optimization for continuous domains (Socha &
// Dorigo, 2008). Maintains an archive of the k best solutions and
// samples new candidates from a mixture of Gaussian kernels centred
// on archive points. Mirrors the Python port; replaces humpday's
// previous discrete-bin "continuous via discretization" ACO that was
// ~457× off mealpy.swarm_based.ACOR on the sphere benchmark.
class AntColonyOpt extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'AntColonyOpt';
    }

    optimize() {
        const n = this.nDim;
        const k = Math.min(50, Math.max(10, Math.floor(this.nTrials / 10)));
        const nAnts = Math.min(25, Math.max(5, Math.floor(this.nTrials / 20)));
        const q = 0.5;
        const xi = 1.0;

        // Rank weights: w_i ∝ Gaussian on rank i, normalised.
        const weights = new Array(k);
        for (let i = 0; i < k; i++) {
            weights[i] = Math.exp(-(i * i) / (2.0 * q * q * k * k)) /
                         (q * k * Math.sqrt(2.0 * Math.PI));
        }
        let wsum = 0;
        for (let i = 0; i < k; i++) wsum += weights[i];
        for (let i = 0; i < k; i++) weights[i] /= wsum;

        // Initial archive: k uniform samples, sorted by f.
        const archive = [];
        for (let i = 0; i < k && this.evaluations < this.nTrials; i++) {
            const x = new Array(n);
            for (let d = 0; d < n; d++) x[d] = MathUtils.randomScalar();
            archive.push({ x, f: this.evaluate(x) });
        }
        if (!archive.length) {
            return {
                bestValue: this.bestValue,
                bestX: this.bestX,
                evaluations: this.evaluations,
                success: true,
                path: this.trackPath ? this.path : null
            };
        }
        archive.sort((a, b) => a.f - b.f);

        while (this.evaluations < this.nTrials) {
            // Per-kernel per-dim sigma = xi · mean |x_l[d] − x_i[d]|.
            const sigmasByKernel = new Array(archive.length);
            for (let i = 0; i < archive.length; i++) {
                const xi_vec = archive[i].x;
                const sigma = new Array(n).fill(0);
                for (let l = 0; l < archive.length; l++) {
                    if (l === i) continue;
                    const xl = archive[l].x;
                    for (let d = 0; d < n; d++) sigma[d] += Math.abs(xl[d] - xi_vec[d]);
                }
                const denom = Math.max(1, archive.length - 1);
                for (let d = 0; d < n; d++) sigma[d] = xi * sigma[d] / denom;
                sigmasByKernel[i] = sigma;
            }

            const newSolutions = [];
            for (let a = 0; a < nAnts; a++) {
                if (this.evaluations >= this.nTrials) break;

                // Roulette-pick a kernel by weights.
                const r = MathUtils.randomScalar();
                let cum = 0;
                let kernelIdx = archive.length - 1;
                for (let i = 0; i < archive.length; i++) {
                    cum += weights[i];
                    if (r <= cum) { kernelIdx = i; break; }
                }

                const center = archive[kernelIdx].x;
                const sigma = sigmasByKernel[kernelIdx];
                const xNew = new Array(n);
                for (let d = 0; d < n; d++) {
                    const s = Math.max(sigma[d], 1e-12);
                    const z = this._gauss();
                    xNew[d] = Math.max(0, Math.min(1, center[d] + s * z));
                }
                newSolutions.push({ x: xNew, f: this.evaluate(xNew) });
            }

            for (const sol of newSolutions) archive.push(sol);
            archive.sort((a, b) => a.f - b.f);
            archive.length = k;
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
        const u = MathUtils.randomScalar();
        const v = MathUtils.randomScalar();
        const r = Math.sqrt(-2 * Math.log(Math.max(u, 1e-300)));
        const theta = 2 * Math.PI * v;
        this._spare = r * Math.sin(theta);
        return r * Math.cos(theta);
    }
}

// Harmony Search
class HarmonySearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'HarmonySearch';
    }

    *_run() {
        // Twin of HarmonySearch._run in
        // humpday/optimizers/evolutionary_algorithms.py.
        const HMS = Math.min(20, Math.max(5, this.nDim * 2));
        const HMCR = 0.9;
        const PAR = 0.3;

        const harmonyMemory = [];
        for (let k = 0; k < HMS; k++) {
            if (this.evaluations >= this.nTrials) break;
            const harmony = MathUtils.randomUniform(this.nDim);
            const fitness = yield harmony;
            harmonyMemory.push({ harmony, fitness });
        }

        while (this.evaluations < this.nTrials) {
            const newHarmony = new Array(this.nDim).fill(0);

            for (let j = 0; j < this.nDim; j++) {
                if (MathUtils.randomScalar() < HMCR) {
                    const selected = MathUtils.choice(harmonyMemory);
                    let value = selected.harmony[j];

                    if (MathUtils.randomScalar() < PAR) {
                        value = Math.max(0.0, Math.min(1.0, value + 0.1 * MathUtils.randomNormal(1)[0]));
                    }

                    newHarmony[j] = value;
                } else {
                    newHarmony[j] = MathUtils.randomScalar();
                }
            }

            const newFitness = yield newHarmony;

            harmonyMemory.sort((a, b) => a.fitness - b.fitness);
            if (newFitness < harmonyMemory[harmonyMemory.length - 1].fitness) {
                harmonyMemory[harmonyMemory.length - 1] = {
                    harmony: newHarmony.slice(),
                    fitness: newFitness,
                };
            }
        }
    }
}

// (μ+λ) Evolution Strategy
class EvolutionStrategy extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'EvolutionStrategy';
    }

    *_run() {
        // Twin of EvolutionStrategy._run ((mu+lambda)-ES) in
        // humpday/optimizers/evolutionary_algorithms.py.
        const mu = 10;
        const lambda_ = Math.min(30, Math.floor(this.nTrials / 3));

        // Mutation strength, adapted by Rechenberg's 1/5 success rule. Previously a `const`,
        // which made this a fixed-radius sampler rather than an evolution strategy. Constants and
        // update order match the Python twin exactly; no RNG is drawn by the adaptation, so the
        // call sequence is unchanged and the two stay bit-exact.
        const TARGET_SUCCESS = 0.2;
        const ADAPT = 0.817;
        const SIGMA_MIN = 1e-12;
        const SIGMA_MAX = 0.5;
        let sigma = 0.2;

        let population = [];
        let fitness = [];
        for (let k = 0; k < mu; k++) {
            if (this.evaluations >= this.nTrials) break;
            const individual = MathUtils.randomUniform(this.nDim);
            const f = yield individual;
            population.push(individual);
            fitness.push(f);
        }

        while (this.evaluations < this.nTrials) {
            const offspring = [];
            const offspringFitness = [];
            let successes = 0;

            for (let k = 0; k < lambda_; k++) {
                if (this.evaluations >= this.nTrials) break;

                const parentIdx = MathUtils.randInt(population.length);
                const parent = population[parentIdx];

                const z = MathUtils.randomNormal(this.nDim);
                const child = MathUtils.clipArray(
                    parent.map((p, i) => p + sigma * z[i]), 0, 1
                );

                const childFitness = yield child;
                if (childFitness < fitness[parentIdx]) successes++;
                offspring.push(child);
                offspringFitness.push(childFitness);
            }

            if (offspring.length) {
                // Adapt before selection, so the rate refers to the parents that produced these
                // offspring — same order as the Python twin.
                const rate = successes / offspring.length;
                if (rate > TARGET_SUCCESS) {
                    sigma = Math.min(sigma / ADAPT, SIGMA_MAX);
                } else if (rate < TARGET_SUCCESS) {
                    sigma = Math.max(sigma * ADAPT, SIGMA_MIN);
                }

                const allIndividuals = population.concat(offspring);
                const allFitness = fitness.concat(offspringFitness);
                const indices = allFitness
                    .map((_, i) => i)
                    .sort((a, b) => allFitness[a] - allFitness[b])
                    .slice(0, mu);
                population = indices.map(i => allIndividuals[i]);
                fitness = indices.map(i => allFitness[i]);
            }
        }
    }
}

// Export evolutionary algorithms
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = {
        DifferentialEvolution, ParticleSwarm, SimulatedAnnealing, GeneticAlgorithm, RandomSearch,
        BayesianOpt, CMAEvolutionStrategy, FireflyAlgorithm, AntColonyOpt,
        HarmonySearch, EvolutionStrategy
    };
} else {
    // Browser environment
    window.DifferentialEvolution = DifferentialEvolution;
    window.ParticleSwarm = ParticleSwarm;
    window.SimulatedAnnealing = SimulatedAnnealing;
    window.GeneticAlgorithm = GeneticAlgorithm;
    window.RandomSearch = RandomSearch;
    window.BayesianOpt = BayesianOpt;
    window.CMAEvolutionStrategy = CMAEvolutionStrategy;
    window.FireflyAlgorithm = FireflyAlgorithm;
    window.AntColonyOpt = AntColonyOpt;
    window.HarmonySearch = HarmonySearch;
    window.EvolutionStrategy = EvolutionStrategy;
}

