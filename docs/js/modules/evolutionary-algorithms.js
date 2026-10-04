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

// Simplified Bayesian Optimization
class BayesianOpt extends Optimizer {
    // A Gaussian process, which is what this is supposed to be and was not (#408).
    //
    // What stood here took the five nearest observations, predicted with inverse-distance
    // weights, and called `exp(-2 * nearestDistance)` the uncertainty. That quantity is
    // largest at the points already sampled, so both it and the "exploration bonus" built
    // from it rewarded proximity to what was already known. With one observation at
    // (0.5, 0.5) the acquisition scored the observed point 0.136 and everywhere else 0.033 --
    // it preferred to resample the point it had. The Python port's GP scores that point
    // 7.7e-26 and the far corners 0.394.
    //
    // This is now the same RBF Gaussian process the Python port conditions: kernel matrix,
    // Cholesky factorisation, and expected improvement from the posterior mean and standard
    // deviation. Not bit-exact with Python -- BayesianOpt is not in JS_EXACT -- but the same
    // algorithm rather than a different one.

    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'BayesianOpt';
        this.observations = [];
        this.lengthScale = 0.2;
        this.signalVariance = 1.0;
        this.noiseVariance = 1e-6;
        this._posteriorStamp = null;
        this._posterior = null;
    }

    optimize() {
        const nInitial = Math.min(5, Math.max(2, this.nDim));

        // Uniform over the cube. The initial design used to put its first three draws in
        // `0.5 + (random - 0.5) * 0.3`, a box around the centre -- which is the defect #387
        // found in the benchmark objectives, here in an optimizer: a start that encodes where
        // the answer tends to be.
        for (let i = 0; i < nInitial && this.evaluations < this.nTrials; i++) {
            const x = MathUtils.randomUniform(this.nDim);
            this.observations.push({ x: [...x], y: this.evaluate(x) });
        }

        // Reserve budget for a final L-BFGS-B descent on the objective from the best point
        // found. This is a hybrid design choice, not a port of scikit-optimize: the comment
        // here used to claim `gp_minimize` finishes by polishing its best observation, and it
        // does not -- its L-BFGS-B minimises the acquisition function, on the surrogate, for
        // free. Ours spends real evaluations, and is worth them because it escapes the RBF
        // smoothing floor the GP alone plateaus at (#408).
        const polishReserve = Math.min(20 * this.nDim, Math.floor(this.nTrials / 2));
        const loopBudget = Math.max(this.evaluations, this.nTrials - polishReserve);

        while (this.evaluations < loopBudget) {
            const nextX = this.acquireNext();
            this.observations.push({ x: [...nextX], y: this.evaluate(nextX) });
        }

        this._lbfgsPolish();

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    // ---- the Gaussian process ----

    _kernel(a, b) {
        let sq = 0.0;
        for (let i = 0; i < a.length; i++) {
            const d = a[i] - b[i];
            sq += d * d;
        }
        return this.signalVariance * Math.exp(-0.5 * sq / (this.lengthScale * this.lengthScale));
    }

    // The kernel solve is cubic in the observation count and happens once per iteration, so
    // an uncapped set makes a run quartic in the budget (#330). Keep the best half and the
    // newest half: where the optimum is, and what the acquisition was last told.
    _conditioningSet() {
        const n = this.observations.length;
        if (n <= 128) return this.observations;
        const half = 64;
        const byValue = this.observations
            .map((o, i) => i)
            .sort((i, j) => this.observations[i].y - this.observations[j].y)
            .slice(0, half);
        const keep = new Set(byValue);
        for (let i = n - half; i < n; i++) keep.add(i);
        return [...keep].sort((a, b) => a - b).map(i => this.observations[i]);
    }

    // Cholesky of a symmetric positive-definite matrix, or null if it is not one.
    _cholesky(A) {
        const n = A.length;
        const L = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                let s = A[i][j];
                for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
                if (i === j) {
                    if (!(s > 0) || !isFinite(s)) return null;
                    L[i][j] = Math.sqrt(s);
                } else {
                    L[i][j] = s / L[j][j];
                }
            }
        }
        return L;
    }

    _forwardSolve(L, b) {
        const n = L.length;
        const y = new Array(n);
        for (let i = 0; i < n; i++) {
            let s = b[i];
            for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
            y[i] = s / L[i][i];
        }
        return y;
    }

    _backSolve(L, y) {
        const n = L.length;
        const x = new Array(n);
        for (let i = n - 1; i >= 0; i--) {
            let s = y[i];
            for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
            x[i] = s / L[i][i];
        }
        return x;
    }

    // Factorise once per observation set, not once per query: the factor depends only on the
    // observations, and rebuilding it per candidate is an O(n^3) solve answering an O(n^2)
    // question. Twin of BayesianOpt._gp_posterior in evolutionary_algorithms.py.
    _gpPosterior() {
        const obs = this._conditioningSet();
        const stamp = `${obs.length}:${this.observations.length}`;
        if (this._posteriorStamp === stamp) return this._posterior;

        const n = obs.length;
        const K = Array.from({ length: n }, (_, i) =>
            Array.from({ length: n }, (_, j) => this._kernel(obs[i].x, obs[j].x)));
        for (let i = 0; i < n; i++) K[i][i] += this.noiseVariance;

        let L = this._cholesky(K);
        let jitter = 1e-8;
        for (let attempt = 0; attempt < 3 && L === null; attempt++) {
            for (let i = 0; i < n; i++) K[i][i] += jitter;
            jitter *= 10;
            L = this._cholesky(K);
        }

        let posterior = null;
        if (L !== null) {
            const y = obs.map(o => o.y);
            posterior = { obs, L, alpha: this._backSolve(L, this._forwardSolve(L, y)) };
        }
        this._posteriorStamp = stamp;
        this._posterior = posterior;
        return posterior;
    }

    _gpPredict(x) {
        const posterior = this._gpPosterior();
        if (posterior === null) {
            const ys = this.observations.map(o => o.y);
            const mu = ys.reduce((a, b) => a + b, 0) / Math.max(1, ys.length);
            const varr = ys.reduce((a, b) => a + (b - mu) * (b - mu), 0) / Math.max(1, ys.length);
            return [mu, Math.sqrt(Math.max(varr, 1e-8))];
        }
        const { obs, L, alpha } = posterior;
        const ks = obs.map(o => this._kernel(o.x, x));
        let mu = 0.0;
        for (let i = 0; i < ks.length; i++) mu += ks[i] * alpha[i];
        const v = this._forwardSolve(L, ks);
        let vv = 0.0;
        for (let i = 0; i < v.length; i++) vv += v[i] * v[i];
        return [mu, Math.sqrt(Math.max(this.signalVariance - vv, 1e-8))];
    }

    // ---- acquisition ----

    expectedImprovement(x) {
        const [mu, sigma] = this._gpPredict(x);
        if (sigma <= 0) return 0.0;
        let bestY = Infinity;
        for (const o of this.observations) if (o.y < bestY) bestY = o.y;
        const improvement = bestY - mu - 0.01;   // xi = 0.01, scikit-optimize's default
        const z = improvement / sigma;
        const cdf = 0.5 * (1 + this.erf(z / Math.SQRT2));
        const pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
        return improvement * cdf + sigma * pdf;
    }

    // Kept under the old name so callers and tests that reach for it still work; it now
    // returns expected improvement from the posterior rather than a distance heuristic.
    acquisitionFunction(x) {
        return this.expectedImprovement(x);
    }

    // A broad sample then a local search of it, the shape scikit-optimize uses (10,000
    // candidates, then L-BFGS-B from the best five). 64 and 6 are the knee measured for the
    // Python port; every evaluation here is of the surrogate, so none is charged to the budget.
    acquireNext() {
        let bestX = null;
        let bestAcq = -Infinity;
        for (let j = 0; j < 64; j++) {
            const candidate = MathUtils.randomUniform(this.nDim);
            const acq = this.expectedImprovement(candidate);
            if (acq > bestAcq) { bestAcq = acq; bestX = candidate; }
        }
        if (bestX === null) return MathUtils.randomUniform(this.nDim);

        let step = 0.25;
        for (let r = 0; r < 6; r++) {
            let improved = false;
            for (let i = 0; i < this.nDim; i++) {
                for (const sign of [1.0, -1.0]) {
                    const trial = [...bestX];
                    trial[i] = MathUtils.clip(trial[i] + sign * step, 0, 1);
                    const acq = this.expectedImprovement(trial);
                    if (acq > bestAcq) { bestAcq = acq; bestX = trial; improved = true; }
                }
            }
            if (!improved) step *= 0.5;
        }
        return MathUtils.clipArray(bestX, 0, 1);
    }

    // Abramowitz and Stegun 7.1.26, worst error 1.5e-7. JavaScript has no Math.erf; Python's
    // side calls math.erf, which is exact, so the two posteriors differ in the seventh digit
    // of the CDF. That is well inside what separates two runs of a stochastic optimizer.
    erf(x) {
        const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        const sign = x >= 0 ? 1 : -1;
        x = Math.abs(x);
        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return sign * y;
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
            for (let i = 0; i < n; i++) mean[i] = 0.3 + 0.4 * Math.random();
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
        const u = Math.random();
        const v = Math.random();
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
            for (let d = 0; d < n; d++) x[d] = Math.random();
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
                const r = Math.random();
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
        const u = Math.random();
        const v = Math.random();
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

