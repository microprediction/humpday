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
        const deBudget = this.nTrials - polishBudget;

        const popSize = Math.max(10, Math.min(20, Math.floor(deBudget / 5)));
        const CR = 0.7;

        // All initial draws happen before the first yield (Python builds
        // the population in a list comprehension, then evaluates).
        const population = [];
        for (let i = 0; i < popSize; i++) population.push(MathUtils.randomUniform(n));
        const fitness = [];
        for (const ind of population) fitness.push(yield ind);

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

        // --- Polish stage: L-BFGS from best DE point ----------------
        // Matches scipy.differential_evolution `polish=True`.
        yield* this._lbfgsPolishGen();
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
class SimulatedAnnealing extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'SimulatedAnnealing';
    }

    // Two-stage algorithm matching scipy.optimize.dual_annealing in
    // spirit:
    //
    //   Stage 1 — multi-restart Metropolis SA explores globally with a
    //             geometric cooling schedule from T = 1.0 to T = 1e-6.
    //   Stage 2 — L-BFGS-B polish from the best SA point (scipy's
    //             dual_annealing uses L-BFGS-B for its local search).
    *_run() {
        // Twin of SimulatedAnnealing._run in
        // humpday/optimizers/evolutionary_algorithms.py.
        const n = this.nDim;
        const polishBudget = Math.max(20, Math.floor(this.nTrials / 2));
        const saBudget = this.nTrials - polishBudget;

        // --- Stage 1: multi-restart Metropolis SA ----------------------
        const numRestarts = Math.max(3, Math.floor(saBudget / 30));
        const trialsPerRestart = Math.max(1, Math.floor(saBudget / numRestarts));

        for (let restart = 0; restart < numRestarts; restart++) {
            if (this.evaluations >= saBudget) break;

            let x;
            if (restart === 0) {
                // Center-biased first restart.
                const u = MathUtils.randomUniform(n);
                x = u.map(v => 0.5 + (v - 0.5) * 0.4);
            } else {
                x = MathUtils.randomUniform(n);
            }
            let fx = yield x;

            const initialTemp = 1.0;
            const finalTemp = 1e-6;
            // portableExp/portableLog on BOTH sides, not pow: libm pow
            // differs across platforms in the last ulp.
            const cooling = MathUtils.portableExp(
                (1.0 / Math.max(1, trialsPerRestart))
                * MathUtils.portableLog(finalTemp / initialTemp)
            );
            let temp = initialTemp;

            for (let iter = 0; iter < trialsPerRestart; iter++) {
                if (this.evaluations >= saBudget) break;

                // Neighbour proposal: step scales with current temp.
                const stepSize = 0.4 * temp;
                const u = MathUtils.randomUniform(n);
                const newX = new Array(n);
                for (let i = 0; i < n; i++) {
                    newX[i] = MathUtils.clip(x[i] + ((u[i] - 0.5) * 2) * stepSize, 0, 1);
                }
                const newFx = yield newX;

                // Metropolis criterion. The acceptance draw happens only
                // when delta >= 0 (short-circuit) — stream position
                // depends on it.
                const delta = newFx - fx;
                if (delta < 0 || MathUtils.randomScalar() < MathUtils.portableExp(-delta / Math.max(temp, 1e-12))) {
                    x = newX;
                    fx = newFx;
                }

                temp *= cooling;
            }
        }

        // --- Stage 2: L-BFGS polish from best SA point -----------------
        yield* this._lbfgsPolishGen();
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
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'BayesianOpt';
        this.observations = [];
    }

    optimize() {
        // Strategic initial sampling with some center-biased points
        const nInitial = Math.min(10, Math.floor(this.nTrials * 0.2));

        // Sample some points near center for sphere-like functions
        for (let i = 0; i < Math.min(3, nInitial) && this.evaluations < this.nTrials; i++) {
            const x = Array(this.nDim).fill(0).map(() => 0.5 + (Math.random() - 0.5) * 0.3);
            const y = this.evaluate(x);
            this.observations.push({ x: [...x], y });
        }

        // Fill remaining initial samples with random points
        for (let i = this.observations.length; i < nInitial && this.evaluations < this.nTrials; i++) {
            const x = Array(this.nDim).fill(0).map(() => Math.random());
            const y = this.evaluate(x);
            this.observations.push({ x: [...x], y });
        }

        // Reserve budget for the L-BFGS-B polish stage. Reference:
        // scikit-optimize's `gp_minimize` finishes with a
        // `minimize(method='L-BFGS-B')` polish on the best observation.
        // The polish takes 2·nDim evals per gradient + a few per line
        // search; reserving 20·nDim evals (≈ 10 polish iterations)
        // closes the residual ~5 orders of magnitude on smooth
        // problems by escaping the GP's RBF smoothing floor.
        const polishReserve = Math.min(20 * this.nDim, Math.floor(this.nTrials / 2));
        const loopBudget = Math.max(this.evaluations, this.nTrials - polishReserve);

        // Bayesian optimization loop with intensification
        while (this.evaluations < loopBudget) {
            const nextX = this.acquireNext();
            const y = this.evaluate(nextX);
            this.observations.push({ x: [...nextX], y });

            // Intensify search around best point if very good solution found
            if (y < 1e-4 && this.evaluations < loopBudget - 5) {
                for (let i = 0; i < Math.min(3, loopBudget - this.evaluations); i++) {
                    const localX = nextX.map(xi => {
                        const noise = (Math.random() - 0.5) * 0.02;
                        return MathUtils.clip(xi + noise, 0, 1);
                    });
                    const localY = this.evaluate(localX);
                    this.observations.push({ x: [...localX], y: localY });
                }
            }
        }

        // Polish: L-BFGS-B from the GP-EI best. Mirrors the Python port.
        this._lbfgsPolish();

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    acquireNext() {
        let bestAcq = -Infinity;
        let nextX = Array(this.nDim).fill(0).map(() => Math.random());

        // Sample candidate points
        for (let j = 0; j < 100; j++) {
            const candidate = Array(this.nDim).fill(0).map(() => Math.random());
            const acq = this.acquisitionFunction(candidate);

            if (acq > bestAcq) {
                bestAcq = acq;
                nextX = candidate;
            }
        }

        return nextX;
    }

    acquisitionFunction(x) {
        if (this.observations.length === 0) return Math.random();

        // Distance-weighted Expected Improvement approximation
        const distances = this.observations.map(obs => ({
            dist: MathUtils.norm(MathUtils.subtract(x, obs.x)),
            y: obs.y
        }));

        distances.sort((a, b) => a.dist - b.dist);
        const kNearest = distances.slice(0, Math.min(5, distances.length));

        if (kNearest.length === 0) return Math.random();

        // Distance-weighted prediction
        const epsilon = 1e-8; // Avoid division by zero
        let weightSum = 0;
        let weightedMean = 0;

        for (const item of kNearest) {
            const weight = 1.0 / (item.dist + epsilon);
            weightSum += weight;
            weightedMean += weight * item.y;
        }

        const predictedMean = weightedMean / weightSum;

        // Estimate uncertainty based on distance to nearest point and local variance
        const uncertainty = Math.exp(-2.0 * kNearest[0].dist);
        const localVariance = kNearest.length > 1 ?
            kNearest.reduce((sum, item) => sum + Math.pow(item.y - predictedMean, 2), 0) / kNearest.length :
            0.1;
        const predictedStd = Math.max(Math.sqrt(localVariance), 0.01) * uncertainty;

        // Expected Improvement: EI = (f_min - mu) * Φ(Z) + σ * φ(Z)
        const bestY = Math.min(...this.observations.map(obs => obs.y));
        const improvement = bestY - predictedMean;

        if (predictedStd <= epsilon) {
            return improvement > 0 ? improvement : 0;
        }

        const z = improvement / predictedStd;

        // Approximate normal CDF and PDF
        const phi = 0.5 * (1 + this.erf(z / Math.sqrt(2))); // CDF
        const pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); // PDF

        const expectedImprovement = improvement * phi + predictedStd * pdf;

        // Add small exploration bonus
        return Math.max(0, expectedImprovement) + 0.01 * uncertainty;
    }

    // Error function approximation for normal CDF
    erf(x) {
        // Abramowitz and Stegun approximation
        const a1 =  0.254829592;
        const a2 = -0.284496736;
        const a3 =  1.421413741;
        const a4 = -1.453152027;
        const a5 =  1.061405429;
        const p  =  0.3275911;

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
                const hsig = psNorm / hsigDenom < 1.4 + 2 / (n + 1) ? 1 : 0;

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

                const base = 1 - c1 - cmu;
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
                sigma = sigma * Math.exp((cs / damps) * (psNorm / Math.sqrt(n) - 1));

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
        const sigma = 0.2;

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

            for (let k = 0; k < lambda_; k++) {
                if (this.evaluations >= this.nTrials) break;

                const parentIdx = MathUtils.randInt(population.length);
                const parent = population[parentIdx];

                const z = MathUtils.randomNormal(this.nDim);
                const child = MathUtils.clipArray(
                    parent.map((p, i) => p + sigma * z[i]), 0, 1
                );

                const childFitness = yield child;
                offspring.push(child);
                offspringFitness.push(childFitness);
            }

            if (offspring.length) {
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

