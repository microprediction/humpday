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

    // Matches scipy.optimize.differential_evolution defaults, mirroring
    // the Python port in humpday/optimizers/evolutionary_algorithms.py:
    //   - best/1/bin mutation (base = current population best, not a
    //     random member);
    //   - dithered F drawn per-generation in [0.5, 1.0];
    //   - CR = 0.7;
    //   - popSize = max(10, min(20, nTrials // 5)).
    // The previous JS port used rand/1/bin with fixed F=0.5 — that's
    // more exploratory but converges slowly on smooth landscapes and
    // was the reason the JS DE lagged Python so badly on the win-rate
    // test.
    optimize() {
        const n = this.nDim;
        // Reserve ~25% of the budget for a coordinate-descent polish,
        // matching scipy DE's default `polish=True` (which uses L-BFGS-B
        // — closest derivative-free equivalent is coord descent with a
        // shrinking step). Without the polish JS DE was ~1000× off
        // scipy DE on the sphere at n_trials=200.
        const polishBudget = Math.max(15, Math.floor(this.nTrials / 4));
        const deBudget = this.nTrials - polishBudget;

        const popSize = Math.max(10, Math.min(20, Math.floor(deBudget / 5)));
        const CR = 0.7;

        // Initialise population uniformly in [0, 1]^n and evaluate.
        const population = [];
        for (let i = 0; i < popSize && this.evaluations < deBudget; i++) {
            const x = new Array(n);
            for (let j = 0; j < n; j++) x[j] = Math.random();
            const f = this.evaluate(x);
            population.push({ x, f });
        }

        while (this.evaluations < deBudget && population.length >= 4) {
            // Dither: pick F uniformly in [0.5, 1.0] each generation.
            const F = 0.5 + 0.5 * Math.random();

            for (let i = 0; i < population.length; i++) {
                if (this.evaluations >= deBudget) break;

                // best/1: base = current population best.
                let bestIdx = 0;
                for (let k = 1; k < population.length; k++) {
                    if (population[k].f < population[bestIdx].f) bestIdx = k;
                }

                // Two donors distinct from i and bestIdx.
                const candidates = [];
                for (let k = 0; k < population.length; k++) {
                    if (k !== i && k !== bestIdx) candidates.push(k);
                }
                if (candidates.length < 2) {
                    for (let k = 0; k < population.length; k++) {
                        if (k !== i && candidates.indexOf(k) < 0) candidates.push(k);
                    }
                }
                // Sample without replacement (or with, if fewer than 2 candidates).
                const replace = candidates.length < 2;
                const b = candidates[Math.floor(Math.random() * candidates.length)];
                let c;
                if (replace) {
                    c = candidates[Math.floor(Math.random() * candidates.length)];
                } else {
                    do {
                        c = candidates[Math.floor(Math.random() * candidates.length)];
                    } while (c === b);
                }

                // Mutation: v = x_best + F * (x_b - x_c), clipped to [0, 1].
                const mutant = new Array(n);
                for (let j = 0; j < n; j++) {
                    mutant[j] = MathUtils.clip(
                        population[bestIdx].x[j] + F * (population[b].x[j] - population[c].x[j]),
                        0, 1
                    );
                }

                // Binomial crossover with one guaranteed coord.
                const jGuaranteed = Math.floor(Math.random() * n);
                const trial = new Array(n);
                for (let j = 0; j < n; j++) {
                    trial[j] = (Math.random() < CR || j === jGuaranteed)
                        ? mutant[j]
                        : population[i].x[j];
                }

                const trialF = this.evaluate(trial);
                if (trialF < population[i].f) {
                    population[i] = { x: trial, f: trialF };
                }
            }
        }

        // --- Polish stage: L-BFGS from best DE point ----------------
        // Matches scipy.differential_evolution `polish=True` (uses
        // L-BFGS-B). Same inlined two-loop recursion as
        // SimulatedAnnealing's polish.
        this._lbfgsPolish();

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    // `_lbfgsPolish` and `_fdGradientForPolish` now live on the base
    // Optimizer class (docs/js/modules/base-optimizer.js); see comment
    // there for the algorithm.
}

// Particle Swarm Optimization implementation
class ParticleSwarm extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'ParticleSwarm';
    }

    optimize() {
        const swarmSize = Math.min(40, Math.max(15, this.nDim * 3)); // Larger swarm for complex functions

        // Initialize swarm with diverse positions
        const particles = [];
        for (let i = 0; i < swarmSize && this.evaluations < this.nTrials; i++) {
            let position;
            if (i < swarmSize / 4) {
                // 25% near center for sphere-like functions
                position = Array(this.nDim).fill(0).map(() => 0.5 + (Math.random() - 0.5) * 0.3);
            } else if (i < swarmSize / 2) {
                // 25% near corners and edges
                position = Array(this.nDim).fill(0).map(() => Math.random() < 0.5 ? 0.1 : 0.9);
            } else {
                // 50% random distribution
                position = Array(this.nDim).fill(0).map(() => Math.random());
            }

            const velocity = Array(this.nDim).fill(0).map(() => (Math.random() - 0.5) * 0.2);
            const fitness = this.evaluate(position);

            particles.push({
                position: [...position],
                velocity: [...velocity],
                bestPosition: [...position],
                bestFitness: fitness,
                stagnationCount: 0
            });
        }

        // PSO loop with adaptive parameters
        let iteration = 0;
        const maxIterations = Math.ceil(this.nTrials / swarmSize);

        while (this.evaluations < this.nTrials) {
            iteration++;

            // Adaptive parameters based on iteration progress
            const progress = iteration / maxIterations;
            const w = 0.9 - 0.5 * progress; // Decreasing inertia (0.9 → 0.4)
            const c1 = 2.5 - 1.0 * progress; // Decreasing cognitive (2.5 → 1.5)
            const c2 = 1.5 + 1.0 * progress; // Increasing social (1.5 → 2.5)

            // Maximum velocity (velocity clamping)
            const vmax = 0.2 * (1 - 0.5 * progress); // Decreasing velocity limit

            for (let i = 0; i < particles.length && this.evaluations < this.nTrials; i++) {
                const particle = particles[i];

                // Update velocity with constriction factor
                for (let j = 0; j < this.nDim; j++) {
                    const r1 = Math.random();
                    const r2 = Math.random();

                    // Standard PSO velocity update
                    particle.velocity[j] = w * particle.velocity[j] +
                        c1 * r1 * (particle.bestPosition[j] - particle.position[j]) +
                        c2 * r2 * (this.bestX[j] - particle.position[j]);

                    // Velocity clamping
                    particle.velocity[j] = MathUtils.clip(particle.velocity[j], -vmax, vmax);
                }

                // Update position
                for (let j = 0; j < this.nDim; j++) {
                    particle.position[j] = MathUtils.clip(
                        particle.position[j] + particle.velocity[j], 0, 1
                    );
                }

                const fitness = this.evaluate(particle.position);

                // Update personal best
                if (fitness < particle.bestFitness) {
                    particle.bestFitness = fitness;
                    particle.bestPosition = [...particle.position];
                    particle.stagnationCount = 0;
                } else {
                    particle.stagnationCount++;
                }

                // Diversification for stagnant particles
                if (particle.stagnationCount > 15 && Math.random() < 0.1) {
                    // Reinitialize position with small probability
                    particle.position = Array(this.nDim).fill(0).map(() => Math.random());
                    particle.velocity = Array(this.nDim).fill(0).map(() => (Math.random() - 0.5) * 0.1);
                    particle.stagnationCount = 0;
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

// Simulated Annealing implementation
class SimulatedAnnealing extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'SimulatedAnnealing';
    }

    // Two-stage algorithm matching scipy.optimize.dual_annealing in
    // spirit (and the Python port in
    // humpday/optimizers/evolutionary_algorithms.py line-for-line):
    //
    //   Stage 1 — multi-restart Metropolis SA explores globally with a
    //             geometric cooling schedule from T = 1.0 to T = 1e-6.
    //   Stage 2 — coordinate-descent polish from the best SA point,
    //             halving the step on each unimproved round, refines to
    //             high precision.
    //
    // scipy's dual_annealing uses L-BFGS-B for stage 2; the closest
    // derivative-free equivalent is a coordinate descent with shrinking
    // step. Without the polish stage humpday's SA was ~1e9× off scipy
    // on sphere and Rosenbrock; the global SA can't reach machine
    // precision because its proposals are noisy.
    optimize() {
        const n = this.nDim;
        const polishBudget = Math.max(20, Math.floor(this.nTrials / 3));
        const saBudget = this.nTrials - polishBudget;

        // --- Stage 1: multi-restart Metropolis SA ----------------------
        const numRestarts = Math.max(3, Math.floor(saBudget / 30));
        const trialsPerRestart = Math.max(1, Math.floor(saBudget / numRestarts));

        for (let restart = 0; restart < numRestarts; restart++) {
            if (this.evaluations >= saBudget) break;

            let x;
            if (restart === 0) {
                // Center-biased first restart (matches Python).
                x = new Array(n);
                for (let i = 0; i < n; i++) x[i] = 0.5 + (Math.random() - 0.5) * 0.4;
            } else {
                // Uniform random subsequent restarts.
                x = new Array(n);
                for (let i = 0; i < n; i++) x[i] = Math.random();
            }
            let fx = this.evaluate(x);

            const initialTemp = 1.0;
            const finalTemp = 1e-6;
            const cooling = Math.pow(finalTemp / initialTemp, 1.0 / Math.max(1, trialsPerRestart));
            let temp = initialTemp;

            for (let iter = 0; iter < trialsPerRestart; iter++) {
                if (this.evaluations >= saBudget) break;

                // Neighbour proposal: step scales with current temp.
                const stepSize = 0.4 * temp;
                const newX = new Array(n);
                for (let i = 0; i < n; i++) {
                    newX[i] = MathUtils.clip(
                        x[i] + (Math.random() - 0.5) * 2 * stepSize, 0, 1
                    );
                }
                const newFx = this.evaluate(newX);

                // Metropolis criterion.
                const delta = newFx - fx;
                if (delta < 0 || Math.random() < Math.exp(-delta / Math.max(temp, 1e-12))) {
                    x = newX;
                    fx = newFx;
                }

                temp *= cooling;
            }
        }

        // --- Stage 2: L-BFGS polish from best SA point -----------------
        // Matches scipy.dual_annealing's L-BFGS-B refinement. Inlined
        // two-loop recursion with FD gradient and Armijo line search —
        // same algorithm the LBFGSB optimizer uses.
        this._lbfgsPolish();

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    // `_lbfgsPolish` and `_fdGradientForPolish` are on the base
    // Optimizer class (docs/js/modules/base-optimizer.js).
}

// Genetic Algorithm implementation
class GeneticAlgorithm extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'GeneticAlgorithm';
    }

    optimize() {
        const popSize = Math.min(50, Math.max(20, this.nDim * 4));
        const mutationRate = 0.1;
        const crossoverRate = 0.8;

        // Initialize population
        let population = [];
        for (let i = 0; i < popSize && this.evaluations < this.nTrials; i++) {
            const individual = Array(this.nDim).fill(0).map(() => Math.random());
            const fitness = this.evaluate(individual);
            population.push({ x: individual, fitness });
        }

        // Evolution loop
        while (this.evaluations < this.nTrials) {
            const newPop = [];
            for (let i = 0; i < popSize && this.evaluations < this.nTrials; i++) {
                const parent1 = this.tournamentSelection(population);
                const parent2 = this.tournamentSelection(population);

                let child = [...parent1.x];

                // Crossover
                if (Math.random() < crossoverRate) {
                    const crossPoint = Math.floor(Math.random() * this.nDim);
                    for (let j = crossPoint; j < this.nDim; j++) {
                        child[j] = parent2.x[j];
                    }
                }

                // Mutation
                for (let j = 0; j < this.nDim; j++) {
                    if (Math.random() < mutationRate) {
                        child[j] = MathUtils.clip(
                            child[j] + (Math.random() - 0.5) * 0.2, 0, 1
                        );
                    }
                }

                const fitness = this.evaluate(child);
                newPop.push({ x: child, fitness });
            }

            population = newPop;
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    tournamentSelection(population) {
        const tournamentSize = 3;
        let best = population[Math.floor(Math.random() * population.length)];

        for (let i = 1; i < tournamentSize; i++) {
            const competitor = population[Math.floor(Math.random() * population.length)];
            if (competitor.fitness < best.fitness) {
                best = competitor;
            }
        }

        return best;
    }
}

// Random Search implementation
class RandomSearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'RandomSearch';
    }

    optimize() {
        while (this.evaluations < this.nTrials) {
            const x = Array(this.nDim).fill(0).map(() => Math.random());
            this.evaluate(x);
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

    // Hansen-standard CMA-ES with rank-1 + rank-μ covariance adaptation
    // and evolution paths. Mirrors `humpday/optimizers/
    // evolutionary_algorithms.py::CMAEvolutionStrategy` step-for-step.
    // Before this rewrite the JS port was sigma-only adaptation with an
    // identity-covariance proposal — it could never learn anisotropy,
    // and the win-rate parity test ran against a Python port that does
    // full rank-1 + rank-μ updates, making the JS side the obvious
    // outlier on Rosenbrock and similar coupled objectives.
    optimize() {
        const n = this.nDim;

        // Hansen's recommended parameters.
        const lambda_ = Math.min(50, 4 + Math.floor(3 * Math.log(n)));
        const mu = Math.floor(lambda_ / 2);

        // Recombination weights w_i = log(μ + 0.5) − log(i + 1), normalised.
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

        // State. Initial mean is a random interior point in [0.3, 0.7]^n
        // (matching the Python port — the previous fixed-centre 0.5*ones
        // gave the same starting point every restart and disadvantaged
        // objectives whose optimum isn't at the centre).
        let mean = new Array(n);
        for (let i = 0; i < n; i++) mean[i] = 0.3 + 0.4 * Math.random();
        let sigma = 0.2;
        let C = Linalg.eye(n);
        let pc = new Array(n).fill(0);
        let ps = new Array(n).fill(0);
        let invsqrtC = Linalg.eye(n);

        let generation = 0;
        const maxGenerations = this.nTrials;

        while (this.evaluations < this.nTrials && generation < maxGenerations) {
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
                if (this.evaluations >= this.nTrials) break;
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
            // Partial-generation guard (same as the Python port — if the
            // budget ran out before μ samples, recombination would be
            // ill-defined).
            if (population.length < mu) break;

            population.sort((a, b) => a.f - b.f);

            // Recombination: new mean = Σ w_i x_i over the μ best.
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

            // Rank-μ update: weighted sum of outer products of the μ
            // best (population[i].x − oldMean) / σ.
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

            // C ← (1 − c1 − cμ) C + c1 (pc pcᵀ) + cμ weightedDiffs.
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

            // Ensure C stays positive definite — bump up by the smallest
            // eigenvalue if it drifted non-PD.
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
                /* leave C as-is — eigh failed; sampling will hit the
                   identity-fallback on the next iteration. */
            }

            // Refresh invsqrtC for the next iteration.
            try {
                const { eigvals: D, eigvecs: B } = Linalg.eigh(C);
                const Dinvsqrt = D.map(d => 1.0 / Math.sqrt(Math.max(d, 1e-14)));
                const Ddiag = Linalg.diag(Dinvsqrt);
                const tmp = Linalg.matmul(B, Ddiag);
                invsqrtC = Linalg.matmul(tmp, Linalg.transpose(B));
            } catch (e) {
                invsqrtC = Linalg.eye(n);
            }

            // Step-size update (CSA). Do NOT cap sigma at 0.5 (pycma has
            // no cap; clipping each x to [0,1]^n already enforces the
            // feasible search space) and do NOT floor at 1e-6 (the floor
            // was preventing precise convergence on smooth basins).
            sigma = sigma * Math.exp((cs / damps) * (psNorm / Math.sqrt(n) - 1));
        }

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

    optimize() {
        const nFireflies = Math.min(25, Math.max(10, this.nDim * 2));
        const alpha = 0.1; // Randomization parameter
        const gamma = 1.0; // Light absorption coefficient

        // Initialize fireflies
        const fireflies = [];
        for (let i = 0; i < nFireflies && this.evaluations < this.nTrials; i++) {
            const position = Array(this.nDim).fill(0).map(() => Math.random());
            const intensity = this.evaluate(position);
            fireflies.push({ position: [...position], intensity });
        }

        while (this.evaluations < this.nTrials) {
            for (let i = 0; i < fireflies.length && this.evaluations < this.nTrials; i++) {
                for (let j = 0; j < fireflies.length; j++) {
                    if (i !== j && fireflies[j].intensity < fireflies[i].intensity) {
                        // Move firefly i towards j
                        const distance = MathUtils.norm(
                            MathUtils.subtract(fireflies[i].position, fireflies[j].position)
                        );
                        const beta = 1.0 / (1.0 + gamma * distance * distance);

                        const newPosition = fireflies[i].position.map((xi, k) => {
                            const attraction = beta * (fireflies[j].position[k] - xi);
                            const randomization = alpha * (Math.random() - 0.5);
                            return MathUtils.clip(xi + attraction + randomization, 0, 1);
                        });

                        const newIntensity = this.evaluate(newPosition);

                        if (newIntensity < fireflies[i].intensity) {
                            fireflies[i].position = newPosition;
                            fireflies[i].intensity = newIntensity;
                        }
                    }
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

    optimize() {
        const HMS = Math.min(20, Math.max(5, this.nDim * 2)); // Harmony Memory Size
        const HMCR = 0.9; // Harmony Memory Considering Rate
        const PAR = 0.3; // Pitch Adjusting Rate

        // Initialize harmony memory
        const harmonyMemory = [];
        for (let i = 0; i < HMS && this.evaluations < this.nTrials; i++) {
            const harmony = Array(this.nDim).fill(0).map(() => Math.random());
            const fitness = this.evaluate(harmony);
            harmonyMemory.push({ harmony: [...harmony], fitness });
        }

        while (this.evaluations < this.nTrials) {
            const newHarmony = [];

            for (let j = 0; j < this.nDim; j++) {
                if (Math.random() < HMCR) {
                    // Pick from harmony memory
                    const selectedHarmony = harmonyMemory[
                        Math.floor(Math.random() * harmonyMemory.length)
                    ];
                    let value = selectedHarmony.harmony[j];

                    // Pitch adjustment
                    if (Math.random() < PAR) {
                        value = MathUtils.clip(value + (Math.random() - 0.5) * 0.1, 0, 1);
                    }

                    newHarmony.push(value);
                } else {
                    // Random selection
                    newHarmony.push(Math.random());
                }
            }

            const newFitness = this.evaluate(newHarmony);

            // Update harmony memory (replace worst if new harmony is better)
            harmonyMemory.sort((a, b) => a.fitness - b.fitness);
            if (newFitness < harmonyMemory[harmonyMemory.length - 1].fitness) {
                harmonyMemory[harmonyMemory.length - 1] = {
                    harmony: [...newHarmony],
                    fitness: newFitness
                };
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

// (μ+λ) Evolution Strategy
class EvolutionStrategy extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'EvolutionStrategy';
    }

    optimize() {
        const mu = Math.min(10, Math.max(2, Math.floor(this.nDim / 2)));
        const lambda = mu * 4;

        // Initialize parent population
        let parents = [];
        for (let i = 0; i < mu && this.evaluations < this.nTrials; i++) {
            const individual = Array(this.nDim).fill(0).map(() => Math.random());
            const fitness = this.evaluate(individual);
            const sigma = Array(this.nDim).fill(0.1); // Strategy parameters
            parents.push({ x: individual, fitness, sigma });
        }

        while (this.evaluations < this.nTrials) {
            const offspring = [];

            // Generate offspring
            for (let i = 0; i < lambda && this.evaluations < this.nTrials; i++) {
                // Select random parent
                const parent = parents[Math.floor(Math.random() * parents.length)];

                // Mutate strategy parameters
                const newSigma = parent.sigma.map(s =>
                    s * Math.exp(0.1 * (Math.random() - 0.5))
                );

                // Generate offspring
                const child = parent.x.map((xi, j) =>
                    MathUtils.clip(xi + newSigma[j] * (Math.random() - 0.5), 0, 1)
                );

                const fitness = this.evaluate(child);
                offspring.push({ x: child, fitness, sigma: newSigma });
            }

            // Select best μ from parents + offspring
            const combined = [...parents, ...offspring];
            combined.sort((a, b) => a.fitness - b.fitness);
            parents = combined.slice(0, mu);
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

