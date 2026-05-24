/**
 * Evolutionary algorithm implementations.
 *
 * These algorithms are inspired by natural evolution processes and include
 * methods like Differential Evolution, Genetic Algorithms, Particle Swarm, etc.
 * They excel at global optimization and handling multimodal landscapes.
 */

// Import base classes and utilities
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    const { Optimizer, MathUtils } = require('./base-optimizer.js');
} else {
    // Browser environment - base classes already loaded
}class DifferentialEvolution extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'DifferentialEvolution';
    }

    optimize() {
        const popSize = Math.min(20, Math.max(8, this.nDim * 3));
        const F = 0.5; // Differential weight
        const CR = 0.7; // Crossover probability

        // Initialize population
        const population = [];
        for (let i = 0; i < popSize && this.evaluations < this.nTrials; i++) {
            const individual = Array(this.nDim).fill(0).map(() => Math.random());
            const fitness = this.evaluate(individual);
            population.push({ x: individual, fitness });
        }

        // Evolution loop
        while (this.evaluations < this.nTrials && population.length >= 4) {
            for (let i = 0; i < population.length && this.evaluations < this.nTrials; i++) {
                // Select three random different individuals
                const indices = Array.from({length: population.length}, (_, idx) => idx)
                    .filter(idx => idx !== i);

                const [a, b, c] = [0, 1, 2].map(j => {
                    const idx = Math.floor(Math.random() * indices.length);
                    const selected = indices[idx];
                    indices.splice(idx, 1);
                    return selected;
                });

                // Mutation: v = a + F * (b - c)
                const mutant = population[a].x.map((xi, j) =>
                    xi + F * (population[b].x[j] - population[c].x[j])
                );

                // Crossover
                const trial = population[i].x.map((xi, j) =>
                    Math.random() < CR ?
                    MathUtils.clip(mutant[j], 0, 1) : xi
                );

                const trialFitness = this.evaluate(trial);

                // Selection
                if (trialFitness < population[i].fitness) {
                    population[i] = { x: trial, fitness: trialFitness };
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

            // Early termination for excellent solutions
            if (this.bestValue < 1e-6) break; // Relaxed termination for visualization
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

    optimize() {
        let globalBestX = null;
        let globalBestFx = Infinity;

        // Multi-restart simulated annealing for better global optimization
        const numRestarts = Math.max(3, Math.floor(this.nTrials / 30));
        const trialsPerRestart = Math.floor(this.nTrials / numRestarts);

        for (let restart = 0; restart < numRestarts && this.evaluations < this.nTrials; restart++) {
            // Initialize with different strategies per restart
            let x;
            if (restart === 0) {
                // First restart: center-biased initialization
                x = Array(this.nDim).fill(0).map(() => 0.5 + (Math.random() - 0.5) * 0.4);
            } else if (restart === 1) {
                // Second restart: near-optimal region
                x = Array(this.nDim).fill(0).map(() => (Math.random() - 0.5) * 0.2 + 0.5);
            } else {
                // Subsequent restarts: random
                x = Array(this.nDim).fill(0).map(() => Math.random());
            }

            let fx = this.evaluate(x);

            // Track best for this restart
            let bestX = x.slice();
            let bestFx = fx;

            // Aggressive initial temperature
            let temperature = Math.max(1.0, bestFx * 2);
            const finalTemp = 1e-8;
            const maxIterations = Math.min(trialsPerRestart, this.nTrials - this.evaluations);

            for (let iter = 0; iter < maxIterations && this.evaluations < this.nTrials; iter++) {
                // Adaptive step size based on current best and temperature
                const progressRatio = iter / maxIterations;
                const tempRatio = temperature / (Math.max(1.0, bestFx * 2));
                let stepSize = 0.3 * tempRatio * (1 - progressRatio) + 0.01 * progressRatio;

                // Generate neighbor with multiple strategies
                const strategy = iter % 3;
                let newX;

                if (strategy === 0) {
                    // Standard perturbation
                    newX = x.map(xi => {
                        const perturbation = (Math.random() - 0.5) * 2 * stepSize;
                        return MathUtils.clip(xi + perturbation, 0, 1);
                    });
                } else if (strategy === 1) {
                    // Move toward current best with noise
                    newX = x.map((xi, i) => {
                        const direction = bestX[i] - xi;
                        const move = direction * 0.3 + (Math.random() - 0.5) * stepSize;
                        return MathUtils.clip(xi + move, 0, 1);
                    });
                } else {
                    // Large jump for exploration
                    newX = x.map(xi => {
                        if (Math.random() < 0.1) {
                            return Math.random(); // Occasional large jump
                        } else {
                            const perturbation = (Math.random() - 0.5) * 2 * stepSize;
                            return MathUtils.clip(xi + perturbation, 0, 1);
                        }
                    });
                }

                const newFx = this.evaluate(newX);

                // Update best for this restart
                if (newFx < bestFx) {
                    bestX = newX.slice();
                    bestFx = newFx;
                }

                // Metropolis criterion
                const delta = newFx - fx;
                if (delta < 0 || (temperature > finalTemp && Math.random() < Math.exp(-delta / temperature))) {
                    x = newX;
                    fx = newFx;
                }

                // Fast exponential cooling with floor
                temperature *= 0.99;
                temperature = Math.max(temperature, finalTemp);

                // Early termination if we find a very good solution
                if (bestFx < 1e-6) break;
            }

            // Update global best
            if (bestFx < globalBestFx) {
                globalBestFx = bestFx;
                globalBestX = bestX.slice();
            }
        }

        return {
            bestValue: globalBestFx,
            bestX: globalBestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }
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

        // Bayesian optimization loop with intensification
        while (this.evaluations < this.nTrials) {
            const nextX = this.acquireNext();
            const y = this.evaluate(nextX);
            this.observations.push({ x: [...nextX], y });

            // Intensify search around best point if very good solution found
            if (y < 1e-4 && this.evaluations < this.nTrials - 5) {
                for (let i = 0; i < Math.min(3, this.nTrials - this.evaluations); i++) {
                    const localX = nextX.map(xi => {
                        const noise = (Math.random() - 0.5) * 0.02; // Small perturbation
                        return MathUtils.clip(xi + noise, 0, 1);
                    });
                    const localY = this.evaluate(localX);
                    this.observations.push({ x: [...localX], y: localY });
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

    optimize() {
        const lambda = Math.min(20, 4 + Math.floor(3 * Math.log(this.nDim)));
        const mu = Math.floor(lambda / 2);

        let mean = Array(this.nDim).fill(0.5);
        let sigma = 0.3;

        // CMA-ES proper weights (log-linear decreasing)
        const weights = Array(mu).fill(0).map((_, i) => Math.log(mu + 0.5) - Math.log(i + 1));
        const sumWeights = weights.reduce((sum, w) => sum + w, 0);
        const normalizedWeights = weights.map(w => w / sumWeights);

        // Simplified covariance matrix (identity scaled by sigma^2)
        let C = Array(this.nDim).fill(0).map(() => Array(this.nDim).fill(0));
        for (let i = 0; i < this.nDim; i++) {
            C[i][i] = 1.0; // Identity matrix initially
        }

        while (this.evaluations < this.nTrials) {
            // Generate offspring using PROPER Gaussian sampling
            const offspring = [];
            for (let i = 0; i < lambda && this.evaluations < this.nTrials; i++) {
                // Sample from multivariate normal N(mean, sigma^2 * C)
                const z = Array(this.nDim).fill(0).map(() => this.boxMullerGaussian());
                const individual = mean.map((m, j) =>
                    MathUtils.clip(m + sigma * z[j], 0, 1)
                );
                const fitness = this.evaluate(individual);
                offspring.push({ x: individual, fitness, z: z });
            }

            // Selection and recombination
            offspring.sort((a, b) => a.fitness - b.fitness);
            const selected = offspring.slice(0, mu);

            // Update mean
            const newMean = Array(this.nDim).fill(0);
            for (let i = 0; i < Math.min(mu, selected.length); i++) {
                if (selected[i] && selected[i].x) {
                    for (let j = 0; j < this.nDim; j++) {
                        newMean[j] += normalizedWeights[i] * selected[i].x[j];
                    }
                }
            }
            mean = newMean;

            // Adapt step size (simplified)
            const improvement = (selected.length > 0 && offspring.length > 0) ?
                (offspring[offspring.length - 1]?.fitness - selected[0]?.fitness) : 0;
            sigma *= improvement > 0 ? 0.95 : 1.05;
            sigma = MathUtils.clip(sigma, 0.01, 1.0);
        }

        return {
            bestValue: this.bestValue,
            bestX: this.bestX,
            evaluations: this.evaluations,
            success: true,
            path: this.trackPath ? this.path : null
        };
    }

    // Box-Muller transform for Gaussian sampling (essential for proper CMA-ES)
    boxMullerGaussian() {
        if (this.spareGaussian !== undefined) {
            const spare = this.spareGaussian;
            this.spareGaussian = undefined;
            return spare;
        }

        const u = Math.random();
        const v = Math.random();
        const r = Math.sqrt(-2 * Math.log(u));
        const theta = 2 * Math.PI * v;

        this.spareGaussian = r * Math.sin(theta);
        return r * Math.cos(theta);
    }
}

// Tabu Search
class TabuSearch extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'TabuSearch';
        this.tabuList = [];
        this.tabuTenure = Math.min(20, Math.max(5, this.nDim));
    }

    optimize() {
        let x = Array(this.nDim).fill(0).map(() => Math.random());
        let fx = this.evaluate(x);

        while (this.evaluations < this.nTrials) {
            let bestNeighbor = null;
            let bestNeighborFx = Infinity;

            // Generate multiple neighbors
            for (let i = 0; i < Math.min(20, this.nTrials - this.evaluations); i++) {
                const neighbor = x.map(xi =>
                    MathUtils.clip(xi + (Math.random() - 0.5) * 0.15, 0, 1)
                );

                // Check if tabu
                const isTabu = this.tabuList.some(tabu =>
                    MathUtils.norm(MathUtils.subtract(neighbor, tabu)) < 0.05
                );

                if (!isTabu) {
                    const neighborFx = this.evaluate(neighbor);
                    if (neighborFx < bestNeighborFx) {
                        bestNeighbor = neighbor;
                        bestNeighborFx = neighborFx;
                    }
                }
            }

            if (bestNeighbor) {
                // Add current solution to tabu list
                this.tabuList.push([...x]);
                if (this.tabuList.length > this.tabuTenure) {
                    this.tabuList.shift();
                }

                x = bestNeighbor;
                fx = bestNeighborFx;
            } else {
                // If all neighbors are tabu, clear tabu list
                this.tabuList = [];
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
class AntColonyOpt extends Optimizer {
    constructor(objective, nTrials, nDim) {
        super(objective, nTrials, nDim);
        this.name = 'AntColonyOpt';
    }

    optimize() {
        const nAnts = Math.min(20, Math.max(8, this.nDim));
        const rho = 0.1; // Evaporation rate

        // Discretize search space
        const nLevels = 20;
        const pheromones = Array(this.nDim).fill(0).map(() => Array(nLevels).fill(1.0));

        while (this.evaluations < this.nTrials) {
            const solutions = [];

            // Construct solutions
            for (let ant = 0; ant < nAnts && this.evaluations < this.nTrials; ant++) {
                const solution = [];
                for (let dim = 0; dim < this.nDim; dim++) {
                    // Probabilistic selection based on pheromones
                    const probabilities = pheromones[dim].map(p => Math.pow(p, 2));
                    const total = probabilities.reduce((sum, p) => sum + p, 0);
                    const normalizedProb = probabilities.map(p => p / total);

                    let selected = 0;
                    const rand = Math.random();
                    let cumProb = 0;
                    for (let level = 0; level < nLevels; level++) {
                        cumProb += normalizedProb[level];
                        if (rand <= cumProb) {
                            selected = level;
                            break;
                        }
                    }

                    solution.push(selected / (nLevels - 1));
                }

                const fitness = this.evaluate(solution);
                solutions.push({ solution, fitness });
            }

            // Evaporate pheromones
            for (let dim = 0; dim < this.nDim; dim++) {
                for (let level = 0; level < nLevels; level++) {
                    pheromones[dim][level] *= (1 - rho);
                }
            }

            // Update pheromones (best solution gets more pheromone)
            solutions.sort((a, b) => a.fitness - b.fitness);
            const bestSol = solutions[0];

            bestSol.solution.forEach((value, dim) => {
                const level = Math.round(value * (nLevels - 1));
                pheromones[dim][level] += 1.0 / (1.0 + bestSol.fitness);
            });
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
        BayesianOpt, CMAEvolutionStrategy, TabuSearch, FireflyAlgorithm, AntColonyOpt,
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
    window.TabuSearch = TabuSearch;
    window.FireflyAlgorithm = FireflyAlgorithm;
    window.AntColonyOpt = AntColonyOpt;
    window.HarmonySearch = HarmonySearch;
    window.EvolutionStrategy = EvolutionStrategy;
}

