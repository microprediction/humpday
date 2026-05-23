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

// Export evolutionary algorithms
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = { DifferentialEvolution, ParticleSwarm, SimulatedAnnealing, GeneticAlgorithm, RandomSearch };
} else {
    // Browser environment
    window.DifferentialEvolution = DifferentialEvolution;
    window.ParticleSwarm = ParticleSwarm;
    window.SimulatedAnnealing = SimulatedAnnealing;
    window.GeneticAlgorithm = GeneticAlgorithm;
    window.RandomSearch = RandomSearch;
}

