"""
Evolutionary algorithm implementations.

These algorithms are inspired by natural evolution processes and include
methods like Differential Evolution, Genetic Algorithms, and Evolution Strategies.
They excel at global optimization and handling multimodal landscapes.
"""

from typing import Tuple

import numpy as np

from .base import BaseOptimizer


class DifferentialEvolution(BaseOptimizer):
    """Differential Evolution algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        pop_size = min(20, self.n_trials // 5)
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover probability

        # Initialize population
        population = np.random.random((pop_size, self.n_dim))
        fitness = np.array([self.evaluate(ind) for ind in population])

        while self.evaluations < self.n_trials:
            for i in range(pop_size):
                if self.evaluations >= self.n_trials:
                    break

                # Select three random individuals (different from current)
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutation
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, 0, 1)

                # Crossover
                trial = population[i].copy()
                for j in range(self.n_dim):
                    if np.random.random() < CR or j == np.random.randint(self.n_dim):
                        trial[j] = mutant[j]

                # Selection
                trial_fitness = self.evaluate(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        return self.best_value, self.best_x


class ParticleSwarm(BaseOptimizer):
    """Particle Swarm Optimization algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        swarm_size = min(40, max(15, self.n_dim * 3))

        # Initialize swarm
        positions = np.random.random((swarm_size, self.n_dim))
        velocities = (np.random.random((swarm_size, self.n_dim)) - 0.5) * 0.2
        personal_best_pos = positions.copy()
        personal_best_fit = np.array([self.evaluate(pos) for pos in positions])

        # PSO parameters
        max_iterations = self.n_trials // swarm_size

        for iteration in range(max_iterations):
            if self.evaluations >= self.n_trials:
                break

            # Adaptive parameters
            w = 0.9 - 0.5 * (iteration / max_iterations)  # Inertia weight
            c1 = 2.5 - 1.0 * (iteration / max_iterations)  # Cognitive
            c2 = 1.5 + 1.0 * (iteration / max_iterations)  # Social

            for i in range(swarm_size):
                if self.evaluations >= self.n_trials:
                    break

                # Update velocity
                r1, r2 = np.random.random(self.n_dim), np.random.random(self.n_dim)
                velocities[i] = (w * velocities[i] +
                               c1 * r1 * (personal_best_pos[i] - positions[i]) +
                               c2 * r2 * (self.best_x - positions[i]))

                # Velocity clamping
                vmax = 0.2 * (1 - 0.5 * iteration / max_iterations)
                velocities[i] = np.clip(velocities[i], -vmax, vmax)

                # Update position
                positions[i] = np.clip(positions[i] + velocities[i], 0, 1)

                # Evaluate fitness
                fitness = self.evaluate(positions[i])

                # Update personal best
                if fitness < personal_best_fit[i]:
                    personal_best_fit[i] = fitness
                    personal_best_pos[i] = positions[i].copy()

        return self.best_value, self.best_x


class SimulatedAnnealing(BaseOptimizer):
    """Simulated Annealing algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        # Multi-restart approach
        num_restarts = max(3, self.n_trials // 30)
        trials_per_restart = self.n_trials // num_restarts

        for restart in range(num_restarts):
            if self.evaluations >= self.n_trials:
                break

            # Initialize
            if restart == 0:
                x = 0.5 + (np.random.random(self.n_dim) - 0.5) * 0.4
            else:
                x = np.random.random(self.n_dim)

            fx = self.evaluate(x)
            best_x, best_fx = x.copy(), fx

            # Temperature schedule
            temp = max(1.0, best_fx * 2)
            final_temp = 1e-8

            for iteration in range(trials_per_restart):
                if self.evaluations >= self.n_trials:
                    break

                # Generate neighbor
                step_size = 0.3 * (temp / max(1.0, best_fx * 2))
                new_x = x + (np.random.random(self.n_dim) - 0.5) * 2 * step_size
                new_x = np.clip(new_x, 0, 1)

                new_fx = self.evaluate(new_x)

                # Update best
                if new_fx < best_fx:
                    best_x, best_fx = new_x.copy(), new_fx

                # Metropolis criterion
                delta = new_fx - fx
                if delta < 0 or (temp > final_temp and np.random.random() < np.exp(-delta / temp)):
                    x, fx = new_x, new_fx

                # Cool down
                temp *= 0.99
                temp = max(temp, final_temp)

        return self.best_value, self.best_x


class GeneticAlgorithm(BaseOptimizer):
    """Genetic Algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        pop_size = min(50, max(20, self.n_dim * 4))
        mutation_rate = 0.1
        crossover_rate = 0.8

        # Initialize population
        population = np.random.random((pop_size, self.n_dim))
        fitness = np.array([self.evaluate(ind) for ind in population])

        generations = self.n_trials // pop_size

        for gen in range(generations):
            if self.evaluations >= self.n_trials:
                break

            new_population = []

            for i in range(pop_size):
                if self.evaluations >= self.n_trials:
                    break

                # Tournament selection
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)

                child = parent1.copy()

                # Crossover
                if np.random.random() < crossover_rate:
                    cross_point = np.random.randint(self.n_dim)
                    child[cross_point:] = parent2[cross_point:]

                # Mutation
                mutation_mask = np.random.random(self.n_dim) < mutation_rate
                child[mutation_mask] += (np.random.random(np.sum(mutation_mask)) - 0.5) * 0.2
                child = np.clip(child, 0, 1)

                new_population.append(child)
                fitness_val = self.evaluate(child)

            population = np.array(new_population)

        return self.best_value, self.best_x

    def tournament_selection(self, population, fitness):
        tournament_size = 3
        competitors = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = competitors[np.argmin(fitness[competitors])]
        return population[best_idx].copy()


class RandomSearch(BaseOptimizer):
    """Random Search algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        while self.evaluations < self.n_trials:
            x = np.random.random(self.n_dim)
            self.evaluate(x)

        return self.best_value, self.best_x