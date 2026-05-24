"""
Evolutionary algorithm implementations.

These algorithms are inspired by natural evolution processes and include
methods like Differential Evolution, Genetic Algorithms, and Evolution Strategies.
They excel at global optimization and handling multimodal landscapes.
"""

import random
from typing import Tuple

import numpy as np

from .base import BaseOptimizer


class DifferentialEvolution(BaseOptimizer):
    """Differential Evolution algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        # Ensure minimum population size for DE (need at least 4: current + 3 others)
        pop_size = max(10, min(20, self.n_trials // 5))
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
                # Ensure we have enough candidates
                candidates = list(range(pop_size))
                candidates.remove(i)

                if len(candidates) < 3:
                    # Fallback: allow replacement if population too small
                    a, b, c = np.random.choice(candidates, 3, replace=True)
                else:
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


class BayesianOpt(BaseOptimizer):
    """Simplified Bayesian Optimization."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        # Random sampling phase
        X_samples = []
        y_samples = []

        # Initial random samples
        n_initial = min(10, self.n_trials // 3)
        for _ in range(n_initial):
            if self.evaluations >= self.n_trials:
                break
            x = np.random.random(self.n_dim)
            y = self.evaluate(x)
            X_samples.append(x)
            y_samples.append(y)

        # Acquisition phase (simplified - just sample around best points)
        while self.evaluations < self.n_trials:
            if len(y_samples) == 0:
                break

            # Find best points
            best_indices = np.argsort(y_samples)[: min(3, len(y_samples))]

            # Sample around best points with decreasing variance
            variance = max(0.05, 0.3 * (1 - self.evaluations / self.n_trials))

            for idx in best_indices:
                if self.evaluations >= self.n_trials:
                    break

                x_best = X_samples[idx]
                x_new = x_best + np.random.normal(0, variance, self.n_dim)
                x_new = np.clip(x_new, 0, 1)

                y_new = self.evaluate(x_new)
                X_samples.append(x_new)
                y_samples.append(y_new)

        return self.best_value, self.best_x


class CMAEvolutionStrategy(BaseOptimizer):
    """Simplified CMA-ES algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n = self.n_dim
        lambda_ = min(20, self.n_trials // 5)
        mu = lambda_ // 2

        # Initialize
        mean = np.random.random(n)
        sigma = 0.3
        C = np.eye(n)

        while self.evaluations < self.n_trials:
            # Generate population
            population = []
            fitness = []

            for _ in range(lambda_):
                if self.evaluations >= self.n_trials:
                    break

                # Sample from multivariate normal
                x = np.random.multivariate_normal(mean, sigma**2 * C)
                x = np.clip(x, 0, 1)
                f = self.evaluate(x)

                population.append(x)
                fitness.append(f)

            if len(fitness) == 0:
                break

            # Selection and update
            indices = np.argsort(fitness)[:mu]
            selected = [population[i] for i in indices]

            # Update mean
            mean = np.mean(selected, axis=0)

            # Simple covariance update
            if len(selected) > 1:
                centered = np.array(selected) - mean
                C = np.cov(centered, rowvar=False) + 1e-6 * np.eye(n)

        return self.best_value, self.best_x


class TabuSearch(BaseOptimizer):
    """Tabu Search algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        x = np.random.random(self.n_dim)
        f = self.evaluate(x)

        tabu_list = []
        tabu_tenure = 5
        step_size = 0.1

        while self.evaluations < self.n_trials:
            best_neighbor = None
            best_neighbor_f = float("inf")

            # Generate neighbors
            for _ in range(min(10, self.n_trials - self.evaluations)):
                if self.evaluations >= self.n_trials:
                    break

                # Random neighbor
                neighbor = x + np.random.normal(0, step_size, self.n_dim)
                neighbor = np.clip(neighbor, 0, 1)

                # Check if tabu
                is_tabu = any(
                    np.linalg.norm(neighbor - tabu_x) < 0.05 for tabu_x in tabu_list
                )

                if not is_tabu:
                    neighbor_f = self.evaluate(neighbor)
                    if neighbor_f < best_neighbor_f:
                        best_neighbor = neighbor
                        best_neighbor_f = neighbor_f

            if best_neighbor is not None:
                x = best_neighbor
                f = best_neighbor_f

                # Update tabu list
                tabu_list.append(x.copy())
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)

        return self.best_value, self.best_x


class FireflyAlgorithm(BaseOptimizer):
    """Firefly Algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n_fireflies = min(15, self.n_trials // 5)
        alpha = 0.2  # Randomness
        beta0 = 1.0  # Attractiveness
        gamma = 1.0  # Absorption

        # Initialize fireflies
        fireflies = [np.random.random(self.n_dim) for _ in range(n_fireflies)]
        intensities = [self.evaluate(f) for f in fireflies]

        while self.evaluations < self.n_trials:
            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    if self.evaluations >= self.n_trials:
                        break

                    if intensities[j] < intensities[i]:  # j is brighter
                        # Distance
                        r = np.linalg.norm(fireflies[i] - fireflies[j])

                        # Attractiveness
                        beta = beta0 * np.exp(-gamma * r**2)

                        # Move towards brighter firefly
                        fireflies[i] = (
                            fireflies[i]
                            + beta * (fireflies[j] - fireflies[i])
                            + alpha * np.random.randn(self.n_dim)
                        )

                        fireflies[i] = np.clip(fireflies[i], 0, 1)

                        if self.evaluations < self.n_trials:
                            intensities[i] = self.evaluate(fireflies[i])

        return self.best_value, self.best_x


class AntColonyOpt(BaseOptimizer):
    """Ant Colony Optimization (continuous version)."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        n_ants = min(15, self.n_trials // 5)
        n_nodes = 10  # Discretization per dimension
        pheromone = np.ones((self.n_dim, n_nodes))
        evaporation = 0.1

        best_path = None
        best_fitness = float("inf")

        while self.evaluations < self.n_trials:
            # Ant solutions
            for ant in range(n_ants):
                if self.evaluations >= self.n_trials:
                    break

                # Construct solution
                solution = np.zeros(self.n_dim)
                for dim in range(self.n_dim):
                    # Probabilistic selection based on pheromone
                    probs = pheromone[dim] / (np.sum(pheromone[dim]) + 1e-10)
                    node = np.random.choice(n_nodes, p=probs)
                    solution[dim] = node / (n_nodes - 1)  # Map to [0,1]

                fitness = self.evaluate(solution)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_path = solution.copy()

            # Update pheromones
            pheromone *= 1 - evaporation
            if best_path is not None:
                for dim in range(self.n_dim):
                    node = int(best_path[dim] * (n_nodes - 1))
                    pheromone[dim, node] += 1.0 / (best_fitness + 1e-10)

        return self.best_value, self.best_x


class EvolutionStrategy(BaseOptimizer):
    """Evolution Strategy (ES) algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        mu = 10  # Parents
        lambda_ = min(30, self.n_trials // 3)  # Offspring
        sigma = 0.2  # Mutation strength

        # Initialize population
        population = []
        fitness = []

        for _ in range(mu):
            if self.evaluations >= self.n_trials:
                break
            individual = np.random.random(self.n_dim)
            f = self.evaluate(individual)
            population.append(individual)
            fitness.append(f)

        while self.evaluations < self.n_trials:
            # Generate offspring
            offspring = []
            offspring_fitness = []

            for _ in range(lambda_):
                if self.evaluations >= self.n_trials:
                    break

                # Select random parent
                parent_idx = np.random.randint(len(population))
                parent = population[parent_idx]

                # Mutate
                child = parent + np.random.normal(0, sigma, self.n_dim)
                child = np.clip(child, 0, 1)

                child_fitness = self.evaluate(child)
                offspring.append(child)
                offspring_fitness.append(child_fitness)

            if len(offspring) > 0:
                # (μ + λ) selection: combine parents and offspring
                all_individuals = population + offspring
                all_fitness = fitness + offspring_fitness

                # Select best μ individuals
                indices = np.argsort(all_fitness)[:mu]
                population = [all_individuals[i] for i in indices]
                fitness = [all_fitness[i] for i in indices]

        return self.best_value, self.best_x


class HillClimbing(BaseOptimizer):
    """Hill climbing with random restarts."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        current = np.random.random(self.n_dim)
        current_value = self.evaluate(current)
        step_size = 0.1

        while self.evaluations < self.n_trials:
            # Random neighbor
            neighbor = current + np.random.normal(0, step_size, self.n_dim)
            neighbor = np.clip(neighbor, 0, 1)

            neighbor_value = self.evaluate(neighbor)

            if neighbor_value < current_value:
                current = neighbor
                current_value = neighbor_value
            else:
                # Random restart occasionally
                if np.random.random() < 0.1:
                    current = np.random.random(self.n_dim)
                    if self.evaluations < self.n_trials:
                        current_value = self.evaluate(current)

        return self.best_value, self.best_x


class HarmonySearch(BaseOptimizer):
    """Harmony Search algorithm."""

    def optimize(self) -> Tuple[float, np.ndarray]:
        HMS = min(20, max(5, self.n_dim * 2))  # Harmony Memory Size
        HMCR = 0.9  # Harmony Memory Considering Rate
        PAR = 0.3  # Pitch Adjusting Rate

        # Initialize harmony memory
        harmony_memory = []
        for _ in range(HMS):
            if self.evaluations >= self.n_trials:
                break
            harmony = np.random.random(self.n_dim)
            fitness = self.evaluate(harmony)
            harmony_memory.append({"harmony": harmony, "fitness": fitness})

        while self.evaluations < self.n_trials:
            new_harmony = np.zeros(self.n_dim)

            for j in range(self.n_dim):
                if np.random.random() < HMCR:
                    # Pick from harmony memory
                    selected = random.choice(harmony_memory)
                    value = selected["harmony"][j]

                    # Pitch adjustment
                    if np.random.random() < PAR:
                        value = np.clip(value + np.random.normal(0, 0.1), 0, 1)

                    new_harmony[j] = value
                else:
                    # Random selection
                    new_harmony[j] = np.random.random()

            new_fitness = self.evaluate(new_harmony)

            # Update harmony memory (replace worst if new harmony is better)
            harmony_memory.sort(key=lambda x: x["fitness"])
            if new_fitness < harmony_memory[-1]["fitness"]:
                harmony_memory[-1] = {
                    "harmony": new_harmony.copy(),
                    "fitness": new_fitness,
                }

        return self.best_value, self.best_x