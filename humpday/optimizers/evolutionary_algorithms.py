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