"""
Pure Python optimization algorithms organized by algorithm family.

This modular organization replaces the previous 1146-line monolithic file
with focused, maintainable modules for better code organization and readability.
"""

# Base class
from .base import BaseOptimizer

# Evolutionary algorithms
from .evolutionary_algorithms import DifferentialEvolution

# PRIMA algorithms (state-of-the-art derivative-free methods)
from .prima_algorithms import PRIMA_BOBYQA, PRIMA_NEWUOA, PRIMA_UOBYQA

# SciPy-based classical methods
from .scipy_algorithms import LBFGSB, NelderMead, Powell

# NOTE: Additional modules to be created:
# - swarm_algorithms.py (ParticleSwarm, AntColonyOpt, FireflyAlgorithm)
# - metaheuristics.py (SimulatedAnnealing, TabuSearch, HarmonySearch)
# - search_algorithms.py (RandomSearch, HillClimbing, CoordinateDescent, etc.)
# - bayesian_algorithms.py (BayesianOpt)

__all__ = [
    # Base class
    "BaseOptimizer",
    # PRIMA algorithms
    "PRIMA_UOBYQA",
    "PRIMA_NEWUOA",
    "PRIMA_BOBYQA",
    # SciPy algorithms
    "NelderMead",
    "Powell",
    "LBFGSB",
    # Evolutionary algorithms
    "DifferentialEvolution",
]
