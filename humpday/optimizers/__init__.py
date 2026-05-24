"""
Pure Python optimization algorithms organized by algorithm family.

**HUMPDAY LIBRARY INTENT - MUST READ:**

HumpDay provides PURE PYTHON implementations of established optimization algorithms.
- NO external dependencies except numpy (no scipy, cvxpy, nlopt, etc.)
- NO wrappers around 3rd party libraries
- NO compilation required - works anywhere Python runs
- Algorithms must be algorithmically CORRECT versions of reference implementations

IMPLEMENTATION RULES:
1. Study reference implementations/papers to understand the algorithm
2. Implement the algorithm correctly in pure Python + numpy
3. Use 3rd party packages ONLY in testing for validation/comparison
4. Never import optimization libraries in main implementation code
5. Goal: small footprint, no dependencies, universal compatibility

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
