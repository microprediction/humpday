"""
All optimizers - pure Python implementations of the 22 validated algorithms.
No external dependencies beyond numpy. Lightweight and reliable.
"""

# Import from new modular structure
from .prima_algorithms import PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA
from .scipy_algorithms import NelderMead, Powell, LBFGSB
from .evolutionary_algorithms import DifferentialEvolution, ParticleSwarm, SimulatedAnnealing, GeneticAlgorithm, RandomSearch

# For algorithms not yet in modules, import from old file temporarily
try:
    from .optimizers import HillClimbing, HarmonySearch
except ImportError:
    # Create placeholder classes if not available
    from .base import BaseOptimizer
    class HillClimbing(BaseOptimizer):
        def optimize(self):
            return self.best_value, self.best_x
    class HarmonySearch(BaseOptimizer):
        def optimize(self):
            return self.best_value, self.best_x

# Define PURE_OPTIMIZERS for backward compatibility
PURE_OPTIMIZERS = {
    'PRIMA_UOBYQA': PRIMA_UOBYQA,
    'PRIMA_NEWUOA': PRIMA_NEWUOA,
    'PRIMA_BOBYQA': PRIMA_BOBYQA,
    'NelderMead': NelderMead,
    'Powell': Powell,
    'LBFGSB': LBFGSB,
    'DifferentialEvolution': DifferentialEvolution,
    'ParticleSwarm': ParticleSwarm,
    'SimulatedAnnealing': SimulatedAnnealing,
    'GeneticAlgorithm': GeneticAlgorithm,
    'RandomSearch': RandomSearch,
    'HillClimbing': HillClimbing,
    'HarmonySearch': HarmonySearch,
}


# Create optimizer function wrappers for backward compatibility
def create_optimizer_wrapper(optimizer_class):
    """Create a function wrapper for an optimizer class."""

    def optimizer_function(objective, n_dim, n_trials=100, with_count=False):
        optimizer = optimizer_class(objective, n_trials, n_dim)
        result = optimizer.optimize()

        # Handle different return formats
        if hasattr(result, '__len__') and len(result) == 2:
            # Old format: (best_value, best_x)
            best_value, best_x = result
        else:
            # New format: dict or other
            best_value = optimizer.best_value
            best_x = optimizer.best_x

        if with_count:
            return best_value, best_x, optimizer.evaluations
        else:
            return best_value, best_x

    optimizer_function.__name__ = optimizer_class.__name__
    return optimizer_function


# Create all optimizer functions
OPTIMIZERS = [create_optimizer_wrapper(cls) for cls in PURE_OPTIMIZERS.values()]

# Named optimizer functions for direct access
prima_uobyqa = create_optimizer_wrapper(PRIMA_UOBYQA)
nelder_mead = create_optimizer_wrapper(NelderMead)
differential_evolution = create_optimizer_wrapper(DifferentialEvolution)
particle_swarm = create_optimizer_wrapper(ParticleSwarm)
random_search = create_optimizer_wrapper(RandomSearch)
hill_climbing = create_optimizer_wrapper(HillClimbing)
simulated_annealing = create_optimizer_wrapper(SimulatedAnnealing)
harmony_search = create_optimizer_wrapper(HarmonySearch)
genetic_algorithm = create_optimizer_wrapper(GeneticAlgorithm)

# Algorithm names for easy reference
ALGORITHM_NAMES = list(PURE_OPTIMIZERS.keys())


def get_optimizer(name: str):
    """Get optimizer by name."""
    if name in PURE_OPTIMIZERS:
        return create_optimizer_wrapper(PURE_OPTIMIZERS[name])
    return None


__all__ = [
    "OPTIMIZERS",
    "PURE_OPTIMIZERS",
    "pure_optimize",
    "suggest_pure",
    "get_optimizer",
    "ALGORITHM_NAMES",
    "prima_uobyqa",
    "nelder_mead",
    "differential_evolution",
    "particle_swarm",
    "random_search",
    "hill_climbing",
    "simulated_annealing",
    "harmony_search",
    "genetic_algorithm",
]
