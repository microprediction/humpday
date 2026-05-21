"""
All optimizers - pure Python implementations of the 22 validated algorithms.
No external dependencies beyond numpy. Lightweight and reliable.
"""

from .optimizers import (
    PURE_OPTIMIZERS,
    pure_optimize,
    suggest_pure,
    PRIMA_UOBYQA,
    NelderMead,
    DifferentialEvolution,
    ParticleSwarm,
    RandomSearch,
    HillClimbing,
    SimulatedAnnealing,
    HarmonySearch,
    GeneticAlgorithm
)

# Create optimizer function wrappers for backward compatibility
def create_optimizer_wrapper(optimizer_class):
    """Create a function wrapper for an optimizer class."""
    def optimizer_function(objective, n_dim, n_trials=100):
        optimizer = optimizer_class(objective, n_trials, n_dim)
        return optimizer.optimize()

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
    'OPTIMIZERS',
    'PURE_OPTIMIZERS',
    'pure_optimize',
    'suggest_pure',
    'get_optimizer',
    'ALGORITHM_NAMES',
    'prima_uobyqa',
    'nelder_mead',
    'differential_evolution',
    'particle_swarm',
    'random_search',
    'hill_climbing',
    'simulated_annealing',
    'harmony_search',
    'genetic_algorithm'
]