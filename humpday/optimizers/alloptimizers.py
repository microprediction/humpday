"""
All optimizers - pure Python implementations of validated algorithms.
No external dependencies beyond numpy. Clean ports, not wrappers.
"""

# Import from new modular structure
from .evolutionary_algorithms import (
    AntColonyOpt,
    BayesianOpt,
    CMAEvolutionStrategy,
    DifferentialEvolution,
    EvolutionStrategy,
    FireflyAlgorithm,
    GeneticAlgorithm,
    HarmonySearch,
    HillClimbing,
    ParticleSwarm,
    RandomSearch,
    SimulatedAnnealing,
    TabuSearch,
)
from .prima_algorithms import PRIMA_BOBYQA, PRIMA_NEWUOA, PRIMA_UOBYQA
from .scipy_algorithms import LBFGSB, NelderMead, Powell
from .search_algorithms import AdaptiveRandomSearch, CoordinateDescent, PatternSearch

# Define PURE_OPTIMIZERS for backward compatibility - all 22 algorithms
PURE_OPTIMIZERS = {
    # PRIMA algorithms
    'PRIMA_UOBYQA': PRIMA_UOBYQA,
    'PRIMA_NEWUOA': PRIMA_NEWUOA,
    'PRIMA_BOBYQA': PRIMA_BOBYQA,
    # SciPy algorithms
    'NelderMead': NelderMead,
    'Powell': Powell,
    'LBFGSB': LBFGSB,
    # Evolutionary algorithms
    'DifferentialEvolution': DifferentialEvolution,
    'ParticleSwarm': ParticleSwarm,
    'SimulatedAnnealing': SimulatedAnnealing,
    'GeneticAlgorithm': GeneticAlgorithm,
    'RandomSearch': RandomSearch,
    'BayesianOpt': BayesianOpt,
    'CMAEvolutionStrategy': CMAEvolutionStrategy,
    'TabuSearch': TabuSearch,
    'FireflyAlgorithm': FireflyAlgorithm,
    'AntColonyOpt': AntColonyOpt,
    'EvolutionStrategy': EvolutionStrategy,
    'HillClimbing': HillClimbing,
    'HarmonySearch': HarmonySearch,
    # Search algorithms
    'AdaptiveRandomSearch': AdaptiveRandomSearch,
    'CoordinateDescent': CoordinateDescent,
    'PatternSearch': PatternSearch,
}


# Create optimizer function interfaces for backward compatibility
def create_optimizer_function(optimizer_class):
    """Create a function interface for an optimizer class."""

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
OPTIMIZERS = [create_optimizer_function(cls) for cls in PURE_OPTIMIZERS.values()]

# Named optimizer functions for direct access
prima_uobyqa = create_optimizer_function(PRIMA_UOBYQA)
prima_newuoa = create_optimizer_function(PRIMA_NEWUOA)
prima_bobyqa = create_optimizer_function(PRIMA_BOBYQA)
nelder_mead = create_optimizer_function(NelderMead)
powell = create_optimizer_function(Powell)
lbfgsb = create_optimizer_function(LBFGSB)
differential_evolution = create_optimizer_function(DifferentialEvolution)
particle_swarm = create_optimizer_function(ParticleSwarm)
simulated_annealing = create_optimizer_function(SimulatedAnnealing)
genetic_algorithm = create_optimizer_function(GeneticAlgorithm)
random_search = create_optimizer_function(RandomSearch)
bayesian_opt = create_optimizer_function(BayesianOpt)
cma_evolution_strategy = create_optimizer_function(CMAEvolutionStrategy)
tabu_search = create_optimizer_function(TabuSearch)
firefly_algorithm = create_optimizer_function(FireflyAlgorithm)
ant_colony_opt = create_optimizer_function(AntColonyOpt)
evolution_strategy = create_optimizer_function(EvolutionStrategy)
hill_climbing = create_optimizer_function(HillClimbing)
harmony_search = create_optimizer_function(HarmonySearch)
adaptive_random_search = create_optimizer_function(AdaptiveRandomSearch)
coordinate_descent = create_optimizer_function(CoordinateDescent)
pattern_search = create_optimizer_function(PatternSearch)

# Algorithm names for easy reference
ALGORITHM_NAMES = list(PURE_OPTIMIZERS.keys())


def get_optimizer(name: str):
    """Get optimizer by name."""
    if name in PURE_OPTIMIZERS:
        return create_optimizer_function(PURE_OPTIMIZERS[name])
    return None


def pure_optimize(objective, algorithm, n_trials, n_dim):
    """Run pure optimization with specified algorithm."""
    if algorithm in PURE_OPTIMIZERS:
        optimizer_class = PURE_OPTIMIZERS[algorithm]
        optimizer = optimizer_class(objective, n_trials, n_dim)
        return optimizer.optimize()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def suggest_pure(n_dim, n_trials):
    """
    Suggest algorithms based on problem characteristics.
    Returns list of algorithm names sorted by expected performance.
    """
    if n_dim <= 2:
        return [
            "NelderMead",
            "PRIMA_UOBYQA",
            "PRIMA_NEWUOA",
            "Powell",
            "LBFGSB",
            "HillClimbing",
            "PatternSearch",
            "CoordinateDescent",
        ]
    elif n_dim <= 10:
        return [
            "DifferentialEvolution",
            "CMAEvolutionStrategy",
            "ParticleSwarm",
            "PRIMA_BOBYQA",
            "BayesianOpt",
            "HarmonySearch",
            "GeneticAlgorithm",
            "PatternSearch",
            "EvolutionStrategy",
            "TabuSearch",
        ]
    elif n_dim <= 50:
        return [
            "CMAEvolutionStrategy",
            "DifferentialEvolution",
            "EvolutionStrategy",
            "ParticleSwarm",
            "AdaptiveRandomSearch",
            "FireflyAlgorithm",
            "AntColonyOpt",
            "RandomSearch",
            "GeneticAlgorithm",
            "SimulatedAnnealing",
        ]
    else:
        return [
            "RandomSearch",
            "AdaptiveRandomSearch",
            "ParticleSwarm",
            "DifferentialEvolution",
            "HillClimbing",
            "CoordinateDescent",
            "SimulatedAnnealing",
            "TabuSearch",
            "EvolutionStrategy",
            "GeneticAlgorithm",
        ]


__all__ = [
    "OPTIMIZERS",
    "PURE_OPTIMIZERS",
    "pure_optimize",
    "suggest_pure",
    "get_optimizer",
    "ALGORITHM_NAMES",
    # Algorithm functions
    "prima_uobyqa",
    "prima_newuoa",
    "prima_bobyqa",
    "nelder_mead",
    "powell",
    "lbfgsb",
    "differential_evolution",
    "particle_swarm",
    "simulated_annealing",
    "genetic_algorithm",
    "random_search",
    "bayesian_opt",
    "cma_evolution_strategy",
    "tabu_search",
    "firefly_algorithm",
    "ant_colony_opt",
    "evolution_strategy",
    "hill_climbing",
    "harmony_search",
    "adaptive_random_search",
    "coordinate_descent",
    "pattern_search",
]
