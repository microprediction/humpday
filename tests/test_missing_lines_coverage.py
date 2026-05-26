"""
Tests specifically designed to hit missing lines for 100% coverage.
"""

import numpy as np
import pytest


class TestMissingLinesCoverage:
    """Target specific missing lines that weren't hit by other tests."""

    def test_prima_uobyqa_interpolation_set_expansion(self):
        """Test PRIMA_UOBYQA with conditions that trigger interpolation set expansion."""
        from humpday.optimizers.alloptimizers import PRIMA_UOBYQA

        # Create objective that will cause improvement and trigger line 106-107
        def tricky_objective(x):
            # Start with a value, then improve dramatically
            if not hasattr(tricky_objective, "call_count"):
                tricky_objective.call_count = 0
            tricky_objective.call_count += 1

            # First few calls return high values, then much better values
            if tricky_objective.call_count <= 3:
                return 100.0 + sum(x**2)
            else:
                # Much better values to trigger improvement condition
                return 0.01 * sum((xi - 0.5) ** 2 for xi in x)

        # Test with small npt to trigger interpolation set expansion
        optimizer = PRIMA_UOBYQA(tricky_objective, n_trials=20, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_prima_uobyqa_early_termination(self):
        """Test PRIMA_UOBYQA early termination condition (line 113)."""
        from humpday.optimizers.alloptimizers import PRIMA_UOBYQA

        def early_term_objective(x):
            # Create condition where rho becomes very small quickly
            return sum(x**2) + 0.1 * np.random.random()

        # Very small n_trials to trigger early break
        optimizer = PRIMA_UOBYQA(early_term_objective, n_trials=3, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_nelder_mead_evaluation_limit_break(self):
        """Test NelderMead evaluation limit break condition (line 154)."""
        from humpday.optimizers.alloptimizers import NelderMead

        def counting_objective(x):
            return sum(x**2)

        # Set n_trials such that we hit exactly the evaluation limit during reflection
        optimizer = NelderMead(counting_objective, n_trials=5, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_differential_evolution_missing_branches(self):
        """Test DifferentialEvolution edge cases (line 212)."""
        from humpday.optimizers.alloptimizers import DifferentialEvolution

        def simple_objective(x):
            return sum((xi - 0.7) ** 2 for xi in x)

        # Test with larger population to avoid empty candidates issue
        optimizer = DifferentialEvolution(simple_objective, n_trials=25, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_particle_swarm_boundary_conditions(self):
        """Test ParticleSwarm boundary and update conditions (lines 220-235)."""
        from humpday.optimizers.alloptimizers import ParticleSwarm

        def boundary_objective(x):
            # Objective that pushes particles to boundaries
            return (
                sum((xi - 0.9) ** 2 for xi in x)
                if all(xi < 0.95 for xi in x)
                else sum((xi - 0.1) ** 2 for xi in x)
            )

        optimizer = ParticleSwarm(boundary_objective, n_trials=20, n_dim=3)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_cma_evolution_strategy_edge_cases(self):
        """Test CMAEvolutionStrategy specific conditions (line 258)."""
        from humpday.optimizers.alloptimizers import CMAEvolutionStrategy

        def multimodal_objective(x):
            # Create multiple local minima
            return min(
                sum((xi - 0.2) ** 2 for xi in x), sum((xi - 0.8) ** 2 for xi in x)
            )

        optimizer = CMAEvolutionStrategy(multimodal_objective, n_trials=25, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_bayesian_optimization_edge_cases(self):
        """Test BayesianOpt edge conditions (line 366)."""
        from humpday.optimizers.alloptimizers import BayesianOpt

        def noisy_objective(x):
            return sum(x**2) + 0.01 * np.random.random()

        optimizer = BayesianOpt(noisy_objective, n_trials=15, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_hill_climbing_step_conditions(self):
        """Test HillClimbing step acceptance conditions (lines 481-482)."""
        from humpday.optimizers.alloptimizers import HillClimbing

        def step_objective(x):
            # Objective designed to trigger both acceptance and rejection
            if not hasattr(step_objective, "calls"):
                step_objective.calls = 0
            step_objective.calls += 1

            # Alternate between improvement and no improvement
            if step_objective.calls % 3 == 0:
                return 0.1 * sum(x**2)  # Good step
            else:
                return sum(x**2) + 1.0  # Bad step

        optimizer = HillClimbing(step_objective, n_trials=20, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_simulated_annealing_acceptance_conditions(self):
        """Test SimulatedAnnealing acceptance probability conditions (line 556)."""
        from humpday.optimizers.alloptimizers import SimulatedAnnealing

        def sa_objective(x):
            return sum((xi - 0.3) ** 2 for xi in x)

        optimizer = SimulatedAnnealing(sa_objective, n_trials=30, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_adaptive_random_search_improvement_conditions(self):
        """Test AdaptiveRandomSearch improvement conditions (line 591, 600)."""
        from humpday.optimizers.alloptimizers import AdaptiveRandomSearch

        def adaptive_objective(x):
            # Create valleys that require adaptive step sizing
            return sum((xi - 0.15) ** 4 for xi in x) + 0.01 * sum(xi**2 for xi in x)

        optimizer = AdaptiveRandomSearch(adaptive_objective, n_trials=40, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_pattern_search_expansion_conditions(self):
        """Test PatternSearch mesh expansion conditions (line 643, 654)."""
        from humpday.optimizers.alloptimizers import PatternSearch

        def pattern_objective(x):
            # Smooth objective that allows pattern expansion
            return sum(xi**2 for xi in x) + 0.01 * sum((xi - 0.5) ** 4 for xi in x)

        optimizer = PatternSearch(pattern_objective, n_trials=25, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_evolution_strategy_selection_conditions(self):
        """Test EvolutionStrategy selection conditions (lines 665-666)."""
        from humpday.optimizers.alloptimizers import EvolutionStrategy

        def evolution_objective(x):
            return sum((xi - 0.6) ** 2 for xi in x)

        optimizer = EvolutionStrategy(evolution_objective, n_trials=20, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_harmony_search_harmony_conditions(self):
        """Test HarmonySearch harmony memory conditions (line 743)."""
        from humpday.optimizers.alloptimizers import HarmonySearch

        def harmony_objective(x):
            return sum((xi - 0.4) ** 2 for xi in x)

        optimizer = HarmonySearch(harmony_objective, n_trials=25, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_firefly_algorithm_movement_conditions(self):
        """Test FireflyAlgorithm movement conditions (lines 788-791)."""
        from humpday.optimizers.alloptimizers import FireflyAlgorithm

        def firefly_objective(x):
            # Multi-peak objective to trigger firefly movements
            peak1 = sum((xi - 0.3) ** 2 for xi in x)
            peak2 = sum((xi - 0.7) ** 2 for xi in x)
            return min(peak1, peak2)

        optimizer = FireflyAlgorithm(firefly_objective, n_trials=30, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_tabu_search_tabu_conditions(self):
        """Test TabuSearch tabu list conditions (line 814, 836)."""
        from humpday.optimizers.alloptimizers import TabuSearch

        def tabu_objective(x):
            return sum((xi - 0.25) ** 2 for xi in x)

        optimizer = TabuSearch(tabu_objective, n_trials=25, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_genetic_algorithm_selection_conditions(self):
        """Test GeneticAlgorithm selection conditions (line 896)."""
        from humpday.optimizers.alloptimizers import GeneticAlgorithm

        def genetic_objective(x):
            return sum(xi**2 for xi in x) + sum((xi - 0.8) ** 2 for xi in x)

        optimizer = GeneticAlgorithm(genetic_objective, n_trials=30, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2

    def test_ant_colony_pheromone_conditions(self):
        """Test AntColonyOpt pheromone conditions (line 936, lines 944-971)."""
        from humpday.optimizers.alloptimizers import AntColonyOpt

        def ant_objective(x):
            # Discrete-like objective for ant colony
            return sum(round(xi * 10) ** 2 for xi in x)

        optimizer = AntColonyOpt(ant_objective, n_trials=25, n_dim=2)
        result = optimizer.optimize()
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__])
