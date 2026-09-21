"""
Tests specifically designed to hit missing lines in adaptive_optimizer.py for 100% coverage.
"""

import os
import tempfile

import pytest


class TestAdaptiveOptimizerMissingCoverage:
    """Target specific missing lines in adaptive_optimizer.py."""

    def test_pairwise_outcome_orders_finite_values(self):
        """Lower is better, ties draw, and the scale does not matter."""
        from humpday.optimizers.adaptive_optimizer import pairwise_outcome

        assert pairwise_outcome(1.0, 2.0) == 1.0
        assert pairwise_outcome(2.0, 1.0) == 0.0
        assert pairwise_outcome(1.0, 1.0) == 0.5
        assert pairwise_outcome(-3.0, -1.0) == 1.0
        # Differences far below any normalisation resolution still decide.
        assert pairwise_outcome(1e-15, 2e-15) == 1.0

    def test_pairwise_outcome_handles_failures(self):
        """A failure loses to any finite value, and two failures are not compared."""
        import math

        from humpday.optimizers.adaptive_optimizer import pairwise_outcome

        for bad in (float("inf"), float("nan"), -float("inf")):
            assert pairwise_outcome(1.0, bad) == 1.0, bad
            assert pairwise_outcome(bad, 1.0) == 0.0, bad

        assert pairwise_outcome(float("inf"), float("inf")) is None
        assert pairwise_outcome(float("nan"), float("inf")) is None
        assert math.isnan(float("nan"))  # guards the premise of the case above

    def test_one_failure_does_not_erase_the_round(self):
        """The regression behind #344: an inf used to make every score NaN.

        With a NaN score no comparison is true, so every pair fell through to
        the equal branch or to the later-listed algorithm, and a round of real
        results was recorded as a round of draws.
        """
        from humpday.optimizers.adaptive_optimizer import (
            EloRatingSystem,
            pairwise_outcome,
        )

        results = {"Good": 1.0, "Middling": 2.0, "Broken": float("inf")}
        elo = EloRatingSystem()
        names = list(results)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                outcome = pairwise_outcome(results[a], results[b])
                if outcome is not None:
                    elo.update_ratings(a, b, outcome)

        assert elo.get_rating("Good") > elo.get_rating("Middling")
        assert elo.get_rating("Middling") > elo.get_rating("Broken")

    def test_elo_system_edge_cases(self):
        """Test EloRatingSystem edge cases (lines 143, 153-155)."""
        from humpday.optimizers.adaptive_optimizer import EloRatingSystem

        elo = EloRatingSystem()

        # Test get_rating for non-existent algorithm
        rating = elo.get_rating("NonExistentAlgorithm")
        assert rating == elo.initial_rating

        # Test update_ratings with tie (score = 0.5)
        elo.update_ratings("RandomSearch", "NelderMead", 0.5)

        # Test update_ratings with complete win/loss
        elo.update_ratings("RandomSearch", "NelderMead", 1.0)
        elo.update_ratings("RandomSearch", "NelderMead", 0.0)

        # Test save against an injected I/O failure. (A supposedly unwritable
        # absolute path is not a reliable stand-in: a privileged environment
        # can create it.)
        import builtins
        from unittest import mock

        real_open = builtins.open

        def denied(path, *args, **kwargs):
            if str(path).endswith("ratings.json"):
                raise PermissionError("injected")
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=denied):
            success = elo.save_ratings("ratings.json")
        assert success is False
        assert isinstance(elo.last_error, PermissionError)

    def test_elo_system_file_operations(self):
        """Test EloRatingSystem file save/load edge cases (lines 185, 187)."""
        from humpday.optimizers.adaptive_optimizer import EloRatingSystem

        elo = EloRatingSystem()

        # Add some ratings
        elo.update_ratings("RandomSearch", "NelderMead", 0.7)
        elo.update_ratings("HillClimbing", "SimulatedAnnealing", 0.3)

        # Test save to a valid temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Test successful save
            success = elo.save_ratings(temp_path)
            assert success is True

            # Test load from existing file
            new_elo = EloRatingSystem()
            success = new_elo.load_ratings(temp_path)
            assert success is True

            # Verify ratings were loaded
            assert "RandomSearch" in new_elo.ratings
            assert "NelderMead" in new_elo.ratings

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        # Test load from non-existent file
        fake_elo = EloRatingSystem()
        success = fake_elo.load_ratings("definitely_nonexistent_file.json")
        assert success is False

    def test_adaptive_optimize_components(self):
        """Test individual components of adaptive_optimize without full tournament."""
        from humpday.optimizers.adaptive_optimizer import (
            EloRatingSystem,
            sphere_variants_generator,
        )

        # Test EloRatingSystem initialization with different parameters
        elo1 = EloRatingSystem(initial_rating=1600.0, k_factor=16.0)
        assert elo1.initial_rating == 1600.0
        assert elo1.k_factor == 16.0

        # Test generator functionality
        generator = sphere_variants_generator(n_dim=3)
        obj1 = next(generator)
        obj2 = next(generator)

        # Test that generator produces valid objective functions
        test_point = [0.1, 0.2, 0.3]
        result1 = obj1(test_point)
        result2 = obj2(test_point)

        assert isinstance(result1, (int, float))
        assert isinstance(result2, (int, float))
        assert result1 >= 0  # Sphere function is non-negative

    def test_tournament_function_signature(self):
        """Test run_algorithm_tournament function exists and has correct signature."""
        import inspect

        from humpday.optimizers.adaptive_optimizer import (
            EloRatingSystem,
            run_algorithm_tournament,
        )

        # Test that function exists and has expected parameters
        sig = inspect.signature(run_algorithm_tournament)
        expected_params = {
            "objective_generator",
            "trials_per_problem",
            "n_problems",
            "n_dim",
            "elo_system",
            "algorithms_to_test",
        }
        actual_params = set(sig.parameters.keys())

        assert expected_params.issubset(actual_params)

        # Test EloRatingSystem creation works
        elo_system = EloRatingSystem()
        assert hasattr(elo_system, "ratings")
        assert hasattr(elo_system, "match_history")

    def test_suggest_algorithm_from_elo_edge_cases(self):
        """Test suggest_algorithm_from_elo with edge cases."""
        from humpday.optimizers.adaptive_optimizer import (
            EloRatingSystem,
            suggest_algorithm_from_elo,
        )

        elo_system = EloRatingSystem()

        # Test with different problem types and dimensions
        result = suggest_algorithm_from_elo(elo_system, n_dim=2, problem_type="smooth")
        assert isinstance(result, str)

        result = suggest_algorithm_from_elo(
            elo_system, n_dim=20, problem_type="multimodal"
        )
        assert isinstance(result, str)

        result = suggest_algorithm_from_elo(elo_system, n_dim=5, problem_type="noisy")
        assert isinstance(result, str)

    def test_get_top_algorithms_edge_cases(self):
        """Test get_top_algorithms with edge cases (lines 326-344)."""
        from humpday.optimizers.adaptive_optimizer import EloRatingSystem

        # Test with empty elo system
        empty_elo = EloRatingSystem()
        top_algs = empty_elo.get_top_algorithms(n=5)
        assert isinstance(top_algs, list)

        # Test with more algorithms requested than exist
        elo_few = EloRatingSystem()
        elo_few.update_ratings("RandomSearch", "NelderMead", 0.8)
        top_many = elo_few.get_top_algorithms(n=50)  # More than available
        assert isinstance(top_many, list)

    def test_objective_generators_edge_cases(self):
        """Test objective generators with edge cases (lines 369-372)."""
        from humpday.optimizers.adaptive_optimizer import (
            rosenbrock_variants_generator,
            sphere_variants_generator,
        )

        # Test sphere generator with 1D (edge case)
        sphere_1d = sphere_variants_generator(n_dim=1)
        obj_1d = next(sphere_1d)
        result_1d = obj_1d([0.5])
        assert isinstance(result_1d, (int, float))

        # Test sphere generator with high dimension
        sphere_high = sphere_variants_generator(n_dim=20)
        obj_high = next(sphere_high)
        result_high = obj_high([0.1] * 20)
        assert isinstance(result_high, (int, float))

        # Test rosenbrock generator with minimum dimension
        rosenbrock_2d = rosenbrock_variants_generator(n_dim=2)  # Minimum for Rosenbrock
        obj_rb_2d = next(rosenbrock_2d)
        result_rb = obj_rb_2d([0.5, 0.5])
        assert isinstance(result_rb, (int, float))

        # Test multiple variants from generator
        sphere_gen = sphere_variants_generator(n_dim=3)
        objectives = [next(sphere_gen) for _ in range(5)]
        test_point = [0.1, 0.2, 0.3]
        for obj in objectives:
            result = obj(test_point)
            assert isinstance(result, (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
