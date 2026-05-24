"""
Tests specifically designed to hit missing lines in adaptive_optimizer.py for 100% coverage.
"""

import os
import tempfile

import pytest


class TestAdaptiveOptimizerMissingCoverage:
    """Target specific missing lines in adaptive_optimizer.py."""

    def test_normalize_scores_edge_cases(self):
        """Test normalize_scores with edge cases (lines 97, 100-101, 108, 114)."""
        from humpday.optimizers.adaptive_optimizer import (
            normalize_performance as normalize_scores,
        )

        # Test with all equal values (should trigger division by zero handling)
        equal_values = [1.0, 1.0, 1.0, 1.0]
        normalized = normalize_scores(equal_values)
        assert all(isinstance(x, (int, float)) for x in normalized)

        # Test with single value
        single_value = [5.0]
        normalized_single = normalize_scores(single_value)
        assert len(normalized_single) == 1

        # Test with negative values
        negative_values = [-1.0, -2.0, -3.0]
        normalized_negative = normalize_scores(negative_values)
        assert len(normalized_negative) == 3

        # Test with zeros and very small differences
        small_diff_values = [0.0, 1e-15, 2e-15]
        normalized_small = normalize_scores(small_diff_values)
        assert len(normalized_small) == 3

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

        # Test save to invalid path (should trigger error handling)
        try:
            success = elo.save_ratings("/invalid/nonexistent/path/ratings.json")
            assert success is False
        except (OSError, PermissionError, FileNotFoundError):
            pass  # Expected error

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

    def test_adaptive_optimize_edge_parameters(self):
        """Test adaptive_optimize with edge parameters (lines 232-233, 236, 240)."""
        from humpday.optimizers.adaptive_optimizer import (
            adaptive_optimize,
            sphere_variants_generator,
        )

        generator = sphere_variants_generator(n_dim=2)

        # Test with minimal parameters that might trigger edge cases
        results = adaptive_optimize(
            objective_generator=generator,
            trials_budget=50,  # Small budget
            n_dim=2,
            n_warmup_problems=1,  # Minimal warmup
            trials_per_warmup=3,  # Very small trials per warmup
            verbose=True,  # Test verbose mode
        )

        assert "elo_system" in results
        assert "top_algorithms" in results

    def test_run_algorithm_tournament_edge_cases(self):
        """Test run_algorithm_tournament edge cases (lines 253-255, 261-270)."""
        from humpday.optimizers.adaptive_optimizer import (
            EloRatingSystem,
            run_algorithm_tournament,
            sphere_variants_generator,
        )

        elo_system = EloRatingSystem()
        generator = sphere_variants_generator(n_dim=2)

        # Test with minimal parameters
        updated_elo = run_algorithm_tournament(
            objective_generator=generator,
            trials_per_problem=3,  # Very small
            n_problems=1,  # Single problem
            n_dim=2,
            elo_system=elo_system,
        )

        assert isinstance(updated_elo, EloRatingSystem)

    def test_run_single_tournament_specific_conditions(self):
        """Test run_single_tournament with specific conditions (lines 292-294, 297-299)."""
        from humpday.optimizers.adaptive_optimizer import (
            EloRatingSystem,
            run_single_tournament,
        )

        elo_system = EloRatingSystem()

        def tournament_objective(x):
            return sum(x**2)

        # Test with very few trials to trigger specific conditions
        updated_elo = run_single_tournament(
            objective=tournament_objective,
            elo_system=elo_system,
            trials_per_algorithm=2,  # Very small
            n_dim=2,
        )

        assert isinstance(updated_elo, EloRatingSystem)

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
