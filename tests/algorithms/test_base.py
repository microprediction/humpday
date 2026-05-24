"""
Tests for BaseOptimizer class.

Tests the common functionality shared by all optimization algorithms including
objective evaluation, path tracking, and performance monitoring.
"""

import numpy as np
import pytest

from humpday.optimizers.base import BaseOptimizer


class TestOptimizer(BaseOptimizer):
    """Minimal test optimizer for testing base functionality."""

    def optimize(self):
        """Simple test optimization - moves toward center."""
        for _ in range(min(10, self.n_trials)):
            if self.evaluations >= self.n_trials:
                break
            # Move toward center of unit cube
            x = (
                self.best_x
                + 0.1 * (0.5 - self.best_x)
                + 0.05 * np.random.randn(self.n_dim)
            )
            x = np.clip(x, 0, 1)
            self.evaluate(x)
        return self.best_value, self.best_x


class TestBaseOptimizer:
    """Test suite for BaseOptimizer class."""

    def test_initialization(self):
        """Test proper initialization of BaseOptimizer."""

        def objective(x):
            return sum(xi**2 for xi in x)

        optimizer = TestOptimizer(objective, n_trials=100, n_dim=3)

        assert optimizer.objective == objective
        assert optimizer.n_trials == 100
        assert optimizer.n_dim == 3
        assert optimizer.evaluations == 0
        assert optimizer.best_value == float("inf")
        assert len(optimizer.best_x) == 3
        assert not optimizer.track_path
        assert len(optimizer.path) == 0

    def test_evaluate_function(self):
        """Test objective evaluation and tracking."""

        def objective(x):
            return sum((xi - 0.3) ** 2 for xi in x)  # Optimum at [0.3, 0.3]

        optimizer = TestOptimizer(objective, n_trials=50, n_dim=2)

        # Test evaluation
        x = np.array([0.5, 0.6])
        value = optimizer.evaluate(x)

        expected = (0.5 - 0.3) ** 2 + (0.6 - 0.3) ** 2
        assert abs(value - expected) < 1e-10
        assert optimizer.evaluations == 1
        assert optimizer.best_value == value
        np.testing.assert_array_equal(optimizer.best_x, x)

    def test_best_value_tracking(self):
        """Test that best value and position are properly tracked."""

        def objective(x):
            return sum(xi**2 for xi in x)  # Optimum at origin

        optimizer = TestOptimizer(objective, n_trials=50, n_dim=2)

        # Evaluate progressively better points
        points = [
            np.array([0.8, 0.9]),  # value = 0.64 + 0.81 = 1.45
            np.array([0.6, 0.7]),  # value = 0.36 + 0.49 = 0.85 (better)
            np.array([0.9, 0.8]),  # value = 0.81 + 0.64 = 1.45 (worse)
            np.array([0.3, 0.4]),  # value = 0.09 + 0.16 = 0.25 (best)
        ]

        expected_best_values = [1.45, 0.85, 0.85, 0.25]

        for i, point in enumerate(points):
            optimizer.evaluate(point)
            assert abs(optimizer.best_value - expected_best_values[i]) < 1e-10
            if expected_best_values[i] <= (
                expected_best_values[i - 1] if i > 0 else float("inf")
            ):
                np.testing.assert_array_equal(optimizer.best_x, point)

    def test_path_tracking(self):
        """Test path tracking functionality."""

        def objective(x):
            return sum(xi**2 for xi in x)

        optimizer = TestOptimizer(objective, n_trials=20, n_dim=2)
        optimizer.track_path = True

        # Run a few evaluations
        optimizer.optimize()

        # Should have recorded some path points
        assert len(optimizer.path) > 0
        assert all(len(point) == 2 for point in optimizer.path)
        assert all(0 <= xi <= 1 for point in optimizer.path for xi in point)

    def test_clipping_bounds(self):
        """Test that points are properly clipped to unit cube."""

        def objective(x):
            return sum(xi**2 for xi in x)

        optimizer = TestOptimizer(objective, n_trials=50, n_dim=2)

        # Try to evaluate points outside unit cube
        out_of_bounds = np.array([-0.5, 1.5])
        value = optimizer.evaluate(out_of_bounds)

        # Should be clipped to [0, 1]
        np.testing.assert_array_equal(optimizer.best_x, np.array([0.0, 1.0]))
        assert value == 1.0  # 0^2 + 1^2 = 1

    def test_evaluation_limit(self):
        """Test that optimization respects evaluation budget."""

        def objective(x):
            return sum(xi**2 for xi in x)

        optimizer = TestOptimizer(objective, n_trials=5, n_dim=2)
        optimizer.optimize()

        # Should not exceed evaluation budget
        assert optimizer.evaluations <= 5

    def test_different_dimensions(self):
        """Test optimizer works with different dimensions."""

        def objective(x):
            return sum(xi**2 for xi in x)

        for n_dim in [1, 2, 5, 10]:
            optimizer = TestOptimizer(objective, n_trials=20, n_dim=n_dim)
            optimizer.optimize()

            assert len(optimizer.best_x) == n_dim
            assert optimizer.best_value < 1.0  # Should find reasonable solution
            assert optimizer.evaluations > 0

    def test_optimization_improves(self):
        """Test that optimization actually improves the objective."""

        def objective(x):
            return sum((xi - 0.7) ** 2 for xi in x)  # Optimum at [0.7, 0.7]

        optimizer = TestOptimizer(objective, n_trials=50, n_dim=2)

        # Record initial best value
        initial_value = optimizer.best_value

        # Run optimization
        optimizer.optimize()

        # Should have improved
        assert optimizer.best_value < initial_value
        assert optimizer.evaluations > 0

        # Best point should be closer to optimum
        distance_to_optimum = np.linalg.norm(optimizer.best_x - 0.7)
        assert distance_to_optimum < 0.5  # Should get reasonably close


class TestObjectiveFunctions:
    """Test different types of objective functions with BaseOptimizer."""

    def test_sphere_function(self):
        """Test sphere function optimization."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        optimizer = TestOptimizer(sphere, n_trials=30, n_dim=3)
        optimizer.optimize()

        # Should find point close to origin
        assert optimizer.best_value < 0.1
        assert np.linalg.norm(optimizer.best_x - 0.5) < 0.3

    def test_rosenbrock_function(self):
        """Test Rosenbrock function optimization."""

        def rosenbrock(x):
            if len(x) < 2:
                return float("inf")
            return sum(
                100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                for i in range(len(x) - 1)
            )

        # Scale to unit cube (Rosenbrock optimum is at [1, 1] in [-2, 2]^2)
        def rosenbrock_unit_cube(x):
            scaled_x = 4 * np.array(x) - 2  # [0,1] -> [-2,2]
            return rosenbrock(scaled_x)

        optimizer = TestOptimizer(rosenbrock_unit_cube, n_trials=50, n_dim=2)
        optimizer.optimize()

        # Should make progress (Rosenbrock is hard, so just check improvement)
        assert optimizer.evaluations > 0
        assert optimizer.best_value < 1000  # Should be reasonable

    def test_noisy_function(self):
        """Test optimization with noisy objective."""

        def noisy_sphere(x):
            clean_value = sum(xi**2 for xi in x)
            noise = 0.01 * np.random.randn()  # Small noise
            return clean_value + noise

        optimizer = TestOptimizer(noisy_sphere, n_trials=40, n_dim=2)
        optimizer.optimize()

        # Should still find reasonable solution despite noise
        assert optimizer.best_value < 0.2  # Allow for noise tolerance

    def test_discontinuous_function(self):
        """Test optimization with discontinuous objective."""

        def step_function(x):
            return sum(int(xi * 10) for xi in x)  # Step function

        optimizer = TestOptimizer(step_function, n_trials=30, n_dim=2)
        optimizer.optimize()

        # Should handle discontinuities
        assert optimizer.evaluations > 0
        assert optimizer.best_value >= 0
