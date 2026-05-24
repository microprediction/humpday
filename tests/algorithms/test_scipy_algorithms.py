"""
Tests for SciPy-based algorithm implementations.

Tests Nelder-Mead, Powell, and L-BFGS-B algorithms for correctness,
convergence behavior, and robustness across different problem types.
"""

import numpy as np
import pytest

from humpday.optimizers.scipy_algorithms import LBFGSB, NelderMead, Powell


class TestSciPyAlgorithms:
    """Test suite for SciPy-based algorithm family."""

    @pytest.fixture
    def sphere_function(self):
        """Simple sphere function."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        return sphere

    @pytest.fixture
    def rosenbrock_function(self):
        """Rosenbrock function (classic optimization test)."""

        def rosenbrock(x):
            if len(x) < 2:
                return 1000.0
            return sum(
                100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                for i in range(len(x) - 1)
            )

        return rosenbrock

    @pytest.fixture
    def beale_function(self):
        """Beale function - has steep ridges."""

        def beale(x):
            if len(x) != 2:
                return 1000.0
            # Scale from [0,1]² to [-4.5, 4.5]²
            x1, x2 = 9 * x[0] - 4.5, 9 * x[1] - 4.5
            term1 = (1.5 - x1 + x1 * x2) ** 2
            term2 = (2.25 - x1 + x1 * x2**2) ** 2
            term3 = (2.625 - x1 + x1 * x2**3) ** 2
            return term1 + term2 + term3

        return beale

    def test_nelder_mead_initialization(self, sphere_function):
        """Test Nelder-Mead initializes correctly."""
        optimizer = NelderMead(sphere_function, n_trials=100, n_dim=3)

        assert optimizer.objective == sphere_function
        assert optimizer.n_trials == 100
        assert optimizer.n_dim == 3
        assert optimizer.evaluations == 0

    def test_nelder_mead_simplex_optimization(self, sphere_function):
        """Test Nelder-Mead simplex method finds good solutions."""
        np.random.seed(42)

        optimizer = NelderMead(sphere_function, n_trials=80, n_dim=2)
        best_value, best_x = optimizer.optimize()

        # Nelder-Mead should work well on smooth functions
        assert optimizer.evaluations > 0
        assert optimizer.evaluations <= 80
        assert best_value < 0.1  # Should find good solution

        # Best point should be near origin
        distance_to_origin = np.linalg.norm(best_x)
        assert distance_to_origin < 0.3

    def test_nelder_mead_rosenbrock(self, rosenbrock_function):
        """Test Nelder-Mead on Rosenbrock function."""
        np.random.seed(123)

        optimizer = NelderMead(rosenbrock_function, n_trials=200, n_dim=2)
        best_value, best_x = optimizer.optimize()

        # Rosenbrock is challenging, but Nelder-Mead should make progress
        assert best_value < 10  # Should be much better than random start
        assert optimizer.evaluations > 0

    def test_powell_initialization(self, sphere_function):
        """Test Powell method initializes correctly."""
        optimizer = Powell(sphere_function, n_trials=60, n_dim=4)

        assert optimizer.objective == sphere_function
        assert optimizer.n_trials == 60
        assert optimizer.n_dim == 4

    def test_powell_conjugate_directions(self, sphere_function):
        """Test Powell's conjugate direction method."""
        np.random.seed(42)

        optimizer = Powell(sphere_function, n_trials=100, n_dim=3)
        best_value, best_x = optimizer.optimize()

        # Powell should work well on quadratic functions
        assert optimizer.evaluations > 0
        assert best_value < 0.2
        assert all(0 <= xi <= 1 for xi in best_x)

    def test_powell_different_dimensions(self, sphere_function):
        """Test Powell method across dimensions."""
        for n_dim in [2, 3, 5]:
            optimizer = Powell(sphere_function, n_trials=n_dim * 20, n_dim=n_dim)
            best_value, best_x = optimizer.optimize()

            assert len(best_x) == n_dim
            assert best_value < 1.0  # Should find reasonable solution
            assert 0 <= optimizer.evaluations <= n_dim * 20

    def test_lbfgsb_initialization(self, sphere_function):
        """Test L-BFGS-B initializes correctly."""
        optimizer = LBFGSB(sphere_function, n_trials=50, n_dim=2)

        assert optimizer.objective == sphere_function
        assert optimizer.n_trials == 50
        assert optimizer.n_dim == 2

    def test_lbfgsb_gradient_based(self, sphere_function):
        """Test L-BFGS-B gradient-based optimization."""
        np.random.seed(42)

        optimizer = LBFGSB(sphere_function, n_trials=60, n_dim=2)
        best_value, best_x = optimizer.optimize()

        # L-BFGS-B uses finite differences, should work well on smooth functions
        assert optimizer.evaluations > 0
        assert best_value < 0.5  # Should make good progress

    def test_lbfgsb_bounds_handling(self, sphere_function):
        """Test L-BFGS-B handles bounds correctly."""
        optimizer = LBFGSB(sphere_function, n_trials=40, n_dim=3)
        optimizer.track_path = True

        best_value, best_x = optimizer.optimize()

        # All points should respect unit cube bounds
        assert all(0 <= xi <= 1 for xi in best_x)

        # Check path points are also bounded
        for point in optimizer.path:
            assert all(0 <= xi <= 1 for xi in point)

    def test_scipy_algorithms_comparison(self, beale_function):
        """Compare SciPy algorithms on challenging function."""
        np.random.seed(456)

        algorithms = [
            ("Nelder-Mead", NelderMead),
            ("Powell", Powell),
            ("L-BFGS-B", LBFGSB),
        ]

        results = {}

        for name, AlgorithmClass in algorithms:
            optimizer = AlgorithmClass(beale_function, n_trials=150, n_dim=2)
            best_value, best_x = optimizer.optimize()

            results[name] = {
                "value": best_value,
                "x": best_x,
                "evaluations": optimizer.evaluations,
            }

            # All should handle the function without crashing
            assert np.isfinite(best_value)
            assert optimizer.evaluations > 0

        print("\nSciPy Algorithm Comparison on Beale function:")
        for name, result in results.items():
            print(
                f"  {name:12}: {result['value']:8.4f} ({result['evaluations']} evals)"
            )

    def test_scipy_robustness(self):
        """Test SciPy algorithms handle edge cases."""

        def noisy_function(x):
            clean = sum((xi - 0.3) ** 2 for xi in x)
            noise = 0.01 * np.random.randn()
            return clean + noise

        for AlgorithmClass in [NelderMead, Powell, LBFGSB]:
            np.random.seed(789)
            optimizer = AlgorithmClass(noisy_function, n_trials=60, n_dim=2)
            best_value, best_x = optimizer.optimize()

            # Should handle noisy functions reasonably
            assert np.isfinite(best_value)
            assert best_value < 0.5  # Should still make progress despite noise

    def test_scipy_convergence(self):
        """Test SciPy algorithms converge properly."""

        def quadratic(x):
            return sum((xi - 0.4) ** 2 for xi in x)

        for AlgorithmClass in [NelderMead, Powell, LBFGSB]:
            np.random.seed(42)
            optimizer = AlgorithmClass(quadratic, n_trials=100, n_dim=2)
            best_value, best_x = optimizer.optimize()

            # Should converge to near-optimal solution
            if optimizer.evaluations >= 30:  # If had enough evaluations
                assert best_value < 0.05
                distance_to_optimum = np.linalg.norm(best_x - 0.4)
                assert distance_to_optimum < 0.2

    def test_scipy_path_tracking(self):
        """Test path tracking with SciPy algorithms."""

        def simple_objective(x):
            return sum(xi**2 for xi in x)

        for AlgorithmClass in [NelderMead, Powell, LBFGSB]:
            optimizer = AlgorithmClass(simple_objective, n_trials=30, n_dim=2)
            optimizer.track_path = True

            best_value, best_x = optimizer.optimize()

            # Should record optimization path
            if len(optimizer.path) > 0:  # Some algorithms might not track every step
                assert all(len(point) == 2 for point in optimizer.path)
                assert all(0 <= xi <= 1 for point in optimizer.path for xi in point)

    def test_scipy_evaluation_budget(self):
        """Test SciPy algorithms respect evaluation budgets."""

        def simple_function(x):
            return sum(xi**2 for xi in x)

        for AlgorithmClass in [NelderMead, Powell, LBFGSB]:
            optimizer = AlgorithmClass(simple_function, n_trials=25, n_dim=2)
            best_value, best_x = optimizer.optimize()

            # Should not exceed evaluation budget
            assert optimizer.evaluations <= 25
            assert optimizer.evaluations > 0

    def test_nelder_mead_simplex_behavior(self):
        """Test Nelder-Mead specific simplex operations."""

        def asymmetric_function(x):
            # Function that benefits from simplex adaptation
            return x[0] ** 2 + 10 * x[1] ** 2

        np.random.seed(42)
        optimizer = NelderMead(asymmetric_function, n_trials=100, n_dim=2)
        best_value, best_x = optimizer.optimize()

        # Should handle anisotropic functions reasonably
        assert best_value < 1.0
        assert abs(best_x[0]) < 0.5  # Should get close to x=0
        assert abs(best_x[1]) < 0.3  # Should get close to y=0 (more important)

    def test_powell_line_search(self):
        """Test Powell method's line search behavior."""

        def ridge_function(x):
            # Function with ridge structure
            return (x[0] - 0.3) ** 2 + 0.1 * (x[1] - 0.7) ** 2

        np.random.seed(42)
        optimizer = Powell(ridge_function, n_trials=80, n_dim=2)
        best_value, best_x = optimizer.optimize()

        # Should handle ridge functions well due to conjugate directions
        assert best_value < 0.1
        assert abs(best_x[0] - 0.3) < 0.2
        assert abs(best_x[1] - 0.7) < 0.2

    def test_lbfgsb_gradient_approximation(self):
        """Test L-BFGS-B finite difference gradient approximation."""

        def smooth_function(x):
            return sum(i * xi**2 for i, xi in enumerate(x, 1))  # Weighted quadratic

        np.random.seed(42)
        optimizer = LBFGSB(smooth_function, n_trials=60, n_dim=3)
        best_value, best_x = optimizer.optimize()

        # Should work well with finite difference gradients
        assert best_value < 0.2
        assert all(abs(xi) < 0.3 for xi in best_x)  # Should be near origin


class TestSciPyPerformance:
    """Performance tests for SciPy algorithms."""

    def test_scipy_efficiency(self):
        """Test SciPy algorithms are reasonably efficient."""

        def simple_quadratic(x):
            return sum((xi - 0.5) ** 2 for xi in x)

        for AlgorithmClass in [NelderMead, Powell, LBFGSB]:
            optimizer = AlgorithmClass(simple_quadratic, n_trials=50, n_dim=3)

            import time

            start_time = time.time()
            best_value, best_x = optimizer.optimize()
            elapsed = time.time() - start_time

            # Should complete quickly for simple problems
            assert elapsed < 0.5  # Should be fast
            assert best_value < 0.1  # Should solve well

    def test_scipy_scalability(self):
        """Test how SciPy algorithms scale with problem size."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        dimensions = [2, 4, 6, 8]

        for n_dim in dimensions:
            for AlgorithmClass in [NelderMead, Powell, LBFGSB]:
                optimizer = AlgorithmClass(sphere, n_trials=n_dim * 15, n_dim=n_dim)
                best_value, best_x = optimizer.optimize()

                # Should still work reasonably in higher dimensions
                assert best_value < 1.0
                assert len(best_x) == n_dim
                assert all(0 <= xi <= 1 for xi in best_x)
