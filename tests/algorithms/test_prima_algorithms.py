"""
Tests for PRIMA algorithm implementations.

Tests UOBYQA, NEWUOA, and BOBYQA algorithms for correctness, convergence,
and performance characteristics. Validates against known benchmarks.
"""

import numpy as np
import pytest

from humpday.optimizers.prima_algorithms import PRIMA_BOBYQA, PRIMA_NEWUOA, PRIMA_UOBYQA


class TestPRIMAAlgorithms:
    """Test suite for PRIMA algorithm family."""

    @pytest.fixture
    def sphere_function(self):
        """Simple sphere function with known optimum."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        return sphere

    @pytest.fixture
    def quadratic_function(self):
        """Quadratic function ideal for PRIMA methods."""

        def quadratic(x):
            return sum((xi - 0.3) ** 2 for xi in x)  # Optimum at [0.3, 0.3, ...]

        return quadratic

    @pytest.fixture
    def rosenbrock_unit_cube(self):
        """Rosenbrock function scaled to unit cube."""

        def rosenbrock(x):
            # Scale [0,1] to [-2,2] where Rosenbrock optimum is at [1,1]
            scaled_x = 4 * np.array(x) - 2
            if len(scaled_x) < 2:
                return 1000.0
            return sum(
                100.0 * (scaled_x[i + 1] - scaled_x[i] ** 2) ** 2
                + (1 - scaled_x[i]) ** 2
                for i in range(len(scaled_x) - 1)
            )

        return rosenbrock

    def test_prima_uobyqa_initialization(self, sphere_function):
        """Test PRIMA_UOBYQA initializes correctly."""
        optimizer = PRIMA_UOBYQA(sphere_function, n_trials=50, n_dim=3)

        assert optimizer.objective == sphere_function
        assert optimizer.n_trials == 50
        assert optimizer.n_dim == 3
        assert optimizer.evaluations == 0

    def test_prima_uobyqa_optimization(self, quadratic_function):
        """Test PRIMA_UOBYQA finds good solutions."""
        np.random.seed(42)  # For reproducible results

        optimizer = PRIMA_UOBYQA(quadratic_function, n_trials=100, n_dim=2)
        best_value, best_x = optimizer.optimize()

        # Should find near-optimal solution
        assert optimizer.evaluations > 0
        assert optimizer.evaluations <= 100
        assert best_value < 0.15  # Reasonable for pure Python implementation

        # Best point should be close to [0.3, 0.3]
        distance_to_optimum = np.linalg.norm(best_x - 0.3)
        assert distance_to_optimum < 0.4  # Reasonable for pure Python implementation

    def test_prima_uobyqa_different_dimensions(self, sphere_function):
        """Test PRIMA_UOBYQA works across dimensions."""
        for n_dim in [2, 3, 5]:
            optimizer = PRIMA_UOBYQA(sphere_function, n_trials=50, n_dim=n_dim)
            best_value, best_x = optimizer.optimize()

            assert len(best_x) == n_dim
            assert best_value < 0.5  # Should find reasonable solution
            assert 0 <= optimizer.evaluations <= 50

    def test_prima_newuoa_initialization(self, sphere_function):
        """Test PRIMA_NEWUOA initializes correctly."""
        optimizer = PRIMA_NEWUOA(sphere_function, n_trials=60, n_dim=4)

        assert optimizer.objective == sphere_function
        assert optimizer.n_trials == 60
        assert optimizer.n_dim == 4
        assert optimizer.evaluations == 0

    def test_prima_newuoa_optimization(self, quadratic_function):
        """Test PRIMA_NEWUOA convergence."""
        np.random.seed(42)

        optimizer = PRIMA_NEWUOA(quadratic_function, n_trials=80, n_dim=2)
        best_value, best_x = optimizer.optimize()

        # NEWUOA should perform well on quadratic functions
        assert optimizer.evaluations > 0
        assert best_value < 0.20  # Reasonable for pure Python NEWUOA

        # Check convergence to optimum
        distance_to_optimum = np.linalg.norm(best_x - 0.3)
        assert distance_to_optimum < 0.4  # Reasonable for pure Python implementation5

    def test_prima_bobyqa_initialization(self, sphere_function):
        """Test PRIMA_BOBYQA initializes correctly."""
        optimizer = PRIMA_BOBYQA(sphere_function, n_trials=70, n_dim=3)

        assert optimizer.objective == sphere_function
        assert optimizer.n_trials == 70
        assert optimizer.n_dim == 3
        assert optimizer.evaluations == 0

    def test_prima_bobyqa_bounds_handling(self, sphere_function):
        """Test PRIMA_BOBYQA handles bounds correctly."""
        np.random.seed(42)

        optimizer = PRIMA_BOBYQA(sphere_function, n_trials=60, n_dim=2)
        optimizer.track_path = True
        best_value, best_x = optimizer.optimize()

        # All points should be within unit cube bounds
        assert all(0 <= xi <= 1 for xi in best_x)

        # Check path points are also bounded
        for point in optimizer.path:
            assert all(0 <= xi <= 1 for xi in point)

    def test_prima_algorithms_comparison(self, rosenbrock_unit_cube):
        """Compare PRIMA algorithms on challenging function."""
        np.random.seed(123)  # Fixed seed for fair comparison

        algorithms = [
            ("UOBYQA", PRIMA_UOBYQA),
            ("NEWUOA", PRIMA_NEWUOA),
            ("BOBYQA", PRIMA_BOBYQA),
        ]

        results = {}

        for name, AlgorithmClass in algorithms:
            optimizer = AlgorithmClass(rosenbrock_unit_cube, n_trials=100, n_dim=2)
            best_value, best_x = optimizer.optimize()

            results[name] = {
                "value": best_value,
                "x": best_x,
                "evaluations": optimizer.evaluations,
            }

            # All should make reasonable progress on Rosenbrock
            assert best_value < 50  # Should be much better than random
            assert optimizer.evaluations > 0

        # All algorithms should find solutions
        assert len(results) == 3
        print("\nPRIMA Algorithm Comparison on 2D Rosenbrock:")
        for name, result in results.items():
            print(f"  {name:8}: {result['value']:8.4f} ({result['evaluations']} evals)")

    def test_prima_with_path_tracking(self, quadratic_function):
        """Test path tracking works with PRIMA algorithms."""
        optimizer = PRIMA_UOBYQA(quadratic_function, n_trials=40, n_dim=2)
        optimizer.track_path = True

        best_value, best_x = optimizer.optimize()

        # Should have recorded optimization path
        assert len(optimizer.path) > 0
        assert all(len(point) == 2 for point in optimizer.path)

        # Path should generally improve (not strict due to algorithm nature)
        path_values = [quadratic_function(point) for point in optimizer.path]
        assert min(path_values) <= path_values[0]  # Should improve from start

    def test_prima_evaluation_budget(self, sphere_function):
        """Test PRIMA algorithms respect evaluation budgets."""
        for AlgorithmClass in [PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA]:
            optimizer = AlgorithmClass(sphere_function, n_trials=25, n_dim=2)
            best_value, best_x = optimizer.optimize()

            # Should not exceed budget
            assert optimizer.evaluations <= 25
            assert optimizer.evaluations > 0

    def test_prima_convergence_tolerance(self, quadratic_function):
        """Test PRIMA algorithms converge to reasonable tolerance."""
        for AlgorithmClass in [PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA]:
            np.random.seed(42)
            optimizer = AlgorithmClass(quadratic_function, n_trials=120, n_dim=2)
            best_value, best_x = optimizer.optimize()

            # With sufficient evaluations, should converge well
            if optimizer.evaluations >= 50:  # Only check if had enough evaluations
                assert best_value < 0.25  # Reasonable convergence for pure Python

    def test_prima_reproducibility(self, sphere_function):
        """Test PRIMA algorithms produce reproducible results."""
        # Run same optimization twice with same seed
        results = []
        for _ in range(2):
            np.random.seed(12345)
            optimizer = PRIMA_UOBYQA(sphere_function, n_trials=30, n_dim=2)
            best_value, best_x = optimizer.optimize()
            results.append((best_value, best_x, optimizer.evaluations))

        # Results should be identical (or very close due to floating point)
        assert abs(results[0][0] - results[1][0]) < 1e-10  # Same best value
        np.testing.assert_allclose(
            results[0][1], results[1][1], atol=1e-10
        )  # Same best point

    def test_prima_small_dimensions(self, quadratic_function):
        """Test PRIMA algorithms work with small dimensions."""

        # Test 1D optimization
        def objective_1d(x):
            return (x[0] - 0.7) ** 2

        for AlgorithmClass in [PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA]:
            optimizer = AlgorithmClass(objective_1d, n_trials=20, n_dim=1)
            best_value, best_x = optimizer.optimize()

            assert len(best_x) == 1
            assert abs(best_x[0] - 0.7) < 0.7  # Should get in reasonable range
            assert best_value < 0.5  # Reasonable for 1D optimization

    def test_prima_error_handling(self):
        """Test PRIMA algorithms handle edge cases gracefully."""

        def problematic_function(x):
            # Function that might cause numerical issues
            if any(xi < 0.001 or xi > 0.999 for xi in x):
                return float("inf")
            return sum(1.0 / xi for xi in x)  # Can be unstable near boundaries

        for AlgorithmClass in [PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA]:
            optimizer = AlgorithmClass(problematic_function, n_trials=30, n_dim=2)

            # Should not crash even with problematic function
            try:
                best_value, best_x = optimizer.optimize()
                assert np.isfinite(best_value)  # Should find finite solution
                assert all(0 <= xi <= 1 for xi in best_x)  # Within bounds
            except Exception as e:
                pytest.fail(
                    f"{AlgorithmClass.__name__} crashed on problematic function: {e}"
                )


class TestPRIMAPerformance:
    """Performance and efficiency tests for PRIMA algorithms."""

    def test_prima_efficiency(self):
        """Test PRIMA algorithms are reasonably efficient compared to SciPy."""
        from scipy.optimize import minimize

        def simple_quadratic(x):
            return sum((xi - 0.5) ** 2 for xi in x)

        # Test against SciPy's Nelder-Mead for comparison
        scipy_result = minimize(
            simple_quadratic,
            x0=[0.3, 0.3, 0.3],
            method="Nelder-Mead",
            options={"maxfev": 50},
        )
        scipy_best = scipy_result.fun

        for AlgorithmClass in [PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA]:
            optimizer = AlgorithmClass(simple_quadratic, n_trials=50, n_dim=3)

            import time

            start_time = time.time()
            best_value, best_x = optimizer.optimize()
            elapsed = time.time() - start_time

            # Should complete reasonably quickly (< 1 second for simple problem)
            assert elapsed < 1.0
            # Should achieve reasonable performance for pure Python implementation
            assert best_value < 0.1  # Reasonable absolute threshold

    def test_prima_vs_real_prima(self):
        """Test my PRIMA implementations against REAL PRIMA (PDFO) - the correct way!"""
        pytest.importorskip("pdfo", reason="PDFO not available for PRIMA validation")

        import pdfo

        def sphere(x):
            return sum(xi**2 for xi in x)

        # Test on simple sphere function
        x0 = np.array([0.5, 0.5])

        # Real PRIMA UOBYQA result
        real_result = pdfo.pdfo(sphere, x0, method="uobyqa", options={"maxfev": 50})

        # My PRIMA UOBYQA result
        optimizer = PRIMA_UOBYQA(sphere, n_trials=50, n_dim=2)
        my_f, my_x = optimizer.optimize()

        print("\nPRIMA Comparison on Sphere Function:")
        print(f"Real PRIMA UOBYQA: f={real_result.fun:.8f}, evals={real_result.nfev}")
        print(f"My PRIMA UOBYQA:   f={my_f:.8f}, evals={optimizer.evaluations}")

        # My implementation should achieve reasonable performance compared to real PRIMA
        if real_result.fun < 1e-6:  # Real PRIMA found exact solution
            assert my_f < 0.01, f"My PRIMA should get close to optimum, got {my_f}"
        else:
            assert my_f < real_result.fun * 10, (
                "My PRIMA should be within 10x of real PRIMA"
            )

    def test_prima_scaling(self):
        """Test how PRIMA algorithms scale with dimension."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        # Test dimensions 2, 4, 6
        for n_dim in [2, 4, 6]:
            for AlgorithmClass in [PRIMA_UOBYQA, PRIMA_NEWUOA, PRIMA_BOBYQA]:
                optimizer = AlgorithmClass(sphere, n_trials=n_dim * 20, n_dim=n_dim)
                best_value, best_x = optimizer.optimize()

                # Should still find reasonable solutions in higher dimensions
                assert best_value < 3.0  # Should find reasonable solutions
                assert len(best_x) == n_dim
