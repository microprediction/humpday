"""
Test the SciPy-style interface with rectangular bounds.

This test suite verifies that our thin wrappers correctly transform problems
from arbitrary rectangular domains to the unit hypercube and back.
"""

import numpy as np
import pytest


class TestSciPyInterface:
    """Test the SciPy-style interface functions."""

    def test_basic_minimize(self):
        """Test basic minimize functionality."""
        from humpday import cube_minimize

        # Simple quadratic with minimum at x=2, y=3
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2

        # Optimize with bounds
        bounds = [(0, 5), (0, 6)]
        result = cube_minimize(objective, bounds=bounds, method='NelderMead')

        # Check result structure
        assert hasattr(result, 'x')
        assert hasattr(result, 'fun')
        assert hasattr(result, 'success')
        assert hasattr(result, 'nfev')

        # Check solution quality
        assert len(result.x) == 2
        assert result.success is True
        assert result.fun < 0.01, f"Didn't find minimum: {result.fun}"

        # Check solution is close to [2, 3]
        error = np.linalg.norm(result.x - np.array([2, 3]))
        assert error < 0.1, f"Solution error too large: {error}, solution: {result.x}"

        # Check bounds are respected
        assert 0 <= result.x[0] <= 5, f"x[0] violates bounds: {result.x[0]}"
        assert 0 <= result.x[1] <= 6, f"x[1] violates bounds: {result.x[1]}"

    def test_cube_minimize_scalar(self):
        """Test scalar minimize functionality."""
        from humpday import cube_minimize_scalar

        # Simple 1D quadratic with minimum at x=3
        def objective(x):
            return (x - 3)**2

        # Optimize with bounds
        result = cube_minimize_scalar(objective, bounds=(0, 5), method='NelderMead')

        # Check result
        assert isinstance(result.x, (int, float, np.number))
        assert result.fun < 0.01, f"Didn't find minimum: {result.fun}"

        # Check solution is close to 3
        assert abs(result.x - 3) < 0.1, f"Solution error: {result.x}"
        assert 0 <= result.x <= 5, f"Solution violates bounds: {result.x}"

    def test_different_algorithms(self):
        """Test that different algorithms work with bounds."""
        from humpday import (cube_nelder_mead, cube_differential_evolution,
                           cube_particle_swarm, cube_cma_es)

        # Simple test function
        def objective(x):
            return np.sum((x - 1.5)**2)

        bounds = [(0, 3), (0, 3)]
        algorithms = [
            cube_nelder_mead,
            cube_differential_evolution,
            cube_particle_swarm,
            cube_cma_es
        ]

        for algorithm in algorithms:
            result = algorithm(objective, bounds=bounds)

            # All should find reasonable solutions
            assert result.fun < 1.0, f"{algorithm.__name__} poor result: {result.fun}"
            assert len(result.x) == 2

            # Check bounds
            for i, xi in enumerate(result.x):
                assert 0 <= xi <= 3, f"{algorithm.__name__} violated bounds: {result.x}"

    def test_bounds_parsing(self):
        """Test different bounds specification formats."""
        from humpday.optimizers.scipy_interface import parse_bounds

        # Test single bound pair for all dimensions
        lower, upper = parse_bounds((-2, 5), 3)
        np.testing.assert_array_equal(lower, [-2, -2, -2])
        np.testing.assert_array_equal(upper, [5, 5, 5])

        # Test individual bounds per dimension
        bounds = [(0, 1), (-1, 2), (3, 10)]
        lower, upper = parse_bounds(bounds, 3)
        np.testing.assert_array_equal(lower, [0, -1, 3])
        np.testing.assert_array_equal(upper, [1, 2, 10])

        # Test None (default unit cube)
        lower, upper = parse_bounds(None, 2)
        np.testing.assert_array_equal(lower, [0, 0])
        np.testing.assert_array_equal(upper, [1, 1])

        # Test invalid bounds
        with pytest.raises(ValueError):
            parse_bounds([(1, 0)], 1)  # lower > upper

    def test_domain_transformations(self):
        """Test domain transformation utilities."""
        from humpday import transform_to_unit_cube, transform_from_unit_cube

        # Test transformation to unit cube
        bounds = [(0, 10), (-5, 5)]
        x_real = np.array([5, 0])  # Middle points
        x_unit = transform_to_unit_cube(x_real, bounds)
        np.testing.assert_array_almost_equal(x_unit, [0.5, 0.5])

        # Test transformation back
        x_recovered = transform_from_unit_cube(x_unit, bounds)
        np.testing.assert_array_almost_equal(x_recovered, x_real)

        # Test corner cases
        x_corner = np.array([0, -5])  # Lower bounds
        x_unit_corner = transform_to_unit_cube(x_corner, bounds)
        np.testing.assert_array_almost_equal(x_unit_corner, [0, 0])

        x_corner2 = np.array([10, 5])  # Upper bounds
        x_unit_corner2 = transform_to_unit_cube(x_corner2, bounds)
        np.testing.assert_array_almost_equal(x_unit_corner2, [1, 1])

    def test_vs_unit_cube_version(self):
        """Test that bounded version gives same result as unit cube version."""
        from humpday import pure_optimize, cube_minimize

        # Define function on [-2, 3] x [-1, 4] with minimum at (1, 2)
        def objective_bounded(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2

        # Same function mapped to unit cube
        def objective_unit(x):
            # Map from [0,1]^2 to [-2,3] x [-1,4]
            x_real = np.array([-2, -1]) + x * np.array([5, 5])
            return (x_real[0] - 1)**2 + (x_real[1] - 2)**2

        # Optimize both versions
        bounds = [(-2, 3), (-1, 4)]
        result_bounded = cube_minimize(objective_bounded, bounds=bounds, method='NelderMead')

        result_unit = pure_optimize(objective_unit, 'NelderMead', 1000, 2)

        # Transform unit cube result to bounded domain
        x_unit_solution = np.array(result_unit[1])
        x_bounded_from_unit = np.array([-2, -1]) + x_unit_solution * np.array([5, 5])

        # Both should find similar solutions
        assert abs(result_bounded.fun - result_unit[0]) < 0.1, \
            f"Function values differ: bounded={result_bounded.fun}, unit={result_unit[0]}"

        # Solutions should be close (allowing for randomness)
        solution_diff = np.linalg.norm(result_bounded.x - x_bounded_from_unit)
        assert solution_diff < 1.0, f"Solutions differ significantly: {solution_diff}"

    def test_rosenbrock_with_bounds(self):
        """Test on Rosenbrock function with non-unit bounds."""
        from humpday import cube_minimize

        def rosenbrock(x):
            """Rosenbrock function with minimum at (1, 1)."""
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

        # Optimize on [-2, 2]^2
        bounds = [(-2, 2), (-2, 2)]
        result = cube_minimize(rosenbrock, bounds=bounds, method='NelderMead')

        # Should find the minimum at (1, 1)
        assert result.fun < 0.1, f"Didn't solve Rosenbrock: {result.fun}"

        expected_solution = np.array([1, 1])
        error = np.linalg.norm(result.x - expected_solution)
        assert error < 0.2, f"Rosenbrock solution error: {error}, solution: {result.x}"

    def test_high_dimensional(self):
        """Test on higher dimensional problem."""
        from humpday import cube_minimize

        n_dim = 10
        target = np.random.random(n_dim) * 4 - 2  # Random target in [-2, 2]

        def objective(x):
            return np.sum((x - target)**2)

        bounds = [(-3, 3)] * n_dim
        result = cube_minimize(objective, bounds=bounds, method='ParticleSwarm')

        # Should find the target
        assert result.fun < 1.0, f"High-dim optimization failed: {result.fun}"

        # Check bounds
        for xi in result.x:
            assert -3 <= xi <= 3, f"High-dim solution violates bounds: {result.x}"

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        from humpday import cube_minimize

        def objective(x):
            return x[0]**2

        # Test unknown method
        with pytest.raises(ValueError, match="Unknown method"):
            cube_minimize(objective, bounds=[(0, 1)], method='UnknownMethod')

        # Test dimension mismatch - objective that requires specific dimension
        def objective_3d_strict(x):
            return x[0]**2 + x[1]**2 + x[2]**2  # Requires exactly 3 dimensions

        with pytest.raises(IndexError):
            cube_minimize(objective_3d_strict, bounds=[(0, 1)])  # Only 1D bounds for 3D function

        # Test missing dimension info
        with pytest.raises(ValueError):
            cube_minimize(objective)  # No bounds or x0


def run_scipy_interface_tests():
    """Run all SciPy interface tests."""
    print("🔧 Testing SciPy-style interface with rectangular bounds...")

    test = TestSciPyInterface()
    test_methods = [m for m in dir(test) if m.startswith('test_')]

    passed = 0
    total = len(test_methods)

    for method_name in test_methods:
        try:
            method = getattr(test, method_name)
            method()
            print(f"  ✅ {method_name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {method_name}: {e}")

    print(f"\n🎯 SciPy Interface Tests: {passed}/{total} passed")

    if passed == total:
        print("🎉 All SciPy interface tests passed!")
        print("📏 Rectangular bounds working correctly!")
    else:
        print("⚠️  Some tests failed - check implementations")


if __name__ == "__main__":
    run_scipy_interface_tests()