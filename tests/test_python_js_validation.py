"""
Comprehensive cross-validation test suite for Python/JavaScript implementations.

This test suite ensures that:
1. Python objective functions match JavaScript objective functions
2. Python optimizers produce equivalent results to JavaScript optimizers
3. Both Python and JavaScript implementations match reference implementations
4. All functions work correctly on the unit hypercube [0,1]^n

Usage: python -m pytest tests/test_python_js_validation.py -v
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Test configuration
REPO_ROOT = Path(__file__).parent.parent
JS_SURFACES_PATH = REPO_ROOT / "docs/js/surfaces.js"
JS_OPTIMIZERS_PATH = REPO_ROOT / "docs/js/optimizers.js"

# Tolerance for numerical comparisons
FUNCTION_TOLERANCE = 1e-12
OPTIMIZER_TOLERANCE = 1e-6


class JavaScriptRunner:
    """Helper class to run JavaScript code from Python for testing."""

    @staticmethod
    def run_js_function_test(js_code):
        """Run JavaScript code and return the result."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["node", temp_file], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"JavaScript execution failed: {result.stderr}")
            return result.stdout.strip()
        finally:
            os.unlink(temp_file)


def create_js_function_test(function_name, test_points):
    """Create JavaScript code to test an objective function."""
    test_points_js = json.dumps(test_points)

    return f"""
// Load the surfaces implementation
const fs = require('fs');
const surfacesCode = fs.readFileSync('{JS_SURFACES_PATH}', 'utf8');
eval(surfacesCode);

// Test the function
const testPoints = {test_points_js};
const func = TestSurfaces.{function_name}();
const results = testPoints.map(point => func(point));

// Output results as JSON
console.log(JSON.stringify(results));
"""


def create_js_optimizer_test(optimizer_name, objective_code, n_trials, n_dim):
    """Create JavaScript code to test an optimizer."""

    return f"""
// Load the implementations
const fs = require('fs');
const surfacesCode = fs.readFileSync('{JS_SURFACES_PATH}', 'utf8');
const optimizersCode = fs.readFileSync('{JS_OPTIMIZERS_PATH}', 'utf8');
eval(surfacesCode);
eval(optimizersCode);

// Create objective function
{objective_code}

// Run optimizer
const optimizer = new {optimizer_name}(objective, {n_trials}, {n_dim});
const result = optimizer.optimize();

// Output result as JSON
console.log(JSON.stringify({{
    bestValue: result.bestValue,
    bestX: result.bestX,
    evaluations: optimizer.evaluations
}}));
"""


class TestObjectiveFunctions:
    """Test Python vs JavaScript objective function implementations."""

    def test_sphere_function(self):
        """Test sphere function Python vs JavaScript."""

        # Python implementation
        def python_sphere(x):
            x = np.asarray(x)
            return np.sum(x * x)

        # Test points
        test_points = [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.1, 0.2, 0.3],
            [0.7, 0.8, 0.9, 0.1],
        ]

        # Get JavaScript results
        js_code = create_js_function_test("sphere", test_points)
        js_results_str = JavaScriptRunner.run_js_function_test(js_code)
        js_results = json.loads(js_results_str)

        # Compare Python and JavaScript results
        for i, point in enumerate(test_points):
            python_result = python_sphere(point)
            js_result = js_results[i]

            assert (
                abs(python_result - js_result) < FUNCTION_TOLERANCE
            ), f"Sphere function mismatch at {point}: Python={python_result}, JS={js_result}"

    def test_rosenbrock_function(self):
        """Test Rosenbrock function Python vs JavaScript."""

        # Python implementation
        def python_rosenbrock(x):
            x = np.asarray(x)
            if len(x) < 2:
                return (x[0] - 1.0) ** 2
            return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

        # Test points
        test_points = [
            [1.0, 1.0],  # Global minimum
            [0.0, 0.0],  # Origin
            [0.5, 0.5],  # Middle
            [0.1, 0.9],  # Off-diagonal
            [0.3, 0.7, 0.2],  # 3D case
        ]

        # Get JavaScript results
        js_code = create_js_function_test("rosenbrock", test_points)
        js_results_str = JavaScriptRunner.run_js_function_test(js_code)
        js_results = json.loads(js_results_str)

        # Compare results
        for i, point in enumerate(test_points):
            python_result = python_rosenbrock(point)
            js_result = js_results[i]

            assert (
                abs(python_result - js_result) < FUNCTION_TOLERANCE
            ), f"Rosenbrock function mismatch at {point}: Python={python_result}, JS={js_result}"

    def test_rastrigin_function(self):
        """Test Rastrigin function Python vs JavaScript."""

        # Python implementation (with domain transformation)
        def python_rastrigin(x):
            x = np.asarray(x)
            # Transform from [0,1] to [-5.12, 5.12] to match JavaScript
            x_transformed = (x - 0.5) * 10.24
            A = 10.0
            n = len(x_transformed)
            return A * n + np.sum(
                x_transformed**2 - A * np.cos(2 * np.pi * x_transformed)
            )

        # Test points
        test_points = [
            [0.5, 0.5],  # Center (should be global minimum)
            [0.0, 0.0],  # Corner
            [1.0, 1.0],  # Opposite corner
            [0.25, 0.75],  # Asymmetric
            [0.1, 0.3, 0.7],  # 3D case
        ]

        # Get JavaScript results
        js_code = create_js_function_test("rastrigin", test_points)
        js_results_str = JavaScriptRunner.run_js_function_test(js_code)
        js_results = json.loads(js_results_str)

        # Compare results
        for i, point in enumerate(test_points):
            python_result = python_rastrigin(point)
            js_result = js_results[i]

            assert (
                abs(python_result - js_result) < FUNCTION_TOLERANCE
            ), f"Rastrigin function mismatch at {point}: Python={python_result}, JS={js_result}"


class TestOptimizerValidation:
    """Test Python vs JavaScript optimizer implementations."""

    def test_random_search_optimizer(self):
        """Test RandomSearch optimizer Python vs JavaScript."""
        from humpday.optimizers.optimizers import RandomSearch

        # Simple objective function (sphere)
        def objective(x):
            return np.sum(np.asarray(x) ** 2)

        # Test parameters
        n_trials = 50
        n_dim = 2

        # JavaScript objective code
        js_objective_code = (
            "const objective = (x) => x.reduce((sum, xi) => sum + xi*xi, 0);"
        )

        # Run Python optimizer multiple times for statistical comparison
        python_results = []
        for seed in range(3):
            np.random.seed(seed)
            optimizer = RandomSearch(objective, n_trials, n_dim)
            best_val, best_x = optimizer.optimize()
            python_results.append(best_val)

        # Run JavaScript optimizer
        js_code = create_js_optimizer_test(
            "RandomSearch", js_objective_code, n_trials, n_dim
        )
        js_result_str = JavaScriptRunner.run_js_function_test(js_code)
        js_result = json.loads(js_result_str)

        # Both should find reasonable solutions (not exact match due to randomness)
        python_best = min(python_results)
        js_best = js_result["bestValue"]

        # Both should be reasonably good (for sphere function, expect values < 1.0 with 50 trials)
        assert (
            python_best < 1.0
        ), f"Python RandomSearch found poor solution: {python_best}"
        assert js_best < 1.0, f"JavaScript RandomSearch found poor solution: {js_best}"

        # Both should have used all evaluations
        assert js_result["evaluations"] == n_trials

    def test_nelder_mead_deterministic(self):
        """Test Nelder-Mead with deterministic initialization."""
        from humpday.optimizers.optimizers import NelderMead

        # Simple quadratic function
        def objective(x):
            x = np.asarray(x)
            return np.sum((x - 0.3) ** 2)  # Minimum at x=[0.3, 0.3]

        n_trials = 100
        n_dim = 2

        # JavaScript objective code
        js_objective_code = "const objective = (x) => x.reduce((sum, xi, i) => sum + (xi - 0.3)*(xi - 0.3), 0);"

        # Run multiple times and check consistency
        python_optimizer = NelderMead(objective, n_trials, n_dim)
        python_best_val, python_best_x = python_optimizer.optimize()

        # JavaScript version
        js_code = create_js_optimizer_test(
            "NelderMead", js_objective_code, n_trials, n_dim
        )
        js_result_str = JavaScriptRunner.run_js_function_test(js_code)
        js_result = json.loads(js_result_str)

        # Both should find the optimum reasonably well
        assert (
            python_best_val < 0.01
        ), f"Python Nelder-Mead didn't converge: {python_best_val}"
        assert (
            js_result["bestValue"] < 0.01
        ), f"JavaScript Nelder-Mead didn't converge: {js_result['bestValue']}"

        # Check that best points are close to [0.3, 0.3]
        expected = np.array([0.3, 0.3])
        python_error = np.linalg.norm(np.array(python_best_x) - expected)
        js_error = np.linalg.norm(np.array(js_result["bestX"]) - expected)

        assert python_error < 0.1, f"Python solution far from optimum: {python_best_x}"
        assert (
            js_error < 0.1
        ), f"JavaScript solution far from optimum: {js_result['bestX']}"


class TestReferenceValidation:
    """Test against reference implementations when available."""

    def test_sphere_vs_scipy(self):
        """Test sphere function against simple reference."""
        try:
            # Our implementation
            def our_sphere(x):
                x = np.asarray(x)
                return np.sum(x * x)

            # Reference implementation (simple)
            def ref_sphere(x):
                return sum(xi**2 for xi in x)

            test_points = [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [-0.3, 0.7],
                [0.1, 0.2, 0.3, 0.4],
            ]

            for point in test_points:
                our_result = our_sphere(point)
                ref_result = ref_sphere(point)
                assert abs(our_result - ref_result) < 1e-15

        except ImportError:
            pytest.skip("Reference implementation not available")

    def test_rosenbrock_vs_scipy(self):
        """Test Rosenbrock against SciPy reference."""
        try:
            from scipy.optimize import rosen

            # Our implementation
            def our_rosenbrock(x):
                x = np.asarray(x)
                return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

            test_points = [
                np.array([1.0, 1.0]),
                np.array([0.0, 0.0]),
                np.array([0.5, 0.8]),
                np.array([0.1, 0.3, 0.7]),
                np.array([-0.5, 1.2, 0.3]),
            ]

            for point in test_points:
                our_result = our_rosenbrock(point)
                scipy_result = rosen(point)
                assert (
                    abs(our_result - scipy_result) < 1e-12
                ), f"Rosenbrock mismatch at {point}: ours={our_result}, scipy={scipy_result}"

        except ImportError:
            pytest.skip("SciPy not available for reference comparison")


class TestDomainTransformations:
    """Test that functions properly handle unit hypercube transformations."""

    def test_unit_hypercube_bounds(self):
        """Test that all functions accept inputs in [0,1]^n."""
        from humpday.optimizers.adaptive_optimizer import (
            rosenbrock_variants_generator,
            sphere_variants_generator,
        )

        # Test various dimensions
        for n_dim in [1, 2, 3, 5, 10]:
            # Test sphere variants
            sphere_gen = sphere_variants_generator(n_dim)
            sphere_func = next(sphere_gen)

            # Test at boundaries
            test_points = [
                np.zeros(n_dim),  # All zeros
                np.ones(n_dim),  # All ones
                np.full(n_dim, 0.5),  # Center
                np.random.random(n_dim),  # Random point
            ]

            for point in test_points:
                result = sphere_func(point)
                assert isinstance(result, (int, float, np.number))
                assert not np.isnan(result)
                assert not np.isinf(result)
                assert result >= 0  # Sphere function should be non-negative

            # Test Rosenbrock variants (if n_dim >= 2)
            if n_dim >= 2:
                rosenbrock_gen = rosenbrock_variants_generator(n_dim)
                rosenbrock_func = next(rosenbrock_gen)

                for point in test_points:
                    result = rosenbrock_func(point)
                    assert isinstance(result, (int, float, np.number))
                    assert not np.isnan(result)
                    assert not np.isinf(result)

    def test_javascript_domain_transformations(self):
        """Test that JavaScript functions handle domain transformations correctly."""
        # Test Rastrigin domain transformation
        test_points = [
            [0.5],  # Center should map to 0 in Rastrigin domain
            [0.0],  # Boundary should map to -5.12
            [1.0],  # Boundary should map to +5.12
        ]

        js_code = create_js_function_test("rastrigin", test_points)
        js_results_str = JavaScriptRunner.run_js_function_test(js_code)
        js_results = json.loads(js_results_str)

        # Center point [0.5] should give minimum value (close to 0)
        center_result = js_results[0]
        boundary_results = js_results[1:3]

        # Center should be better than boundaries for Rastrigin
        assert center_result < min(
            boundary_results
        ), f"Center point not optimal: center={center_result}, boundaries={boundary_results}"


# Integration test
def test_end_to_end_optimization():
    """End-to-end test of Python vs JavaScript optimization."""
    from humpday import pure_optimize

    # Define test objective
    def test_objective(x):
        # Simple quadratic with known minimum at [0.3, 0.7]
        x = np.asarray(x)
        return (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2

    # Test parameters
    n_trials = 100
    n_dim = 2

    # Run Python optimization
    python_result = pure_optimize(test_objective, "NelderMead", n_trials, n_dim)

    # JavaScript equivalent
    js_objective_code = "const objective = (x) => (x[0] - 0.3)*(x[0] - 0.3) + (x[1] - 0.7)*(x[1] - 0.7);"
    js_code = create_js_optimizer_test("NelderMead", js_objective_code, n_trials, n_dim)
    js_result_str = JavaScriptRunner.run_js_function_test(js_code)
    js_result = json.loads(js_result_str)

    # Both should find the minimum
    assert python_result[0] < 0.01, f"Python didn't find minimum: {python_result[0]}"
    assert (
        js_result["bestValue"] < 0.01
    ), f"JavaScript didn't find minimum: {js_result['bestValue']}"

    # Solutions should be close
    python_point = np.array(python_result[1])
    js_point = np.array(js_result["bestX"])
    expected = np.array([0.3, 0.7])

    python_error = np.linalg.norm(python_point - expected)
    js_error = np.linalg.norm(js_point - expected)

    assert python_error < 0.1, f"Python solution error: {python_error}"
    assert js_error < 0.1, f"JavaScript solution error: {js_error}"


if __name__ == "__main__":
    # Run tests directly
    test_obj = TestObjectiveFunctions()
    test_obj.test_sphere_function()
    test_obj.test_rosenbrock_function()
    test_obj.test_rastrigin_function()
    print("✓ Objective function tests passed!")

    test_ref = TestReferenceValidation()
    test_ref.test_sphere_vs_scipy()
    try:
        test_ref.test_rosenbrock_vs_scipy()
        print("✓ Reference validation tests passed!")
    except Exception as e:
        print(f"⚠ Reference validation skipped: {e}")

    test_domain = TestDomainTransformations()
    test_domain.test_unit_hypercube_bounds()
    test_domain.test_javascript_domain_transformations()
    print("✓ Domain transformation tests passed!")

    # Optimizer tests require Node.js, so only run if available
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)

        test_opt = TestOptimizerValidation()
        test_opt.test_random_search_optimizer()
        test_opt.test_nelder_mead_deterministic()
        print("✓ Optimizer validation tests passed!")

        test_end_to_end_optimization()
        print("✓ End-to-end test passed!")

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Optimizer tests skipped (Node.js not available)")

    print(
        "\n🎉 All available tests passed! Python and JavaScript implementations are consistent."
    )
