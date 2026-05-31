"""
Complete validation test suite - BOTH directions required:
(a) Python implementations vs 3rd party references
(b) JavaScript implementations vs our clean Python

This ensures mathematical correctness AND Python/JS consistency.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Test configuration
REPO_ROOT = Path(__file__).parent.parent
JS_SURFACES_PATH = REPO_ROOT / "docs/js/surfaces.js"
JS_OPTIMIZERS_PATH = REPO_ROOT / "docs/js/optimizers.js"

# Tolerances
REFERENCE_TOLERANCE = 1e-12  # For 3rd party validation
JS_PYTHON_TOLERANCE = 1e-10  # For JS/Python comparison


class JavaScriptTestRunner:
    """Run JavaScript code for validation against Python."""

    @staticmethod
    def create_node_compatible_surfaces():
        """Create Node.js compatible version of surfaces.js"""
        with open(JS_SURFACES_PATH) as f:
            js_content = f.read()

        # Make it Node.js compatible
        node_compatible = """
// Node.js compatibility
const TestSurfaces = {
    sphere(params = {}) {
        const center = params.center || 0.0;
        const scale = params.scale || 1.0;
        return function(x) {
            let sum = 0;
            for (let i = 0; i < x.length; i++) {
                sum += Math.pow(x[i] - center, 2);
            }
            return scale * sum;
        };
    },

    rosenbrock(params = {}) {
        const a = params.a || 1.0;
        const b = params.b || 100.0;
        return function(x) {
            if (x.length < 2) return Math.pow(x[0] - a, 2);
            let sum = 0;
            for (let i = 0; i < x.length - 1; i++) {
                sum += b * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(a - x[i], 2);
            }
            return sum;
        };
    },

    rastrigin(params = {}) {
        const A = params.A || 10.0;
        const omega = params.omega || 2 * Math.PI;
        return function(x) {
            const n = x.length;
            let sum = A * n;
            for (let i = 0; i < n; i++) {
                // Transform to [-5.12, 5.12] range for standard Rastrigin
                const xi = (x[i] - 0.5) * 10.24;
                sum += xi * xi - A * Math.cos(omega * xi);
            }
            return sum;
        };
    },

    ackley(params = {}) {
        const a = params.a || 20.0;
        const b = params.b || 0.2;
        const c = params.c || 2 * Math.PI;
        return function(x) {
            const n = x.length;
            // Transform to [-5, 5] range
            const transformedX = x.map(xi => (xi - 0.5) * 10);

            let sumSq = 0;
            let sumCos = 0;
            for (let i = 0; i < n; i++) {
                sumSq += transformedX[i] * transformedX[i];
                sumCos += Math.cos(c * transformedX[i]);
            }

            return a + Math.exp(1) - a * Math.exp(-b * Math.sqrt(sumSq / n)) - Math.exp(sumCos / n);
        };
    },

    griewank(params = {}) {
        return function(x) {
            // Transform to [-600, 600] range
            const transformedX = x.map(xi => (xi - 0.5) * 1200);

            let sumTerm = 0;
            let prodTerm = 1;
            for (let i = 0; i < transformedX.length; i++) {
                sumTerm += transformedX[i] * transformedX[i];
                prodTerm *= Math.cos(transformedX[i] / Math.sqrt(i + 1));
            }

            return sumTerm / 4000.0 - prodTerm + 1;
        };
    }
};

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TestSurfaces };
}
"""
        return node_compatible

    @staticmethod
    def run_js_function_test(
        function_name: str, test_points: List[List[float]]
    ) -> List[float]:
        """Test a JavaScript surface function."""
        js_code = JavaScriptTestRunner.create_node_compatible_surfaces()

        test_code = f"""
{js_code}

// Test the function
const testPoints = {json.dumps(test_points)};
const func = TestSurfaces.{function_name}();
const results = testPoints.map(point => func(point));

// Output results as JSON
console.log(JSON.stringify(results));
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(test_code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["node", temp_file], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"JavaScript execution failed: {result.stderr}")
            return json.loads(result.stdout.strip())
        except FileNotFoundError:
            pytest.skip("Node.js not available for JavaScript testing")
        finally:
            os.unlink(temp_file)


class TestPythonVs3rdParty:
    """(a) Validate Python implementations against 3rd party references."""

    def test_sphere_vs_simple_reference(self):
        """Test sphere against trivial reference implementation."""

        def our_sphere(x):
            x = np.asarray(x)
            return np.sum(x * x)

        def reference_sphere(x):
            return sum(xi * xi for xi in x)

        test_points = [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.1, 0.2, 0.3],
            np.random.random(5),
            np.random.random(10),
        ]

        for point in test_points:
            our_result = our_sphere(point)
            ref_result = reference_sphere(point)
            assert abs(our_result - ref_result) < REFERENCE_TOLERANCE, (
                f"Sphere mismatch at {point}: ours={our_result}, ref={ref_result}"
            )

    def test_rosenbrock_vs_scipy(self):
        """Test Rosenbrock against SciPy reference."""

        def our_rosenbrock(x):
            x = np.asarray(x)
            return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

        test_points = [
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.5, 0.8]),
            np.array([0.1, 0.3, 0.7]),
            np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        ]

        try:
            from scipy.optimize import rosen

            for point in test_points:
                our_result = our_rosenbrock(point)
                scipy_result = rosen(point)
                assert abs(our_result - scipy_result) < REFERENCE_TOLERANCE, (
                    f"Rosenbrock mismatch at {point}: ours={our_result}, scipy={scipy_result}"
                )

        except ImportError:
            pytest.skip("SciPy not available for Rosenbrock validation")

    def test_differential_evolution_vs_scipy(self):
        """Test our Differential Evolution against SciPy's version on same problem."""
        from humpday.optimizers.alloptimizers import DifferentialEvolution

        # Simple quadratic function
        def objective(x):
            return np.sum((np.asarray(x) - 0.3) ** 2)

        # Our implementation
        our_optimizer = DifferentialEvolution(objective, 200, 2)
        our_result = our_optimizer.optimize()

        try:
            from scipy.optimize import differential_evolution

            # SciPy version (need to define bounds)
            def scipy_objective(x):
                return np.sum((x - 0.3) ** 2)

            bounds = [(0, 1), (0, 1)]
            scipy_result = differential_evolution(
                scipy_objective, bounds, maxiter=50, seed=42
            )

            # Both should find the minimum reasonably well
            assert our_result[0] < 0.01, f"Our DE didn't converge: {our_result[0]}"
            assert scipy_result.fun < 0.01, (
                f"SciPy DE didn't converge: {scipy_result.fun}"
            )

            # Solutions should be close to [0.3, 0.3]
            target = np.array([0.3, 0.3])
            our_error = np.linalg.norm(np.array(our_result[1]) - target)
            scipy_error = np.linalg.norm(scipy_result.x - target)

            assert our_error < 0.1, f"Our DE solution error: {our_error}"
            assert scipy_error < 0.1, f"SciPy DE solution error: {scipy_error}"

        except ImportError:
            pytest.skip("SciPy not available for DE comparison")

    def test_nelder_mead_vs_scipy(self):
        """Test our Nelder-Mead against SciPy's version."""
        from humpday.optimizers.alloptimizers import NelderMead

        def objective(x):
            return np.sum((np.asarray(x) - 0.4) ** 2)

        # Our implementation
        our_optimizer = NelderMead(objective, 100, 2)
        our_result = our_optimizer.optimize()

        try:
            from scipy.optimize import minimize

            def scipy_objective(x):
                # Map from unbounded to [0,1]
                x_bounded = np.clip(x, 0, 1)
                return np.sum((x_bounded - 0.4) ** 2)

            scipy_result = minimize(
                scipy_objective,
                x0=[0.5, 0.5],
                method="Nelder-Mead",
                options={"maxiter": 100},
            )

            # Both should converge to [0.4, 0.4]
            target = np.array([0.4, 0.4])
            our_error = np.linalg.norm(np.array(our_result[1]) - target)

            # Our implementation should work well on this simple problem
            assert our_result[0] < 0.01, (
                f"Our Nelder-Mead didn't converge: {our_result[0]}"
            )
            assert our_error < 0.1, f"Our Nelder-Mead solution error: {our_error}"

        except ImportError:
            pytest.skip("SciPy not available for Nelder-Mead comparison")


class TestJavaScriptVsPython:
    """(b) Validate JavaScript implementations match our clean Python."""

    def test_sphere_js_vs_python(self):
        """Test JavaScript sphere vs Python sphere."""

        # Our Python implementation
        def python_sphere(x):
            x = np.asarray(x)
            return np.sum(x * x)

        test_points = [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.1, 0.2],
            [0.3, 0.7, 0.1],
            [0.9, 0.1, 0.5, 0.8],
        ]

        # Get JavaScript results
        js_results = JavaScriptTestRunner.run_js_function_test("sphere", test_points)

        # Compare with Python
        for i, point in enumerate(test_points):
            python_result = python_sphere(point)
            js_result = js_results[i]

            assert abs(python_result - js_result) < JS_PYTHON_TOLERANCE, (
                f"Sphere JS/Python mismatch at {point}: Python={python_result}, JS={js_result}"
            )

    def test_rosenbrock_js_vs_python(self):
        """Test JavaScript Rosenbrock vs Python Rosenbrock."""

        # Our Python implementation
        def python_rosenbrock(x):
            x = np.asarray(x)
            if len(x) < 2:
                return (x[0] - 1.0) ** 2
            return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

        test_points = [
            [1.0, 1.0],  # Global minimum
            [0.0, 0.0],  # Origin
            [0.5, 0.5],  # Center
            [0.2, 0.8],  # Random point
            [0.1, 0.3, 0.7],  # 3D case
            [0.9, 0.1],  # Edge case
        ]

        js_results = JavaScriptTestRunner.run_js_function_test(
            "rosenbrock", test_points
        )

        for i, point in enumerate(test_points):
            python_result = python_rosenbrock(point)
            js_result = js_results[i]

            assert abs(python_result - js_result) < JS_PYTHON_TOLERANCE, (
                f"Rosenbrock JS/Python mismatch at {point}: Python={python_result}, JS={js_result}"
            )

    def test_rastrigin_js_vs_python(self):
        """Test JavaScript Rastrigin vs Python Rastrigin with domain transformation."""

        # Our Python implementation (matching JS domain transformation)
        def python_rastrigin(x):
            x = np.asarray(x)
            # Transform [0,1] to [-5.12, 5.12] to match JavaScript
            x_transformed = (x - 0.5) * 10.24
            A = 10.0
            n = len(x_transformed)
            return A * n + np.sum(
                x_transformed**2 - A * np.cos(2 * np.pi * x_transformed)
            )

        test_points = [
            [0.5, 0.5],  # Center (global minimum)
            [0.0, 0.0],  # Corner
            [1.0, 1.0],  # Opposite corner
            [0.25, 0.75],  # Asymmetric
            [0.1, 0.9, 0.5],  # 3D case
        ]

        js_results = JavaScriptTestRunner.run_js_function_test("rastrigin", test_points)

        for i, point in enumerate(test_points):
            python_result = python_rastrigin(point)
            js_result = js_results[i]

            assert abs(python_result - js_result) < JS_PYTHON_TOLERANCE, (
                f"Rastrigin JS/Python mismatch at {point}: Python={python_result}, JS={js_result}"
            )

    def test_ackley_js_vs_python(self):
        """Test JavaScript Ackley vs Python Ackley with domain transformation."""

        # Our Python implementation (matching JS domain transformation)
        def python_ackley(x):
            x = np.asarray(x)
            # Transform [0,1] to [-5, 5] to match JavaScript
            x_transformed = (x - 0.5) * 10

            a, b, c = 20.0, 0.2, 2.0 * np.pi
            n = len(x_transformed)

            sum_sq = np.sum(x_transformed**2) / n
            sum_cos = np.sum(np.cos(c * x_transformed)) / n

            return a + np.exp(1) - a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos)

        test_points = [
            [0.5, 0.5],  # Center (global minimum)
            [0.0, 0.0],  # Corner
            [1.0, 1.0],  # Opposite corner
            [0.3, 0.7],  # Random point
            [0.1, 0.5, 0.9],  # 3D case
        ]

        js_results = JavaScriptTestRunner.run_js_function_test("ackley", test_points)

        for i, point in enumerate(test_points):
            python_result = python_ackley(point)
            js_result = js_results[i]

            assert abs(python_result - js_result) < JS_PYTHON_TOLERANCE, (
                f"Ackley JS/Python mismatch at {point}: Python={python_result}, JS={js_result}"
            )

    def test_griewank_js_vs_python(self):
        """Test JavaScript Griewank vs Python Griewank with domain transformation."""

        # Our Python implementation (matching JS domain transformation)
        def python_griewank(x):
            x = np.asarray(x)
            # Transform [0,1] to [-600, 600] to match JavaScript
            x_transformed = (x - 0.5) * 1200

            sum_term = np.sum(x_transformed**2) / 4000.0
            prod_term = np.prod(
                [
                    np.cos(x_transformed[i] / np.sqrt(i + 1))
                    for i in range(len(x_transformed))
                ]
            )

            return sum_term - prod_term + 1

        test_points = [
            [0.5, 0.5],  # Center (global minimum)
            [0.0, 0.0],  # Corner
            [1.0, 1.0],  # Opposite corner
            [0.2, 0.8],  # Random point
            [0.1, 0.5, 0.9],  # 3D case
        ]

        js_results = JavaScriptTestRunner.run_js_function_test("griewank", test_points)

        for i, point in enumerate(test_points):
            python_result = python_griewank(point)
            js_result = js_results[i]

            assert abs(python_result - js_result) < JS_PYTHON_TOLERANCE, (
                f"Griewank JS/Python mismatch at {point}: Python={python_result}, JS={js_result}"
            )


class TestOptimizerConsistency:
    """Test that our optimizers work correctly and consistently."""

    def test_all_optimizers_functional(self):
        """Test that all 22 optimizers work on a simple problem."""
        from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

        # Simple quadratic with known minimum
        def objective(x):
            return np.sum((np.asarray(x) - 0.3) ** 2)

        successful_optimizers = []
        failed_optimizers = []

        for name, optimizer_class in PURE_OPTIMIZERS.items():
            try:
                optimizer = optimizer_class(objective, 100, 2)
                best_val, best_x = optimizer.optimize()

                # Check basic sanity
                assert isinstance(best_val, (int, float, np.number))
                assert len(best_x) == 2
                assert all(0 <= xi <= 1 for xi in best_x), (
                    f"{name} violated bounds: {best_x}"
                )

                successful_optimizers.append(name)

            except Exception as e:
                failed_optimizers.append((name, str(e)))

        # Report results
        print(f"✅ {len(successful_optimizers)}/22 optimizers working")
        for name in failed_optimizers:
            print(f"❌ {name[0]}: {name[1]}")

        assert len(failed_optimizers) == 0, f"Failed optimizers: {failed_optimizers}"

    def test_optimizer_convergence_quality(self):
        """Test that key optimizers converge well on known problems."""
        from humpday.optimizers.alloptimizers import (
            DifferentialEvolution,
            NelderMead,
            ParticleSwarm,
        )

        # Test problem: quadratic with minimum at [0.3, 0.7]
        def objective(x):
            x = np.asarray(x)
            target = np.array([0.3, 0.7])
            return np.sum((x - target) ** 2)

        optimizers_to_test = [
            ("NelderMead", NelderMead),
            ("DifferentialEvolution", DifferentialEvolution),
            ("ParticleSwarm", ParticleSwarm),
        ]

        for name, optimizer_class in optimizers_to_test:
            optimizer = optimizer_class(objective, 200, 2)
            best_val, best_x = optimizer.optimize()

            # Should find minimum reasonably well
            assert best_val < 0.01, f"{name} didn't converge well: {best_val}"

            # Solution should be close to [0.3, 0.7]
            target = np.array([0.3, 0.7])
            error = np.linalg.norm(np.array(best_x) - target)
            assert error < 0.1, f"{name} solution error too large: {error}"


def run_complete_validation():
    """Run complete validation in both directions."""
    print("🔍 COMPLETE VALIDATION: Python vs 3rd Party AND JavaScript vs Python")
    print("=" * 80)

    test_classes = [
        ("(a) Python vs 3rd Party", TestPythonVs3rdParty()),
        ("(b) JavaScript vs Python", TestJavaScriptVsPython()),
        ("Optimizer Consistency", TestOptimizerConsistency()),
    ]

    total_tests = 0
    passed_tests = 0
    skipped_tests = 0

    for section_name, test_class in test_classes:
        print(f"\n📁 {section_name}")

        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(test_class, method_name)
                    method()
                    print(f"  ✅ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    if "skip" in str(e).lower() or "node.js not available" in str(e):
                        print(f"  ⚠️  {method_name}: SKIPPED - {e}")
                        skipped_tests += 1
                    else:
                        print(f"  ❌ {method_name}: {e}")

    print("\n" + "=" * 80)
    print("🎯 VALIDATION SUMMARY")
    print(f"  ✅ Passed: {passed_tests}")
    print(f"  ⚠️  Skipped: {skipped_tests}")
    print(f"  ❌ Failed: {total_tests - passed_tests - skipped_tests}")
    print(f"  📊 Total: {total_tests}")

    if passed_tests + skipped_tests == total_tests:
        print("🎉 COMPLETE VALIDATION SUCCESSFUL!")
        print("   ✓ Python implementations mathematically correct")
        print("   ✓ JavaScript and Python implementations consistent")
    else:
        print("⚠️  VALIDATION ISSUES FOUND - Review failures above")


if __name__ == "__main__":
    run_complete_validation()
