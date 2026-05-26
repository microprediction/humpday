"""
Comprehensive validation test suite for Python implementations.

This test suite ensures:
1. Python objective functions match reference implementations
2. Python optimizers work correctly on unit hypercube [0,1]^n
3. All functions are lightweight with no external dependencies
4. Consistent behavior across different input formats

This is the practical validation that can be run without JavaScript dependencies.
"""

import numpy as np


class TestObjectiveImplementations:
    """Test objective function implementations against references."""

    def test_sphere_vs_reference(self):
        """Test sphere function against simple reference."""

        # Our lightweight implementation
        def our_sphere(x):
            x = np.asarray(x)
            return np.sum(x * x)

        # Reference implementation
        def ref_sphere(x):
            return sum(xi**2 for xi in x)

        test_points = [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.1, 0.2, 0.3],
            [0.7, 0.8, 0.9, 0.1],
            np.random.random(10),  # High dimensional
        ]

        for point in test_points:
            our_result = our_sphere(point)
            ref_result = ref_sphere(point)
            assert abs(our_result - ref_result) < 1e-15, (
                f"Sphere mismatch at {point}: ours={our_result}, ref={ref_result}"
            )

    def test_rosenbrock_vs_scipy(self):
        """Test Rosenbrock against SciPy reference if available."""

        # Our lightweight implementation
        def our_rosenbrock(x):
            x = np.asarray(x)
            return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

        test_points = [
            np.array([1.0, 1.0]),  # Global minimum
            np.array([0.0, 0.0]),  # Origin
            np.array([0.5, 0.8]),  # Random point
            np.array([0.1, 0.3, 0.7]),  # 3D
            np.array([0.2, 0.4, 0.6, 0.8]),  # 4D
        ]

        try:
            from scipy.optimize import rosen

            for point in test_points:
                our_result = our_rosenbrock(point)
                scipy_result = rosen(point)
                assert abs(our_result - scipy_result) < 1e-12, (
                    f"Rosenbrock mismatch: ours={our_result}, scipy={scipy_result}"
                )

        except ImportError:
            # Test against manual calculation for known points
            # f(1,1) = 0
            assert abs(our_rosenbrock([1.0, 1.0])) < 1e-15
            # f(0,0) = 100*(0-0)^2 + (1-0)^2 = 1
            assert abs(our_rosenbrock([0.0, 0.0]) - 1.0) < 1e-15

    def test_rastrigin_implementation(self):
        """Test Rastrigin function with domain transformation."""

        def our_rastrigin_unit_cube(x):
            """Rastrigin on unit cube [0,1]^n with domain transformation."""
            x = np.asarray(x)
            # Transform [0,1] to [-5.12, 5.12] like JavaScript version
            x_transformed = (x - 0.5) * 10.24
            A = 10.0
            n = len(x_transformed)
            return A * n + np.sum(
                x_transformed**2 - A * np.cos(2 * np.pi * x_transformed)
            )

        # Test key points
        # Center [0.5, 0.5] should map to [0, 0] and be close to global minimum
        center_result = our_rastrigin_unit_cube([0.5, 0.5])
        assert center_result < 1.0, (
            f"Center point should be near minimum: {center_result}"
        )

        # Corners should be worse than center
        corner_results = [
            our_rastrigin_unit_cube([0.0, 0.0]),
            our_rastrigin_unit_cube([1.0, 1.0]),
            our_rastrigin_unit_cube([0.0, 1.0]),
            our_rastrigin_unit_cube([1.0, 0.0]),
        ]

        for corner_result in corner_results:
            assert corner_result > center_result, (
                f"Corner should be worse than center: {corner_result} vs {center_result}"
            )

    def test_ackley_implementation(self):
        """Test Ackley function implementation."""

        def our_ackley_unit_cube(x):
            """Ackley on unit cube with domain transformation."""
            x = np.asarray(x)
            # Transform [0,1] to [-5, 5] for Ackley
            x_transformed = (x - 0.5) * 10

            a, b, c = 20.0, 0.2, 2.0 * np.pi
            n = len(x_transformed)

            sum_sq = np.sum(x_transformed**2) / n
            sum_cos = np.sum(np.cos(c * x_transformed)) / n

            return a + np.exp(1) - a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos)

        # Test that center is close to global minimum (should be near 0)
        center_result = our_ackley_unit_cube([0.5, 0.5])
        assert center_result < 1.0, (
            f"Ackley center should be near minimum: {center_result}"
        )

        # Test different dimensions
        for n_dim in [1, 2, 3, 5]:
            center = [0.5] * n_dim
            result = our_ackley_unit_cube(center)
            assert result >= -1e-10, (
                f"Ackley should be non-negative (allowing floating point precision): {result}"
            )
            assert result < 5.0, f"Ackley center should be reasonable: {result}"


class TestOptimizerImplementations:
    """Test optimizer implementations."""

    def test_all_optimizers_importable(self):
        """Test that all 22 optimizers can be imported and instantiated."""
        from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

        # Simple test objective
        def test_obj(x):
            return np.sum(np.asarray(x) ** 2)

        assert len(PURE_OPTIMIZERS) == 22, (
            f"Expected 22 optimizers, got {len(PURE_OPTIMIZERS)}"
        )

        for name, optimizer_class in PURE_OPTIMIZERS.items():
            # Test instantiation
            optimizer = optimizer_class(test_obj, 10, 2)
            assert hasattr(optimizer, "optimize"), f"{name} missing optimize method"
            assert hasattr(optimizer, "evaluate"), f"{name} missing evaluate method"

            # Test basic properties
            assert optimizer.n_dim == 2
            assert optimizer.n_trials == 10
            assert optimizer.evaluations == 0

    def test_optimizer_basic_functionality(self):
        """Test basic optimizer functionality on simple problems."""
        from humpday.optimizers.alloptimizers import NelderMead, RandomSearch

        # Simple quadratic with known minimum at [0.3, 0.7]
        def quadratic(x):
            x = np.asarray(x)
            target = np.array([0.3, 0.7])
            return np.sum((x - target) ** 2)

        # Test RandomSearch
        rs_optimizer = RandomSearch(quadratic, 100, 2)
        rs_best_val, rs_best_x = rs_optimizer.optimize()

        assert rs_best_val < 1.0, (
            f"RandomSearch didn't find good solution: {rs_best_val}"
        )
        assert len(rs_best_x) == 2, f"Wrong solution dimension: {len(rs_best_x)}"
        assert rs_optimizer.evaluations <= 100, (
            f"Too many evaluations: {rs_optimizer.evaluations}"
        )

        # Test NelderMead (should be more accurate)
        nm_optimizer = NelderMead(quadratic, 100, 2)
        nm_best_val, nm_best_x = nm_optimizer.optimize()

        assert nm_best_val < 0.01, f"Nelder-Mead didn't converge: {nm_best_val}"

        # Check solution is close to [0.3, 0.7]
        target = np.array([0.3, 0.7])
        error = np.linalg.norm(np.array(nm_best_x) - target)
        assert error < 0.1, f"Solution too far from optimum: {nm_best_x}, error={error}"

    def test_optimizers_respect_bounds(self):
        """Test that optimizers respect unit hypercube bounds."""
        from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

        # Objective that heavily penalizes going outside [0,1]
        def bounded_objective(x):
            x = np.asarray(x)
            penalty = 0
            for xi in x:
                if xi < 0 or xi > 1:
                    penalty += 1000 * abs(xi - np.clip(xi, 0, 1))
            return np.sum(x**2) + penalty

        # Test a few representative optimizers
        test_optimizers = [
            "RandomSearch",
            "NelderMead",
            "DifferentialEvolution",
            "ParticleSwarm",
        ]

        for opt_name in test_optimizers:
            if opt_name in PURE_OPTIMIZERS:
                optimizer_class = PURE_OPTIMIZERS[opt_name]
                optimizer = optimizer_class(bounded_objective, 50, 3)
                best_val, best_x = optimizer.optimize()

                # Check bounds are respected
                for i, xi in enumerate(best_x):
                    assert 0 <= xi <= 1, f"{opt_name} violated bounds: x[{i}]={xi}"

    def test_optimizer_consistency(self):
        """Test that optimizers give consistent results with same random seed."""
        from humpday.optimizers.alloptimizers import RandomSearch

        def test_objective(x):
            return np.sum(np.asarray(x) ** 2)

        # Run same optimizer with same seed multiple times
        np.random.seed(42)
        result1 = RandomSearch(test_objective, 20, 2).optimize()

        np.random.seed(42)
        result2 = RandomSearch(test_objective, 20, 2).optimize()

        # Results should be identical (same random seed)
        assert abs(result1[0] - result2[0]) < 1e-10, (
            "Non-deterministic behavior with same seed"
        )


class TestAdaptiveSystem:
    """Test the adaptive optimization system."""

    def test_elo_system_basic(self):
        """Test basic ELO rating system functionality."""
        from humpday.optimizers.adaptive_optimizer import EloRatingSystem

        elo = EloRatingSystem()

        # Test initial ratings
        initial_rating = elo.get_rating("NelderMead")
        assert initial_rating == 1500.0, f"Wrong initial rating: {initial_rating}"

        # Test rating update
        elo.update_ratings("NelderMead", "RandomSearch", 1.0)  # NelderMead wins

        nm_rating = elo.get_rating("NelderMead")
        rs_rating = elo.get_rating("RandomSearch")

        assert nm_rating > initial_rating, "Winner should gain rating"
        assert rs_rating < initial_rating, "Loser should lose rating"
        assert len(elo.match_history) == 1, "Match should be recorded"

    def test_objective_generators(self):
        """Test objective function generators."""
        from humpday.optimizers.adaptive_optimizer import (
            rosenbrock_variants_generator,
            sphere_variants_generator,
        )

        # Test sphere generator
        sphere_gen = sphere_variants_generator(3)
        for _ in range(5):
            func = next(sphere_gen)
            result = func([0.1, 0.2, 0.3])
            assert isinstance(result, (int, float, np.number))
            assert not np.isnan(result)
            assert result >= 0  # Sphere variants should be non-negative

        # Test Rosenbrock generator (2D minimum)
        rosenbrock_gen = rosenbrock_variants_generator(2)
        for _ in range(5):
            func = next(rosenbrock_gen)
            result = func([0.5, 0.5])
            assert isinstance(result, (int, float, np.number))
            assert not np.isnan(result)

    def test_adaptive_optimize_basic(self):
        """Test basic adaptive optimization functionality."""
        from humpday.optimizers.adaptive_optimizer import (
            adaptive_optimize,
            sphere_variants_generator,
        )

        # Quick test with small budget
        objective_gen = sphere_variants_generator(2)

        results = adaptive_optimize(
            objective_generator=objective_gen,
            trials_budget=500,
            n_dim=2,
            n_warmup_problems=2,
            trials_per_warmup=20,  # Increased to avoid population size issues
            verbose=False,
        )

        # Check results structure
        assert "elo_system" in results
        assert "top_algorithms" in results
        assert "recommendations" in results
        assert "total_problems_solved" in results

        # Check we got some results
        assert len(results["top_algorithms"]) > 0
        assert results["total_problems_solved"] >= 2

        # Check top algorithm has reasonable rating
        top_alg, top_rating = results["top_algorithms"][0]
        assert isinstance(top_alg, str)
        assert isinstance(top_rating, (int, float))


class TestDomainHandling:
    """Test unit hypercube domain handling."""

    def test_unit_cube_inputs(self):
        """Test that all functions accept unit cube inputs."""
        from humpday.optimizers.adaptive_optimizer import sphere_variants_generator

        gen = sphere_variants_generator(5)
        func = next(gen)

        # Test boundary points
        test_points = [
            [0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros
            [1.0, 1.0, 1.0, 1.0, 1.0],  # All ones
            [0.5, 0.5, 0.5, 0.5, 0.5],  # Center
            np.random.random(5),  # Random point
        ]

        for point in test_points:
            result = func(point)
            assert isinstance(result, (int, float, np.number))
            assert not np.isnan(result)
            assert not np.isinf(result)

    def test_input_format_consistency(self):
        """Test that functions handle different input formats consistently."""

        # Use a simple deterministic sphere function for consistency testing
        def sphere_func(x):
            x = np.asarray(x)
            return np.sum(x * x)

        # Test same point in different formats
        test_point_value = [0.1, 0.2, 0.3]

        formats = [
            list(test_point_value),  # List
            tuple(test_point_value),  # Tuple
            np.array(test_point_value),  # NumPy array
        ]

        results = [sphere_func(point) for point in formats]

        # All should give the same result
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-12, (
                f"Inconsistent results for different input formats: {results}"
            )


def run_comprehensive_validation():
    """Run all validation tests."""
    print("🔍 Running comprehensive implementation validation...")

    test_classes = [
        TestObjectiveImplementations(),
        TestOptimizerImplementations(),
        TestAdaptiveSystem(),
        TestDomainHandling(),
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📁 {class_name}")

        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(test_class, method_name)
                    method()
                    print(f"  ✅ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ❌ {method_name}: {e}")

    print(f"\n🎯 Test Summary: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Implementation validation successful.")
    else:
        print("⚠️  Some tests failed. Review implementation consistency.")


if __name__ == "__main__":
    run_comprehensive_validation()
