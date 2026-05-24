#!/usr/bin/env python3
"""
Demo of the Comprehensive Cross-Validation Framework

This script demonstrates the key capabilities of the cross-validation framework:
1. Mathematical equivalence testing
2. Statistical validation
3. Benchmark problem evaluation
4. Cross-implementation consistency

This demo runs quickly with reduced parameters to show the framework in action.

Usage: python demo_validation_framework.py
"""

import sys
from pathlib import Path

import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from benchmark_suite import BenchmarkSuite
    from cross_validation_framework import CrossValidationFramework, StandardBenchmarks
    from humpday.optimizers.prima_algorithms import PRIMA_UOBYQA
    from humpday.optimizers.scipy_algorithms import NelderMead, Powell
    from statistical_validation import StatisticalValidator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all modules are available")
    sys.exit(1)


def demo_basic_validation():
    """Demonstrate basic validation functionality."""
    print("🎯 DEMO: Basic Algorithm Validation")
    print("=" * 40)

    # Create simple benchmark problem
    def sphere_2d(x):
        x = np.asarray(x)
        scaled = (x - 0.5) * 10  # Transform [0,1] to [-5,5]
        return np.sum(scaled**2)

    print("Testing algorithms on 2D Sphere function...")

    # Test algorithms
    algorithms = {
        "NelderMead": NelderMead,
        "Powell": Powell,
        "PRIMA_UOBYQA": PRIMA_UOBYQA,
    }

    results = {}

    for alg_name, alg_class in algorithms.items():
        values = []

        # Run multiple times for statistics
        for run in range(3):
            np.random.seed(run * 42)
            optimizer = alg_class(sphere_2d, 50, 2)  # 50 trials, 2D
            best_val, best_x = optimizer.optimize()
            values.append(best_val)

        mean_val = np.mean(values)
        std_val = np.std(values)
        results[alg_name] = (mean_val, std_val)

        print(f"  {alg_name}: {mean_val:.6f} ± {std_val:.6f}")

    # Compare algorithms
    best_algorithm = min(results.items(), key=lambda x: x[1][0])
    print(f"\n🏆 Best performer: {best_algorithm[0]} ({best_algorithm[1][0]:.6f})")

    return results


def demo_statistical_validation():
    """Demonstrate statistical validation capabilities."""
    print("\n📊 DEMO: Statistical Validation")
    print("=" * 35)

    validator = StatisticalValidator(significance_level=0.05)

    # Simulate two algorithms with similar but slightly different performance
    np.random.seed(42)
    algorithm_a_results = np.random.exponential(1.0, 20).tolist()
    algorithm_b_results = np.random.exponential(1.1, 20).tolist()

    print("Comparing two algorithm performance distributions...")
    print(f"Algorithm A mean: {np.mean(algorithm_a_results):.4f}")
    print(f"Algorithm B mean: {np.mean(algorithm_b_results):.4f}")

    # Statistical comparison
    comparison = validator.compare_performance_distributions(
        algorithm_a_results, algorithm_b_results, "Algorithm_A", "Algorithm_B"
    )

    print("\nStatistical Analysis:")
    print(f"  Mean difference: {comparison.mean_difference:.4f}")
    print(f"  Variance ratio: {comparison.variance_ratio:.4f}")
    print(f"  KS test p-value: {comparison.ks_pvalue:.4f}")
    print(f"  Mann-Whitney p-value: {comparison.mannwhitney_pvalue:.4f}")
    print(f"  Algorithms equivalent: {'✅ Yes' if comparison.equivalent else '❌ No'}")

    return comparison


def demo_benchmark_suite():
    """Demonstrate benchmark suite capabilities."""
    print("\n🎯 DEMO: Benchmark Suite")
    print("=" * 30)

    # Create benchmark suite
    suite = BenchmarkSuite()

    # Add some problems
    from benchmark_suite import RastriginProblem, RosenbrockProblem, SphereProblem

    suite.add_problem(SphereProblem(2))
    suite.add_problem(RosenbrockProblem(2))
    suite.add_problem(RastriginProblem(2))

    print(f"Created benchmark suite with {len(suite.problems)} problems:")
    for problem_name in suite.problems:
        problem = suite.problems[problem_name]
        print(
            f"  • {problem_name} - {problem.metadata.difficulty} ({problem.metadata.problem_class})"
        )

    # Test an algorithm on benchmarks
    def simple_algorithm(objective, n_trials, n_dim):
        """Simple random search for demonstration."""
        best_value = float("inf")
        best_x = None

        for _ in range(n_trials):
            x = np.random.random(n_dim)
            value = objective(x)
            if value < best_value:
                best_value = value
                best_x = x

        return best_value, best_x

    print("\nTesting simple random search algorithm:")

    results = suite.evaluate_algorithm(
        simple_algorithm,
        problem_names=list(suite.problems.keys()),
        n_runs=3,
        n_trials=30,
    )

    for problem_name, result in results.items():
        if "statistics" in result:
            stats = result["statistics"]
            print(f"  {problem_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return results


def demo_convergence_analysis():
    """Demonstrate convergence analysis."""
    print("\n📈 DEMO: Convergence Analysis")
    print("=" * 35)

    validator = StatisticalValidator()

    # Simulate convergence curves for two algorithms
    # Algorithm A: fast initial convergence, then slow
    conv_a = []
    val = 10.0
    for i in range(20):
        val *= 0.8 + 0.1 * np.random.random()  # Exponential decay with noise
        conv_a.append(val)

    # Algorithm B: steady convergence
    conv_b = []
    val = 10.0
    for i in range(20):
        val *= 0.85 + 0.05 * np.random.random()  # Different decay rate
        conv_b.append(val)

    print(f"Algorithm A: {conv_a[0]:.3f} → {conv_a[-1]:.3f}")
    print(f"Algorithm B: {conv_b[0]:.3f} → {conv_b[-1]:.3f}")

    # Analyze convergence behavior
    analysis = validator.analyze_convergence_behavior(
        conv_a, conv_b, "Fast_Start", "Steady"
    )

    print("\nConvergence Analysis:")
    print(f"  Rate A: {analysis.convergence_rate_a:.4f}")
    print(f"  Rate B: {analysis.convergence_rate_b:.4f}")
    print(f"  Rate similarity: {analysis.rate_similarity:.4f}")
    print(f"  Path correlation: {analysis.path_correlation:.4f}")
    print(
        f"  Equivalent behavior: {'✅ Yes' if analysis.passed_equivalence else '❌ No'}"
    )

    return analysis


def demo_comprehensive_framework():
    """Demonstrate the complete framework (quick version)."""
    print("\n🚀 DEMO: Comprehensive Framework (Quick)")
    print("=" * 45)

    # Initialize framework
    framework = CrossValidationFramework("demo_results")

    print("Running quick mathematical correctness validation...")

    # Run mathematical correctness tests only (fastest)
    try:
        math_results = framework.run_mathematical_correctness_validation()

        # Count passed/failed tests
        total_tests = len(framework.results)
        passed_tests = sum(1 for r in framework.results if r.passed)

        print("\nQuick Validation Results:")
        print(f"  Tests run: {total_tests}")
        print(f"  Tests passed: {passed_tests}")
        print(
            f"  Pass rate: {passed_tests / total_tests * 100:.1f}%"
            if total_tests > 0
            else "  Pass rate: N/A"
        )

        # Show some specific results
        if framework.results:
            print("\nSample Results:")
            for result in framework.results[:3]:  # Show first 3 results
                status = "✅" if result.passed else "❌"
                print(f"  {status} {result.algorithm_name} - {result.test_name}")

    except Exception as e:
        print(f"⚠️ Framework validation had issues: {e}")

    return framework.results


def main():
    """Run all demos."""
    print("🔬 Cross-Validation Framework Demo")
    print("=" * 40)
    print("This demo shows key capabilities of the validation framework")
    print("with reduced parameters for quick execution.\n")

    # Run demos
    try:
        demo_basic_validation()
        demo_statistical_validation()
        demo_benchmark_suite()
        demo_convergence_analysis()
        demo_comprehensive_framework()

        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("\nNext Steps:")
        print("• Run full validation: python run_comprehensive_validation.py")
        print("• Quick validation: python run_comprehensive_validation.py --quick")
        print("• View results in: ./validation_results/ or ./demo_results/")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
