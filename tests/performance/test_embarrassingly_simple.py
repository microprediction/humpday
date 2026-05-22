#!/usr/bin/env python3
"""
Simple test of embarrassingly library techniques without full HumpDay imports.
"""

import time

import numpy as np
from embarrassingly.shy import Shy
from scipy.optimize import minimize


def create_expensive_sphere(time_scale: float = 0.05) -> callable:
    """Create sphere function with variable computation time."""

    def expensive_sphere(x):
        x = np.array(x)
        # More expensive computation near boundaries
        boundary_dist = min(np.min(x), np.min(1 - x))  # Distance to nearest boundary
        compute_time = max(
            0.001, time_scale * (0.1 - boundary_dist) if boundary_dist < 0.1 else 0.001
        )
        time.sleep(compute_time)

        # Standard sphere function: minimize sum of squares
        scaled_x = 10 * x - 5  # Scale [0,1] to [-5,5]
        return np.sum(scaled_x**2)

    return expensive_sphere


def run_scipy_optimizer(objective_func, method="Powell", max_evals=30):
    """Run scipy optimizer and measure performance."""
    eval_count = [0]  # Mutable counter

    def wrapped_objective(x):
        eval_count[0] += 1
        return objective_func(x)

    start_time = time.time()

    try:
        # Random starting point in [0,1]^2
        x0 = np.random.rand(2)

        result = minimize(
            wrapped_objective,
            x0,
            method=method,
            bounds=[(0, 1), (0, 1)],
            options={"maxfev": max_evals},
        )

        elapsed = time.time() - start_time

        return {
            "best_value": float(result.fun),
            "elapsed_time": elapsed,
            "evaluations": eval_count[0],
            "success": result.success,
            "evals_per_sec": eval_count[0] / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        return {
            "best_value": float("inf"),
            "elapsed_time": time.time() - start_time,
            "evaluations": eval_count[0],
            "success": False,
            "error": str(e),
        }


def test_shy_effectiveness():
    """Test whether Shy wrapper actually improves optimization on expensive functions."""

    print("🧪 Testing Embarrassingly Shy Wrapper")
    print("=" * 50)

    # Create expensive objective function
    expensive_func = create_expensive_sphere(time_scale=0.02)
    bounds = [[0, 1], [0, 1]]

    methods = ["Powell", "Nelder-Mead", "SLSQP"]
    max_evals = 25

    results = []

    for method in methods:
        print(f"\n📊 Testing {method} optimizer:")

        # Test standard function (multiple runs for average)
        standard_results = []
        shy_results = []

        for run in range(3):  # 3 runs per method
            np.random.seed(42 + run)  # Reproducible but different seeds

            # Standard evaluation
            standard_result = run_scipy_optimizer(expensive_func, method, max_evals)
            standard_results.append(standard_result)

            # Shy evaluation
            shy_func = Shy(expensive_func, bounds=bounds, t_unit=0.01, d_unit=0.1)
            shy_result = run_scipy_optimizer(shy_func, method, max_evals)
            shy_results.append(shy_result)

        # Calculate averages
        avg_standard_time = np.mean([r["elapsed_time"] for r in standard_results])
        avg_standard_value = np.mean([r["best_value"] for r in standard_results])
        avg_standard_evals = np.mean([r["evaluations"] for r in standard_results])

        avg_shy_time = np.mean([r["elapsed_time"] for r in shy_results])
        avg_shy_value = np.mean([r["best_value"] for r in shy_results])
        avg_shy_evals = np.mean([r["evaluations"] for r in shy_results])

        time_improvement = (avg_standard_time - avg_shy_time) / avg_standard_time * 100
        value_improvement = (
            (avg_standard_value - avg_shy_value) / avg_standard_value * 100
            if avg_standard_value != 0
            else 0
        )

        print(
            f"  Standard: {avg_standard_value:.4f} in {avg_standard_time:.2f}s ({avg_standard_evals:.0f} evals)"
        )
        print(
            f"  Shy:      {avg_shy_value:.4f} in {avg_shy_time:.2f}s ({avg_shy_evals:.0f} evals)"
        )

        if abs(time_improvement) > 5:
            if time_improvement > 0:
                print(f"  🚀 Shy is {time_improvement:.1f}% faster!")
            else:
                print(f"  🐌 Shy is {abs(time_improvement):.1f}% slower")
        else:
            print(f"  ≈ Similar time performance ({time_improvement:+.1f}%)")

        if abs(value_improvement) > 5:
            if value_improvement > 0:
                print(f"  🎯 Shy finds {value_improvement:.1f}% better solutions!")
            else:
                print(f"  📉 Shy finds {abs(value_improvement):.1f}% worse solutions")
        else:
            print(f"  ≈ Similar solution quality ({value_improvement:+.1f}%)")

        results.append(
            {
                "method": method,
                "time_improvement": time_improvement,
                "value_improvement": value_improvement,
                "standard_time": avg_standard_time,
                "shy_time": avg_shy_time,
                "standard_value": avg_standard_value,
                "shy_value": avg_shy_value,
            }
        )

    # Overall analysis
    print("\n📈 OVERALL ANALYSIS")
    print("-" * 30)

    avg_time_improvement = np.mean([r["time_improvement"] for r in results])
    avg_value_improvement = np.mean([r["value_improvement"] for r in results])

    print(f"Average time improvement: {avg_time_improvement:+.1f}%")
    print(f"Average solution quality improvement: {avg_value_improvement:+.1f}%")

    # Count significant improvements
    time_wins = sum(1 for r in results if r["time_improvement"] > 5)
    value_wins = sum(1 for r in results if r["value_improvement"] > 5)

    print(f"Methods with >5% time improvement: {time_wins}/{len(results)}")
    print(f"Methods with >5% solution improvement: {value_wins}/{len(results)}")

    # Verdict
    print("\n🏆 VERDICT:")
    if avg_time_improvement > 10 or avg_value_improvement > 10:
        print("Shy wrapper shows STRONG benefits! 🚀")
    elif avg_time_improvement > 5 or avg_value_improvement > 5:
        print("Shy wrapper shows MODERATE benefits 👍")
    elif avg_time_improvement > 0 and avg_value_improvement > 0:
        print("Shy wrapper shows SLIGHT benefits 🤔")
    else:
        print("Shy wrapper shows NO CLEAR benefits ❌")

    return results


def test_parallel_benefit():
    """Test whether Parallel wrapper helps with evaluation efficiency."""

    print("\n\n🔄 Testing Embarrassingly Parallel Wrapper")
    print("=" * 50)

    try:
        from embarrassingly.parallel import Parallel

        # Create a simple objective that could benefit from parallelization
        def simple_objective(x):
            # Simulate some computation
            time.sleep(0.01)  # 10ms per evaluation
            scaled_x = 10 * np.array(x) - 5
            return np.sum(scaled_x**2)

        print("Creating parallel wrapper...")
        parallel_obj = Parallel(lambda worker, x: simple_objective(x), num_workers=2)

        # Test with a simple optimization
        start_time = time.time()
        for i in range(10):
            x = np.random.rand(2)
            result = parallel_obj(x)
        sequential_time = time.time() - start_time

        print(f"Sequential-style calls through Parallel: {sequential_time:.2f}s")
        print(
            "✅ Parallel wrapper works (but doesn't show benefits without true parallel optimization)"
        )

    except ImportError:
        print("❌ Parallel not available")
    except Exception as e:
        print(f"❌ Parallel test failed: {e}")


if __name__ == "__main__":
    print("🚀 Testing Embarrassingly Library Techniques (Simple Version)")
    print("=" * 70)

    # Test Shy wrapper
    shy_results = test_shy_effectiveness()

    # Test Parallel wrapper (basic functionality)
    test_parallel_benefit()

    print("\n🏁 TESTING COMPLETE!")
    print("Results show whether embarrassingly techniques provide real benefits.")
