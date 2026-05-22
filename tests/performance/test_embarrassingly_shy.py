#!/usr/bin/env python3
"""
Test whether embarrassingly.Shy actually improves optimizer performance
on expensive/variable-time functions.
"""

import time
from typing import Callable, Dict

import numpy as np
from embarrassingly.shy import Shy
from humpday.optimizers.scipycube import (
    scipy_lbfgsb_cube,
    scipy_powell_cube,
    scipy_slsqp_cube,
)


def create_expensive_sphere(time_scale: float = 0.1) -> Callable:
    """Create sphere function with variable computation time."""

    def expensive_sphere(x):
        x = np.array(x)
        # Expensive computation in certain regions
        r = np.linalg.norm(x - 0.5)  # Distance from center
        compute_time = max(0.001, time_scale * (0.5 - r))  # More expensive near edges
        time.sleep(compute_time)

        # Standard sphere function
        scaled_x = 10 * x - 5
        return np.sum(scaled_x**2)

    return expensive_sphere


def create_expensive_rastrigin(time_scale: float = 0.1) -> Callable:
    """Create Rastrigin function with variable computation time."""

    def expensive_rastrigin(x):
        x = np.array(x)
        # Variable computation cost
        r = np.linalg.norm(x)
        compute_time = max(0.001, time_scale * r)  # More expensive far from origin
        time.sleep(compute_time)

        # Rastrigin function
        scaled_x = 10.24 * x - 5.12
        n = len(scaled_x)
        return 10 * n + np.sum(scaled_x**2 - 10 * np.cos(2 * np.pi * scaled_x))

    return expensive_rastrigin


def run_optimizer_test(
    optimizer_func, objective_func, n_trials: int = 25, n_dim: int = 2
) -> Dict:
    """Run optimizer on objective function and measure performance."""
    start_time = time.time()

    try:
        result = optimizer_func(
            objective_func, n_trials=n_trials, n_dim=n_dim, with_count=True
        )
        best_value, best_params, reported_trials = result
        elapsed = time.time() - start_time

        return {
            "best_value": float(best_value),
            "elapsed_time": elapsed,
            "reported_trials": reported_trials,
            "success": True,
            "evaluations_per_second": reported_trials / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        return {
            "best_value": float("inf"),
            "elapsed_time": time.time() - start_time,
            "reported_trials": 0,
            "success": False,
            "error": str(e),
            "evaluations_per_second": 0,
        }


def test_shy_vs_standard():
    """Test whether Shy wrapper improves performance on expensive functions."""

    print("🧪 Testing Embarrassingly Shy vs Standard Evaluation")
    print("=" * 60)

    # Test configuration
    optimizers = [
        ("Powell", scipy_powell_cube),
        ("SLSQP", scipy_slsqp_cube),
        ("L-BFGS-B", scipy_lbfgsb_cube),
    ]

    expensive_functions = [
        ("Expensive Sphere", create_expensive_sphere(0.05)),
        ("Expensive Rastrigin", create_expensive_rastrigin(0.05)),
    ]

    bounds = [[0, 1], [0, 1]]  # 2D unit square
    n_trials = 20  # Fewer trials due to expense
    n_dim = 2

    results = []

    for func_name, expensive_func in expensive_functions:
        print(f"\n📊 Testing {func_name}")
        print("-" * 40)

        # Create Shy wrapper
        shy_func = Shy(expensive_func, bounds=bounds, t_unit=0.01, d_unit=0.1)

        for opt_name, optimizer in optimizers:
            print(f"\n  Testing {opt_name}:")

            # Test standard function
            print("    Standard evaluation...", end=" ", flush=True)
            standard_result = run_optimizer_test(
                optimizer, expensive_func, n_trials, n_dim
            )
            print(
                f"✓ {standard_result['best_value']:.4f} in {standard_result['elapsed_time']:.2f}s"
            )

            # Test with Shy wrapper
            print("    Shy evaluation...     ", end=" ", flush=True)
            shy_result = run_optimizer_test(optimizer, shy_func, n_trials, n_dim)
            print(
                f"✓ {shy_result['best_value']:.4f} in {shy_result['elapsed_time']:.2f}s"
            )

            # Calculate improvements
            time_improvement = (
                standard_result["elapsed_time"] - shy_result["elapsed_time"]
            ) / standard_result["elapsed_time"]
            value_improvement = (
                (standard_result["best_value"] - shy_result["best_value"])
                / abs(standard_result["best_value"])
                if standard_result["best_value"] != 0
                else 0
            )

            results.append(
                {
                    "function": func_name,
                    "optimizer": opt_name,
                    "standard_time": standard_result["elapsed_time"],
                    "shy_time": shy_result["elapsed_time"],
                    "time_improvement": time_improvement,
                    "standard_value": standard_result["best_value"],
                    "shy_value": shy_result["best_value"],
                    "value_improvement": value_improvement,
                    "shy_evals_per_sec": shy_result["evaluations_per_second"],
                    "standard_evals_per_sec": standard_result["evaluations_per_second"],
                }
            )

            # Show improvements
            if time_improvement > 0.05:  # More than 5% faster
                print(f"      🚀 Shy is {time_improvement * 100:.1f}% faster!")
            elif time_improvement < -0.05:  # More than 5% slower
                print(f"      🐌 Shy is {abs(time_improvement) * 100:.1f}% slower")
            else:
                print("      ≈ Similar performance")

            if abs(value_improvement) > 0.1:  # More than 10% different
                if value_improvement > 0:
                    print(
                        f"      🎯 Shy finds {value_improvement * 100:.1f}% better solution!"
                    )
                else:
                    print(
                        f"      📉 Shy finds {abs(value_improvement) * 100:.1f}% worse solution"
                    )

    # Summary analysis
    print("\n📈 SUMMARY ANALYSIS")
    print("=" * 60)

    time_improvements = [r["time_improvement"] for r in results]
    value_improvements = [r["value_improvement"] for r in results]

    avg_time_improvement = np.mean(time_improvements)
    avg_value_improvement = np.mean(value_improvements)

    print(f"Average time improvement: {avg_time_improvement * 100:.1f}%")
    print(f"Average solution quality improvement: {avg_value_improvement * 100:.1f}%")

    # Count wins
    time_wins = sum(1 for t in time_improvements if t > 0.05)
    value_wins = sum(1 for v in value_improvements if v > 0.05)

    print(f"Shy was significantly faster in {time_wins}/{len(results)} cases")
    print(f"Shy found better solutions in {value_wins}/{len(results)} cases")

    # Verdict
    if avg_time_improvement > 0.1 or avg_value_improvement > 0.1:
        print("\n🏆 VERDICT: Shy wrapper shows measurable benefits!")
    elif avg_time_improvement > 0.05 or avg_value_improvement > 0.05:
        print("\n🤔 VERDICT: Shy wrapper shows modest benefits")
    else:
        print("\n❌ VERDICT: Shy wrapper doesn't provide clear benefits on these tests")

    return results


def test_plateau_finding():
    """Test whether Underpromoted helps find plateau regions."""

    print("\n\n🏔️ Testing Embarrassingly Underpromoted (Plateau Finding)")
    print("=" * 60)

    try:
        from embarrassingly.underpromoted import Underpromoted2d
    except ImportError:
        print("❌ Underpromoted not available, skipping plateau test")
        return []

    # Create a function with a plateau
    def plateau_function(x):
        x = np.array(x)
        # Create a "landing pad" - flat region around [0.3, 0.7]
        target = np.array([0.3, 0.7])
        dist = np.linalg.norm(x - target)

        if dist < 0.1:  # Plateau region
            return 1.0 + 0.1 * dist  # Slight slope toward center
        else:
            return 1.0 + dist**2  # Steeper outside

    bounds = [[0, 1], [0, 1]]
    optimizers = [("Powell", scipy_powell_cube), ("SLSQP", scipy_slsqp_cube)]

    plateau_results = []

    for opt_name, optimizer in optimizers:
        print(f"\n  Testing {opt_name} on plateau function:")

        # Standard function
        print("    Standard function... ", end="", flush=True)
        standard_result = run_optimizer_test(optimizer, plateau_function, 25, 2)
        print(f"✓ {standard_result['best_value']:.4f}")

        # Underpromoted version
        plateau_enhanced = Underpromoted2d(plateau_function, bounds=bounds, radius=0.05)
        print("    Plateau-enhanced...  ", end="", flush=True)
        plateau_result = run_optimizer_test(optimizer, plateau_enhanced, 25, 2)
        print(f"✓ {plateau_result['best_value']:.4f}")

        improvement = (
            standard_result["best_value"] - plateau_result["best_value"]
        ) / standard_result["best_value"]

        if improvement > 0.05:
            print(
                f"      🎯 Underpromoted found {improvement * 100:.1f}% better solution!"
            )
        elif improvement < -0.05:
            print(
                f"      📉 Underpromoted found {abs(improvement) * 100:.1f}% worse solution"
            )
        else:
            print("      ≈ Similar performance")

        plateau_results.append(
            {
                "optimizer": opt_name,
                "standard_value": standard_result["best_value"],
                "plateau_value": plateau_result["best_value"],
                "improvement": improvement,
            }
        )

    avg_plateau_improvement = np.mean([r["improvement"] for r in plateau_results])
    print(
        f"\nAverage plateau-finding improvement: {avg_plateau_improvement * 100:.1f}%"
    )

    return plateau_results


if __name__ == "__main__":
    print("🚀 Testing Embarrassingly Library Techniques with HumpDay")
    print("=" * 80)

    # Test 1: Shy wrapper for expensive functions
    shy_results = test_shy_vs_standard()

    # Test 2: Plateau finding
    plateau_results = test_plateau_finding()

    print("\n\n🏁 FINAL CONCLUSIONS")
    print("=" * 80)
    print(
        "✅ Tests completed - check results above to see if embarrassingly techniques work!"
    )
    print("📊 Integration recommendation depends on measurable benefits shown")
