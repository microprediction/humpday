#!/usr/bin/env python3
"""
Example demonstrating the indicative scale feature for unbounded optimization.

This shows how providing a scale hint can improve optimization performance
when you know roughly what magnitude the solution should have.
"""

import numpy as np
from humpday import minimize
import time


def test_scale_effectiveness():
    """Test how different scale values affect optimization performance."""

    print("=" * 60)
    print("DEMONSTRATING INDICATIVE SCALE FOR UNBOUNDED OPTIMIZATION")
    print("=" * 60)

    # Test function with known minimum at [100, -50]
    def objective(x):
        return (x[0] - 100)**2 + (x[1] + 50)**2

    true_minimum = np.array([100, -50])

    print(f"Objective: minimize (x[0] - 100)² + (x[1] + 50)²")
    print(f"True minimum: {true_minimum}")
    print(f"Expected function value: 0.0")
    print()

    # Test different scale values
    scales_to_test = [
        (None, "Default (scale=1.0)"),
        (1.0, "scale=1.0"),
        (10.0, "scale=10.0"),
        (100.0, "scale=100.0 (optimal)"),
        ([100.0, 50.0], "scale=[100.0, 50.0] (optimal per-dimension)")
    ]

    print(f"{'Scale':<25} {'Final Value':<12} {'Solution':<25} {'Error':<10} {'Time (s)'}")
    print("-" * 80)

    for scale, description in scales_to_test:
        start_time = time.time()

        try:
            # Run optimization with different scales
            result = minimize(objective, x0=[0, 0], scale=scale,
                            method='DifferentialEvolution',
                            options={'maxiter': 500})

            elapsed = time.time() - start_time
            error = np.linalg.norm(result.x - true_minimum)

            print(f"{description:<25} {result.fun:<12.2e} "
                  f"[{result.x[0]:6.1f}, {result.x[1]:6.1f}] {error:>8.2f} {elapsed:>8.2f}")

        except Exception as e:
            print(f"{description:<25} {'FAILED':<12} {str(e)[:40]}")


def test_high_magnitude_problem():
    """Test with a problem where the solution has very high magnitude."""

    print("\n" + "=" * 60)
    print("HIGH-MAGNITUDE PROBLEM TEST")
    print("=" * 60)

    # Problem with solution at very high magnitude
    def high_mag_objective(x):
        target = np.array([1000, -2000, 500])
        return np.sum((x - target)**2)

    true_minimum = np.array([1000, -2000, 500])

    print(f"Objective: 3D sphere with minimum at {true_minimum}")
    print()

    # Compare no scale vs appropriate scale
    test_cases = [
        (None, "No scale hint"),
        (1500.0, "scale=1500 (appropriate magnitude)"),
        ([1000.0, 2000.0, 500.0], "Per-dimension scales")
    ]

    for scale, description in test_cases:
        start_time = time.time()

        result = minimize(high_mag_objective, x0=[0, 0, 0], scale=scale,
                         method='ParticleSwarm', options={'maxiter': 800})

        elapsed = time.time() - start_time
        error = np.linalg.norm(result.x - true_minimum)

        print(f"{description:30}: error={error:8.2f}, "
              f"time={elapsed:5.2f}s, value={result.fun:.2e}")
        print(f"{'':30}  solution={result.x}")


def test_mixed_scale_problem():
    """Test with a problem where different dimensions have very different scales."""

    print("\n" + "=" * 60)
    print("MIXED-SCALE PROBLEM TEST")
    print("=" * 60)

    def mixed_scale_objective(x):
        # Different scales: x[0] around 0.1, x[1] around 1000
        return (x[0] - 0.1)**2 + ((x[1] - 1000) / 1000)**2

    true_minimum = np.array([0.1, 1000])

    print("Objective: (x[0] - 0.1)² + ((x[1] - 1000)/1000)²")
    print(f"True minimum: {true_minimum}")
    print()

    # Compare different scaling strategies
    test_cases = [
        (None, "No scale hint"),
        (100.0, "Uniform scale=100"),
        ([0.5, 1000.0], "Per-dimension: [0.5, 1000]")
    ]

    for scale, description in test_cases:
        result = minimize(mixed_scale_objective, x0=[1, 1], scale=scale,
                         method='CMAEvolutionStrategy', options={'maxiter': 400})

        error = np.linalg.norm(result.x - true_minimum)

        print(f"{description:25}: error={error:8.2f}, value={result.fun:.2e}")
        print(f"{'':25}  solution=[{result.x[0]:.4f}, {result.x[1]:.1f}]")


def main():
    """Run all scale demonstration examples."""
    print("🎯 HUMPDAY INDICATIVE SCALE DEMONSTRATION")
    print("Showing how scale hints improve unbounded optimization performance")

    try:
        test_scale_effectiveness()
        test_high_magnitude_problem()
        test_mixed_scale_problem()

        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print("✅ Scale hints significantly improve optimization when:")
        print("   • Solution magnitude is much different from 1.0")
        print("   • You know the approximate scale of the solution")
        print("   • Different dimensions have very different scales")
        print()
        print("💡 RECOMMENDATIONS:")
        print("   • Use scale=X where X matches expected solution magnitude")
        print("   • Use scale=[X1, X2, ...] for different scales per dimension")
        print("   • Default scale=1.0 works well for solutions around [-1, 1]")

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()