#!/usr/bin/env python3
"""
Test PRIMA constraint handling - ensure methods stay within [0,1]^n.
"""

import sys

import numpy as np

sys.path.append("/Users/petercotton/github/humpday/humpday/optimizers")
from primacube import prima_newuoa_cube, prima_uobyqa_cube


def test_constraint_handling():
    """Test that PRIMA methods respect unit cube constraints."""

    print("🔒 Testing PRIMA Unit Cube Constraint Handling")
    print("=" * 50)

    # Test function that has optimum outside [0,1]^n
    def shifted_sphere(x):
        # Optimum at [2.0, 2.0] - outside unit cube
        target = np.array([2.0, 2.0])
        return np.sum((np.array(x) - target) ** 2)

    print("Test function: Sphere with optimum at [2.0, 2.0] (outside unit cube)")
    print("Expected: Methods should find best solution within [0,1]^2")
    print()

    methods = {"PRIMA_UOBYQA": prima_uobyqa_cube, "PRIMA_NEWUOA": prima_newuoa_cube}

    for method_name, method_func in methods.items():
        print(f"Testing {method_name}:")

        # Run multiple times to check consistency
        for run in range(3):
            np.random.seed(run * 42)

            val, x, evals = method_func(shifted_sphere, 50, 2, with_count=True)

            # Check constraints
            in_bounds = np.all((x >= 0.0) & (x <= 1.0))
            closest_to_optimum = np.linalg.norm(
                x - np.array([1.0, 1.0])
            )  # [1,1] is closest feasible point

            print(
                f"  Run {run + 1}: x = [{x[0]:.3f}, {x[1]:.3f}], f = {val:.4f}, "
                f"evals = {evals}, in_bounds = {in_bounds}"
            )

            if not in_bounds:
                print("    ❌ CONSTRAINT VIOLATION!")

        print()

    # Test with function that has optimum at boundary
    def boundary_optimum(x):
        # Optimum at [0.0, 0.0] - on boundary
        return np.sum(np.array(x) ** 2)

    print("Boundary test: Function with optimum at [0.0, 0.0]")

    for method_name, method_func in methods.items():
        np.random.seed(123)
        val, x, evals = method_func(boundary_optimum, 30, 2, with_count=True)

        in_bounds = np.all((x >= 0.0) & (x <= 1.0))
        print(
            f"  {method_name}: x = [{x[0]:.4f}, {x[1]:.4f}], f = {val:.6f}, "
            f"in_bounds = {in_bounds}"
        )

    print("\n✅ Constraint handling validation complete!")


if __name__ == "__main__":
    test_constraint_handling()
