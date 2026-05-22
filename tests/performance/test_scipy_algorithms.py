#!/usr/bin/env python3

"""
Simple test to validate SciPy algorithm implementations
"""

import sys

import numpy as np

# Add the current directory to path to import local modules
sys.path.append(".")


def test_scipy_algorithms():
    """Test SciPy algorithms directly"""

    # Test functions
    def sphere_2d(x):
        return x[0] ** 2 + x[1] ** 2

    def rosenbrock_2d(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    print("🧪 Testing SciPy Algorithms Direct Validation")
    print("=" * 60)

    # Test SciPy Nelder-Mead
    try:
        from scipy.optimize import minimize

        print("\n✅ SciPy available - testing Nelder-Mead:")

        # Test on 2D sphere
        x0 = np.array([0.5, 0.5])
        result = minimize(
            sphere_2d, x0, method="Nelder-Mead", options={"maxfev": 100, "disp": False}
        )

        print(f"  2D Sphere: f = {result.fun:.6f}, x = {result.x}")
        print(f"  Success: {result.success}, Evaluations: {result.nfev}")

        # Test on Rosenbrock
        result2 = minimize(
            rosenbrock_2d,
            x0,
            method="Nelder-Mead",
            options={"maxfev": 200, "disp": False},
        )

        print(f"  Rosenbrock: f = {result2.fun:.6f}, x = {result2.x}")
        print(f"  Success: {result2.success}, Evaluations: {result2.nfev}")

    except ImportError as e:
        print(f"❌ SciPy not available: {e}")
    except Exception as e:
        print(f"❌ SciPy test failed: {e}")

    # Test Differential Evolution
    try:
        from scipy.optimize import differential_evolution

        print("\n✅ Testing Differential Evolution:")

        # Test on 2D sphere
        bounds = [(0, 1), (0, 1)]
        result = differential_evolution(sphere_2d, bounds, maxiter=50, seed=42)

        print(f"  2D Sphere: f = {result.fun:.6f}, x = {result.x}")
        print(f"  Success: {result.success}, Evaluations: {result.nfev}")

    except Exception as e:
        print(f"❌ Differential Evolution test failed: {e}")

    print("\n" + "=" * 60)
    print("Direct SciPy algorithm validation complete!")


if __name__ == "__main__":
    test_scipy_algorithms()
