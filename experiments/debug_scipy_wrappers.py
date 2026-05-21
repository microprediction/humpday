#!/usr/bin/env python3
"""
Debug SciPy wrapper failures, especially Nelder-Mead's 0% success rate.
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def debug_scipy_methods():
    """Debug why SciPy methods are failing."""

    print("🔧 Debugging SciPy Wrapper Issues")
    print("=" * 40)

    def simple_sphere(x):
        """Simple sphere function for testing."""
        return np.sum((np.array(x) - 0.5)**2)

    x0 = np.array([0.1, 0.9])  # Start far from optimum
    print(f"Test function: Sphere with minimum at [0.5, 0.5]")
    print(f"Starting point: {x0}")
    print(f"Expected minimum value: 0.0")
    print()

    methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'SLSQP', 'COBYLA']

    for method in methods:
        print(f"🧪 Testing {method}:")
        print("-" * 30)

        # Test 1: No bounds (completely unconstrained)
        print("Test 1: No bounds")
        try:
            result = minimize(simple_sphere, x0, method=method, options={'maxfev': 50})
            print(f"  Success: {result.success}")
            print(f"  Message: {result.message}")
            print(f"  Final x: {result.x}")
            print(f"  Final f: {result.fun}")
            print(f"  Evaluations: {result.nfev}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()

        # Test 2: With bounds [0,1]^n
        print("Test 2: With bounds [(0,1), (0,1)]")
        try:
            result = minimize(simple_sphere, x0, method=method,
                            bounds=[(0, 1), (0, 1)], options={'maxfev': 50})
            print(f"  Success: {result.success}")
            print(f"  Message: {result.message}")
            print(f"  Final x: {result.x}")
            print(f"  Final f: {result.fun}")
            print(f"  Evaluations: {result.nfev}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()

        # Test 3: With tighter bounds to keep in [0,1]
        print("Test 3: With tight bounds [(0.001, 0.999), (0.001, 0.999)]")
        try:
            result = minimize(simple_sphere, x0, method=method,
                            bounds=[(0.001, 0.999), (0.001, 0.999)], options={'maxfev': 50})
            print(f"  Success: {result.success}")
            print(f"  Message: {result.message}")
            print(f"  Final x: {result.x}")
            print(f"  Final f: {result.fun}")
            print(f"  Evaluations: {result.nfev}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print("\n" + "="*50 + "\n")

def test_wrapper_approaches():
    """Test different wrapper approaches."""

    print("🔧 Testing Different Wrapper Approaches")
    print("=" * 45)

    def sphere(x):
        return np.sum((np.array(x) - 0.5)**2)

    # Approach 1: Penalty method
    print("🧪 Approach 1: Penalty Method")
    print("-" * 35)

    def penalty_wrapper(x):
        x = np.array(x)
        # Penalty for going outside [0,1]
        penalty = 0
        for xi in x:
            if xi < 0 or xi > 1:
                penalty += 1000 * (max(0, -xi) + max(0, xi - 1))**2
        return sphere(np.clip(x, 0, 1)) + penalty

    for method in ['Nelder-Mead', 'Powell']:
        print(f"{method}:")
        try:
            result = minimize(penalty_wrapper, [0.1, 0.9], method=method, options={'maxfev': 100})
            print(f"  Success: {result.success}, Final f: {result.fun:.6f}, Final x: {result.x}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print()

    # Approach 2: Transform to unbounded space
    print("🧪 Approach 2: Sigmoid Transform")
    print("-" * 35)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_wrapper(x_unbounded):
        x_bounded = sigmoid(np.array(x_unbounded))
        return sphere(x_bounded)

    # Initial point in unbounded space that maps to [0.1, 0.9] in bounded space
    x0_unbounded = np.log(np.array([0.1, 0.9]) / (1 - np.array([0.1, 0.9])))

    for method in ['Nelder-Mead', 'Powell']:
        print(f"{method}:")
        try:
            result = minimize(sigmoid_wrapper, x0_unbounded, method=method, options={'maxfev': 100})
            final_x_bounded = sigmoid(result.x)
            print(f"  Success: {result.success}, Final f: {result.fun:.6f}, Final x: {final_x_bounded}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n🎯 Diagnosis Summary:")
    print("=" * 25)
    print("1. Test which bounds approaches work for each method")
    print("2. Identify why Nelder-Mead shows 0% success rate")
    print("3. Fix wrappers to properly handle unit cube constraints")

if __name__ == "__main__":
    debug_scipy_methods()
    print()
    test_wrapper_approaches()