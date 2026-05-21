#!/usr/bin/env python3
"""
Debug PRIMA evaluation counting issue.
"""

import numpy as np
import sys
sys.path.append('/Users/petercotton/github/humpday/humpday/optimizers')

from pdfo import uobyqa, newuoa

def debug_prima_evaluations():
    """Debug why PRIMA methods only use 1 evaluation."""

    print("🔍 Debugging PRIMA Evaluation Counting")
    print("=" * 45)

    def simple_sphere(x):
        print(f"  Function call: x = {x}, f(x) = {np.sum((x-0.5)**2)}")
        return np.sum((x - 0.5)**2)  # Minimum at [0.5, 0.5]

    x0 = np.array([0.1, 0.9])  # Start far from optimum
    n_trials = 20

    print(f"Starting point: {x0}")
    print(f"Target optimum: [0.5, 0.5]")
    print(f"Budget: {n_trials} evaluations")
    print()

    # Test 1: UOBYQA with current settings
    print("📊 Test 1: UOBYQA with current settings")
    print("-" * 40)

    eval_count = [0]

    def counting_sphere(x):
        eval_count[0] += 1
        val = simple_sphere(x)
        print(f"    Eval #{eval_count[0]}: f({x}) = {val}")
        return val

    try:
        result = uobyqa(
            counting_sphere,
            x0,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            options={
                'maxfev': n_trials,
                'ftol': 1e-12,
                'rhobeg': 0.1,
                'rhoend': 1e-8
            }
        )

        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        print(f"  Final x: {result.x}")
        print(f"  Final f: {result.fun}")
        print(f"  Function evals: {result.nfev}")
        print(f"  Our counter: {eval_count[0]}")

    except Exception as e:
        print(f"Error: {e}")

    print()

    # Test 2: Try with looser tolerances
    print("📊 Test 2: UOBYQA with looser tolerances")
    print("-" * 40)

    eval_count = [0]

    try:
        result = uobyqa(
            counting_sphere,
            x0,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            options={
                'maxfev': n_trials,
                'ftol': 1e-6,     # Looser tolerance
                'rhobeg': 0.5,    # Larger initial radius
                'rhoend': 1e-4    # Looser final radius
            }
        )

        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        print(f"  Final x: {result.x}")
        print(f"  Final f: {result.fun}")
        print(f"  Function evals: {result.nfev}")
        print(f"  Our counter: {eval_count[0]}")

    except Exception as e:
        print(f"Error: {e}")

    print()

    # Test 3: Try without bounds
    print("📊 Test 3: UOBYQA without bounds")
    print("-" * 35)

    eval_count = [0]

    try:
        result = uobyqa(
            counting_sphere,
            x0,
            options={
                'maxfev': n_trials,
                'ftol': 1e-6,
                'rhobeg': 0.1,
                'rhoend': 1e-4
            }
        )

        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        print(f"  Final x: {result.x}")
        print(f"  Final f: {result.fun}")
        print(f"  Function evals: {result.nfev}")
        print(f"  Our counter: {eval_count[0]}")

    except Exception as e:
        print(f"Error: {e}")

    print()

    # Test 4: NEWUOA comparison
    print("📊 Test 4: NEWUOA comparison")
    print("-" * 30)

    eval_count = [0]

    try:
        result = newuoa(
            counting_sphere,
            x0,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            options={
                'maxfev': n_trials,
                'ftol': 1e-6,
                'rhobeg': 0.1,
                'rhoend': 1e-4
            }
        )

        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        print(f"  Final x: {result.x}")
        print(f"  Final f: {result.fun}")
        print(f"  Function evals: {result.nfev}")
        print(f"  Our counter: {eval_count[0]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_prima_evaluations()