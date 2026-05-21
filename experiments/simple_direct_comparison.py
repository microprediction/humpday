#!/usr/bin/env python3
"""
Simple direct comparison - no wrappers, just raw performance.
"""

import numpy as np
import time
import sys
from scipy.optimize import minimize

# Import PRIMA directly
sys.path.append('/Users/petercotton/github/humpday/humpday/optimizers')
from primacube import prima_uobyqa_cube, prima_newuoa_cube

def simple_sphere(x):
    """Simple sphere function."""
    x = np.array(x)
    scaled_x = 4 * x - 2  # Scale [0,1] to [-2,2]
    return np.sum(scaled_x**2)

def direct_scipy_test(objective, n_trials, n_dim, method):
    """Direct SciPy test without complex wrappers."""

    best_val = float('inf')

    # Try multiple random starts
    for start in range(3):
        x0 = np.random.rand(n_dim)

        try:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=[(0.01, 0.99)] * n_dim,
                options={'maxfev': n_trials // 3}
            )

            if result.success and result.fun < best_val:
                best_val = result.fun

        except:
            continue

    return best_val

def quick_comparison():
    """Quick direct comparison."""

    print("🎯 Quick Direct Optimizer Comparison")
    print("=" * 42)

    n_trials = 50
    n_runs = 10

    dimensions = [2, 5]

    for dim in dimensions:
        print(f"\n📊 {dim}D Sphere Function:")
        print("-" * 25)

        results = {
            'PRIMA_UOBYQA': [],
            'PRIMA_NEWUOA': [],
            'SciPy_Powell': [],
            'SciPy_BFGS': []
        }

        for run in range(n_runs):
            np.random.seed(run * 42)

            # PRIMA UOBYQA
            try:
                val = prima_uobyqa_cube(simple_sphere, n_trials, dim, with_count=False)
                results['PRIMA_UOBYQA'].append(val)
            except:
                results['PRIMA_UOBYQA'].append(float('inf'))

            # PRIMA NEWUOA
            try:
                val = prima_newuoa_cube(simple_sphere, n_trials, dim, with_count=False)
                results['PRIMA_NEWUOA'].append(val)
            except:
                results['PRIMA_NEWUOA'].append(float('inf'))

            # SciPy Powell
            try:
                val = direct_scipy_test(simple_sphere, n_trials, dim, 'Powell')
                results['SciPy_Powell'].append(val)
            except:
                results['SciPy_Powell'].append(float('inf'))

            # SciPy BFGS
            try:
                val = direct_scipy_test(simple_sphere, n_trials, dim, 'L-BFGS-B')
                results['SciPy_BFGS'].append(val)
            except:
                results['SciPy_BFGS'].append(float('inf'))

        # Analyze results
        for method, values in results.items():
            finite_vals = [v for v in values if np.isfinite(v)]

            if finite_vals:
                success_rate = len(finite_vals) / n_runs * 100
                mean_val = np.mean(finite_vals)
                std_val = np.std(finite_vals)
                print(f"  {method:12}: {success_rate:3.0f}% success | {mean_val:8.4f} ± {std_val:6.4f}")
            else:
                print(f"  {method:12}: Failed all runs")

    # Simple ranking
    print(f"\n🏆 Key Takeaways:")
    print("=" * 20)

    # Test a single challenging case directly
    print("Direct head-to-head on 5D Rosenbrock:")

    def rosenbrock(x):
        x = np.array(x)
        scaled_x = 2 * x - 1  # Scale to [-1,1]
        result = 0
        for i in range(len(scaled_x)-1):
            result += 100*(scaled_x[i+1] - scaled_x[i]**2)**2 + (1 - scaled_x[i])**2
        return result

    # Single test case
    np.random.seed(12345)

    print(f"  PRIMA UOBYQA: ", end="")
    start = time.time()
    val_u, _, evals_u = prima_uobyqa_cube(rosenbrock, 100, 5, with_count=True)
    time_u = time.time() - start
    print(f"{val_u:.4f} in {evals_u} evals ({time_u:.3f}s)")

    print(f"  PRIMA NEWUOA: ", end="")
    start = time.time()
    val_n, _, evals_n = prima_newuoa_cube(rosenbrock, 100, 5, with_count=True)
    time_n = time.time() - start
    print(f"{val_n:.4f} in {evals_n} evals ({time_n:.3f}s)")

    print(f"  SciPy Powell: ", end="")
    start = time.time()
    x0 = np.random.rand(5)
    try:
        result = minimize(rosenbrock, x0, method='Powell',
                         bounds=[(0.01,0.99)]*5, options={'maxfev': 100})
        val_p = result.fun if result.success else float('inf')
        time_p = time.time() - start
        print(f"{val_p:.4f} ({'success' if result.success else 'failed'}) ({time_p:.3f}s)")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    quick_comparison()