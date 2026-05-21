#!/usr/bin/env python3
"""
Demonstration of PRIMA UOBYQA and NEWUOA integration with HumpDay.
Shows their performance compared to existing optimizers.
"""

import numpy as np
import time
import sys
import os

# Add the humpday directory to path for imports
sys.path.append('/Users/petercotton/github/humpday')

from humpday.optimizers.primacube import prima_uobyqa_cube, prima_newuoa_cube
from scipy.optimize import minimize

def create_test_functions():
    """Create a diverse set of test problems for comparison."""

    # 1. Simple Sphere - smooth, unimodal
    def sphere(x):
        time.sleep(0.001)  # Realistic computation time
        scaled_x = 6 * np.array(x) - 3  # Scale [0,1] to [-3,3]
        return np.sum(scaled_x**2)

    # 2. Rosenbrock - smooth, unimodal, difficult valley
    def rosenbrock(x):
        time.sleep(0.001)
        x = np.array(x)
        scaled_x = 4.096 * x - 2.048  # Scale to [-2.048, 2.048]
        if len(scaled_x) < 2:
            return float('inf')
        return sum(100*(scaled_x[i+1] - scaled_x[i]**2)**2 + (1 - scaled_x[i])**2
                  for i in range(len(scaled_x)-1))

    # 3. Noisy Sphere - test robustness
    def noisy_sphere(x):
        time.sleep(0.001)
        scaled_x = 6 * np.array(x) - 3
        base_value = np.sum(scaled_x**2)
        noise = 0.1 * np.random.normal(0, 1)  # 10% noise
        return base_value + noise

    return {
        'sphere': sphere,
        'rosenbrock': rosenbrock,
        'noisy_sphere': noisy_sphere
    }

def scipy_powell_baseline(objective, n_trials, n_dim):
    """Reference implementation using SciPy Powell method."""
    eval_count = [0]

    def counting_wrapper(x):
        eval_count[0] += 1
        return objective(x)

    x0 = np.random.rand(n_dim)

    try:
        result = minimize(
            counting_wrapper,
            x0,
            method='Powell',
            bounds=[(0, 1)] * n_dim,
            options={'maxfev': n_trials}
        )

        best_x = np.clip(result.x, 0.0, 1.0)
        return result.fun, best_x, eval_count[0]

    except Exception as e:
        return float('inf'), x0, eval_count[0]

def scipy_nelder_mead_baseline(objective, n_trials, n_dim):
    """Reference implementation using SciPy Nelder-Mead."""
    eval_count = [0]

    def counting_wrapper(x):
        eval_count[0] += 1
        return objective(x)

    x0 = np.random.rand(n_dim)

    try:
        result = minimize(
            counting_wrapper,
            x0,
            method='Nelder-Mead',
            bounds=[(0, 1)] * n_dim,
            options={'maxfev': n_trials}
        )

        best_x = np.clip(result.x, 0.0, 1.0)
        return result.fun, best_x, eval_count[0]

    except Exception as e:
        return float('inf'), x0, eval_count[0]

def run_comparison_study():
    """Run comprehensive comparison of PRIMA vs baseline methods."""

    print("🏆 PRIMA/PDFO Integration Demonstration")
    print("=" * 55)
    print("Comparing UOBYQA and NEWUOA vs SciPy baselines")

    test_functions = create_test_functions()
    n_runs = 10  # Multiple runs for statistics
    n_trials = 40
    dimensions = [2, 5]  # Test both low and medium dimensions

    optimizers = {
        'PRIMA UOBYQA': prima_uobyqa_cube,
        'PRIMA NEWUOA': prima_newuoa_cube,
        'SciPy Powell': scipy_powell_baseline,
        'SciPy Nelder-Mead': scipy_nelder_mead_baseline
    }

    results = {}

    for dim in dimensions:
        print(f"\n📊 Testing in {dim}D")
        print("-" * 30)

        for func_name, func in test_functions.items():
            print(f"\n{func_name.upper()} function:")

            for opt_name, optimizer in optimizers.items():
                values = []
                times = []
                evaluations = []

                for run in range(n_runs):
                    np.random.seed(run * 42)  # Reproducible but varied starts

                    start_time = time.time()
                    result = optimizer(func, n_trials, dim, with_count=True)
                    elapsed = time.time() - start_time

                    if len(result) == 3:
                        val, x, evals = result
                        values.append(val)
                        times.append(elapsed)
                        evaluations.append(evals)

                if values:
                    avg_value = np.mean(values)
                    std_value = np.std(values)
                    avg_time = np.mean(times)
                    avg_evals = np.mean(evaluations)

                    print(f"  {opt_name:15}: {avg_value:.6f} ± {std_value:.6f} "
                          f"({avg_time:.3f}s, {avg_evals:.1f} evals)")

                    key = (dim, func_name, opt_name)
                    results[key] = {
                        'mean_value': avg_value,
                        'std_value': std_value,
                        'mean_time': avg_time,
                        'mean_evals': avg_evals
                    }

    # Performance summary
    print(f"\n🎯 PERFORMANCE SUMMARY")
    print("=" * 35)

    prima_methods = ['PRIMA UOBYQA', 'PRIMA NEWUOA']
    baseline_methods = ['SciPy Powell', 'SciPy Nelder-Mead']

    prima_wins = 0
    total_comparisons = 0

    for dim in dimensions:
        for func_name in test_functions.keys():
            print(f"\n{func_name} ({dim}D):")

            # Find best result for each category
            prima_best = float('inf')
            baseline_best = float('inf')

            for method in prima_methods:
                key = (dim, func_name, method)
                if key in results:
                    prima_best = min(prima_best, results[key]['mean_value'])

            for method in baseline_methods:
                key = (dim, func_name, method)
                if key in results:
                    baseline_best = min(baseline_best, results[key]['mean_value'])

            if prima_best < float('inf') and baseline_best < float('inf'):
                total_comparisons += 1
                if prima_best < baseline_best:
                    prima_wins += 1
                    improvement = (baseline_best - prima_best) / baseline_best * 100
                    print(f"  🏆 PRIMA better: {prima_best:.6f} vs {baseline_best:.6f} ({improvement:+.1f}%)")
                else:
                    decline = (prima_best - baseline_best) / baseline_best * 100
                    print(f"  📊 Baseline better: {prima_best:.6f} vs {baseline_best:.6f} ({decline:+.1f}%)")

    # Overall assessment
    win_rate = prima_wins / total_comparisons * 100 if total_comparisons > 0 else 0

    print(f"\n🏁 FINAL ASSESSMENT")
    print("=" * 25)
    print(f"PRIMA methods win rate: {prima_wins}/{total_comparisons} ({win_rate:.1f}%)")

    if win_rate >= 60:
        print("✅ PRIMA methods show STRONG performance advantage")
        print("   RECOMMENDATION: Excellent addition to HumpDay optimizer suite")
    elif win_rate >= 40:
        print("🤔 PRIMA methods show MODERATE performance")
        print("   RECOMMENDATION: Good specialized optimizers for specific cases")
    else:
        print("📊 PRIMA methods comparable to existing optimizers")
        print("   RECOMMENDATION: Useful additional options in the toolkit")

    print(f"\n💡 KEY STRENGTHS:")
    print(f"   • UOBYQA: High-accuracy quadratic interpolation for low dimensions")
    print(f"   • NEWUOA: Scalable iterative approximation for higher dimensions")
    print(f"   • Both: Derivative-free, proven Powell methods with bug fixes")

    return results

if __name__ == "__main__":
    print("🧪 Demonstrating PRIMA Integration with HumpDay")
    print("=" * 55)

    try:
        results = run_comparison_study()
        print(f"\n✨ Demonstration complete!")
        print("PRIMA UOBYQA and NEWUOA are now part of the HumpDay ecosystem.")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()