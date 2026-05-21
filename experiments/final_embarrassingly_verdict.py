#!/usr/bin/env python3
"""
Final test to give clear verdict on embarrassingly library techniques.
Focus on most promising approach: controlled parallelization test.
"""

import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from embarrassingly.parallel import Parallel
from scipy.optimize import minimize

def simple_sphere(x):
    """Simple sphere function with artificial delay to test parallelization."""
    # Add small delay to simulate real computation
    time.sleep(0.001)  # 1ms per evaluation
    scaled_x = 10 * np.array(x) - 5
    return np.sum(scaled_x**2)

def worker_sphere(worker_id, x):
    """Sphere function that accepts worker_id for Parallel wrapper."""
    return simple_sphere(x)

def test_parallel_optimization_benefit():
    """Test if Parallel wrapper provides benefits in optimization context."""

    print("🔄 Testing Embarrassingly Parallel in Real Optimization Context")
    print("=" * 65)

    # Test 1: Direct function evaluation comparison
    print("\n📊 Test 1: Direct Function Evaluation Speed")
    print("-" * 45)

    test_points = [np.random.rand(2) for _ in range(20)]

    # Sequential evaluation
    start_time = time.time()
    sequential_results = [simple_sphere(x) for x in test_points]
    sequential_time = time.time() - start_time

    # Parallel wrapper evaluation
    parallel_func = Parallel(worker_sphere, num_workers=2)
    start_time = time.time()
    parallel_results = [parallel_func(x) for x in test_points]
    parallel_time = time.time() - start_time

    time_improvement = (sequential_time - parallel_time) / sequential_time * 100

    print(f"Sequential evaluation: {sequential_time:.3f}s")
    print(f"Parallel wrapper:      {parallel_time:.3f}s")

    if time_improvement > 5:
        print(f"🚀 Parallel is {time_improvement:.1f}% faster!")
    elif time_improvement < -5:
        print(f"🐌 Parallel is {abs(time_improvement):.1f}% slower (overhead)")
    else:
        print(f"≈ Similar performance ({time_improvement:+.1f}%)")

    # Test 2: Integration with scipy optimizer
    print(f"\n📊 Test 2: Integration with SciPy Optimize")
    print("-" * 45)

    def run_optimization_with_function(obj_func, label):
        """Run optimization and return timing/quality results."""
        eval_count = [0]

        def counting_wrapper(x):
            eval_count[0] += 1
            return obj_func(x)

        start_time = time.time()

        result = minimize(
            counting_wrapper,
            x0=np.array([0.7, 0.3]),  # Start away from optimum
            method='Powell',
            bounds=[(0, 1), (0, 1)],
            options={'maxfev': 30}
        )

        elapsed = time.time() - start_time

        print(f"  {label}: {result.fun:.6f} in {elapsed:.3f}s ({eval_count[0]} evals)")
        return {
            'value': result.fun,
            'time': elapsed,
            'evals': eval_count[0],
            'evals_per_sec': eval_count[0] / elapsed if elapsed > 0 else 0
        }

    # Run with standard function
    standard_result = run_optimization_with_function(simple_sphere, "Standard function")

    # Run with parallel wrapper
    parallel_result = run_optimization_with_function(parallel_func, "Parallel wrapper ")

    opt_time_improvement = (standard_result['time'] - parallel_result['time']) / standard_result['time'] * 100

    if opt_time_improvement > 5:
        print(f"  🚀 Parallel optimization is {opt_time_improvement:.1f}% faster!")
    elif opt_time_improvement < -5:
        print(f"  🐌 Parallel optimization is {abs(opt_time_improvement):.1f}% slower")
    else:
        print(f"  ≈ Similar optimization time ({opt_time_improvement:+.1f}%)")

    return {
        'eval_time_improvement': time_improvement,
        'opt_time_improvement': opt_time_improvement,
        'parallel_overhead': parallel_time > sequential_time
    }

def test_memorable_caching():
    """Test if memoization provides benefits."""

    print(f"\n🧠 Testing Embarrassingly Memorable (Caching)")
    print("=" * 50)

    try:
        from embarrassingly.memorable import np_cache

        # Create cached version of expensive function
        @np_cache(maxsize=128)
        def cached_sphere(x):
            time.sleep(0.002)  # 2ms computation
            scaled_x = 10 * np.array(x) - 5
            return float(np.sum(scaled_x**2))

        def uncached_sphere(x):
            time.sleep(0.002)  # Same computation time
            scaled_x = 10 * np.array(x) - 5
            return float(np.sum(scaled_x**2))

        # Test with repeated evaluations (simulating optimizer re-visiting points)
        test_points = [np.random.rand(2) for _ in range(10)]
        repeated_points = test_points + test_points[:5]  # Some repeats

        # Uncached evaluation
        start_time = time.time()
        uncached_results = [uncached_sphere(x) for x in repeated_points]
        uncached_time = time.time() - start_time

        # Cached evaluation
        start_time = time.time()
        cached_results = [cached_sphere(x) for x in repeated_points]
        cached_time = time.time() - start_time

        cache_improvement = (uncached_time - cached_time) / uncached_time * 100

        print(f"Uncached evaluation: {uncached_time:.3f}s")
        print(f"Cached evaluation:   {cached_time:.3f}s")

        if cache_improvement > 10:
            print(f"🚀 Caching provides {cache_improvement:.1f}% speedup!")
            print(f"💾 Cache info: {cached_sphere.cache_info()}")
        else:
            print(f"≈ No significant caching benefit ({cache_improvement:+.1f}%)")

        return cache_improvement > 10

    except ImportError:
        print("❌ Memorable (caching) not available")
        return False

def final_verdict():
    """Give final recommendation on embarrassingly library integration."""

    print(f"\n🏁 FINAL VERDICT ON EMBARRASSINGLY LIBRARY")
    print("=" * 55)

    # Run all tests
    parallel_results = test_parallel_optimization_benefit()
    caching_works = test_memorable_caching()

    benefits_found = []
    issues_found = []

    # Analyze parallel results
    if parallel_results['eval_time_improvement'] > 10:
        benefits_found.append("Parallel evaluation shows significant speedup")
    elif parallel_results['parallel_overhead']:
        issues_found.append("Parallel wrapper adds overhead without benefits")

    if parallel_results['opt_time_improvement'] > 10:
        benefits_found.append("Parallel integration with optimizers works well")
    elif parallel_results['opt_time_improvement'] < -10:
        issues_found.append("Parallel wrapper slows down optimization")

    # Analyze caching
    if caching_works:
        benefits_found.append("Caching provides measurable benefits for repeated evaluations")
    else:
        issues_found.append("Caching doesn't show clear benefits")

    # Final recommendation
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"Benefits found: {len(benefits_found)}")
    for benefit in benefits_found:
        print(f"  ✅ {benefit}")

    print(f"Issues found: {len(issues_found)}")
    for issue in issues_found:
        print(f"  ❌ {issue}")

    # Overall recommendation
    if len(benefits_found) >= 2:
        print(f"\n🎯 RECOMMENDATION: INTEGRATE embarrassingly library!")
        print("The techniques show measurable benefits for HumpDay.")
    elif len(benefits_found) >= 1:
        print(f"\n🤔 RECOMMENDATION: SELECTIVE integration of embarrassingly library")
        print("Some techniques work well, integrate the beneficial ones.")
    else:
        print(f"\n❌ RECOMMENDATION: SKIP embarrassingly library for now")
        print("Techniques don't show clear benefits in current testing.")

if __name__ == "__main__":
    print("🧪 Final Verdict: Do Embarrassingly Techniques Actually Work?")
    print("=" * 70)

    final_verdict()

    print(f"\n✨ Testing complete! Check recommendations above.")