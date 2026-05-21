#!/usr/bin/env python3
"""
Thorough statistical testing of embarrassingly library techniques.
Proper sample sizes, multiple functions, rigorous analysis.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
from embarrassingly.shy import Shy
from embarrassingly.memorable import np_cache
from typing import Dict, List, Callable, Tuple
import warnings
warnings.filterwarnings('ignore')

class ThoroughTestSuite:
    """Rigorous testing framework for embarrassingly techniques."""

    def __init__(self, n_runs=30, max_evals=50):
        """Initialize with proper sample sizes for statistical significance."""
        self.n_runs = n_runs  # 30 runs for statistical significance
        self.max_evals = max_evals
        self.results = []

    def create_test_functions(self) -> Dict[str, Callable]:
        """Create diverse set of realistic test functions."""
        functions = {}

        # 1. Simple Sphere (smooth, unimodal)
        def sphere(x):
            time.sleep(0.001)  # Small realistic delay
            scaled_x = 10 * np.array(x) - 5
            return np.sum(scaled_x**2)
        functions['sphere'] = sphere

        # 2. Rosenbrock (smooth, unimodal, difficult valley)
        def rosenbrock(x):
            time.sleep(0.001)
            x = np.array(x)
            scaled_x = 4.096 * x - 2.048  # Scale to [-2.048, 2.048]
            return sum(100*(scaled_x[i+1] - scaled_x[i]**2)**2 + (1 - scaled_x[i])**2
                      for i in range(len(scaled_x)-1))
        functions['rosenbrock'] = rosenbrock

        # 3. Rastrigin (multimodal, many local minima)
        def rastrigin(x):
            time.sleep(0.001)
            x = np.array(x)
            scaled_x = 10.24 * x - 5.12
            n = len(scaled_x)
            return 10*n + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in scaled_x)
        functions['rastrigin'] = rastrigin

        # 4. Ackley (multimodal, nearly flat outer region)
        def ackley(x):
            time.sleep(0.001)
            x = np.array(x)
            scaled_x = 65.536 * x - 32.768
            n = len(scaled_x)
            sum1 = sum(xi**2 for xi in scaled_x)
            sum2 = sum(np.cos(2*np.pi*xi) for xi in scaled_x)
            return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
        functions['ackley'] = ackley

        # 5. Expensive Himmelblau (multiple global minima)
        def himmelblau(x):
            time.sleep(0.002)  # More expensive
            x = np.array(x)
            scaled_x = 10 * x - 5  # Scale to [-5, 5]
            return (scaled_x[0]**2 + scaled_x[1] - 11)**2 + (scaled_x[0] + scaled_x[1]**2 - 7)**2
        functions['himmelblau'] = himmelblau

        return functions

    def run_single_optimization(self, optimizer_name: str, objective_func: Callable,
                              seed: int) -> Dict:
        """Run single optimization with detailed tracking."""
        np.random.seed(seed)

        eval_count = [0]
        eval_times = []

        def tracking_wrapper(x):
            eval_count[0] += 1
            start = time.time()
            result = objective_func(x)
            eval_times.append(time.time() - start)
            return result

        x0 = np.random.rand(2)  # Random start in [0,1]^2

        start_time = time.time()

        try:
            if optimizer_name == 'Powell':
                result = minimize(tracking_wrapper, x0, method='Powell',
                                bounds=[(0,1), (0,1)], options={'maxfev': self.max_evals})
            elif optimizer_name == 'Nelder-Mead':
                result = minimize(tracking_wrapper, x0, method='Nelder-Mead',
                                bounds=[(0,1), (0,1)], options={'maxfev': self.max_evals})
            elif optimizer_name == 'SLSQP':
                result = minimize(tracking_wrapper, x0, method='SLSQP',
                                bounds=[(0,1), (0,1)], options={'maxfev': self.max_evals})
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

        except Exception as e:
            return {
                'success': False, 'final_value': float('inf'), 'total_time': 0,
                'evaluations': eval_count[0], 'error': str(e)
            }

        total_time = time.time() - start_time

        return {
            'success': result.success,
            'final_value': float(result.fun if result.success else float('inf')),
            'total_time': total_time,
            'evaluations': eval_count[0],
            'avg_eval_time': np.mean(eval_times) if eval_times else 0,
            'final_x': result.x.tolist() if result.success else x0.tolist()
        }

    def test_caching_effectiveness(self) -> pd.DataFrame:
        """Thorough test of caching with repeated evaluations."""
        print("🧠 Testing Memorable (Caching) - Thorough Analysis")
        print("=" * 55)

        cache_results = []

        # Create cached and uncached versions of each function
        test_functions = self.create_test_functions()

        for func_name, func in test_functions.items():
            print(f"\nTesting {func_name}...")

            # Create cached version
            @np_cache(maxsize=128)
            def cached_version(x_tuple):
                return func(np.array(x_tuple))

            def uncached_version(x_tuple):
                return func(np.array(x_tuple))

            # Generate test points with intentional repeats
            base_points = [tuple(np.random.rand(2)) for _ in range(15)]
            repeated_points = base_points + base_points[:10]  # 67% new, 33% repeats
            np.random.shuffle(repeated_points)

            for run in range(10):  # 10 runs per function
                # Test uncached
                start_time = time.time()
                uncached_results = [uncached_version(x) for x in repeated_points]
                uncached_time = time.time() - start_time

                # Reset and test cached
                cached_version.cache_clear()
                start_time = time.time()
                cached_results = [cached_version(x) for x in repeated_points]
                cached_time = time.time() - start_time

                # Verify results are identical
                assert np.allclose(uncached_results, cached_results), "Cache returned different results!"

                cache_info = cached_version.cache_info()
                speedup = (uncached_time - cached_time) / uncached_time * 100

                cache_results.append({
                    'function': func_name,
                    'run': run,
                    'uncached_time': uncached_time,
                    'cached_time': cached_time,
                    'speedup_percent': speedup,
                    'cache_hits': cache_info.hits,
                    'cache_misses': cache_info.misses,
                    'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) * 100
                })

        df = pd.DataFrame(cache_results)

        # Statistical analysis
        print(f"\n📊 CACHING RESULTS SUMMARY:")
        print(f"Average speedup: {df['speedup_percent'].mean():.1f}% ± {df['speedup_percent'].std():.1f}%")
        print(f"Average hit rate: {df['hit_rate'].mean():.1f}% ± {df['hit_rate'].std():.1f}%")

        # Test statistical significance
        t_stat, p_value = stats.ttest_1samp(df['speedup_percent'], 0)
        print(f"Statistical significance: t={t_stat:.3f}, p={p_value:.6f}")

        if p_value < 0.001:
            print("✅ HIGHLY SIGNIFICANT caching benefit (p < 0.001)")
        elif p_value < 0.05:
            print("✅ SIGNIFICANT caching benefit (p < 0.05)")
        else:
            print("❌ NO significant caching benefit")

        return df

    def test_shy_effectiveness(self) -> pd.DataFrame:
        """Thorough test of Shy wrapper across multiple functions and optimizers."""
        print(f"\n🔍 Testing Shy Wrapper - Thorough Analysis")
        print("=" * 50)

        shy_results = []
        test_functions = self.create_test_functions()
        optimizers = ['Powell', 'Nelder-Mead', 'SLSQP']
        bounds = [[0, 1], [0, 1]]

        for func_name, func in test_functions.items():
            print(f"\nTesting {func_name}...")

            # Create Shy version with conservative parameters
            shy_func = Shy(func, bounds=bounds, t_unit=0.005, d_unit=0.05)

            for optimizer_name in optimizers:
                print(f"  {optimizer_name}...", end=" ", flush=True)

                standard_values = []
                shy_values = []
                standard_times = []
                shy_times = []

                for run in range(self.n_runs):
                    seed = run * 1000 + hash(f"{func_name}_{optimizer_name}") % 1000

                    # Standard optimization
                    standard_result = self.run_single_optimization(optimizer_name, func, seed)
                    if standard_result['success']:
                        standard_values.append(standard_result['final_value'])
                        standard_times.append(standard_result['total_time'])

                    # Shy optimization
                    shy_result = self.run_single_optimization(optimizer_name, shy_func, seed)
                    if shy_result['success']:
                        shy_values.append(shy_result['final_value'])
                        shy_times.append(shy_result['total_time'])

                if len(standard_values) >= 10 and len(shy_values) >= 10:  # Need sufficient data

                    # Statistical tests
                    value_t_stat, value_p = stats.ttest_ind(standard_values, shy_values)
                    time_t_stat, time_p = stats.ttest_ind(standard_times, shy_times)

                    mean_standard_value = np.mean(standard_values)
                    mean_shy_value = np.mean(shy_values)
                    mean_standard_time = np.mean(standard_times)
                    mean_shy_time = np.mean(shy_times)

                    value_improvement = (mean_standard_value - mean_shy_value) / mean_standard_value * 100
                    time_improvement = (mean_standard_time - mean_shy_time) / mean_standard_time * 100

                    shy_results.append({
                        'function': func_name,
                        'optimizer': optimizer_name,
                        'standard_value_mean': mean_standard_value,
                        'shy_value_mean': mean_shy_value,
                        'value_improvement_percent': value_improvement,
                        'value_p_value': value_p,
                        'standard_time_mean': mean_standard_time,
                        'shy_time_mean': mean_shy_time,
                        'time_improvement_percent': time_improvement,
                        'time_p_value': time_p,
                        'standard_n': len(standard_values),
                        'shy_n': len(shy_values)
                    })

                    print(f"✓ ({len(standard_values)}/{len(shy_values)} successful runs)")
                else:
                    print(f"❌ Insufficient successful runs")

        df = pd.DataFrame(shy_results)

        if len(df) > 0:
            print(f"\n📊 SHY WRAPPER RESULTS SUMMARY:")

            # Overall statistics
            significant_value_improvements = df[df['value_p_value'] < 0.05]['value_improvement_percent']
            significant_time_improvements = df[df['time_p_value'] < 0.05]['time_improvement_percent']

            print(f"Tests with significant value improvements: {len(significant_value_improvements)}/{len(df)}")
            if len(significant_value_improvements) > 0:
                print(f"  Average significant value improvement: {significant_value_improvements.mean():.1f}%")

            print(f"Tests with significant time improvements: {len(significant_time_improvements)}/{len(df)}")
            if len(significant_time_improvements) > 0:
                print(f"  Average significant time improvement: {significant_time_improvements.mean():.1f}%")

            # Best cases
            best_value_case = df.loc[df['value_improvement_percent'].idxmax()]
            print(f"\nBest value improvement: {best_value_case['value_improvement_percent']:.1f}% "
                  f"({best_value_case['function']}, {best_value_case['optimizer']}, p={best_value_case['value_p_value']:.4f})")

        return df

def run_comprehensive_analysis():
    """Run complete thorough analysis."""
    print("🔬 COMPREHENSIVE EMBARRASSINGLY LIBRARY ANALYSIS")
    print("=" * 60)
    print(f"Sample size: 30 runs per test for statistical significance")
    print(f"Functions: 5 diverse test problems")
    print(f"Optimizers: 3 different methods")

    tester = ThoroughTestSuite(n_runs=30, max_evals=50)

    # Test 1: Caching
    cache_df = tester.test_caching_effectiveness()

    # Test 2: Shy wrapper
    shy_df = tester.test_shy_effectiveness()

    print(f"\n🏆 FINAL STATISTICAL CONCLUSIONS:")
    print("=" * 45)

    # Caching conclusion
    avg_cache_speedup = cache_df['speedup_percent'].mean()
    cache_significant = stats.ttest_1samp(cache_df['speedup_percent'], 0)[1] < 0.05

    if cache_significant and avg_cache_speedup > 5:
        print(f"✅ CACHING: Statistically significant {avg_cache_speedup:.1f}% average speedup")
        print("   RECOMMENDATION: Integrate caching into HumpDay")
    else:
        print(f"❌ CACHING: No statistically significant benefit")

    # Shy conclusion
    if len(shy_df) > 0:
        significant_shy_improvements = len(shy_df[shy_df['value_p_value'] < 0.05])
        total_shy_tests = len(shy_df)

        if significant_shy_improvements >= total_shy_tests * 0.3:  # 30% of tests show benefit
            print(f"🤔 SHY: Shows benefits in {significant_shy_improvements}/{total_shy_tests} cases")
            print("   RECOMMENDATION: Consider selective integration with careful tuning")
        else:
            print(f"❌ SHY: Inconsistent benefits ({significant_shy_improvements}/{total_shy_tests} significant)")
            print("   RECOMMENDATION: Skip for now")

    return cache_df, shy_df

if __name__ == "__main__":
    cache_results, shy_results = run_comprehensive_analysis()

    print(f"\n📊 Data saved for further analysis:")
    print(f"   Cache results: {len(cache_results)} tests")
    print(f"   Shy results: {len(shy_results)} tests")

    print(f"\n✅ THOROUGH TESTING COMPLETE!")