#!/usr/bin/env python3
"""
Comprehensive optimizer ranking and testing framework.
Tests all available HumpDay optimizers with proper statistical methodology.
"""

import numpy as np
import pandas as pd
import time
import sys
from typing import Dict, List, Tuple, Callable
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import individual optimizer collections to avoid dependency issues
sys.path.append('/Users/petercotton/github/humpday/humpday/optimizers')

# Import what we can without breaking dependencies
from primacube import PRIMA_OPTIMIZERS, PRIMA_AVAILABLE
from scipycube import scipy_minimize_cube

# Test functions with diverse characteristics
class OptimizationTestSuite:
    """Comprehensive test suite for optimizer benchmarking."""

    def __init__(self):
        self.results = []

    def create_test_functions(self) -> Dict[str, Tuple[Callable, str]]:
        """Create diverse test functions with descriptions."""

        functions = {}

        # 1. Smooth Unimodal - Sphere
        def sphere(x):
            time.sleep(0.0005)  # Realistic computation time
            scaled_x = 4 * np.array(x) - 2  # Scale [0,1] to [-2,2]
            return np.sum(scaled_x**2)
        functions['sphere'] = (sphere, "Smooth unimodal (sphere)")

        # 2. Smooth Unimodal - Rosenbrock
        def rosenbrock(x):
            time.sleep(0.0005)
            x = np.array(x)
            scaled_x = 4.096 * x - 2.048  # Scale to [-2.048, 2.048]
            if len(scaled_x) < 2:
                return float('inf')
            return sum(100*(scaled_x[i+1] - scaled_x[i]**2)**2 + (1 - scaled_x[i])**2
                      for i in range(len(scaled_x)-1))
        functions['rosenbrock'] = (rosenbrock, "Smooth unimodal with valley (Rosenbrock)")

        # 3. Multimodal - Rastrigin
        def rastrigin(x):
            time.sleep(0.0005)
            x = np.array(x)
            scaled_x = 10.24 * x - 5.12  # Scale to [-5.12, 5.12]
            n = len(scaled_x)
            return 10*n + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in scaled_x)
        functions['rastrigin'] = (rastrigin, "Multimodal with many local minima (Rastrigin)")

        # 4. Multimodal - Ackley
        def ackley(x):
            time.sleep(0.0005)
            x = np.array(x)
            scaled_x = 65.536 * x - 32.768  # Scale to [-32.768, 32.768]
            n = len(scaled_x)
            sum1 = sum(xi**2 for xi in scaled_x)
            sum2 = sum(np.cos(2*np.pi*xi) for xi in scaled_x)
            return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
        functions['ackley'] = (ackley, "Multimodal with flat outer region (Ackley)")

        # 5. Noisy Sphere - robustness test
        def noisy_sphere(x):
            time.sleep(0.0005)
            scaled_x = 4 * np.array(x) - 2
            base = np.sum(scaled_x**2)
            noise = 0.1 * base * np.random.normal(0, 1)  # 10% multiplicative noise
            return max(0, base + noise)  # Ensure non-negative
        functions['noisy_sphere'] = (noisy_sphere, "Noisy smooth function (robustness test)")

        return functions

    def get_available_optimizers(self) -> Dict[str, Callable]:
        """Get all available optimizers we can test."""

        optimizers = {}

        # PRIMA optimizers (our new additions)
        if PRIMA_AVAILABLE:
            optimizers['PRIMA_UOBYQA'] = PRIMA_OPTIMIZERS[0]  # prima_uobyqa_cube
            optimizers['PRIMA_NEWUOA'] = PRIMA_OPTIMIZERS[1]  # prima_newuoa_cube

        # SciPy baseline
        optimizers['SciPy_Powell'] = lambda obj, n_trials, n_dim, with_count=False: self.scipy_wrapper(obj, n_trials, n_dim, 'Powell', with_count)
        optimizers['SciPy_NelderMead'] = lambda obj, n_trials, n_dim, with_count=False: self.scipy_wrapper(obj, n_trials, n_dim, 'Nelder-Mead', with_count)
        optimizers['SciPy_SLSQP'] = lambda obj, n_trials, n_dim, with_count=False: self.scipy_wrapper(obj, n_trials, n_dim, 'SLSQP', with_count)

        # Try to import other optimizers safely
        try:
            from nloptcube import nlopt_bobyqa_cube, nlopt_cobyla_cube
            optimizers['NLopt_BOBYQA'] = nlopt_bobyqa_cube
            optimizers['NLopt_COBYLA'] = nlopt_cobyla_cube
        except ImportError:
            pass

        try:
            from shgocube import shgo_cube
            optimizers['SHGO'] = shgo_cube
        except ImportError:
            pass

        return optimizers

    def scipy_wrapper(self, objective, n_trials, n_dim, method, with_count=False):
        """Wrapper for SciPy methods to match HumpDay interface."""
        from scipy.optimize import minimize

        eval_count = [0]
        def counting_wrapper(x):
            eval_count[0] += 1
            return objective(x)

        x0 = np.random.rand(n_dim)

        try:
            result = minimize(
                counting_wrapper,
                x0,
                method=method,
                bounds=[(0, 1)] * n_dim,
                options={'maxfev': n_trials}
            )

            best_x = np.clip(result.x, 0.0, 1.0)
            best_val = result.fun
            n_evaluations = min(eval_count[0], n_trials)

            if with_count:
                return best_val, best_x, n_evaluations
            else:
                return best_val

        except Exception as e:
            if with_count:
                return float('inf'), x0, eval_count[0]
            else:
                return float('inf')

    def run_single_test(self, optimizer_name: str, optimizer_func: Callable,
                       objective: Callable, n_trials: int, n_dim: int,
                       seed: int) -> Dict:
        """Run single optimization test with detailed tracking."""

        np.random.seed(seed)

        start_time = time.time()

        try:
            result = optimizer_func(objective, n_trials, n_dim, with_count=True)

            if len(result) == 3:
                best_val, best_x, evaluations = result
                success = True
            else:
                # Handle case where with_count=True not supported
                best_val = result
                best_x = None
                evaluations = n_trials
                success = True

        except Exception as e:
            best_val = float('inf')
            best_x = None
            evaluations = 0
            success = False

        elapsed_time = time.time() - start_time

        return {
            'optimizer': optimizer_name,
            'best_value': best_val,
            'success': success,
            'evaluations': evaluations,
            'time': elapsed_time,
            'seed': seed
        }

    def run_comprehensive_benchmark(self, n_runs: int = 15, n_trials: int = 50) -> pd.DataFrame:
        """Run comprehensive benchmark across all optimizers and functions."""

        print("🏁 Comprehensive Optimizer Ranking")
        print("=" * 50)

        test_functions = self.create_test_functions()
        optimizers = self.get_available_optimizers()
        dimensions = [2, 5, 10]  # Test different dimensionalities

        print(f"Testing {len(optimizers)} optimizers")
        print(f"Testing {len(test_functions)} functions")
        print(f"Testing {len(dimensions)} dimensions")
        print(f"Running {n_runs} trials each")
        print(f"Budget: {n_trials} evaluations per trial")
        print()

        all_results = []

        for dim in dimensions:
            print(f"📊 Testing {dim}D problems...")

            for func_name, (func, desc) in test_functions.items():
                print(f"  {func_name}: {desc}")

                for opt_name, opt_func in optimizers.items():
                    print(f"    {opt_name}...", end=" ", flush=True)

                    run_results = []

                    for run in range(n_runs):
                        seed = run * 1000 + hash(f"{opt_name}_{func_name}_{dim}") % 1000

                        result = self.run_single_test(
                            opt_name, opt_func, func, n_trials, dim, seed
                        )

                        result.update({
                            'function': func_name,
                            'dimension': dim,
                            'run': run,
                            'description': desc
                        })

                        run_results.append(result)
                        all_results.append(result)

                    # Quick summary for this optimizer
                    successful_runs = [r for r in run_results if r['success']]
                    if successful_runs:
                        avg_val = np.mean([r['best_value'] for r in successful_runs])
                        success_rate = len(successful_runs) / len(run_results) * 100
                        print(f"✓ {success_rate:.0f}% success, avg: {avg_val:.4f}")
                    else:
                        print("❌ All runs failed")

        return pd.DataFrame(all_results)

    def analyze_results(self, df: pd.DataFrame) -> None:
        """Analyze and rank optimizer performance."""

        print("\n" + "="*60)
        print("🏆 OPTIMIZER PERFORMANCE ANALYSIS")
        print("="*60)

        # Overall success rates
        print("\n📊 Success Rates (% of runs that completed successfully):")
        success_rates = df.groupby('optimizer')['success'].mean() * 100
        success_rates = success_rates.sort_values(ascending=False)

        for opt, rate in success_rates.items():
            print(f"  {opt:15}: {rate:5.1f}%")

        # Performance by function type
        print("\n🎯 Performance by Problem Type:")

        # Filter successful runs only
        successful_df = df[df['success'] == True].copy()

        if len(successful_df) == 0:
            print("❌ No successful runs to analyze!")
            return

        # Normalize values by function and dimension for fair comparison
        successful_df['normalized_value'] = successful_df.groupby(['function', 'dimension'])['best_value'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
        )

        # Overall ranking by normalized performance
        print("\n🏆 Overall Rankings (lower normalized score = better):")
        overall_perf = successful_df.groupby('optimizer').agg({
            'normalized_value': ['mean', 'std', 'count'],
            'best_value': 'mean',
            'time': 'mean',
            'evaluations': 'mean'
        }).round(4)

        overall_perf.columns = ['norm_mean', 'norm_std', 'count', 'raw_mean', 'time_mean', 'eval_mean']
        overall_perf = overall_perf.sort_values('norm_mean')

        for i, (opt, row) in enumerate(overall_perf.iterrows(), 1):
            print(f"  {i:2}. {opt:15}: {row['norm_mean']:.3f} ± {row['norm_std']:.3f} "
                  f"({row['count']:3} successful runs, {row['time_mean']:.2f}s avg)")

        # Function-specific rankings
        print("\n📈 Best Performer by Problem Type:")

        for func in successful_df['function'].unique():
            func_data = successful_df[successful_df['function'] == func]
            func_perf = func_data.groupby('optimizer')['best_value'].mean().sort_values()

            if len(func_perf) > 0:
                best_opt = func_perf.index[0]
                best_val = func_perf.iloc[0]
                print(f"  {func:12}: {best_opt:15} ({best_val:.4f})")

        # Dimension scaling analysis
        print("\n📏 Performance by Dimension:")

        for dim in sorted(successful_df['dimension'].unique()):
            print(f"\n  {dim}D Problems:")
            dim_data = successful_df[successful_df['dimension'] == dim]
            dim_perf = dim_data.groupby('optimizer')['normalized_value'].mean().sort_values()

            for i, (opt, score) in enumerate(dim_perf.head(5).items(), 1):
                print(f"    {i}. {opt:15}: {score:.3f}")

        # Statistical significance testing
        print("\n🔬 Statistical Significance Analysis:")

        # Compare top performers
        top_3_optimizers = overall_perf.head(3).index.tolist()

        if len(top_3_optimizers) >= 2:
            for i, opt1 in enumerate(top_3_optimizers[:-1]):
                for opt2 in top_3_optimizers[i+1:]:

                    opt1_values = successful_df[successful_df['optimizer'] == opt1]['normalized_value']
                    opt2_values = successful_df[successful_df['optimizer'] == opt2]['normalized_value']

                    if len(opt1_values) > 0 and len(opt2_values) > 0:
                        t_stat, p_value = stats.ttest_ind(opt1_values, opt2_values)

                        significance = ""
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = "ns"

                        print(f"  {opt1} vs {opt2}: p={p_value:.4f} {significance}")

        return overall_perf

def main():
    """Run comprehensive optimizer benchmarking."""

    print("🚀 Starting Comprehensive Optimizer Benchmarking")
    print("=" * 55)

    suite = OptimizationTestSuite()

    # Run the benchmark
    start_time = time.time()
    results_df = suite.run_comprehensive_benchmark(n_runs=15, n_trials=50)
    total_time = time.time() - start_time

    print(f"\n⏱️  Total benchmarking time: {total_time:.1f} seconds")
    print(f"📊 Total test runs: {len(results_df)}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"optimizer_benchmark_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"💾 Results saved to: {filename}")

    # Analyze results
    performance_summary = suite.analyze_results(results_df)

    print(f"\n✨ Benchmarking complete!")
    print("Check the rankings above to see how optimizers compare.")

    return results_df, performance_summary

if __name__ == "__main__":
    try:
        results_df, performance_summary = main()
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()