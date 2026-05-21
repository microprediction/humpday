#!/usr/bin/env python3
"""
Focused optimizer ranking testing PRIMA vs SciPy baselines.
Clean implementation without complex dependencies.
"""

import numpy as np
import pandas as pd
import time
import sys
from typing import Dict, List, Callable
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Direct import of PRIMA optimizers
sys.path.append('/Users/petercotton/github/humpday/humpday/optimizers')
from primacube import prima_uobyqa_cube, prima_newuoa_cube, PRIMA_AVAILABLE

class FocusedOptimizerTest:
    """Focused test comparing key optimizers with statistical rigor."""

    def __init__(self):
        self.results = []

    def create_test_functions(self) -> Dict[str, Callable]:
        """Create diverse test functions."""

        functions = {}

        # 1. Smooth Unimodal - Sphere
        def sphere(x):
            time.sleep(0.0008)  # Realistic computation time
            scaled_x = 6 * np.array(x) - 3  # Scale [0,1] to [-3,3]
            return np.sum(scaled_x**2)
        functions['sphere'] = sphere

        # 2. Smooth Unimodal - Rosenbrock
        def rosenbrock(x):
            time.sleep(0.0008)
            x = np.array(x)
            scaled_x = 4.096 * x - 2.048
            if len(scaled_x) < 2:
                return float('inf')
            return sum(100*(scaled_x[i+1] - scaled_x[i]**2)**2 + (1 - scaled_x[i])**2
                      for i in range(len(scaled_x)-1))
        functions['rosenbrock'] = rosenbrock

        # 3. Multimodal - Rastrigin
        def rastrigin(x):
            time.sleep(0.0008)
            x = np.array(x)
            scaled_x = 10.24 * x - 5.12
            n = len(scaled_x)
            return 10*n + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in scaled_x)
        functions['rastrigin'] = rastrigin

        # 4. Noisy function - robustness test
        def noisy_sphere(x):
            time.sleep(0.0008)
            scaled_x = 6 * np.array(x) - 3
            base = np.sum(scaled_x**2)
            noise = 0.05 * base * np.random.normal(0, 1)  # 5% noise
            return max(0.001, base + noise)  # Small minimum to avoid log issues
        functions['noisy_sphere'] = noisy_sphere

        return functions

    def scipy_optimizer(self, objective, n_trials, n_dim, method, with_count=False):
        """SciPy optimizer wrapper."""
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
            best_val = result.fun if result.success else float('inf')
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

    def get_optimizers(self) -> Dict[str, Callable]:
        """Get available optimizers for testing."""

        optimizers = {}

        # PRIMA optimizers (if available)
        if PRIMA_AVAILABLE:
            optimizers['PRIMA_UOBYQA'] = prima_uobyqa_cube
            optimizers['PRIMA_NEWUOA'] = prima_newuoa_cube

        # SciPy baselines
        optimizers['SciPy_Powell'] = lambda obj, nt, nd, wc=False: self.scipy_optimizer(obj, nt, nd, 'Powell', wc)
        optimizers['SciPy_NelderMead'] = lambda obj, nt, nd, wc=False: self.scipy_optimizer(obj, nt, nd, 'Nelder-Mead', wc)
        optimizers['SciPy_BFGS'] = lambda obj, nt, nd, wc=False: self.scipy_optimizer(obj, nt, nd, 'L-BFGS-B', wc)

        return optimizers

    def run_single_test(self, optimizer_name: str, optimizer_func: Callable,
                       objective: Callable, n_trials: int, n_dim: int, seed: int):
        """Run single optimization test."""

        np.random.seed(seed)
        start_time = time.time()

        try:
            result = optimizer_func(objective, n_trials, n_dim, with_count=True)

            if isinstance(result, tuple) and len(result) >= 3:
                best_val, best_x, evaluations = result[:3]
                success = True
            elif isinstance(result, (int, float)):
                best_val = float(result)
                evaluations = n_trials
                success = True
            else:
                best_val = float('inf')
                evaluations = 0
                success = False

            # Validate result
            if np.isnan(best_val) or np.isinf(best_val):
                best_val = float('inf')
                success = False

        except Exception as e:
            best_val = float('inf')
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

    def run_benchmark(self, n_runs: int = 20, n_trials: int = 50):
        """Run focused benchmark."""

        print("🏁 Focused Optimizer Ranking")
        print("=" * 40)

        test_functions = self.create_test_functions()
        optimizers = self.get_optimizers()
        dimensions = [2, 5, 10]

        print(f"Optimizers: {list(optimizers.keys())}")
        print(f"Functions: {list(test_functions.keys())}")
        print(f"Dimensions: {dimensions}")
        print(f"Runs per test: {n_runs}")
        print(f"Evaluation budget: {n_trials}")
        print()

        all_results = []
        total_tests = len(dimensions) * len(test_functions) * len(optimizers) * n_runs

        test_count = 0

        for dim in dimensions:
            print(f"\n📊 Testing {dim}D Problems")
            print("-" * 30)

            for func_name, func in test_functions.items():
                print(f"\n{func_name.upper()}:")

                function_results = []

                for opt_name, opt_func in optimizers.items():
                    print(f"  {opt_name:15}...", end=" ", flush=True)

                    optimizer_results = []

                    for run in range(n_runs):
                        test_count += 1
                        seed = run * 1000 + hash(f"{opt_name}_{func_name}_{dim}") % 1000

                        result = self.run_single_test(
                            opt_name, opt_func, func, n_trials, dim, seed
                        )

                        result.update({
                            'function': func_name,
                            'dimension': dim,
                            'run': run
                        })

                        optimizer_results.append(result)
                        all_results.append(result)

                        # Progress indicator
                        if test_count % 50 == 0:
                            progress = test_count / total_tests * 100
                            print(f"({progress:.0f}%)", end="")

                    # Analyze this optimizer's performance on this function
                    successful_runs = [r for r in optimizer_results if r['success']]

                    if successful_runs:
                        values = [r['best_value'] for r in successful_runs]
                        times = [r['time'] for r in successful_runs]

                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        success_rate = len(successful_runs) / n_runs * 100
                        mean_time = np.mean(times)

                        print(f"✓ {success_rate:3.0f}% | {mean_val:.4f}±{std_val:.4f} | {mean_time:.3f}s")

                        function_results.append({
                            'optimizer': opt_name,
                            'mean_value': mean_val,
                            'std_value': std_val,
                            'success_rate': success_rate,
                            'mean_time': mean_time
                        })
                    else:
                        print("❌ All failed")

                # Rank optimizers for this function
                if function_results:
                    function_results.sort(key=lambda x: x['mean_value'])
                    print(f"    Best: {function_results[0]['optimizer']} ({function_results[0]['mean_value']:.4f})")

        return pd.DataFrame(all_results)

    def analyze_comprehensive_results(self, df: pd.DataFrame):
        """Comprehensive analysis of benchmark results."""

        print("\n" + "="*60)
        print("🏆 COMPREHENSIVE RANKING ANALYSIS")
        print("="*60)

        # Filter successful runs
        successful_df = df[df['success'] == True].copy()
        if len(successful_df) == 0:
            print("❌ No successful runs to analyze!")
            return

        print(f"\n📊 Dataset Summary:")
        print(f"Total runs: {len(df)}")
        print(f"Successful runs: {len(successful_df)} ({len(successful_df)/len(df)*100:.1f}%)")

        # Success rates by optimizer
        print(f"\n🎯 Success Rates by Optimizer:")
        success_rates = df.groupby('optimizer')['success'].mean() * 100
        success_rates = success_rates.sort_values(ascending=False)

        for opt, rate in success_rates.items():
            print(f"  {opt:15}: {rate:5.1f}%")

        # Normalized performance analysis
        print(f"\n📈 Performance Analysis (Normalized by Function):")

        # Normalize by function and dimension for fair comparison
        for func in successful_df['function'].unique():
            for dim in successful_df['dimension'].unique():
                mask = (successful_df['function'] == func) & (successful_df['dimension'] == dim)
                if mask.sum() > 0:
                    values = successful_df.loc[mask, 'best_value']
                    min_val = values.min()
                    max_val = values.max()
                    if max_val > min_val:
                        successful_df.loc[mask, 'normalized_value'] = (values - min_val) / (max_val - min_val)
                    else:
                        successful_df.loc[mask, 'normalized_value'] = 0.0

        # Overall ranking
        print(f"\n🏆 OVERALL OPTIMIZER RANKINGS:")
        print("(Lower score = better performance, 0.0 = best on that problem)")

        overall_stats = successful_df.groupby('optimizer').agg({
            'normalized_value': ['mean', 'std', 'count'],
            'best_value': 'mean',
            'time': 'mean'
        }).round(4)

        overall_stats.columns = ['norm_mean', 'norm_std', 'n_tests', 'raw_mean', 'time_mean']
        overall_stats = overall_stats.sort_values('norm_mean')

        print(f"\n{'Rank':<4} {'Optimizer':<15} {'Score':<8} {'±Std':<8} {'Tests':<6} {'Time(s)':<8}")
        print("-" * 60)

        for rank, (opt, row) in enumerate(overall_stats.iterrows(), 1):
            print(f"{rank:<4} {opt:<15} {row['norm_mean']:<8.3f} {row['norm_std']:<8.3f} "
                  f"{row['n_tests']:<6.0f} {row['time_mean']:<8.3f}")

        # Statistical significance testing
        print(f"\n🔬 Statistical Significance Tests:")

        optimizers = overall_stats.index.tolist()
        significant_pairs = []

        for i, opt1 in enumerate(optimizers[:-1]):
            for opt2 in optimizers[i+1:]:
                opt1_scores = successful_df[successful_df['optimizer'] == opt1]['normalized_value']
                opt2_scores = successful_df[successful_df['optimizer'] == opt2]['normalized_value']

                if len(opt1_scores) > 5 and len(opt2_scores) > 5:  # Minimum sample size
                    t_stat, p_value = stats.ttest_ind(opt1_scores, opt2_scores)

                    if p_value < 0.05:
                        winner = opt1 if opt1_scores.mean() < opt2_scores.mean() else opt2
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
                        significant_pairs.append({
                            'winner': winner,
                            'loser': opt2 if winner == opt1 else opt1,
                            'p_value': p_value,
                            'significance': significance
                        })

        if significant_pairs:
            significant_pairs.sort(key=lambda x: x['p_value'])
            print(f"\nSignificant differences found:")
            for pair in significant_pairs:
                print(f"  {pair['winner']} > {pair['loser']} (p={pair['p_value']:.4f} {pair['significance']})")
        else:
            print("No statistically significant differences found between optimizers.")

        # Function-specific performance
        print(f"\n🎯 Best Performer by Problem Type:")

        for func in successful_df['function'].unique():
            func_data = successful_df[successful_df['function'] == func]
            func_performance = func_data.groupby('optimizer')['best_value'].mean().sort_values()

            if len(func_performance) > 0:
                best_opt = func_performance.index[0]
                best_val = func_performance.iloc[0]
                print(f"  {func:<15}: {best_opt} ({best_val:.4f})")

        # Dimension scaling
        print(f"\n📏 Performance by Dimension:")

        for dim in sorted(successful_df['dimension'].unique()):
            dim_data = successful_df[successful_df['dimension'] == dim]
            dim_performance = dim_data.groupby('optimizer')['normalized_value'].mean().sort_values()

            print(f"\n  {dim}D Problems:")
            for rank, (opt, score) in enumerate(dim_performance.items(), 1):
                print(f"    {rank}. {opt:<15}: {score:.3f}")

        return overall_stats

def main():
    """Run the focused optimizer ranking."""

    print("🚀 Starting Focused Optimizer Ranking")
    print("=" * 45)

    if not PRIMA_AVAILABLE:
        print("⚠️  PRIMA optimizers not available - testing SciPy methods only")

    tester = FocusedOptimizerTest()

    start_time = time.time()
    results_df = tester.run_benchmark(n_runs=20, n_trials=50)
    total_time = time.time() - start_time

    print(f"\n⏱️  Total benchmark time: {total_time:.1f} seconds")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"focused_optimizer_ranking_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"💾 Results saved to: {filename}")

    # Comprehensive analysis
    performance_stats = tester.analyze_comprehensive_results(results_df)

    print(f"\n✨ Ranking complete! Check the analysis above.")

    return results_df, performance_stats

if __name__ == "__main__":
    try:
        results_df, performance_stats = main()
    except Exception as e:
        print(f"❌ Ranking failed: {e}")
        import traceback
        traceback.print_exc()