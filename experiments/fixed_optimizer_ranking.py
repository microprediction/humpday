#!/usr/bin/env python3
"""
Fixed optimizer ranking with proper SciPy integration.
"""

import numpy as np
import pandas as pd
import time
import sys
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import PRIMA optimizers
sys.path.append('/Users/petercotton/github/humpday/humpday/optimizers')
from primacube import prima_uobyqa_cube, prima_newuoa_cube, PRIMA_AVAILABLE

class FixedOptimizerRanking:
    """Fixed optimizer ranking with proper SciPy wrappers."""

    def create_test_functions(self):
        """Simpler, well-behaved test functions."""

        functions = {}

        # 1. Simple Sphere
        def sphere(x):
            time.sleep(0.001)
            x = np.array(x)
            # Scale [0,1] to [-2,2]
            scaled_x = 4 * x - 2
            return np.sum(scaled_x**2)
        functions['sphere'] = sphere

        # 2. Rosenbrock
        def rosenbrock(x):
            time.sleep(0.001)
            x = np.array(x)
            if len(x) < 2:
                return 1000.0
            # Scale [0,1] to [-1, 1] for better conditioning
            scaled_x = 2 * x - 1
            result = 0
            for i in range(len(scaled_x)-1):
                result += 100*(scaled_x[i+1] - scaled_x[i]**2)**2 + (1 - scaled_x[i])**2
            return result
        functions['rosenbrock'] = rosenbrock

        # 3. Simple Rastrigin
        def rastrigin(x):
            time.sleep(0.001)
            x = np.array(x)
            # Scale [0,1] to [-2,2] for easier optimization
            scaled_x = 4 * x - 2
            n = len(scaled_x)
            return 10*n + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in scaled_x)
        functions['rastrigin'] = rastrigin

        return functions

    def scipy_optimizer_fixed(self, objective, n_trials, n_dim, method):
        """Fixed SciPy optimizer wrapper."""

        eval_count = [0]

        def counting_wrapper(x):
            eval_count[0] += 1
            # Ensure we're in [0,1] bounds
            x = np.clip(x, 0.0, 1.0)
            try:
                result = objective(x)
                return float(result) if np.isfinite(result) else 1e6
            except:
                return 1e6

        # Multiple random starts for robustness
        best_result = None
        best_value = float('inf')

        max_starts = min(5, n_trials // 10)  # Use up to 5 starts
        evals_per_start = n_trials // max(1, max_starts)

        for start in range(max_starts):
            if eval_count[0] >= n_trials:
                break

            # Random starting point
            x0 = np.random.rand(n_dim)
            remaining_evals = n_trials - eval_count[0]

            try:
                result = minimize(
                    counting_wrapper,
                    x0,
                    method=method,
                    bounds=[(0.001, 0.999)] * n_dim,  # Slightly inside bounds
                    options={'maxfev': min(remaining_evals, evals_per_start)}
                )

                if result.success and result.fun < best_value:
                    best_result = result
                    best_value = result.fun

            except Exception as e:
                continue

        if best_result is not None and best_result.success:
            best_x = np.clip(best_result.x, 0.0, 1.0)
            return best_value, best_x, eval_count[0]
        else:
            # Fallback: return best random sample
            x_random = np.random.rand(n_dim)
            val_random = counting_wrapper(x_random)
            return val_random, x_random, eval_count[0]

    def run_comprehensive_ranking(self):
        """Run comprehensive ranking with fixed optimizers."""

        print("🏁 Fixed Optimizer Ranking")
        print("=" * 35)

        if not PRIMA_AVAILABLE:
            print("❌ PRIMA not available!")
            return

        test_functions = self.create_test_functions()

        optimizers = {
            'PRIMA_UOBYQA': prima_uobyqa_cube,
            'PRIMA_NEWUOA': prima_newuoa_cube,
            'SciPy_Powell': lambda obj, nt, nd, wc=False: self.scipy_optimizer_fixed(obj, nt, nd, 'Powell') + (wc,) if wc else self.scipy_optimizer_fixed(obj, nt, nd, 'Powell')[0],
            'SciPy_NelderMead': lambda obj, nt, nd, wc=False: self.scipy_optimizer_fixed(obj, nt, nd, 'Nelder-Mead') + (wc,) if wc else self.scipy_optimizer_fixed(obj, nt, nd, 'Nelder-Mead')[0],
            'SciPy_BFGS': lambda obj, nt, nd, wc=False: self.scipy_optimizer_fixed(obj, nt, nd, 'L-BFGS-B') + (wc,) if wc else self.scipy_optimizer_fixed(obj, nt, nd, 'L-BFGS-B')[0]
        }

        dimensions = [2, 5, 10]
        n_runs = 15
        n_trials = 60

        print(f"Testing {len(optimizers)} optimizers")
        print(f"Testing {len(test_functions)} functions")
        print(f"Testing {dimensions} dimensions")
        print(f"{n_runs} runs × {n_trials} evaluations each")
        print()

        all_results = []

        for dim in dimensions:
            print(f"\n📊 {dim}D Problems")
            print("-" * 25)

            for func_name, func in test_functions.items():
                print(f"\n{func_name.upper()}:")
                function_results = {}

                for opt_name, opt_func in optimizers.items():
                    print(f"  {opt_name:15}...", end=" ", flush=True)

                    run_results = []
                    successful_runs = []

                    for run in range(n_runs):
                        np.random.seed(run * 123 + hash(f"{opt_name}_{func_name}_{dim}") % 1000)

                        start_time = time.time()

                        try:
                            result = opt_func(func, n_trials, dim, with_count=True)

                            if isinstance(result, tuple) and len(result) >= 3:
                                val, x, evals = result[:3]
                                success = True
                            else:
                                val = float('inf')
                                success = False

                        except Exception as e:
                            val = float('inf')
                            success = False

                        elapsed = time.time() - start_time

                        run_result = {
                            'optimizer': opt_name,
                            'function': func_name,
                            'dimension': dim,
                            'run': run,
                            'value': val,
                            'time': elapsed,
                            'success': success
                        }

                        run_results.append(run_result)
                        all_results.append(run_result)

                        if success and np.isfinite(val):
                            successful_runs.append(run_result)

                    # Summarize performance for this optimizer
                    if successful_runs:
                        values = [r['value'] for r in successful_runs]
                        times = [r['time'] for r in successful_runs]

                        success_rate = len(successful_runs) / n_runs * 100
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        mean_time = np.mean(times)

                        print(f"✓ {success_rate:3.0f}% | {mean_val:8.3f}±{std_val:6.3f} | {mean_time:.3f}s")

                        function_results[opt_name] = {
                            'mean_value': mean_val,
                            'success_rate': success_rate,
                            'std_value': std_val
                        }

                    else:
                        print("❌ Failed")

                # Show best for this function
                if function_results:
                    best_opt = min(function_results.keys(),
                                 key=lambda x: function_results[x]['mean_value'])
                    best_val = function_results[best_opt]['mean_value']
                    print(f"    → Best: {best_opt} ({best_val:.3f})")

        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)

        # Comprehensive analysis
        self.analyze_final_results(df)

        return df

    def analyze_final_results(self, df):
        """Final comprehensive analysis."""

        print("\n" + "="*60)
        print("🏆 FINAL RANKING ANALYSIS")
        print("="*60)

        # Success rates
        print("\n📊 Success Rates:")
        success_rates = df.groupby('optimizer')['success'].mean() * 100
        success_rates = success_rates.sort_values(ascending=False)

        for opt, rate in success_rates.items():
            print(f"  {opt:15}: {rate:5.1f}%")

        # Performance analysis on successful runs
        successful_df = df[df['success'] == True].copy()

        if len(successful_df) == 0:
            print("\n❌ No successful runs to analyze!")
            return

        print(f"\n📈 Performance on Successful Runs:")

        # Normalize scores within each function/dimension for fair comparison
        for func in successful_df['function'].unique():
            for dim in successful_df['dimension'].unique():
                mask = (successful_df['function'] == func) & (successful_df['dimension'] == dim)

                if mask.sum() > 1:
                    values = successful_df.loc[mask, 'value']
                    min_val = values.min()
                    max_val = values.max()

                    if max_val > min_val:
                        successful_df.loc[mask, 'normalized_score'] = (values - min_val) / (max_val - min_val)
                    else:
                        successful_df.loc[mask, 'normalized_score'] = 0.0
                else:
                    successful_df.loc[mask, 'normalized_score'] = 0.0

        # Overall rankings
        print(f"\n🏆 OVERALL OPTIMIZER RANKINGS:")
        print(f"{'Rank':<4} {'Optimizer':<15} {'Score':<8} {'StdDev':<8} {'Tests':<6}")
        print("-" * 50)

        overall_perf = successful_df.groupby('optimizer').agg({
            'normalized_score': ['mean', 'std', 'count'],
            'value': 'mean',
            'time': 'mean'
        }).round(4)

        overall_perf.columns = ['norm_mean', 'norm_std', 'n_tests', 'raw_mean', 'time_mean']
        overall_perf = overall_perf.sort_values('norm_mean')

        for rank, (opt, row) in enumerate(overall_perf.iterrows(), 1):
            print(f"{rank:<4} {opt:<15} {row['norm_mean']:<8.3f} {row['norm_std']:<8.3f} {row['n_tests']:<6.0f}")

        # Statistical significance
        print(f"\n🔬 Statistical Tests (t-test between top performers):")

        top_optimizers = overall_perf.index[:3].tolist()

        for i, opt1 in enumerate(top_optimizers[:-1]):
            for opt2 in top_optimizers[i+1:]:

                opt1_scores = successful_df[successful_df['optimizer'] == opt1]['normalized_score']
                opt2_scores = successful_df[successful_df['optimizer'] == opt2]['normalized_score']

                if len(opt1_scores) >= 5 and len(opt2_scores) >= 5:
                    t_stat, p_val = stats.ttest_ind(opt1_scores, opt2_scores)

                    significance = ""
                    if p_val < 0.001: significance = " ***"
                    elif p_val < 0.01: significance = " **"
                    elif p_val < 0.05: significance = " *"

                    mean1, mean2 = opt1_scores.mean(), opt2_scores.mean()
                    better = opt1 if mean1 < mean2 else opt2

                    print(f"  {opt1} vs {opt2}: p={p_val:.4f}{significance}")
                    if significance:
                        print(f"    → {better} significantly better")

def main():
    """Run the fixed optimizer ranking."""

    print("🚀 Fixed Comprehensive Optimizer Ranking")
    print("=" * 45)

    ranker = FixedOptimizerRanking()

    start_time = time.time()
    results_df = ranker.run_comprehensive_ranking()
    elapsed = time.time() - start_time

    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"fixed_optimizer_ranking_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"💾 Saved: {filename}")

    print(f"\n✨ Comprehensive ranking complete!")

    return results_df

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"❌ Ranking failed: {e}")
        import traceback
        traceback.print_exc()