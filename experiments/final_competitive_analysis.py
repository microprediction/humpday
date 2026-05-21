#!/usr/bin/env python3
"""
Final competitive analysis using our existing stochastic surface infrastructure.
Now that PRIMA is fixed, let's see the true rankings!
"""

import numpy as np
import pandas as pd
import time
import sys
from scipy.optimize import minimize, differential_evolution

# Import our existing stochastic surfaces
sys.path.append('/Users/petercotton/github/humpday/humpday/objectives')
from stochastic_surfaces import StochasticSurfaceGenerator

# Import fixed PRIMA optimizers
sys.path.append('/Users/petercotton/github/humpday/humpday/optimizers')
from primacube import prima_uobyqa_cube, prima_newuoa_cube

class FinalCompetitiveAnalysis:
    """Final competitive analysis using existing stochastic surfaces."""

    def __init__(self):
        self.surface_generator = StochasticSurfaceGenerator(seed=12345)

    def get_optimizers(self):
        """Get all optimizers for competitive analysis."""

        optimizers = {}

        # Our fixed PRIMA methods
        optimizers['PRIMA_UOBYQA'] = prima_uobyqa_cube
        optimizers['PRIMA_NEWUOA'] = prima_newuoa_cube

        # SciPy methods with robust wrappers
        def make_scipy_optimizer(method_name):
            def optimizer(objective, n_trials, n_dim, with_count=False):
                best_val = float('inf')
                best_x = np.random.rand(n_dim)
                total_evals = 0

                # Multiple starts for robustness
                n_starts = min(3, n_trials // 15)
                evals_per_start = n_trials // n_starts

                for start in range(n_starts):
                    if total_evals >= n_trials:
                        break

                    eval_count = [0]

                    def counting_objective(x):
                        eval_count[0] += 1
                        return objective(np.clip(x, 0.0, 1.0))

                    x0 = np.random.rand(n_dim)

                    try:
                        result = minimize(
                            counting_objective,
                            x0,
                            method=method_name,
                            bounds=[(0.001, 0.999)] * n_dim,
                            options={'maxfev': evals_per_start}
                        )

                        total_evals += eval_count[0]

                        if result.success and result.fun < best_val:
                            best_val = result.fun
                            best_x = np.clip(result.x, 0.0, 1.0)

                    except:
                        total_evals += evals_per_start

                if with_count:
                    return best_val, best_x, min(total_evals, n_trials)
                else:
                    return best_val

            return optimizer

        optimizers['SciPy_BFGS'] = make_scipy_optimizer('L-BFGS-B')
        optimizers['SciPy_NelderMead'] = make_scipy_optimizer('Nelder-Mead')

        return optimizers

    def run_stochastic_benchmark(self, n_surfaces=20, n_runs=10, n_trials=60):
        """Run benchmark on stochastic surfaces."""

        print("🎲 Final Competitive Analysis - Stochastic Surfaces")
        print("=" * 60)

        optimizers = self.get_optimizers()
        dimensions = [2, 5]

        print(f"Optimizers: {list(optimizers.keys())}")
        print(f"Dimensions: {dimensions}")
        print(f"Surfaces per dimension: {n_surfaces}")
        print(f"Runs per surface: {n_runs}")
        print(f"Evaluation budget: {n_trials}")
        print()

        all_results = []

        for dim in dimensions:
            print(f"\n🏔️  {dim}D Stochastic Surfaces")
            print("-" * 35)

            # Get random function suite for this dimension
            function_suite = self.surface_generator.get_random_function_suite(n_surfaces)

            for surface_idx, (func_name, surface_func) in enumerate(function_suite.items()):
                print(f"Surface {surface_idx+1:2d}: {func_name:<15} ", end="")

                surface_results = []

                for opt_name, opt_func in optimizers.items():
                    run_values = []
                    run_times = []
                    run_evals = []

                    for run in range(n_runs):
                        seed = (surface_idx * 1000) + (run * 100) + hash(opt_name) % 100
                        np.random.seed(seed)

                        start_time = time.time()

                        try:
                            result = opt_func(surface_func, n_trials, dim, with_count=True)

                            if isinstance(result, tuple) and len(result) >= 3:
                                val, x, evals = result[:3]
                                success = np.isfinite(val) and val < 1e8
                            else:
                                val = float('inf')
                                success = False
                                evals = 0

                        except:
                            val = float('inf')
                            success = False
                            evals = 0

                        elapsed = time.time() - start_time

                        if success:
                            run_values.append(val)
                            run_times.append(elapsed)
                            run_evals.append(evals)

                        # Store detailed result
                        all_results.append({
                            'optimizer': opt_name,
                            'dimension': dim,
                            'surface': surface_idx,
                            'surface_type': func_name,
                            'run': run,
                            'value': val,
                            'success': success,
                            'evaluations': evals,
                            'time': elapsed
                        })

                    # Summarize this optimizer's performance on this surface
                    if run_values:
                        success_rate = len(run_values) / n_runs * 100
                        mean_val = np.mean(run_values)
                        surface_results.append((opt_name, mean_val, success_rate))

                # Show best performer for this surface
                if surface_results:
                    surface_results.sort(key=lambda x: x[1])  # Sort by mean value
                    best_opt, best_val, best_rate = surface_results[0]
                    print(f"Best: {best_opt} ({best_val:.4f})")
                else:
                    print("All optimizers failed")

        return pd.DataFrame(all_results)

    def analyze_final_results(self, df):
        """Final competitive analysis with definitive rankings."""

        print("\n" + "="*70)
        print("🏆 FINAL COMPETITIVE RANKINGS - STOCHASTIC SURFACES")
        print("="*70)

        successful_df = df[df['success'] == True].copy()

        if len(successful_df) == 0:
            print("❌ No successful runs!")
            return

        print(f"\n📊 Overall Performance Summary:")
        print(f"Total tests: {len(df)}")
        print(f"Successful: {len(successful_df)} ({len(successful_df)/len(df)*100:.1f}%)")

        # Success rates
        print(f"\n🎯 Reliability Rankings (Success Rate):")
        success_rates = df.groupby('optimizer')['success'].mean() * 100
        success_rates = success_rates.sort_values(ascending=False)

        for rank, (opt, rate) in enumerate(success_rates.items(), 1):
            status = "🟢" if rate >= 85 else "🟡" if rate >= 60 else "🔴"
            print(f"  {rank}. {status} {opt:15}: {rate:5.1f}%")

        # Performance on successful runs
        print(f"\n🚀 Performance Rankings (Solution Quality):")

        # Normalize scores within each surface for fair comparison
        for surface_idx in successful_df['surface'].unique():
            for dim in successful_df['dimension'].unique():
                mask = (successful_df['surface'] == surface_idx) & (successful_df['dimension'] == dim)

                if mask.sum() > 1:
                    values = successful_df.loc[mask, 'value']
                    min_val = values.min()
                    max_val = values.max()

                    if max_val > min_val:
                        successful_df.loc[mask, 'normalized_score'] = (values - min_val) / (max_val - min_val)
                    else:
                        successful_df.loc[mask, 'normalized_score'] = 0.0

        performance_stats = successful_df.groupby('optimizer').agg({
            'normalized_score': ['mean', 'std', 'count'],
            'value': 'mean',
            'evaluations': 'mean'
        }).round(4)

        performance_stats.columns = ['norm_mean', 'norm_std', 'n_tests', 'raw_mean', 'avg_evals']
        performance_stats = performance_stats.sort_values('norm_mean')

        for rank, (opt, row) in enumerate(performance_stats.iterrows(), 1):
            print(f"  {rank}. {opt:15}: {row['norm_mean']:.3f} ± {row['norm_std']:.3f} "
                  f"({row['avg_evals']:.0f} evals avg)")

        # Combined final ranking
        print(f"\n🏆 FINAL OVERALL RANKINGS:")
        print("(Combines reliability + performance)")
        print(f"{'Rank':<4} {'Optimizer':<15} {'Score':<8} {'Success%':<8} {'Performance':<11}")
        print("-" * 60)

        # Combine reliability and performance
        final_scores = {}
        for opt in success_rates.index:
            reliability_score = success_rates[opt] / 100  # 0-1 scale
            if opt in performance_stats.index:
                performance_score = 1 - performance_stats.loc[opt, 'norm_mean']  # Invert: higher is better
            else:
                performance_score = 0

            # Weighted combination: 40% reliability, 60% performance
            final_scores[opt] = 0.4 * reliability_score + 0.6 * performance_score

        final_rankings = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (opt, score) in enumerate(final_rankings, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            success_pct = success_rates[opt]
            perf_score = performance_stats.loc[opt, 'norm_mean'] if opt in performance_stats.index else 1.0

            print(f"{rank:<4} {medal} {opt:<15} {score:<8.3f} {success_pct:<8.1f} {perf_score:<11.3f}")

        # Statistical significance testing
        print(f"\n🔬 Statistical Significance (Wilcoxon tests):")
        from scipy.stats import ranksums

        top_3 = [opt for opt, _ in final_rankings[:3]]
        for i, opt1 in enumerate(top_3[:-1]):
            for opt2 in top_3[i+1:]:
                if opt1 in performance_stats.index and opt2 in performance_stats.index:
                    opt1_scores = successful_df[successful_df['optimizer'] == opt1]['normalized_score']
                    opt2_scores = successful_df[successful_df['optimizer'] == opt2]['normalized_score']

                    if len(opt1_scores) >= 10 and len(opt2_scores) >= 10:
                        stat, p_val = ranksums(opt1_scores, opt2_scores)
                        significance = " ***" if p_val < 0.001 else " **" if p_val < 0.01 else " *" if p_val < 0.05 else ""
                        print(f"  {opt1} vs {opt2}: p={p_val:.4f}{significance}")

        print(f"\n🎯 Key Insights:")
        winner = final_rankings[0][0]
        winner_success = success_rates[winner]
        print(f"🥇 OVERALL WINNER: {winner}")
        print(f"   → {winner_success:.1f}% success rate")

        if winner.startswith('PRIMA'):
            print("   → PRIMA methods now competitive after bug fix!")
        elif winner.startswith('SciPy'):
            print("   → Established SciPy methods remain strong")

        return final_rankings

def main():
    """Run final competitive analysis."""

    print("🚀 Final Competitive Analysis - Now with Fixed PRIMA!")
    print("=" * 55)

    analyzer = FinalCompetitiveAnalysis()

    start_time = time.time()
    results_df = analyzer.run_stochastic_benchmark(
        n_surfaces=15,
        n_runs=8,
        n_trials=60
    )
    elapsed = time.time() - start_time

    print(f"\n⏱️  Analysis completed in {elapsed/60:.1f} minutes")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"final_competitive_analysis_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"💾 Results saved: {filename}")

    # Final analysis
    final_rankings = analyzer.analyze_final_results(results_df)

    print(f"\n✨ COMPETITIVE ANALYSIS COMPLETE!")
    print("🏆 We now have definitive rankings across diverse stochastic surfaces!")

    return results_df, final_rankings

if __name__ == "__main__":
    try:
        results, rankings = main()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()