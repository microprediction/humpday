#!/usr/bin/env python3
"""
Final comprehensive optimizer rankings with working methods.
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution, minimize

# Import PRIMA optimizers
sys.path.append("/Users/petercotton/github/humpday/humpday/optimizers")
from primacube import prima_newuoa_cube, prima_uobyqa_cube


class FinalOptimizerRankings:
    """Final comprehensive optimizer rankings."""

    def create_test_suite(self):
        """Create comprehensive test suite."""

        functions = {}

        # 1. Smooth Unimodal - Sphere
        def sphere(x):
            time.sleep(0.001)  # Realistic computation time
            x = np.array(x)
            scaled_x = 4 * x - 2  # Scale [0,1] to [-2,2]
            return np.sum(scaled_x**2)

        functions["sphere"] = (sphere, "Smooth unimodal")

        # 2. Smooth Valley - Rosenbrock
        def rosenbrock(x):
            time.sleep(0.001)
            x = np.array(x)
            if len(x) < 2:
                return 1000.0
            scaled_x = 2 * x - 1  # Scale to [-1,1]
            result = 0
            for i in range(len(scaled_x) - 1):
                result += (
                    100 * (scaled_x[i + 1] - scaled_x[i] ** 2) ** 2
                    + (1 - scaled_x[i]) ** 2
                )
            return result

        functions["rosenbrock"] = (rosenbrock, "Smooth valley (difficult)")

        # 3. Multimodal - Rastrigin
        def rastrigin(x):
            time.sleep(0.001)
            x = np.array(x)
            scaled_x = 4 * x - 2  # Scale to [-2,2]
            n = len(scaled_x)
            return 10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in scaled_x)

        functions["rastrigin"] = (rastrigin, "Multimodal (many local minima)")

        # 4. Noisy function
        def noisy_sphere(x):
            time.sleep(0.001)
            x = np.array(x)
            scaled_x = 4 * x - 2
            base = np.sum(scaled_x**2)
            noise = 0.1 * np.random.normal(0, base * 0.1)  # Proportional noise
            return max(0.01, base + noise)

        functions["noisy_sphere"] = (noisy_sphere, "Noisy function (robustness test)")

        return functions

    def run_optimizer(self, name, objective, n_trials, n_dim, seed):
        """Run single optimizer with error handling."""

        np.random.seed(seed)
        start_time = time.time()

        try:
            if name == "PRIMA_UOBYQA":
                result = prima_uobyqa_cube(objective, n_trials, n_dim, with_count=True)
                val, x, evals = result
                success = True

            elif name == "PRIMA_NEWUOA":
                result = prima_newuoa_cube(objective, n_trials, n_dim, with_count=True)
                val, x, evals = result
                success = True

            elif name == "SciPy_BFGS":
                x0 = np.random.rand(n_dim)
                result = minimize(
                    objective,
                    x0,
                    method="L-BFGS-B",
                    bounds=[(0.001, 0.999)] * n_dim,
                    options={"maxfev": n_trials},
                )
                val = result.fun if result.success else float("inf")
                evals = result.nfev if hasattr(result, "nfev") else n_trials
                success = result.success

            elif name == "SciPy_NelderMead":
                x0 = np.random.rand(n_dim)
                result = minimize(
                    objective,
                    x0,
                    method="Nelder-Mead",
                    bounds=[(0.001, 0.999)] * n_dim,
                    options={"maxfev": n_trials},
                )
                val = result.fun if result.success else float("inf")
                evals = result.nfev if hasattr(result, "nfev") else n_trials
                success = result.success

            elif name == "SciPy_DiffEvol":
                # Differential Evolution - good global optimizer
                result = differential_evolution(
                    objective,
                    [(0.001, 0.999)] * n_dim,
                    maxiter=n_trials // 10,
                    seed=seed,
                )
                val = result.fun if result.success else float("inf")
                evals = result.nfev if hasattr(result, "nfev") else n_trials
                success = result.success

            else:
                raise ValueError(f"Unknown optimizer: {name}")

        except Exception:
            val = float("inf")
            evals = 0
            success = False

        elapsed_time = time.time() - start_time

        return {
            "optimizer": name,
            "value": val,
            "evaluations": evals,
            "time": elapsed_time,
            "success": success,
        }

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across all optimizers and problems."""

        print("🏆 Final Comprehensive Optimizer Rankings")
        print("=" * 50)

        functions = self.create_test_suite()

        optimizers = [
            "PRIMA_UOBYQA",
            "PRIMA_NEWUOA",
            "SciPy_BFGS",
            "SciPy_NelderMead",
            "SciPy_DiffEvol",
        ]

        dimensions = [2, 5, 10]
        n_runs = 20
        n_trials = 75

        print(f"Testing {len(optimizers)} optimizers")
        print(f"Testing {len(functions)} functions")
        print(f"Testing {len(dimensions)} dimensions")
        print(f"{n_runs} runs × {n_trials} evaluations each")
        print()

        all_results = []

        for dim in dimensions:
            print(f"\n📊 {dim}D Problems")
            print("-" * 30)

            for func_name, (func, desc) in functions.items():
                print(f"\n{func_name.upper()} ({desc}):")

                for opt_name in optimizers:
                    print(f"  {opt_name:15}...", end=" ", flush=True)

                    run_results = []

                    for run in range(n_runs):
                        seed = run * 1000 + hash(f"{opt_name}_{func_name}_{dim}") % 1000

                        result = self.run_optimizer(opt_name, func, n_trials, dim, seed)
                        result.update(
                            {
                                "function": func_name,
                                "dimension": dim,
                                "run": run,
                                "description": desc,
                            }
                        )

                        run_results.append(result)
                        all_results.append(result)

                    # Analyze this optimizer's performance
                    successful_runs = [r for r in run_results if r["success"]]

                    if successful_runs:
                        values = [r["value"] for r in successful_runs]
                        times = [r["time"] for r in successful_runs]
                        evals = [r["evaluations"] for r in successful_runs]

                        success_rate = len(successful_runs) / n_runs * 100
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        mean_time = np.mean(times)
                        mean_evals = np.mean(evals)

                        print(
                            f"✓ {success_rate:3.0f}% | {mean_val:8.3f}±{std_val:6.3f} | {mean_evals:4.1f}ev | {mean_time:.3f}s"
                        )

                    else:
                        print("❌ All failed")

        return pd.DataFrame(all_results)

    def comprehensive_analysis(self, df):
        """Comprehensive analysis with rankings and insights."""

        print("\n" + "=" * 70)
        print("🏆 COMPREHENSIVE ANALYSIS & FINAL RANKINGS")
        print("=" * 70)

        # Overall success rates
        print("\n📊 Success Rates by Optimizer:")
        success_rates = df.groupby("optimizer")["success"].mean() * 100
        success_rates = success_rates.sort_values(ascending=False)

        for opt, rate in success_rates.items():
            reliability = "🟢" if rate >= 90 else "🟡" if rate >= 60 else "🔴"
            print(f"  {reliability} {opt:15}: {rate:5.1f}%")

        # Performance analysis on successful runs
        successful_df = df[df["success"] == True].copy()

        if len(successful_df) == 0:
            print("\n❌ No successful runs to analyze!")
            return

        print(f"\n📈 Performance Analysis ({len(successful_df)} successful runs):")

        # Normalize performance by function and dimension
        for func in successful_df["function"].unique():
            for dim in successful_df["dimension"].unique():
                mask = (successful_df["function"] == func) & (
                    successful_df["dimension"] == dim
                )

                if mask.sum() > 1:
                    values = successful_df.loc[mask, "value"]
                    min_val = values.min()
                    max_val = values.max()

                    if max_val > min_val:
                        # Normalized score: 0 = best, 1 = worst on this problem
                        successful_df.loc[mask, "norm_score"] = (values - min_val) / (
                            max_val - min_val
                        )
                    else:
                        successful_df.loc[mask, "norm_score"] = 0.0

        # Overall rankings
        print("\n🏆 OVERALL OPTIMIZER RANKINGS:")
        print(
            f"{'Rank':<4} {'Optimizer':<15} {'Score':<8} {'StdDev':<8} {'Success%':<8} {'Tests':<6}"
        )
        print("-" * 65)

        # Combine performance and success rate
        overall_stats = (
            successful_df.groupby("optimizer")
            .agg({"norm_score": ["mean", "std", "count"]})
            .round(4)
        )

        overall_stats.columns = ["norm_mean", "norm_std", "n_successful"]

        # Add success rates
        overall_stats["success_rate"] = success_rates

        # Combined score: weighted average of normalized performance and success rate
        overall_stats["combined_score"] = (
            0.7 * overall_stats["norm_mean"]  # 70% performance
            + 0.3
            * (1 - overall_stats["success_rate"] / 100)  # 30% reliability (inverted)
        )

        overall_stats = overall_stats.sort_values("combined_score")

        for rank, (opt, row) in enumerate(overall_stats.iterrows(), 1):
            medal = (
                "🥇"
                if rank == 1
                else "🥈"
                if rank == 2
                else "🥉"
                if rank == 3
                else "  "
            )
            print(
                f"{rank:<4} {medal} {opt:<15} {row['norm_mean']:<8.3f} {row['norm_std']:<8.3f} "
                f"{row['success_rate']:<8.1f} {row['n_successful']:<6.0f}"
            )

        # Function-specific analysis
        print("\n🎯 Best Performer by Problem Type:")

        problem_winners = {}
        for func in successful_df["function"].unique():
            func_data = successful_df[successful_df["function"] == func]
            func_performance = func_data.groupby("optimizer")["value"].agg(
                ["mean", "count"]
            )

            # Only consider optimizers with reasonable sample size
            reliable_performers = func_performance[func_performance["count"] >= 10]

            if len(reliable_performers) > 0:
                best_opt = reliable_performers["mean"].idxmin()
                best_val = reliable_performers.loc[best_opt, "mean"]
                problem_winners[func] = best_opt

                print(f"  {func:<15}: {best_opt} ({best_val:.4f})")

        # Statistical significance testing
        print("\n🔬 Statistical Significance (Wilcoxon rank-sum tests):")

        top_3 = overall_stats.index[:3].tolist()
        significant_differences = []

        for i, opt1 in enumerate(top_3[:-1]):
            for opt2 in top_3[i + 1 :]:
                opt1_scores = successful_df[successful_df["optimizer"] == opt1][
                    "norm_score"
                ]
                opt2_scores = successful_df[successful_df["optimizer"] == opt2][
                    "norm_score"
                ]

                if len(opt1_scores) >= 10 and len(opt2_scores) >= 10:
                    from scipy.stats import ranksums

                    stat, p_val = ranksums(opt1_scores, opt2_scores)

                    significance = ""
                    if p_val < 0.001:
                        significance = " ***"
                    elif p_val < 0.01:
                        significance = " **"
                    elif p_val < 0.05:
                        significance = " *"

                    better = (
                        opt1 if opt1_scores.median() < opt2_scores.median() else opt2
                    )

                    print(f"  {opt1} vs {opt2}: p={p_val:.4f}{significance}")
                    if significance:
                        significant_differences.append(
                            (better, opt2 if better == opt1 else opt1)
                        )

        # Insights and recommendations
        print("\n🧠 Key Insights & Recommendations:")
        print("=" * 40)

        winner = overall_stats.index[0]
        winner_stats = overall_stats.loc[winner]

        print(f"🥇 OVERALL WINNER: {winner}")
        print(f"   → Success Rate: {winner_stats['success_rate']:.1f}%")
        print(f"   → Performance Score: {winner_stats['norm_mean']:.3f}")

        # Categorize optimizers
        reliable_optimizers = overall_stats[
            overall_stats["success_rate"] >= 80
        ].index.tolist()
        high_performance = overall_stats[
            overall_stats["norm_mean"] <= 0.3
        ].index.tolist()

        print("\n🛡️  RELIABLE OPTIMIZERS (≥80% success):")
        for opt in reliable_optimizers:
            print(f"   • {opt}")

        print("\n🚀 HIGH PERFORMANCE (score ≤0.3):")
        for opt in high_performance:
            print(f"   • {opt}")

        # Practical recommendations
        print("\n💡 PRACTICAL RECOMMENDATIONS:")

        if winner in reliable_optimizers and winner in high_performance:
            print(f"   ✨ Use {winner} as primary optimizer - best overall balance")

        if "SciPy_BFGS" in reliable_optimizers:
            print("   🎯 SciPy L-BFGS-B excellent for smooth functions")

        if (
            "PRIMA_NEWUOA" in reliable_optimizers
            or "PRIMA_UOBYQA" in reliable_optimizers
        ):
            print("   🔬 PRIMA methods provide consistent performance")

        if "SciPy_DiffEvol" in reliable_optimizers:
            print("   🌍 Differential Evolution good for global optimization")

        return overall_stats, problem_winners


def main():
    """Run final comprehensive optimizer rankings."""

    print("🚀 Starting Final Comprehensive Optimizer Rankings")
    print("=" * 60)

    ranker = FinalOptimizerRankings()

    start_time = time.time()
    results_df = ranker.run_comprehensive_benchmark()
    total_time = time.time() - start_time

    print(f"\n⏱️  Total benchmark time: {total_time:.1f} seconds")

    # Save raw results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"final_optimizer_rankings_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"💾 Raw results saved: {filename}")

    # Comprehensive analysis
    overall_stats, problem_winners = ranker.comprehensive_analysis(results_df)

    print("\n✨ Final comprehensive rankings complete!")
    print("🎯 Check the analysis above for detailed insights and recommendations.")

    return results_df, overall_stats, problem_winners


if __name__ == "__main__":
    try:
        results_df, stats, winners = main()
    except Exception as e:
        print(f"❌ Rankings failed: {e}")
        import traceback

        traceback.print_exc()
