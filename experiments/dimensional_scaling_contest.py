#!/usr/bin/env python3
"""
Dimensional scaling and convergence speed contest.
Test how optimizers perform at different dimensions and evaluation budgets.
"""

import numpy as np
import pandas as pd
import time
import sys
from scipy.optimize import minimize, differential_evolution
# import matplotlib.pyplot as plt  # Not available

# Import optimizers and surfaces
sys.path.append('/Users/petercotton/github/humpday/humpday/optimizers')
from primacube import prima_uobyqa_cube, prima_newuoa_cube

sys.path.append('/Users/petercotton/github/humpday/humpday/objectives')
from stochastic_surfaces import StochasticSurfaceGenerator

class DimensionalScalingContest:
    """Contest between optimizers across dimensions and evaluation budgets."""

    def __init__(self):
        self.surface_generator = StochasticSurfaceGenerator(seed=42)

    def get_contest_optimizers(self):
        """Get optimizers for the contest."""

        optimizers = {}

        # PRIMA methods
        optimizers['PRIMA_UOBYQA'] = prima_uobyqa_cube
        optimizers['PRIMA_NEWUOA'] = prima_newuoa_cube

        # SciPy with progressive tracking
        def make_progressive_scipy(method_name):
            def optimizer(objective, n_trials, n_dim, with_count=False, track_progress=False):

                if track_progress:
                    # Return progressive results
                    eval_count = [0]
                    eval_history = []

                    def tracking_objective(x):
                        eval_count[0] += 1
                        val = objective(np.clip(x, 0.0, 1.0))
                        eval_history.append((eval_count[0], val))
                        return val

                    x0 = np.random.rand(n_dim)

                    try:
                        result = minimize(
                            tracking_objective,
                            x0,
                            method=method_name,
                            bounds=[(0.001, 0.999)] * n_dim,
                            options={'maxfev': n_trials}
                        )

                        best_val = result.fun if result.success else float('inf')
                        best_x = np.clip(result.x, 0.0, 1.0) if result.success else x0

                        return eval_history, best_val, best_x

                    except:
                        return eval_history, float('inf'), x0

                else:
                    # Standard call
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
                            options={'maxfev': n_trials}
                        )

                        best_val = result.fun if result.success else float('inf')
                        best_x = np.clip(result.x, 0.0, 1.0) if result.success else x0
                        n_evals = eval_count[0]

                        if with_count:
                            return best_val, best_x, n_evals
                        else:
                            return best_val

                    except:
                        if with_count:
                            return float('inf'), x0, eval_count[0]
                        else:
                            return float('inf')

            return optimizer

        optimizers['SciPy_BFGS'] = make_progressive_scipy('L-BFGS-B')
        optimizers['SciPy_Powell'] = make_progressive_scipy('Powell')

        return optimizers

    def create_test_surfaces(self, n_dim):
        """Create simple test surfaces for dimensional scaling."""

        surfaces = {}

        # 1. Simple sphere - scales predictably
        def sphere(x):
            time.sleep(0.0005)  # Realistic delay
            scaled_x = 4 * np.array(x) - 2  # Scale [0,1] to [-2,2]
            return np.sum(scaled_x**2)

        # 2. Rosenbrock - classic high-dimensional test
        def rosenbrock(x):
            time.sleep(0.0005)
            x = np.array(x)
            if len(x) < 2:
                return 1000.0
            scaled_x = 2 * x - 1  # Scale to [-1,1]
            result = 0
            for i in range(len(scaled_x)-1):
                result += 100*(scaled_x[i+1] - scaled_x[i]**2)**2 + (1 - scaled_x[i])**2
            return result

        # 3. Sum of different powers - separable but challenging
        def sum_of_powers(x):
            time.sleep(0.0005)
            x = np.array(x)
            scaled_x = 2 * x - 1  # Scale to [-1,1]
            result = 0
            for i, xi in enumerate(scaled_x):
                power = 2 + i % 4  # Powers 2, 3, 4, 5, cycling
                result += abs(xi) ** power
            return result

        surfaces['sphere'] = (sphere, "Separable quadratic")
        surfaces['rosenbrock'] = (rosenbrock, "Nonseparable valley")
        surfaces['sum_powers'] = (sum_of_powers, "Mixed separable powers")

        return surfaces

    def run_progressive_contest(self, evaluation_checkpoints=[10, 20, 30, 40, 50, 60, 80, 100]):
        """Run contest tracking performance at different evaluation counts."""

        print("🏁 Progressive Evaluation Contest")
        print("=" * 40)

        optimizers = self.get_contest_optimizers()
        dimensions = [2, 5, 10, 15, 20]
        max_evals = max(evaluation_checkpoints)

        print(f"Optimizers: {list(optimizers.keys())}")
        print(f"Dimensions: {dimensions}")
        print(f"Checkpoints: {evaluation_checkpoints}")
        print()

        all_results = []

        for dim in dimensions:
            print(f"\n🔍 {dim}D Problems")
            print("-" * 25)

            surfaces = self.create_test_surfaces(dim)

            for surface_name, (surface_func, description) in surfaces.items():
                print(f"\n{surface_name.upper()} ({description}):")

                for opt_name, opt_func in optimizers.items():
                    print(f"  {opt_name:15}: ", end="", flush=True)

                    # Run multiple trials
                    n_trials = 5
                    checkpoint_results = {cp: [] for cp in evaluation_checkpoints}

                    for trial in range(n_trials):
                        np.random.seed(trial * 100 + dim * 10 + hash(opt_name) % 100)

                        start_time = time.time()

                        try:
                            # PRIMA methods: sample at checkpoints
                            if opt_name.startswith('PRIMA'):
                                for checkpoint in evaluation_checkpoints:
                                    result = opt_func(surface_func, checkpoint, dim, with_count=True)
                                    if isinstance(result, tuple) and len(result) >= 3:
                                        val, x, evals = result[:3]
                                        checkpoint_results[checkpoint].append(val if np.isfinite(val) else 1e6)
                                    else:
                                        checkpoint_results[checkpoint].append(1e6)

                            # SciPy methods: track progressive performance
                            else:
                                eval_history, final_val, final_x = opt_func(
                                    surface_func, max_evals, dim, track_progress=True
                                )

                                # Extract values at checkpoints
                                for checkpoint in evaluation_checkpoints:
                                    # Find best value up to this checkpoint
                                    relevant_evals = [val for count, val in eval_history if count <= checkpoint]
                                    if relevant_evals:
                                        best_so_far = min(relevant_evals)
                                        checkpoint_results[checkpoint].append(best_so_far if np.isfinite(best_so_far) else 1e6)
                                    else:
                                        checkpoint_results[checkpoint].append(1e6)

                        except Exception as e:
                            # Fill with large values on failure
                            for checkpoint in evaluation_checkpoints:
                                checkpoint_results[checkpoint].append(1e6)

                        elapsed = time.time() - start_time

                    # Summarize results for this optimizer
                    avg_results = {}
                    for checkpoint in evaluation_checkpoints:
                        valid_results = [r for r in checkpoint_results[checkpoint] if r < 1e5]
                        if valid_results:
                            avg_results[checkpoint] = np.mean(valid_results)
                        else:
                            avg_results[checkpoint] = float('inf')

                    # Print summary
                    early_perf = avg_results[evaluation_checkpoints[1]]  # 20 evals typically
                    final_perf = avg_results[evaluation_checkpoints[-1]]  # Final result
                    print(f"Early: {early_perf:8.3f}, Final: {final_perf:8.3f}")

                    # Store detailed results
                    for checkpoint in evaluation_checkpoints:
                        all_results.append({
                            'optimizer': opt_name,
                            'dimension': dim,
                            'surface': surface_name,
                            'checkpoint': checkpoint,
                            'avg_value': avg_results[checkpoint],
                            'n_trials': len([r for r in checkpoint_results[checkpoint] if r < 1e5])
                        })

        return pd.DataFrame(all_results)

    def analyze_contest_results(self, df):
        """Analyze dimensional scaling and convergence speed results."""

        print("\n" + "="*70)
        print("🏆 DIMENSIONAL SCALING & CONVERGENCE ANALYSIS")
        print("="*70)

        successful_df = df[df['avg_value'] < 1e5].copy()

        # 1. Dimensional scaling analysis
        print(f"\n📈 DIMENSIONAL SCALING PERFORMANCE:")
        print("(Average across all surfaces and checkpoints)")

        dim_scaling = successful_df.groupby(['optimizer', 'dimension'])['avg_value'].mean().unstack()
        dim_scaling = dim_scaling.fillna(float('inf'))

        print(f"\n{'Optimizer':<15}", end="")
        for dim in sorted(dim_scaling.columns):
            print(f"{dim:>8}D", end="")
        print()
        print("-" * 70)

        for opt in dim_scaling.index:
            print(f"{opt:<15}", end="")
            for dim in sorted(dim_scaling.columns):
                val = dim_scaling.loc[opt, dim]
                if val == float('inf'):
                    print(f"{'FAIL':>8}", end="")
                else:
                    print(f"{val:>8.3f}", end="")
            print()

        # 2. Convergence speed analysis
        print(f"\n⚡ CONVERGENCE SPEED ANALYSIS:")
        print("(Who's best at different evaluation budgets)")

        checkpoints = sorted(df['checkpoint'].unique())
        convergence_analysis = successful_df.groupby(['optimizer', 'checkpoint'])['avg_value'].mean().unstack()
        convergence_analysis = convergence_analysis.fillna(float('inf'))

        print(f"\n{'Optimizer':<15}", end="")
        for cp in checkpoints[:6]:  # Show first 6 checkpoints
            print(f"{cp:>6}ev", end="")
        print()
        print("-" * 70)

        for opt in convergence_analysis.index:
            print(f"{opt:<15}", end="")
            for cp in checkpoints[:6]:
                if cp in convergence_analysis.columns:
                    val = convergence_analysis.loc[opt, cp]
                    if val == float('inf'):
                        print(f"{'FAIL':>6}", end="")
                    else:
                        print(f"{val:>6.2f}", end="")
                else:
                    print(f"{'--':>6}", end="")
            print()

        # 3. Contest winners by category
        print(f"\n🏆 CONTEST WINNERS BY CATEGORY:")

        # Early performance (20 evals)
        early_checkpoint = 20
        if early_checkpoint in convergence_analysis.columns:
            early_winner = convergence_analysis[early_checkpoint].idxmin()
            early_score = convergence_analysis.loc[early_winner, early_checkpoint]
            print(f"🏃 EARLY LEADER (20 evals): {early_winner} ({early_score:.3f})")

        # Final performance (100 evals)
        final_checkpoint = max(checkpoints)
        if final_checkpoint in convergence_analysis.columns:
            final_winner = convergence_analysis[final_checkpoint].idxmin()
            final_score = convergence_analysis.loc[final_winner, final_checkpoint]
            print(f"🏁 FINAL WINNER ({final_checkpoint} evals): {final_winner} ({final_score:.3f})")

        # High-dimensional champion (average of 15D, 20D)
        high_dim_cols = [col for col in dim_scaling.columns if col >= 15]
        if high_dim_cols:
            high_dim_avg = dim_scaling[high_dim_cols].mean(axis=1)
            high_dim_winner = high_dim_avg.idxmin()
            high_dim_score = high_dim_avg[high_dim_winner]
            print(f"🏔️  HIGH-DIM CHAMPION (15D+): {high_dim_winner} ({high_dim_score:.3f})")

        # 4. Efficiency analysis (performance per evaluation)
        print(f"\n⚡ EFFICIENCY RANKINGS:")
        print("(Best performance per evaluation used)")

        # Calculate efficiency metric: performance improvement per evaluation
        efficiency_scores = {}
        for opt in convergence_analysis.index:
            early_val = convergence_analysis.loc[opt, 20] if 20 in convergence_analysis.columns else float('inf')
            final_val = convergence_analysis.loc[opt, final_checkpoint]

            if early_val != float('inf') and final_val != float('inf'):
                improvement = max(0, early_val - final_val)  # How much better we got
                evaluations_used = final_checkpoint - 20
                efficiency = improvement / max(1, evaluations_used) if evaluations_used > 0 else 0
                efficiency_scores[opt] = efficiency

        if efficiency_scores:
            sorted_efficiency = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (opt, score) in enumerate(sorted_efficiency, 1):
                print(f"  {rank}. {opt:15}: {score:.5f} improvement/eval")

        return dim_scaling, convergence_analysis

def main():
    """Run dimensional scaling and convergence contest."""

    print("🚀 Dimensional Scaling & Convergence Speed Contest")
    print("=" * 55)

    contest = DimensionalScalingContest()

    start_time = time.time()
    results_df = contest.run_progressive_contest()
    elapsed = time.time() - start_time

    print(f"\n⏱️  Contest completed in {elapsed/60:.1f} minutes")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"dimensional_scaling_contest_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"💾 Results saved: {filename}")

    # Analysis
    dim_scaling, convergence = contest.analyze_contest_results(results_df)

    print(f"\n✨ CONTEST COMPLETE!")
    print("🏆 Now we know who wins at different dimensions and speeds!")

    return results_df, dim_scaling, convergence

if __name__ == "__main__":
    try:
        results, scaling, convergence = main()
    except Exception as e:
        print(f"❌ Contest failed: {e}")
        import traceback
        traceback.print_exc()