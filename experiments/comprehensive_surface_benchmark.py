#!/usr/bin/env python3
"""
Comprehensive competitive comparison across diverse optimization surfaces.
Tests all optimizers on realistic, varied landscapes with optima scattered throughout [0,1]^n.
"""

import numpy as np
import pandas as pd
import time
import sys
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our fixed PRIMA optimizers
sys.path.append('/Users/petercotton/github/humpday/humpday/optimizers')
from primacube import prima_uobyqa_cube, prima_newuoa_cube

# Import stochastic surfaces
sys.path.append('/Users/petercotton/github/humpday/humpday/objectives')

class DiverseSurfaceBenchmark:
    """Comprehensive benchmark across diverse optimization surfaces."""

    def create_diverse_surfaces(self, n_dim: int) -> dict:
        """Create diverse optimization surfaces with scattered optima."""

        surfaces = {}

        # 1. Shifted Sphere - optimum anywhere in cube
        def shifted_sphere(shift_point):
            def surface(x):
                time.sleep(0.0005)
                x = np.array(x)
                return np.sum((x - shift_point)**2)
            return surface

        # Create multiple shifted spheres with different optima
        shifts = [
            np.array([0.1, 0.9] if n_dim == 2 else [0.1] + [0.9] * (n_dim-1)),
            np.array([0.8, 0.3] if n_dim == 2 else [0.8] + [0.3] * (n_dim-1)),
            np.array([0.5] * n_dim),  # Center
            np.array([0.05] * n_dim), # Near boundary
            np.array([0.95] * n_dim)  # Near opposite boundary
        ]

        for i, shift in enumerate(shifts):
            surfaces[f'sphere_shift_{i+1}'] = (
                shifted_sphere(shift),
                f"Sphere optimum at {shift[:2]}" + ("..." if n_dim > 2 else "")
            )

        # 2. Rosenbrock variants with different scaling/shifting
        def shifted_rosenbrock(shift, scale):
            def surface(x):
                time.sleep(0.0005)
                x = np.array(x)
                # Shift and scale to move optimum around unit cube
                scaled_x = scale * (x - shift)
                if len(scaled_x) < 2:
                    return 1000.0
                result = 0
                for i in range(len(scaled_x)-1):
                    result += 100*(scaled_x[i+1] - scaled_x[i]**2)**2 + (1 - scaled_x[i])**2
                return result
            return surface

        # Rosenbrock variants
        rosenbrock_configs = [
            ([0.2, 0.7], 2.0),  # Different optima locations
            ([0.6, 0.1], 1.5),
            ([0.9, 0.8], 0.8)
        ]

        for i, (shift_2d, scale) in enumerate(rosenbrock_configs):
            shift = np.array(shift_2d if n_dim == 2 else shift_2d + [0.5] * (n_dim-2))
            surfaces[f'rosenbrock_var_{i+1}'] = (
                shifted_rosenbrock(shift, scale),
                f"Rosenbrock variant {i+1}"
            )

        # 3. Multi-modal surfaces with scattered local minima
        def scattered_multimodal(centers, depths):
            def surface(x):
                time.sleep(0.0005)
                x = np.array(x)
                # Multiple Gaussian wells at different depths
                total = 10.0  # Base level
                for center, depth in zip(centers, depths):
                    dist_sq = np.sum((x - center)**2)
                    total -= depth * np.exp(-50 * dist_sq)  # Sharp wells
                return total
            return surface

        # Scattered multimodal configurations
        if n_dim == 2:
            multimodal_centers = [
                np.array([0.2, 0.3]),
                np.array([0.7, 0.8]),
                np.array([0.9, 0.1]),
                np.array([0.4, 0.9])
            ]
        else:
            # For higher dimensions, create scattered centers
            np.random.seed(12345)  # Reproducible
            multimodal_centers = [np.random.rand(n_dim) for _ in range(4)]

        multimodal_depths = [8.0, 10.0, 6.0, 7.0]  # Different well depths

        surfaces['multimodal_scattered'] = (
            scattered_multimodal(multimodal_centers, multimodal_depths),
            "Scattered multimodal (4 wells)"
        )

        # 4. Noisy surfaces with optima at various locations
        def noisy_surface_generator(base_func, noise_level):
            def surface(x):
                base_val = base_func(x)
                noise = noise_level * base_val * np.random.normal(0, 0.1)
                return max(0.001, base_val + noise)
            return surface

        # Add noise to some base functions
        surfaces['noisy_sphere_corner'] = (
            noisy_surface_generator(shifted_sphere(np.array([0.1] * n_dim)), 0.15),
            "Noisy sphere (corner optimum)"
        )

        # 5. Asymmetric surfaces
        def asymmetric_surface(bias_direction):
            def surface(x):
                time.sleep(0.0005)
                x = np.array(x)
                # Asymmetric penalty - harder to optimize in one direction
                penalty = np.sum(np.maximum(0, x - bias_direction)**3)  # Cubic penalty
                base = np.sum((x - 0.5)**2)  # Base quadratic
                return base + 2 * penalty
            return surface

        surfaces['asymmetric'] = (
            asymmetric_surface(np.array([0.3] * n_dim)),
            "Asymmetric surface"
        )

        return surfaces

    def get_all_optimizers(self):
        """Get all available optimizers for testing."""

        optimizers = {}

        # Fixed PRIMA optimizers
        optimizers['PRIMA_UOBYQA'] = prima_uobyqa_cube
        optimizers['PRIMA_NEWUOA'] = prima_newuoa_cube

        # SciPy optimizers with proper wrappers
        def scipy_wrapper(method_name):
            def optimizer(objective, n_trials, n_dim, with_count=False):
                eval_count = [0]

                def counting_wrapper(x):
                    eval_count[0] += 1
                    return objective(np.clip(x, 0.0, 1.0))

                best_val = float('inf')
                best_x = None
                n_evaluations = 0

                # Multiple random starts for robustness
                n_starts = min(3, max(1, n_trials // 20))
                evals_per_start = n_trials // n_starts

                for start in range(n_starts):
                    if eval_count[0] >= n_trials:
                        break

                    x0 = np.random.rand(n_dim)
                    remaining_evals = n_trials - eval_count[0]

                    try:
                        result = minimize(
                            counting_wrapper,
                            x0,
                            method=method_name,
                            bounds=[(0.001, 0.999)] * n_dim,
                            options={'maxfev': min(remaining_evals, evals_per_start)}
                        )

                        if result.success and result.fun < best_val:
                            best_val = result.fun
                            best_x = np.clip(result.x, 0.0, 1.0)
                            n_evaluations = eval_count[0]

                    except Exception:
                        continue

                if best_x is None:
                    best_x = np.random.rand(n_dim)
                    best_val = counting_wrapper(best_x)
                    n_evaluations = eval_count[0]

                if with_count:
                    return best_val, best_x, n_evaluations
                else:
                    return best_val

            return optimizer

        optimizers['SciPy_BFGS'] = scipy_wrapper('L-BFGS-B')
        optimizers['SciPy_NelderMead'] = scipy_wrapper('Nelder-Mead')
        optimizers['SciPy_Powell'] = scipy_wrapper('Powell')

        # Differential Evolution wrapper
        def diffevol_wrapper(objective, n_trials, n_dim, with_count=False):
            eval_count = [0]

            def counting_wrapper(x):
                eval_count[0] += 1
                return objective(x)

            try:
                result = differential_evolution(
                    counting_wrapper,
                    [(0.001, 0.999)] * n_dim,
                    maxiter=n_trials // (15 * n_dim),  # Reasonable population-based budget
                    seed=np.random.randint(10000),
                    atol=1e-6,
                    tol=1e-6
                )

                best_val = result.fun if result.success else float('inf')
                best_x = result.x if result.success else np.random.rand(n_dim)
                n_evaluations = min(eval_count[0], n_trials)

            except Exception:
                best_x = np.random.rand(n_dim)
                best_val = counting_wrapper(best_x)
                n_evaluations = eval_count[0]

            if with_count:
                return best_val, best_x, n_evaluations
            else:
                return best_val

        optimizers['SciPy_DiffEvol'] = diffevol_wrapper

        return optimizers

    def run_comprehensive_benchmark(self, dimensions=[2, 5], n_runs=20, n_trials=75):
        """Run comprehensive benchmark across diverse surfaces."""

        print("🌍 Comprehensive Diverse Surface Benchmark")
        print("=" * 50)

        optimizers = self.get_all_optimizers()

        print(f"Testing {len(optimizers)} optimizers")
        print(f"Dimensions: {dimensions}")
        print(f"Runs per test: {n_runs}")
        print(f"Evaluation budget: {n_trials}")
        print()

        all_results = []

        for dim in dimensions:
            print(f"\n🏔️  {dim}D Diverse Surfaces")
            print("=" * 35)

            surfaces = self.create_diverse_surfaces(dim)
            print(f"Generated {len(surfaces)} diverse surfaces")

            for surface_name, (surface_func, description) in surfaces.items():
                print(f"\n📊 {surface_name}: {description}")

                for opt_name, opt_func in optimizers.items():
                    print(f"  {opt_name:15}...", end=" ", flush=True)

                    run_results = []

                    for run in range(n_runs):
                        np.random.seed(run * 1000 + hash(f"{opt_name}_{surface_name}_{dim}") % 1000)

                        start_time = time.time()

                        try:
                            result = opt_func(surface_func, n_trials, dim, with_count=True)

                            if isinstance(result, tuple) and len(result) >= 3:
                                val, x, evals = result[:3]
                                success = np.isfinite(val) and val < 1e10
                            else:
                                val = float('inf')
                                success = False
                                evals = 0

                        except Exception as e:
                            val = float('inf')
                            success = False
                            evals = 0

                        elapsed = time.time() - start_time

                        run_result = {
                            'optimizer': opt_name,
                            'surface': surface_name,
                            'dimension': dim,
                            'run': run,
                            'value': val,
                            'success': success,
                            'evaluations': evals,
                            'time': elapsed,
                            'description': description
                        }

                        run_results.append(run_result)
                        all_results.append(run_result)

                    # Summarize this optimizer's performance
                    successful_runs = [r for r in run_results if r['success']]

                    if successful_runs:
                        values = [r['value'] for r in successful_runs]
                        times = [r['time'] for r in successful_runs]
                        evals = [r['evaluations'] for r in successful_runs]

                        success_rate = len(successful_runs) / n_runs * 100
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        mean_evals = np.mean(evals)

                        print(f"✓ {success_rate:3.0f}% | {mean_val:8.4f}±{std_val:6.4f} | {mean_evals:4.0f}ev")

                    else:
                        print("❌ All failed")

        return pd.DataFrame(all_results)

    def analyze_comprehensive_results(self, df):
        """Comprehensive analysis of diverse surface benchmark results."""

        print("\n" + "="*70)
        print("🏆 COMPREHENSIVE DIVERSE SURFACE ANALYSIS")
        print("="*70)

        # Filter successful runs
        successful_df = df[df['success'] == True].copy()

        if len(successful_df) == 0:
            print("❌ No successful runs to analyze!")
            return

        print(f"\n📊 Dataset Overview:")
        print(f"Total test runs: {len(df)}")
        print(f"Successful runs: {len(successful_df)} ({len(successful_df)/len(df)*100:.1f}%)")

        # Success rates by optimizer
        print(f"\n🎯 Success Rates by Optimizer:")
        success_rates = df.groupby('optimizer')['success'].mean() * 100
        success_rates = success_rates.sort_values(ascending=False)

        for opt, rate in success_rates.items():
            reliability = "🟢" if rate >= 90 else "🟡" if rate >= 70 else "🔴"
            print(f"  {reliability} {opt:15}: {rate:5.1f}%")

        # Normalize performance within each surface for fair comparison
        for surface in successful_df['surface'].unique():
            for dim in successful_df['dimension'].unique():
                mask = (successful_df['surface'] == surface) & (successful_df['dimension'] == dim)

                if mask.sum() > 1:
                    values = successful_df.loc[mask, 'value']
                    min_val = values.min()
                    max_val = values.max()

                    if max_val > min_val:
                        successful_df.loc[mask, 'norm_score'] = (values - min_val) / (max_val - min_val)
                    else:
                        successful_df.loc[mask, 'norm_score'] = 0.0
                else:
                    successful_df.loc[mask, 'norm_score'] = 0.0

        # Overall rankings with combined success rate and performance
        print(f"\n🏆 FINAL OPTIMIZER RANKINGS ON DIVERSE SURFACES:")
        print(f"{'Rank':<4} {'Optimizer':<15} {'Score':<8} {'Success%':<8} {'Wins':<5} {'Tests':<6}")
        print("-" * 65)

        overall_stats = successful_df.groupby('optimizer').agg({
            'norm_score': ['mean', 'std', 'count'],
            'value': 'mean'
        }).round(4)

        overall_stats.columns = ['norm_mean', 'norm_std', 'n_tests', 'raw_mean']
        overall_stats['success_rate'] = success_rates

        # Count wins (best performance on each surface)
        wins_count = {}
        for surface in successful_df['surface'].unique():
            surface_data = successful_df[successful_df['surface'] == surface]
            if len(surface_data) > 0:
                surface_performance = surface_data.groupby('optimizer')['value'].mean()
                if len(surface_performance) > 0:
                    winner = surface_performance.idxmin()
                    wins_count[winner] = wins_count.get(winner, 0) + 1

        overall_stats['wins'] = [wins_count.get(opt, 0) for opt in overall_stats.index]

        # Combined ranking score
        overall_stats['combined_score'] = (
            0.4 * overall_stats['norm_mean'] +  # 40% normalized performance
            0.3 * (1 - overall_stats['success_rate']/100) +  # 30% reliability (inverted)
            0.3 * (1 - overall_stats['wins']/len(successful_df['surface'].unique()))  # 30% win rate (inverted)
        )

        overall_stats = overall_stats.sort_values('combined_score')

        for rank, (opt, row) in enumerate(overall_stats.iterrows(), 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{rank:<4} {medal} {opt:<15} {row['norm_mean']:<8.3f} {row['success_rate']:<8.1f} "
                  f"{row['wins']:<5.0f} {row['n_tests']:<6.0f}")

        # Surface-specific winners
        print(f"\n🎯 Winners by Surface Type:")
        for surface in sorted(successful_df['surface'].unique()):
            surface_data = successful_df[successful_df['surface'] == surface]
            surface_performance = surface_data.groupby('optimizer')['value'].agg(['mean', 'count'])

            # Only consider optimizers with reasonable sample size
            reliable = surface_performance[surface_performance['count'] >= 5]
            if len(reliable) > 0:
                winner = reliable['mean'].idxmin()
                best_val = reliable.loc[winner, 'mean']
                print(f"  {surface:<20}: {winner} ({best_val:.4f})")

        return overall_stats

def main():
    """Run comprehensive diverse surface benchmark."""

    print("🚀 Starting Comprehensive Diverse Surface Benchmark")
    print("=" * 60)

    benchmark = DiverseSurfaceBenchmark()

    start_time = time.time()
    results_df = benchmark.run_comprehensive_benchmark(
        dimensions=[2, 5, 10],
        n_runs=15,
        n_trials=80
    )
    total_time = time.time() - start_time

    print(f"\n⏱️  Total benchmark time: {total_time/60:.1f} minutes")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"diverse_surface_benchmark_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"💾 Results saved: {filename}")

    # Comprehensive analysis
    overall_stats = benchmark.analyze_comprehensive_results(results_df)

    print(f"\n✨ Comprehensive diverse surface benchmark complete!")
    print("🏆 Now we know which optimizers truly perform across varied landscapes!")

    return results_df, overall_stats

if __name__ == "__main__":
    try:
        results_df, stats = main()
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()