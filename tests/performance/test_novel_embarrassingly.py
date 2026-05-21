#!/usr/bin/env python3
"""
Thorough testing of the NOVEL embarrassingly techniques:
- Shy: Adaptive expensive function evaluation
- Underpromoted: Plateau-seeking modification

Skip trivial caching - focus on academically interesting ideas.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
from embarrassingly.shy import Shy
from embarrassingly.underpromoted import Underpromoted2d
from typing import Dict, List, Callable, Tuple
import warnings
warnings.filterwarnings('ignore')

class NovelTechniqueAnalysis:
    """Rigorous testing of novel embarrassingly techniques."""

    def __init__(self, n_runs=25):
        self.n_runs = n_runs

    def create_expensive_functions(self) -> Dict[str, Callable]:
        """Create functions with realistic variable computation costs."""
        functions = {}

        # 1. Region-dependent expensive sphere
        def expensive_sphere(x):
            x = np.array(x)
            # More expensive near boundaries (realistic for many problems)
            boundary_penalty = min(np.min(x), np.min(1-x))
            compute_time = 0.001 + 0.02 * np.exp(-10 * boundary_penalty)  # Exponential cost near boundaries
            time.sleep(compute_time)

            scaled_x = 10 * x - 5
            return np.sum(scaled_x**2)

        functions['expensive_sphere'] = expensive_sphere

        # 2. Distance-dependent expensive Rastrigin
        def expensive_rastrigin(x):
            x = np.array(x)
            center_dist = np.linalg.norm(x - 0.5)
            compute_time = 0.001 + 0.015 * center_dist  # More expensive far from center
            time.sleep(compute_time)

            scaled_x = 10.24 * x - 5.12
            n = len(scaled_x)
            return 10*n + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in scaled_x)

        functions['expensive_rastrigin'] = expensive_rastrigin

        # 3. Gradient-dependent expensive Rosenbrock
        def expensive_rosenbrock(x):
            x = np.array(x)
            # More expensive in high-gradient regions (realistic for finite-difference scenarios)
            scaled_x = 4.096 * x - 2.048

            # Estimate gradient magnitude (expensive regions)
            if len(scaled_x) >= 2:
                grad_approx = abs(400*scaled_x[0]*(scaled_x[0]**2 - scaled_x[1]) + 2*(scaled_x[0] - 1))
                compute_time = 0.001 + 0.01 * min(grad_approx / 1000, 0.5)  # Cap at reasonable level
            else:
                compute_time = 0.001

            time.sleep(compute_time)

            return sum(100*(scaled_x[i+1] - scaled_x[i]**2)**2 + (1 - scaled_x[i])**2
                      for i in range(len(scaled_x)-1))

        functions['expensive_rosenbrock'] = expensive_rosenbrock

        return functions

    def create_plateau_landscapes(self) -> Dict[str, Callable]:
        """Create landscapes with plateau regions for testing Underpromoted."""
        landscapes = {}

        # 1. Single broad plateau
        def broad_plateau(x):
            x = np.array(x)
            target = np.array([0.3, 0.7])  # Plateau center
            dist = np.linalg.norm(x - target)

            if dist < 0.2:  # Broad plateau
                return 0.1 + 0.01 * dist  # Very flat
            else:
                return 0.1 + 2.0 * (dist - 0.2)**2  # Steep outside

        landscapes['broad_plateau'] = broad_plateau

        # 2. Multiple plateaus vs sharp minimum
        def plateau_vs_sharp(x):
            x = np.array(x)

            # Plateau 1: Good, stable region
            p1_dist = np.linalg.norm(x - np.array([0.2, 0.8]))
            if p1_dist < 0.12:
                return 0.05 + 0.005 * p1_dist

            # Plateau 2: Decent region
            p2_dist = np.linalg.norm(x - np.array([0.8, 0.2]))
            if p2_dist < 0.10:
                return 0.12 + 0.005 * p2_dist

            # Sharp global minimum (unstable/fragile)
            sharp_dist = np.linalg.norm(x - np.array([0.5, 0.5]))
            if sharp_dist < 0.03:
                return 0.001 + 1000 * sharp_dist**4  # Very sharp

            # Background
            return 1.0 + np.sum((x - 0.5)**2)

        landscapes['plateau_vs_sharp'] = plateau_vs_sharp

        # 3. Valley with plateau floor
        def valley_plateau(x):
            x = np.array(x)

            # Main valley direction
            valley_center = 0.3 + 0.4 * x[0]  # Diagonal valley
            cross_valley_dist = abs(x[1] - valley_center)

            if cross_valley_dist < 0.15:  # In valley
                along_valley = abs(x[0] - 0.6)  # Distance along valley
                if along_valley < 0.2:  # Plateau region in valley
                    return 0.05 + 0.01 * along_valley
                else:
                    return 0.05 + 0.5 * (along_valley - 0.2)**2
            else:
                return 0.05 + 5.0 * (cross_valley_dist - 0.15)**2

        landscapes['valley_plateau'] = valley_plateau

        return landscapes

    def run_optimization_battery(self, optimizer_name: str, objective_func: Callable,
                               function_name: str) -> List[Dict]:
        """Run battery of optimizations with different starting points."""
        results = []

        # Use diverse starting points
        start_points = [
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],  # Corners
            [0.5, 0.5],  # Center
            [0.3, 0.7], [0.7, 0.3],  # Off-center
        ]

        # Add random starts
        np.random.seed(42)
        random_starts = [np.random.rand(2).tolist() for _ in range(self.n_runs - len(start_points))]
        start_points.extend(random_starts)

        for i, x0 in enumerate(start_points[:self.n_runs]):
            eval_count = [0]
            eval_times = []

            def tracking_wrapper(x):
                eval_count[0] += 1
                start = time.time()
                result = objective_func(x)
                eval_times.append(time.time() - start)
                return result

            start_time = time.time()

            try:
                result = minimize(
                    tracking_wrapper, x0, method=optimizer_name,
                    bounds=[(0,1), (0,1)], options={'maxfev': 40}
                )

                total_time = time.time() - start_time
                avg_eval_time = np.mean(eval_times) if eval_times else 0

                results.append({
                    'run': i,
                    'success': result.success,
                    'final_value': float(result.fun) if result.success else float('inf'),
                    'total_time': total_time,
                    'evaluations': eval_count[0],
                    'avg_eval_time': avg_eval_time,
                    'final_x': result.x.tolist() if result.success else x0,
                    'start_x': x0
                })

            except Exception as e:
                results.append({
                    'run': i,
                    'success': False,
                    'final_value': float('inf'),
                    'total_time': 0,
                    'evaluations': 0,
                    'avg_eval_time': 0,
                    'final_x': x0,
                    'start_x': x0,
                    'error': str(e)
                })

        return results

    def test_shy_adaptive_evaluation(self):
        """Comprehensive test of Shy's adaptive evaluation strategy."""
        print("🔍 TESTING SHY ADAPTIVE EVALUATION")
        print("=" * 50)
        print("Hypothesis: Shy should save time on expensive functions while maintaining solution quality")

        expensive_functions = self.create_expensive_functions()
        optimizers = ['Powell', 'Nelder-Mead']
        bounds = [[0, 1], [0, 1]]

        all_results = []

        for func_name, func in expensive_functions.items():
            print(f"\n📊 Testing {func_name}:")

            # Test different Shy parameters
            shy_configs = [
                {'t_unit': 0.01, 'd_unit': 0.05, 'name': 'conservative'},
                {'t_unit': 0.005, 'd_unit': 0.1, 'name': 'moderate'},
                {'t_unit': 0.002, 'd_unit': 0.2, 'name': 'aggressive'}
            ]

            for optimizer_name in optimizers:
                print(f"\n  {optimizer_name} optimizer:")

                # Standard (baseline)
                print("    Standard version... ", end="", flush=True)
                standard_results = self.run_optimization_battery(optimizer_name, func, func_name)
                successful_standard = [r for r in standard_results if r['success']]

                if len(successful_standard) < 5:
                    print(f"❌ Too few successful runs ({len(successful_standard)})")
                    continue

                std_avg_value = np.mean([r['final_value'] for r in successful_standard])
                std_avg_time = np.mean([r['total_time'] for r in successful_standard])
                std_avg_eval_time = np.mean([r['avg_eval_time'] for r in successful_standard])

                print(f"✓ {len(successful_standard)}/{len(standard_results)} successful")
                print(f"      Value: {std_avg_value:.4f}, Total time: {std_avg_time:.3f}s, Eval time: {std_avg_eval_time:.4f}s")

                # Test each Shy configuration
                for shy_config in shy_configs:
                    shy_func = Shy(func, bounds=bounds, **{k:v for k,v in shy_config.items() if k != 'name'})

                    print(f"    Shy {shy_config['name']}...    ", end="", flush=True)
                    shy_results = self.run_optimization_battery(optimizer_name, shy_func, f"{func_name}_shy")
                    successful_shy = [r for r in shy_results if r['success']]

                    if len(successful_shy) < 5:
                        print(f"❌ Too few successful runs ({len(successful_shy)})")
                        continue

                    shy_avg_value = np.mean([r['final_value'] for r in successful_shy])
                    shy_avg_time = np.mean([r['total_time'] for r in successful_shy])
                    shy_avg_eval_time = np.mean([r['avg_eval_time'] for r in successful_shy])

                    # Statistical tests
                    value_t, value_p = stats.ttest_ind(
                        [r['final_value'] for r in successful_standard],
                        [r['final_value'] for r in successful_shy]
                    )

                    time_t, time_p = stats.ttest_ind(
                        [r['total_time'] for r in successful_standard],
                        [r['total_time'] for r in successful_shy]
                    )

                    value_improvement = (std_avg_value - shy_avg_value) / std_avg_value * 100
                    time_improvement = (std_avg_time - shy_avg_time) / std_avg_time * 100

                    print(f"✓ {len(successful_shy)}/{len(shy_results)} successful")
                    print(f"      Value: {shy_avg_value:.4f} ({value_improvement:+.1f}%, p={value_p:.3f})")
                    print(f"      Time:  {shy_avg_time:.3f}s ({time_improvement:+.1f}%, p={time_p:.3f})")

                    # Record results
                    all_results.append({
                        'function': func_name,
                        'optimizer': optimizer_name,
                        'shy_config': shy_config['name'],
                        'standard_value': std_avg_value,
                        'shy_value': shy_avg_value,
                        'value_improvement': value_improvement,
                        'value_p_value': value_p,
                        'standard_time': std_avg_time,
                        'shy_time': shy_avg_time,
                        'time_improvement': time_improvement,
                        'time_p_value': time_p,
                        'n_standard': len(successful_standard),
                        'n_shy': len(successful_shy)
                    })

        # Analysis
        df = pd.DataFrame(all_results)
        if len(df) > 0:
            print(f"\n📈 SHY EVALUATION ANALYSIS:")
            print(f"Total comparisons: {len(df)}")

            # Significant improvements
            sig_value_improvements = df[(df['value_p_value'] < 0.05) & (df['value_improvement'] > 0)]
            sig_time_improvements = df[(df['time_p_value'] < 0.05) & (df['time_improvement'] > 0)]

            print(f"Significant value improvements: {len(sig_value_improvements)}/{len(df)}")
            print(f"Significant time improvements: {len(sig_time_improvements)}/{len(df)}")

            if len(sig_value_improvements) > 0:
                print(f"Best value improvement: {sig_value_improvements['value_improvement'].max():.1f}%")
            if len(sig_time_improvements) > 0:
                print(f"Best time improvement: {sig_time_improvements['time_improvement'].max():.1f}%")

            # Overall verdict on Shy
            success_rate = (len(sig_value_improvements) + len(sig_time_improvements)) / len(df)
            if success_rate > 0.3:
                print("🎯 VERDICT: Shy shows promising results!")
            elif success_rate > 0.1:
                print("🤔 VERDICT: Shy shows occasional benefits")
            else:
                print("❌ VERDICT: Shy doesn't consistently help")

        return df

    def test_underpromoted_plateau_finding(self):
        """Test whether Underpromoted helps find robust plateau solutions."""
        print(f"\n🏔️ TESTING UNDERPROMOTED PLATEAU FINDING")
        print("=" * 55)
        print("Hypothesis: Underpromoted should prefer stable plateau regions over sharp minima")

        plateau_landscapes = self.create_plateau_landscapes()
        optimizers = ['Powell', 'Nelder-Mead']
        bounds = [[0, 1], [0, 1]]

        all_results = []

        for landscape_name, landscape_func in plateau_landscapes.items():
            print(f"\n📊 Testing {landscape_name}:")

            # Test different radius parameters
            radius_configs = [0.03, 0.06, 0.12]  # Small, medium, large influence

            for optimizer_name in optimizers:
                print(f"\n  {optimizer_name} optimizer:")

                # Standard optimization
                print("    Standard version... ", end="", flush=True)
                standard_results = self.run_optimization_battery(optimizer_name, landscape_func, landscape_name)
                successful_standard = [r for r in standard_results if r['success']]

                if len(successful_standard) < 5:
                    print(f"❌ Too few successful runs")
                    continue

                std_avg_value = np.mean([r['final_value'] for r in successful_standard])

                print(f"✓ {len(successful_standard)}/{len(standard_results)} successful")
                print(f"      Average value: {std_avg_value:.4f}")

                # Analyze solution locations for standard
                standard_locations = [r['final_x'] for r in successful_standard]

                # Test each radius configuration
                for radius in radius_configs:
                    plateau_func = Underpromoted2d(landscape_func, bounds=bounds, radius=radius)

                    print(f"    Plateau radius {radius:.2f}... ", end="", flush=True)
                    plateau_results = self.run_optimization_battery(optimizer_name, plateau_func, f"{landscape_name}_plateau")
                    successful_plateau = [r for r in plateau_results if r['success']]

                    if len(successful_plateau) < 5:
                        print(f"❌ Too few successful runs")
                        continue

                    plateau_avg_value = np.mean([r['final_value'] for r in successful_plateau])

                    # Statistical test
                    value_t, value_p = stats.ttest_ind(
                        [r['final_value'] for r in successful_standard],
                        [r['final_value'] for r in successful_plateau]
                    )

                    value_improvement = (std_avg_value - plateau_avg_value) / std_avg_value * 100

                    print(f"✓ {len(successful_plateau)}/{len(plateau_results)} successful")
                    print(f"      Average value: {plateau_avg_value:.4f} ({value_improvement:+.1f}%, p={value_p:.3f})")

                    # Record results
                    all_results.append({
                        'landscape': landscape_name,
                        'optimizer': optimizer_name,
                        'radius': radius,
                        'standard_value': std_avg_value,
                        'plateau_value': plateau_avg_value,
                        'value_improvement': value_improvement,
                        'value_p_value': value_p,
                        'n_standard': len(successful_standard),
                        'n_plateau': len(successful_plateau)
                    })

        # Analysis
        df = pd.DataFrame(all_results)
        if len(df) > 0:
            print(f"\n📈 PLATEAU FINDING ANALYSIS:")
            print(f"Total comparisons: {len(df)}")

            # Significant improvements
            sig_improvements = df[(df['value_p_value'] < 0.05) & (df['value_improvement'] > 0)]

            print(f"Significant improvements: {len(sig_improvements)}/{len(df)}")

            if len(sig_improvements) > 0:
                print(f"Best improvement: {sig_improvements['value_improvement'].max():.1f}%")
                best_config = sig_improvements.loc[sig_improvements['value_improvement'].idxmax()]
                print(f"Best config: {best_config['landscape']} with {best_config['optimizer']} (radius={best_config['radius']:.2f})")

            # Overall verdict
            success_rate = len(sig_improvements) / len(df)
            if success_rate > 0.3:
                print("🏆 VERDICT: Underpromoted effectively finds plateau regions!")
            elif success_rate > 0.1:
                print("🤔 VERDICT: Underpromoted shows occasional plateau-finding benefits")
            else:
                print("❌ VERDICT: Underpromoted doesn't consistently find better plateaus")

        return df

if __name__ == "__main__":
    print("🔬 COMPREHENSIVE TESTING: NOVEL EMBARRASSINGLY TECHNIQUES")
    print("=" * 65)
    print("Focus: Shy adaptive evaluation & Underpromoted plateau finding")
    print("(Skipping trivial caching - focus on academically interesting ideas)")

    analyzer = NovelTechniqueAnalysis(n_runs=25)

    # Test the novel techniques
    shy_results = analyzer.test_shy_adaptive_evaluation()
    plateau_results = analyzer.test_underpromoted_plateau_finding()

    print(f"\n🎯 FINAL ACADEMIC ASSESSMENT:")
    print("=" * 40)
    print("Results show whether these novel optimization techniques provide")
    print("measurable, statistically significant benefits for derivative-free optimization.")

    print(f"\nData collected:")
    print(f"  Shy evaluations: {len(shy_results)} comparisons")
    print(f"  Plateau finding: {len(plateau_results)} comparisons")

    print(f"\n✅ THOROUGH ACADEMIC ANALYSIS COMPLETE!")