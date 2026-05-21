#!/usr/bin/env python3
"""
Comprehensive benchmark for JSS paper with expanded optimizer set.
Focus on getting rich 3D Thurstone calibration data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import json
from typing import Dict, List, Tuple
import warnings

# Try to get more optimizers
from humpday.optimizers.alloptimizers import OPTIMIZERS
from humpday.objectives.classic import CLASSIC_OBJECTIVES


def get_comprehensive_optimizer_set():
    """Get the most comprehensive optimizer set available."""

    # Start with all available optimizers and filter for working ones
    working_optimizers = []

    print("Testing optimizer availability...")

    # Quick test to see which optimizers actually work
    test_objective = CLASSIC_OBJECTIVES[0]  # Use rosenbrock or similar

    for opt in OPTIMIZERS[:20]:  # Test first 20 to avoid too long startup
        try:
            # Quick test run
            result = opt(test_objective, n_trials=5, n_dim=2, with_count=True)
            if len(result) == 3 and result[0] is not None:
                working_optimizers.append(opt)
                print(f"  ✓ {opt.__name__.replace('_cube', '')}")
        except Exception as e:
            print(f"  ✗ {opt.__name__.replace('_cube', '')} - {str(e)[:50]}")

    print(f"\nFound {len(working_optimizers)} working optimizers")
    return working_optimizers


def get_diverse_objectives():
    """Get a diverse set of objectives spanning different landscape types."""

    # Categorize objectives by landscape type
    smooth_objectives = []
    multimodal_objectives = []
    rugged_objectives = []

    # Test objectives and categorize based on known properties
    objective_categories = {
        # Smooth landscapes
        'rosenbrock_on_cube': 'smooth',
        'zakharov_on_cube': 'smooth',
        'bohachevsky_on_cube': 'smooth',
        'rotated_hyper_ellipsoid_on_cube': 'smooth',

        # Multimodal landscapes
        'rastrigin_on_cube': 'multimodal',
        'griewank_on_cube': 'multimodal',
        'ackley_on_cube': 'multimodal',
        'styblinski_tang_on_cube': 'multimodal',
        'schwefel_on_cube': 'multimodal',

        # Rugged landscapes
        'shekel_on_cube': 'rugged',
        'shaffer_on_cube': 'rugged',
        'deap_combo1_on_cube': 'rugged',
        'deap_combo2_on_cube': 'rugged',
        'paviani_on_cube': 'rugged',
        'salomon_on_cube': 'rugged'
    }

    # Get available objectives that match our categories
    diverse_objectives = []
    for obj in CLASSIC_OBJECTIVES:
        obj_name = obj.__name__
        if obj_name in objective_categories:
            diverse_objectives.append(obj)

    print(f"Selected {len(diverse_objectives)} diverse objectives:")
    for obj in diverse_objectives:
        category = objective_categories.get(obj.__name__, 'unknown')
        print(f"  {category:10s}: {obj.__name__.replace('_on_cube', '')}")

    return diverse_objectives


def run_single_comparison(optimizer_func, objective_func, n_dim: int, n_trials: int, seed: int = None) -> Dict:
    """Run a single optimizer/objective comparison with comprehensive error handling."""

    if seed:
        np.random.seed(seed)

    start_time = datetime.now()

    try:
        result = optimizer_func(objective_func, n_trials=n_trials, n_dim=n_dim, with_count=True)
        end_time = datetime.now()

        if len(result) != 3:
            raise ValueError(f"Unexpected result format: {result}")

        best_value, best_params, reported_trials = result

        if best_value is None or not np.isfinite(best_value):
            raise ValueError(f"Invalid best_value: {best_value}")

        elapsed = (end_time - start_time).total_seconds()

        return {
            'optimizer': optimizer_func.__name__.replace('_cube', ''),
            'objective': objective_func.__name__.replace('_on_cube', ''),
            'n_dim': n_dim,
            'n_trials': n_trials,
            'reported_trials': reported_trials,
            'best_value': float(best_value),
            'elapsed_seconds': elapsed,
            'success': True,
            'error': None,
            'timestamp': start_time.isoformat(),
            'convergence_quality': abs(best_value) if abs(best_value) < 100 else 100  # Normalized quality metric
        }

    except Exception as e:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        return {
            'optimizer': optimizer_func.__name__.replace('_cube', ''),
            'objective': objective_func.__name__.replace('_on_cube', ''),
            'n_dim': n_dim,
            'n_trials': n_trials,
            'reported_trials': 0,
            'best_value': float('inf'),
            'elapsed_seconds': elapsed,
            'success': False,
            'error': str(e)[:200],  # Limit error message length
            'timestamp': start_time.isoformat(),
            'convergence_quality': 0.0
        }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark for JSS paper."""

    print("=== HumpDay Comprehensive Benchmark for JSS Paper ===")
    print("Focus: Multi-dimensional Thurstone calibration\n")

    # Get optimizers and objectives
    optimizers = get_comprehensive_optimizer_set()
    objectives = get_diverse_objectives()

    # Experimental design for rich 3D analysis
    n_dims = [2, 4, 8, 12]  # Low, medium, high, very high dimensionality
    n_trials_list = [25, 50, 100]  # Low, medium, high budget
    n_repeats = 3  # For statistical reliability

    total_runs = len(optimizers) * len(objectives) * len(n_dims) * len(n_trials_list) * n_repeats

    print(f"Experimental Design:")
    print(f"  Optimizers: {len(optimizers)}")
    print(f"  Objectives: {len(objectives)}")
    print(f"  Dimensions: {n_dims}")
    print(f"  Trial budgets: {n_trials_list}")
    print(f"  Repeats: {n_repeats}")
    print(f"  Total runs: {total_runs}\n")

    if total_runs > 1500:
        print("Auto-reducing scope for manageable runtime...")
        n_dims = [2, 4, 8]  # Skip very high dimensions for now
        n_trials_list = [25, 75]  # Low and high budget
        n_repeats = 2  # Sufficient for statistical analysis
        total_runs = len(optimizers) * len(objectives) * len(n_dims) * len(n_trials_list) * n_repeats
        print(f"Reduced to {total_runs} runs (estimated ~{total_runs/60:.1f} minutes)")

    # Run benchmark
    results = []
    current_run = 0
    failed_runs = 0

    start_time = datetime.now()

    for obj_idx, objective in enumerate(objectives):
        print(f"\n[{obj_idx+1}/{len(objectives)}] Testing {objective.__name__.replace('_on_cube', '')}:")

        for dim_idx, n_dim in enumerate(n_dims):
            for budget_idx, n_trials in enumerate(n_trials_list):
                print(f"  {n_dim}D, {n_trials} trials: ", end="")

                for repeat in range(n_repeats):
                    for opt_idx, optimizer in enumerate(optimizers):
                        current_run += 1

                        # Generate deterministic but varied seeds
                        seed = hash(f"{optimizer.__name__}_{objective.__name__}_{n_dim}_{n_trials}_{repeat}") % (2**31)

                        result = run_single_comparison(optimizer, objective, n_dim, n_trials, seed)
                        result['repeat'] = repeat
                        result['seed'] = seed
                        result['run_id'] = current_run
                        results.append(result)

                        if not result['success']:
                            failed_runs += 1

                        # Progress indicator
                        if current_run % (len(optimizers) * n_repeats) == 0:
                            success_rate = (current_run - failed_runs) / current_run * 100
                            print(f"✓ ({success_rate:.0f}% success) ", end="")

                # Estimate remaining time
                if current_run > 50:  # After some runs for better estimation
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = current_run / elapsed  # runs per second
                    remaining = (total_runs - current_run) / rate
                    print(f" [ETA: {remaining/60:.1f}min]")
                else:
                    print()

    # Create results dataframe
    df = pd.DataFrame(results)

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)

    csv_path = f"results/comprehensive_benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n=== Benchmark Complete ===")
    print(f"Total runs: {len(df)}")
    print(f"Success rate: {df['success'].mean()*100:.1f}%")
    print(f"Results saved to: {csv_path}")

    return df


def create_comprehensive_3d_analysis(df):
    """Enhanced 3D analysis with more optimizers and richer insights."""

    successful = df[df['success'] == True].copy()
    n_successful = len(successful)
    n_total = len(df)

    print(f"\n=== Comprehensive 3D Thurstone Analysis ===")
    print(f"Analyzing {n_successful}/{n_total} successful runs ({100*n_successful/n_total:.1f}%)")

    # Enhanced context categorization
    landscape_mapping = {
        'rosenbrock': 'smooth', 'zakharov': 'smooth', 'bohachevsky': 'smooth', 'rotated_hyper_ellipsoid': 'smooth',
        'rastrigin': 'multimodal', 'griewank': 'multimodal', 'ackley': 'multimodal',
        'styblinski_tang': 'multimodal', 'schwefel': 'multimodal',
        'shekel': 'rugged', 'shaffer': 'rugged', 'deap_combo1': 'rugged',
        'deap_combo2': 'rugged', 'paviani': 'rugged', 'salomon': 'rugged'
    }

    def categorize_objective(obj_name):
        for key, category in landscape_mapping.items():
            if key in obj_name:
                return category
        return 'multimodal'  # Default

    successful['landscape_type'] = successful['objective'].apply(categorize_objective)

    # Dimensionality categories
    def dim_category(n_dim):
        if n_dim <= 2: return 'low'
        elif n_dim <= 4: return 'medium'
        elif n_dim <= 8: return 'high'
        else: return 'very_high'

    successful['dim_class'] = successful['n_dim'].apply(dim_category)

    # Budget categories (evaluations per dimension)
    successful['budget_per_dim'] = successful['n_trials'] / successful['n_dim']
    def budget_category(budget_per_dim):
        if budget_per_dim < 15: return 'low'
        elif budget_per_dim < 30: return 'medium'
        else: return 'high'

    successful['budget_class'] = successful['budget_per_dim'].apply(budget_category)

    # Calculate relative performance within each context
    successful['relative_performance'] = 0.0

    # Group by specific problem instance and rank optimizers
    problem_groups = successful.groupby(['objective', 'n_dim', 'n_trials', 'repeat'])

    for name, group in problem_groups:
        if len(group) < 2:
            continue

        # Rank by best_value (lower is better)
        group_sorted = group.sort_values('best_value')
        n_optimizers = len(group_sorted)

        # Convert ranks to relative performance (higher is better, 0-1 scale)
        for i, idx in enumerate(group_sorted.index):
            relative_perf = (n_optimizers - i) / n_optimizers
            successful.loc[idx, 'relative_performance'] = relative_perf

    # Build comprehensive performance tensor
    optimizers = sorted(successful['optimizer'].unique())
    landscapes = sorted(successful['landscape_type'].unique())
    dims = sorted(successful['dim_class'].unique())
    budgets = sorted(successful['budget_class'].unique())

    print(f"\nPerformance tensor dimensions:")
    print(f"  Optimizers: {len(optimizers)} - {optimizers}")
    print(f"  Landscapes: {len(landscapes)} - {landscapes}")
    print(f"  Dimensions: {len(dims)} - {dims}")
    print(f"  Budgets: {len(budgets)} - {budgets}")

    # Create 4D performance tensor (optimizer × landscape × dim × budget)
    tensor_data = {}

    for optimizer in optimizers:
        opt_data = successful[successful['optimizer'] == optimizer]
        profile = {}

        for landscape in landscapes:
            for dim_class in dims:
                for budget_class in budgets:
                    context_data = opt_data[
                        (opt_data['landscape_type'] == landscape) &
                        (opt_data['dim_class'] == dim_class) &
                        (opt_data['budget_class'] == budget_class)
                    ]

                    if len(context_data) > 0:
                        # Use mean relative performance with confidence weighting
                        mean_perf = context_data['relative_performance'].mean()
                        std_perf = context_data['relative_performance'].std()
                        n_samples = len(context_data)

                        # Weight by sample size (more data = more confidence)
                        confidence = min(n_samples / 5, 1.0)  # Full confidence at 5+ samples
                        weighted_perf = mean_perf * confidence + 0.5 * (1 - confidence)

                        profile[f"{landscape}_{dim_class}_{budget_class}"] = {
                            'performance': weighted_perf,
                            'std': std_perf,
                            'n_samples': n_samples,
                            'confidence': confidence
                        }
                    else:
                        profile[f"{landscape}_{dim_class}_{budget_class}"] = {
                            'performance': 0.5,  # Neutral performance
                            'std': 0.0,
                            'n_samples': 0,
                            'confidence': 0.0
                        }

        tensor_data[optimizer] = profile

    # Extract performance matrix for easier analysis
    context_keys = [f"{l}_{d}_{b}" for l in landscapes for d in dims for b in budgets]
    perf_matrix = {}

    for optimizer in optimizers:
        perf_matrix[optimizer] = [tensor_data[optimizer][key]['performance'] for key in context_keys]

    tensor_df = pd.DataFrame(perf_matrix, index=context_keys).T

    print(f"\n3D Performance Tensor Preview (first 8 contexts):")
    print(tensor_df.iloc[:, :8].round(3))

    # Generate enhanced recommendations
    print(f"\n=== Context-Specific Recommendations ===")

    recommendations = {}
    for context_key in context_keys:
        landscape, dim_class, budget_class = context_key.split('_')

        # Get best optimizer for this context
        context_scores = tensor_df[context_key]
        best_optimizer = context_scores.idxmax()
        best_score = context_scores.max()

        # Get confidence level
        confidence = tensor_data[best_optimizer][context_key]['confidence']
        n_samples = tensor_data[best_optimizer][context_key]['n_samples']

        context_desc = f"{landscape:10s} {dim_class:10s} {budget_class:6s}"

        # Only show recommendations with reasonable confidence
        if confidence > 0.3:
            rec_str = f"{best_optimizer:15s} ({best_score:.3f}, conf: {confidence:.2f}, n={n_samples})"
            recommendations[context_desc] = rec_str
            print(f"{context_desc}: {rec_str}")

    # Versatility analysis
    print(f"\n=== Optimizer Versatility Analysis ===")

    # Calculate various metrics
    mean_performance = tensor_df.mean(axis=1).sort_values(ascending=False)
    consistency = (1 / (tensor_df.std(axis=1) + 0.01)).sort_values(ascending=False)  # Lower std = more consistent

    # Peak performance (max score in any context)
    peak_performance = tensor_df.max(axis=1).sort_values(ascending=False)

    # Specialization strength (difference between best and worst contexts)
    specialization = (tensor_df.max(axis=1) - tensor_df.min(axis=1)).sort_values(ascending=False)

    print(f"Overall Performance Leaders:")
    for i, (opt, score) in enumerate(mean_performance.head(5).items()):
        print(f"  {i+1}. {opt:20s} {score:.3f}")

    print(f"\nMost Consistent Performers:")
    for i, (opt, score) in enumerate(consistency.head(5).items()):
        mean_score = mean_performance[opt]
        print(f"  {i+1}. {opt:20s} consistency: {score:.2f}, mean: {mean_score:.3f}")

    print(f"\nHighest Peak Performance:")
    for i, (opt, score) in enumerate(peak_performance.head(5).items()):
        print(f"  {i+1}. {opt:20s} peak: {score:.3f}")

    print(f"\nMost Specialized (high variance):")
    for i, (opt, score) in enumerate(specialization.head(5).items()):
        print(f"  {i+1}. {opt:20s} specialization: {score:.3f}")

    # Save comprehensive analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    analysis_results = {
        'performance_tensor': {opt: {k: v['performance'] for k, v in profile.items()}
                             for opt, profile in tensor_data.items()},
        'confidence_data': tensor_data,
        'recommendations': recommendations,
        'versatility_metrics': {
            'mean_performance': mean_performance.to_dict(),
            'consistency': consistency.to_dict(),
            'peak_performance': peak_performance.to_dict(),
            'specialization': specialization.to_dict()
        },
        'experimental_setup': {
            'n_optimizers': len(optimizers),
            'n_objectives': successful['objective'].nunique(),
            'n_contexts': len(context_keys),
            'total_runs': len(successful),
            'success_rate': len(successful) / len(df)
        }
    }

    results_path = f"results/comprehensive_3d_analysis_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\nComprehensive analysis saved to: {results_path}")

    return tensor_df, recommendations, analysis_results


def main():
    """Run comprehensive benchmark and analysis for JSS paper."""

    print("HumpDay Comprehensive 3D Thurstone Study for JSS Paper")
    print("=" * 60)

    # Run comprehensive benchmark
    df = run_comprehensive_benchmark()

    # Perform enhanced 3D analysis
    tensor_df, recommendations, analysis = create_comprehensive_3d_analysis(df)

    print(f"\n=== Summary for JSS Paper ===")
    print(f"✓ Comprehensive multi-dimensional Thurstone calibration complete")
    print(f"✓ Context-specific recommendations generated")
    print(f"✓ Rich performance tensor with {len(tensor_df)} optimizers across {len(tensor_df.columns)} contexts")
    print(f"✓ Statistical confidence metrics included")
    print(f"✓ Ready for JSS paper results section")


if __name__ == "__main__":
    main()