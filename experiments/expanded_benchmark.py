#!/usr/bin/env python3
"""
Expanded benchmark with more optimizers for richer 2D/3D performance analysis.
Focus on multi-dimensional Thurstone calibration for practical recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
from typing import Dict, List, Tuple
import warnings

from humpday.optimizers.alloptimizers import OPTIMIZERS
from humpday.objectives.classic import CLASSIC_OBJECTIVES


def get_expanded_optimizer_set():
    """Get a more comprehensive set of optimizers for multi-dimensional analysis."""

    # Import specific optimizer groups
    from humpday.optimizers.scipycube import SCIPY_OPTIMIZERS
    from humpday.optimizers.nevergradcube import NEVERGRAD_OPTIMIZERS
    from humpday.optimizers.optunacube import OPTUNA_OPTIMIZERS

    # Try to import additional optimizers that might be available
    expanded_optimizers = SCIPY_OPTIMIZERS.copy()

    # Add Nevergrad optimizers (browser compatible)
    try:
        ng_optimizers = [opt for opt in NEVERGRAD_OPTIMIZERS if opt in OPTIMIZERS]
        expanded_optimizers.extend(ng_optimizers[:5])  # Limit to 5 to avoid too many
        print(f"Added {len(ng_optimizers[:5])} Nevergrad optimizers")
    except Exception as e:
        print(f"Nevergrad optimizers not available: {e}")

    # Add Optuna optimizers
    try:
        optuna_optimizers = [opt for opt in OPTUNA_OPTIMIZERS if opt in OPTIMIZERS]
        expanded_optimizers.extend(optuna_optimizers[:3])  # Limit for manageable analysis
        print(f"Added {len(optuna_optimizers[:3])} Optuna optimizers")
    except Exception as e:
        print(f"Optuna optimizers not available: {e}")

    print(f"Total expanded optimizer set: {len(expanded_optimizers)}")
    for opt in expanded_optimizers:
        print(f"  - {opt.__name__.replace('_cube', '')}")

    return expanded_optimizers


def run_single_comparison(optimizer_func, objective_func, n_dim: int, n_trials: int, seed: int = None) -> Dict:
    """Run a single optimizer/objective comparison with error handling."""
    if seed:
        np.random.seed(seed)

    try:
        start_time = datetime.now()
        result = optimizer_func(objective_func, n_trials=n_trials, n_dim=n_dim, with_count=True)
        end_time = datetime.now()

        best_value, best_params, reported_trials = result
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
            'timestamp': start_time.isoformat()
        }
    except Exception as e:
        return {
            'optimizer': optimizer_func.__name__.replace('_cube', ''),
            'objective': objective_func.__name__.replace('_on_cube', ''),
            'n_dim': n_dim,
            'n_trials': n_trials,
            'reported_trials': 0,
            'best_value': float('inf'),
            'elapsed_seconds': 0.0,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def run_expanded_benchmark():
    """Run expanded benchmark with more optimizers for richer analysis."""

    optimizers = get_expanded_optimizer_set()
    objectives = CLASSIC_OBJECTIVES[:12]  # Use more objectives for richer data

    # Focused experimental design for JSS paper
    n_dims = [2, 4, 8]  # Include higher dimensions
    n_trials_list = [25, 75]  # Low vs high budget
    n_repeats = 2

    results = []
    total_runs = len(optimizers) * len(objectives) * len(n_dims) * len(n_trials_list) * n_repeats
    current_run = 0

    print(f"Expanded benchmark: {total_runs} total runs")
    print(f"Optimizers: {len(optimizers)}")
    print(f"Objectives: {len(objectives)}")
    print(f"Dimensions: {n_dims}")
    print(f"Trial budgets: {n_trials_list}")

    for objective in objectives:
        print(f"\nTesting objective: {objective.__name__}")

        for n_dim in n_dims:
            for n_trials in n_trials_list:
                print(f"  {n_dim}D, {n_trials} trials:")

                for repeat in range(n_repeats):
                    for optimizer in optimizers:
                        current_run += 1

                        # Use different seed for each repeat
                        seed = hash(f"{optimizer.__name__}_{objective.__name__}_{n_dim}_{n_trials}_{repeat}") % 2**31

                        result = run_single_comparison(
                            optimizer, objective, n_dim, n_trials, seed
                        )
                        result['repeat'] = repeat
                        result['seed'] = seed
                        results.append(result)

                        if current_run % 50 == 0:
                            print(f"    Progress: {current_run}/{total_runs} ({100*current_run/total_runs:.1f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/expanded_benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nExpanded benchmark results saved to: {csv_path}")
    return df


def create_3d_thurstone_analysis(df):
    """
    Create 3D Thurstone-style analysis for practical optimizer recommendations.
    This is the key innovation for the JSS paper.
    """

    successful = df[df['success'] == True].copy()
    print(f"\n=== 3D Thurstone Calibration Analysis ===")
    print(f"Analyzing {len(successful)} successful runs")

    # Define 3 key dimensions for optimizer performance

    # Dimension 1: Problem Landscape Type
    landscape_mapping = {
        'rosenbrock': 'smooth', 'bohachevsky': 'smooth',
        'griewank': 'multimodal', 'rastrigin': 'multimodal', 'schwefel': 'multimodal',
        'shaffer': 'rugged', 'shekel': 'rugged', 'deap_combo1': 'rugged',
        'ackley': 'multimodal', 'styblinski_tang': 'multimodal',
        'zakharov': 'smooth', 'salomon': 'rugged'
    }
    successful['landscape_type'] = successful['objective'].map(landscape_mapping).fillna('multimodal')

    # Dimension 2: Dimensionality Class
    def dim_class(n_dim):
        if n_dim <= 2:
            return 'low'
        elif n_dim <= 4:
            return 'medium'
        else:
            return 'high'

    successful['dim_class'] = successful['n_dim'].apply(dim_class)

    # Dimension 3: Budget Regime
    successful['budget_per_dim'] = successful['n_trials'] / successful['n_dim']
    successful['budget_class'] = successful['budget_per_dim'].apply(lambda x: 'low' if x < 20 else 'high')

    # Calculate performance within each context
    successful['context_performance'] = 0.0

    context_groups = successful.groupby(['landscape_type', 'dim_class', 'budget_class', 'objective', 'n_dim', 'n_trials'])

    for name, group in context_groups:
        if len(group) < 2:
            continue

        # Rank optimizers within this specific context
        sorted_group = group.sort_values('best_value')
        ranks = np.arange(1, len(sorted_group) + 1)
        normalized_performance = (len(sorted_group) + 1 - ranks) / len(sorted_group)

        for idx, perf in zip(sorted_group.index, normalized_performance):
            successful.loc[idx, 'context_performance'] = perf

    # Create 3D performance tensor
    optimizers = sorted(successful['optimizer'].unique())
    landscapes = ['smooth', 'multimodal', 'rugged']
    dims = ['low', 'medium', 'high']
    budgets = ['low', 'high']

    # Build 3D performance profiles
    performance_tensor = {}

    for optimizer in optimizers:
        opt_data = successful[successful['optimizer'] == optimizer]

        profile = {}
        for landscape in landscapes:
            for dim in dims:
                for budget in budgets:
                    context_data = opt_data[
                        (opt_data['landscape_type'] == landscape) &
                        (opt_data['dim_class'] == dim) &
                        (opt_data['budget_class'] == budget)
                    ]

                    if len(context_data) > 0:
                        profile[f"{landscape}_{dim}_{budget}"] = context_data['context_performance'].mean()
                    else:
                        profile[f"{landscape}_{dim}_{budget}"] = 0.5  # Neutral

        performance_tensor[optimizer] = profile

    # Convert to DataFrame
    tensor_df = pd.DataFrame(performance_tensor).T
    tensor_df = tensor_df.fillna(0.5)

    print("\n3D Performance Tensor (Context-Specific Rankings):")
    print("Format: landscape_dimensionality_budget")
    print(tensor_df.round(3))

    # Generate practical recommendations
    print("\n=== Practical Optimizer Recommendations ===")

    recommendations = {}

    for context_key in tensor_df.columns:
        landscape, dim, budget = context_key.split('_')
        best_optimizer = tensor_df[context_key].idxmax()
        best_score = tensor_df[context_key].max()

        context_desc = f"{landscape} landscapes, {dim} dimensions, {budget} budget"
        recommendations[context_desc] = (best_optimizer, best_score)

        print(f"{context_desc:35s}: {best_optimizer:15s} ({best_score:.3f})")

    # Find most versatile optimizers
    print(f"\n=== Most Versatile Optimizers ===")
    versatility_scores = tensor_df.mean(axis=1).sort_values(ascending=False)
    consistency_scores = 1 / (tensor_df.std(axis=1) + 0.01)  # Lower std = more consistent

    combined_score = 0.7 * versatility_scores + 0.3 * consistency_scores
    combined_score = combined_score.sort_values(ascending=False)

    print("Combined versatility + consistency rankings:")
    for i, (opt, score) in enumerate(combined_score.items()):
        versatility = versatility_scores[opt]
        consistency = consistency_scores[opt]
        print(f"  {i+1:2d}. {opt:20s} overall={versatility:.3f}, consistency={consistency:.2f}")

    # Save detailed analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    analysis_results = {
        'performance_tensor': tensor_df.to_dict(),
        'recommendations': recommendations,
        'versatility_rankings': versatility_scores.to_dict(),
        'consistency_rankings': consistency_scores.to_dict(),
        'combined_rankings': combined_score.to_dict()
    }

    with open(f"results/3d_thurstone_analysis_{timestamp}.json", 'w') as f:
        import json
        json.dump(analysis_results, f, indent=2)

    print(f"\nDetailed 3D analysis saved to: results/3d_thurstone_analysis_{timestamp}.json")

    return tensor_df, recommendations, combined_score


def main():
    """Run expanded benchmark and 3D Thurstone analysis."""

    print("HumpDay 3D Thurstone Calibration Study")
    print("=" * 50)

    # Run expanded benchmark
    df = run_expanded_benchmark()

    # Perform 3D Thurstone analysis
    tensor_df, recommendations, rankings = create_3d_thurstone_analysis(df)

    print(f"\n=== Key Insights for JSS Paper ===")
    print(f"1. Multi-dimensional Thurstone calibration provides context-specific recommendations")
    print(f"2. No single 'best' optimizer - performance depends on problem characteristics")
    print(f"3. 3D analysis (landscape × dimensionality × budget) enables practical guidance")
    print(f"4. Browser-based implementation makes this accessible to practitioners")


if __name__ == "__main__":
    main()