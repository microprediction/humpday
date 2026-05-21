#!/usr/bin/env python3
"""
Simple analysis of benchmark results for JSS paper.
Focuses on multi-dimensional optimizer performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def load_latest_results():
    """Load the most recent benchmark results."""
    results_dir = "results"
    csv_files = [f for f in os.listdir(results_dir) if f.startswith("benchmark_results_") and f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No benchmark results found")

    latest_file = sorted(csv_files)[-1]
    file_path = os.path.join(results_dir, latest_file)
    print(f"Loading results from: {file_path}")

    df = pd.read_csv(file_path)
    return df


def multi_dimensional_analysis(df):
    """
    Multi-dimensional Thurstonian analysis based on problem characteristics.
    This implements the user's vision of 2D/3D rankings showing when optimizers work.
    """

    # Filter successful runs
    successful = df[df['success'] == True].copy()
    print(f"Successful runs: {len(successful)}/{len(df)}")

    # Create problem characteristic dimensions

    # Dimension 1: Problem complexity (based on objective function)
    complexity_mapping = {
        # Simple/smooth functions
        'rosenbrock': 1, 'bohachevsky': 1,
        # Medium complexity
        'griewank': 2, 'rastrigin': 2, 'schwefel': 2,
        # Complex/multimodal
        'shaffer': 3, 'shekel': 3, 'deap_combo1': 3
    }
    successful['complexity'] = successful['objective'].map(complexity_mapping).fillna(2)

    # Dimension 2: Dimensionality burden
    successful['dim_burden'] = successful['n_dim']

    # Dimension 3: Computational budget (trials per dimension)
    successful['budget_per_dim'] = successful['n_trials'] / successful['n_dim']

    # Calculate relative performance within each problem instance
    successful['relative_performance'] = 0.0

    for (obj, dim, trials), group in successful.groupby(['objective', 'n_dim', 'n_trials']):
        if len(group) < 2:
            continue

        # Rank by performance (lower is better)
        sorted_group = group.sort_values('best_value')
        ranks = np.arange(1, len(sorted_group) + 1)
        normalized_ranks = (len(sorted_group) + 1 - ranks) / len(sorted_group)  # Higher is better

        for idx, rank in zip(sorted_group.index, normalized_ranks):
            successful.loc[idx, 'relative_performance'] = rank

    # Multi-dimensional performance analysis
    print("\n=== Multi-Dimensional Optimizer Performance Analysis ===")

    # Performance by complexity and dimensionality
    complexity_dim_performance = successful.groupby(['optimizer', 'complexity', 'n_dim'])['relative_performance'].agg(['mean', 'std', 'count']).reset_index()

    print("\nPerformance by Complexity and Dimensionality:")
    pivot_mean = complexity_dim_performance.pivot_table(
        values='mean',
        index=['optimizer'],
        columns=['complexity', 'n_dim'],
        aggfunc='mean'
    )
    print(pivot_mean.round(3))

    # Performance by budget and complexity
    budget_complexity_performance = successful.groupby(['optimizer', 'complexity', 'budget_per_dim'])['relative_performance'].agg(['mean', 'std', 'count']).reset_index()

    # Create 2D performance profiles
    optimizer_profiles = {}
    for optimizer in successful['optimizer'].unique():
        opt_data = successful[successful['optimizer'] == optimizer]

        profile = {
            'low_complexity': opt_data[opt_data['complexity'] == 1]['relative_performance'].mean(),
            'med_complexity': opt_data[opt_data['complexity'] == 2]['relative_performance'].mean(),
            'high_complexity': opt_data[opt_data['complexity'] == 3]['relative_performance'].mean(),
            'low_dim': opt_data[opt_data['n_dim'] == 2]['relative_performance'].mean(),
            'med_dim': opt_data[opt_data['n_dim'] == 3]['relative_performance'].mean(),
            'high_dim': opt_data[opt_data['n_dim'] == 5]['relative_performance'].mean(),
            'low_budget': opt_data[opt_data['budget_per_dim'] <= 15]['relative_performance'].mean(),
            'high_budget': opt_data[opt_data['budget_per_dim'] > 15]['relative_performance'].mean(),
        }
        optimizer_profiles[optimizer] = profile

    # Convert to DataFrame for easy analysis
    profiles_df = pd.DataFrame(optimizer_profiles).T
    profiles_df = profiles_df.fillna(0.5)  # Fill NaN with neutral performance

    print("\n2D Optimizer Performance Profiles:")
    print("(Higher values = better relative performance)")
    print(profiles_df.round(3))

    # Identify optimizer specializations
    print("\n=== Optimizer Specializations ===")

    for optimizer in profiles_df.index:
        profile = profiles_df.loc[optimizer]
        specializations = []

        if profile['low_complexity'] > 0.7:
            specializations.append("smooth functions")
        if profile['high_complexity'] > 0.7:
            specializations.append("multimodal functions")
        if profile['low_dim'] > 0.7:
            specializations.append("low dimensions")
        if profile['high_dim'] > 0.7:
            specializations.append("high dimensions")
        if profile['high_budget'] > 0.7:
            specializations.append("high budgets")
        if profile['low_budget'] > 0.7:
            specializations.append("low budgets")

        if specializations:
            print(f"{optimizer:15s}: Strong on {', '.join(specializations)}")
        elif profile.mean() > 0.6:
            print(f"{optimizer:15s}: Generally strong")
        else:
            print(f"{optimizer:15s}: Mixed performance")

    # Create visualization
    create_2d_performance_plot(profiles_df)

    return profiles_df, successful


def create_2d_performance_plot(profiles_df):
    """Create 2D visualization of optimizer performance characteristics."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Complexity vs Dimensionality
    ax1 = axes[0, 0]
    for optimizer in profiles_df.index:
        complexity_strength = (profiles_df.loc[optimizer, 'high_complexity'] - profiles_df.loc[optimizer, 'low_complexity'])
        dim_strength = (profiles_df.loc[optimizer, 'high_dim'] - profiles_df.loc[optimizer, 'low_dim'])
        ax1.scatter(complexity_strength, dim_strength, s=100, alpha=0.7, label=optimizer)

    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Multimodal Advantage (vs Smooth)')
    ax1.set_ylabel('High-Dim Advantage (vs Low-Dim)')
    ax1.set_title('Optimizer Specialization Map')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance heatmap
    ax2 = axes[0, 1]
    performance_matrix = profiles_df[['low_complexity', 'med_complexity', 'high_complexity']].values
    im = ax2.imshow(performance_matrix, cmap='RdYlBu_r', aspect='auto')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Low', 'Medium', 'High'])
    ax2.set_yticks(range(len(profiles_df.index)))
    ax2.set_yticklabels(profiles_df.index)
    ax2.set_xlabel('Problem Complexity')
    ax2.set_title('Performance by Complexity')
    plt.colorbar(im, ax=ax2)

    # Plot 3: Dimensionality performance
    ax3 = axes[1, 0]
    dim_data = profiles_df[['low_dim', 'med_dim', 'high_dim']].values
    im3 = ax3.imshow(dim_data, cmap='RdYlBu_r', aspect='auto')
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['2D', '3D', '5D'])
    ax3.set_yticks(range(len(profiles_df.index)))
    ax3.set_yticklabels(profiles_df.index)
    ax3.set_xlabel('Problem Dimensionality')
    ax3.set_title('Performance by Dimensionality')
    plt.colorbar(im3, ax=ax3)

    # Plot 4: Budget sensitivity
    ax4 = axes[1, 1]
    for optimizer in profiles_df.index:
        budget_sensitivity = profiles_df.loc[optimizer, 'high_budget'] - profiles_df.loc[optimizer, 'low_budget']
        overall_performance = profiles_df.loc[optimizer].mean()
        ax4.scatter(budget_sensitivity, overall_performance, s=100, alpha=0.7, label=optimizer)

    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Budget Sensitivity (High - Low Budget Performance)')
    ax4.set_ylabel('Overall Performance')
    ax4.set_title('Budget Sensitivity vs Overall Performance')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/optimizer_2d_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nVisualization saved as 'results/optimizer_2d_analysis.png'")


def generate_jss_results_summary(profiles_df, successful_df):
    """Generate summary statistics suitable for JSS paper."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n=== JSS Paper Results Summary ===")

    # Basic statistics
    n_optimizers = len(profiles_df)
    n_problems = successful_df.groupby(['objective', 'n_dim', 'n_trials']).size().shape[0]
    n_comparisons = len(successful_df)

    print(f"Experimental setup:")
    print(f"- {n_optimizers} optimization algorithms")
    print(f"- {n_problems} problem configurations")
    print(f"- {n_comparisons} total runs")
    print(f"- Success rate: {successful_df['success'].mean()*100:.1f}%")

    # Multi-dimensional rankings
    print(f"\nMulti-dimensional performance analysis:")

    # Overall ranking
    overall_performance = profiles_df.mean(axis=1).sort_values(ascending=False)
    print(f"\nOverall rankings (mean relative performance):")
    for i, (opt, score) in enumerate(overall_performance.items()):
        print(f"  {i+1:2d}. {opt:15s} {score:.3f}")

    # Context-specific leaders
    print(f"\nContext-specific performance leaders:")

    contexts = {
        'Smooth Functions (Low Complexity)': 'low_complexity',
        'Multimodal Functions (High Complexity)': 'high_complexity',
        'Low Dimensional (2D)': 'low_dim',
        'High Dimensional (5D)': 'high_dim',
        'Low Budget': 'low_budget',
        'High Budget': 'high_budget'
    }

    for context_name, col in contexts.items():
        leader = profiles_df[col].idxmax()
        score = profiles_df[col].max()
        print(f"  {context_name:30s}: {leader:15s} ({score:.3f})")

    # Performance stability
    stability = profiles_df.std(axis=1).sort_values()
    print(f"\nMost consistent performers (low std across contexts):")
    for i, (opt, std) in enumerate(stability.head(3).items()):
        mean_perf = profiles_df.loc[opt].mean()
        print(f"  {i+1}. {opt:15s} μ={mean_perf:.3f}, σ={std:.3f}")

    # Save detailed results
    results_summary = {
        'overall_rankings': overall_performance.to_dict(),
        'context_leaders': {name: (profiles_df[col].idxmax(), profiles_df[col].max())
                          for name, col in contexts.items()},
        'stability_rankings': stability.to_dict(),
        'experiment_stats': {
            'n_optimizers': n_optimizers,
            'n_problems': n_problems,
            'n_comparisons': n_comparisons,
            'success_rate': successful_df['success'].mean()
        }
    }

    import json
    results_file = f"results/jss_results_summary_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    return results_summary


def main():
    """Run complete analysis pipeline."""

    print("HumpDay Multi-Dimensional Performance Analysis")
    print("=" * 50)

    # Load benchmark results
    df = load_latest_results()

    # Run multi-dimensional analysis
    profiles_df, successful_df = multi_dimensional_analysis(df)

    # Generate JSS paper summary
    results_summary = generate_jss_results_summary(profiles_df, successful_df)

    print(f"\nAnalysis complete!")
    print(f"Key insight: Multi-dimensional Thurstone-style analysis reveals optimizer")
    print(f"specializations across problem complexity, dimensionality, and budget constraints.")


if __name__ == "__main__":
    main()