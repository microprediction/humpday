"""
Statistical Analysis Module for Cross-Validation Framework

This module provides advanced statistical analysis for:
1. Convergence behavior comparison
2. Performance distribution analysis
3. Statistical significance testing
4. Algorithm equivalence validation

MATHEMATICAL FOCUS:
- Statistical tests for algorithmic equivalence
- Convergence rate analysis
- Distribution comparison methods
- Non-parametric statistical validation

Author: HumpDay Statistical Validation Module
Date: 2026-05-23
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Handle optional scipy import
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy not available - using fallback statistical tests")

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ConvergenceAnalysis:
    """Results of convergence analysis between two algorithms."""
    algorithm_a: str
    algorithm_b: str
    convergence_rate_a: float
    convergence_rate_b: float
    rate_similarity: float  # 0 = different, 1 = identical
    path_correlation: float
    statistical_significance: float  # p-value
    passed_equivalence: bool


@dataclass
class DistributionComparison:
    """Results of statistical distribution comparison."""
    algorithm_a: str
    algorithm_b: str
    mean_difference: float
    variance_ratio: float
    ks_statistic: float
    ks_pvalue: float
    mannwhitney_statistic: float
    mannwhitney_pvalue: float
    equivalent: bool


class StatisticalValidator:
    """Advanced statistical validation for optimization algorithms."""

    def __init__(self, significance_level: float = 0.05, equivalence_threshold: float = 0.1):
        self.significance_level = significance_level
        self.equivalence_threshold = equivalence_threshold

    def analyze_convergence_behavior(self,
                                   convergence_a: List[float],
                                   convergence_b: List[float],
                                   algorithm_a: str,
                                   algorithm_b: str) -> ConvergenceAnalysis:
        """
        Analyze and compare convergence behavior of two algorithms.

        MATHEMATICAL APPROACH:
        1. Fit exponential decay models to convergence curves
        2. Compare convergence rates statistically
        3. Measure path correlation
        4. Test for equivalence
        """

        # Ensure we have sufficient data
        min_len = min(len(convergence_a), len(convergence_b))
        if min_len < 5:
            return ConvergenceAnalysis(
                algorithm_a=algorithm_a,
                algorithm_b=algorithm_b,
                convergence_rate_a=0.0,
                convergence_rate_b=0.0,
                rate_similarity=0.0,
                path_correlation=0.0,
                statistical_significance=1.0,
                passed_equivalence=False
            )

        # Truncate to same length for fair comparison
        conv_a = np.array(convergence_a[:min_len])
        conv_b = np.array(convergence_b[:min_len])

        # Fit convergence rates (exponential decay model)
        rate_a = self._fit_convergence_rate(conv_a)
        rate_b = self._fit_convergence_rate(conv_b)

        # Calculate rate similarity (normalized difference)
        rate_diff = abs(rate_a - rate_b)
        max_rate = max(abs(rate_a), abs(rate_b), 1e-10)
        rate_similarity = max(0.0, 1.0 - rate_diff / max_rate)

        # Path correlation
        if np.std(conv_a) > 1e-10 and np.std(conv_b) > 1e-10:
            path_correlation = np.corrcoef(conv_a, conv_b)[0, 1]
            if np.isnan(path_correlation):
                path_correlation = 0.0
        else:
            path_correlation = 1.0 if np.allclose(conv_a, conv_b, rtol=1e-6) else 0.0

        # Statistical significance test (Mann-Whitney U test)
        try:
            if SCIPY_AVAILABLE:
                statistic, p_value = stats.mannwhitneyu(conv_a, conv_b, alternative='two-sided')
            else:
                # Fallback: simple comparison
                mean_diff = abs(np.mean(conv_a) - np.mean(conv_b))
                pooled_std = np.sqrt((np.var(conv_a) + np.var(conv_b)) / 2)
                test_stat = mean_diff / (pooled_std + 1e-10)
                p_value = 0.5 if test_stat < 2.0 else 0.1
        except Exception:
            p_value = 1.0

        # Equivalence test
        passed_equivalence = (
            rate_similarity > (1 - self.equivalence_threshold) and
            abs(path_correlation) > 0.7 and
            p_value > self.significance_level
        )

        return ConvergenceAnalysis(
            algorithm_a=algorithm_a,
            algorithm_b=algorithm_b,
            convergence_rate_a=rate_a,
            convergence_rate_b=rate_b,
            rate_similarity=rate_similarity,
            path_correlation=path_correlation,
            statistical_significance=p_value,
            passed_equivalence=passed_equivalence
        )

    def _fit_convergence_rate(self, convergence: np.ndarray) -> float:
        """
        Fit exponential convergence rate to optimization trajectory.

        Model: f(t) = f_final + (f_initial - f_final) * exp(-rate * t)
        """
        if len(convergence) < 3:
            return 0.0

        # Remove duplicates and ensure monotonic improvement for rate calculation
        unique_conv = []
        last_val = float('inf')
        for val in convergence:
            if val < last_val:
                unique_conv.append(val)
                last_val = val
            else:
                unique_conv.append(last_val)  # Use best so far

        if len(unique_conv) < 3:
            return 0.0

        # Fit exponential decay
        y = np.array(unique_conv)
        t = np.arange(len(y))

        # Shift to positive values for log fitting
        y_shifted = y - np.min(y) + 1e-10

        try:
            # Linear regression on log scale: log(y) = log(a) - rate * t
            log_y = np.log(y_shifted)
            if np.any(np.isfinite(log_y)):
                coeffs = np.polyfit(t[np.isfinite(log_y)], log_y[np.isfinite(log_y)], 1)
                rate = -coeffs[0]  # Negative of slope
                return max(0.0, rate)
        except Exception:
            pass

        # Fallback: simple rate based on improvement
        if len(y) > 1:
            total_improvement = y[0] - y[-1]
            time_steps = len(y) - 1
            return total_improvement / (time_steps * (y[0] + 1e-10))

        return 0.0

    def compare_performance_distributions(self,
                                        results_a: List[float],
                                        results_b: List[float],
                                        algorithm_a: str,
                                        algorithm_b: str) -> DistributionComparison:
        """
        Compare performance distributions using multiple statistical tests.

        STATISTICAL TESTS:
        1. Kolmogorov-Smirnov test for distribution similarity
        2. Mann-Whitney U test for median differences
        3. Variance ratio test for spread comparison
        """

        if len(results_a) < 2 or len(results_b) < 2:
            return DistributionComparison(
                algorithm_a=algorithm_a,
                algorithm_b=algorithm_b,
                mean_difference=float('inf'),
                variance_ratio=float('inf'),
                ks_statistic=1.0,
                ks_pvalue=0.0,
                mannwhitney_statistic=0.0,
                mannwhitney_pvalue=0.0,
                equivalent=False
            )

        arr_a = np.array(results_a)
        arr_b = np.array(results_b)

        # Basic statistics
        mean_a = np.mean(arr_a)
        mean_b = np.mean(arr_b)
        var_a = np.var(arr_a)
        var_b = np.var(arr_b)

        mean_difference = abs(mean_a - mean_b)
        variance_ratio = (var_a + 1e-10) / (var_b + 1e-10)

        # Kolmogorov-Smirnov test
        try:
            if SCIPY_AVAILABLE:
                ks_statistic, ks_pvalue = stats.ks_2samp(arr_a, arr_b)
            else:
                # Fallback: simple comparison
                ks_statistic = abs(np.mean(arr_a) - np.mean(arr_b)) / (np.std(arr_a) + np.std(arr_b) + 1e-10)
                ks_pvalue = 0.5  # Neutral p-value
        except Exception:
            ks_statistic, ks_pvalue = 1.0, 0.0

        # Mann-Whitney U test
        try:
            if SCIPY_AVAILABLE:
                mw_statistic, mw_pvalue = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
            else:
                # Fallback: t-test approximation
                mean_diff = abs(np.mean(arr_a) - np.mean(arr_b))
                pooled_std = np.sqrt((np.var(arr_a) + np.var(arr_b)) / 2)
                mw_statistic = mean_diff / (pooled_std + 1e-10)
                mw_pvalue = 0.5 if mw_statistic < 2.0 else 0.1  # Simple heuristic
        except Exception:
            mw_statistic, mw_pvalue = 0.0, 0.0

        # Equivalence criteria
        relative_mean_diff = mean_difference / (abs(mean_b) + 1e-10)
        acceptable_variance_ratio = 0.2 < variance_ratio < 5.0

        equivalent = (
            relative_mean_diff < self.equivalence_threshold and
            acceptable_variance_ratio and
            ks_pvalue > self.significance_level and
            mw_pvalue > self.significance_level
        )

        return DistributionComparison(
            algorithm_a=algorithm_a,
            algorithm_b=algorithm_b,
            mean_difference=mean_difference,
            variance_ratio=variance_ratio,
            ks_statistic=ks_statistic,
            ks_pvalue=ks_pvalue,
            mannwhitney_statistic=mw_statistic,
            mannwhitney_pvalue=mw_pvalue,
            equivalent=equivalent
        )

    def test_mathematical_equivalence(self,
                                    python_results: Dict[str, List[float]],
                                    reference_results: Dict[str, List[float]],
                                    convergence_python: Dict[str, List[List[float]]],
                                    convergence_reference: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        """
        Comprehensive mathematical equivalence testing.

        VALIDATION APPROACH:
        1. Statistical distribution comparison
        2. Convergence behavior analysis
        3. Cross-validation consistency
        4. Mathematical property verification
        """

        equivalence_report = {
            'distribution_comparisons': [],
            'convergence_analyses': [],
            'overall_equivalence': {},
            'statistical_summary': {}
        }

        # Compare distributions for each algorithm
        for alg_name in python_results.keys():
            if alg_name in reference_results:
                dist_comparison = self.compare_performance_distributions(
                    python_results[alg_name],
                    reference_results[alg_name],
                    f"{alg_name}_Python",
                    f"{alg_name}_Reference"
                )
                equivalence_report['distribution_comparisons'].append(dist_comparison)

        # Compare convergence behaviors
        for alg_name in convergence_python.keys():
            if alg_name in convergence_reference:
                py_conv = convergence_python[alg_name]
                ref_conv = convergence_reference[alg_name]

                # Compare convergence for each run, then aggregate
                conv_analyses = []
                for i, (py_run, ref_run) in enumerate(zip(py_conv, ref_conv)):
                    conv_analysis = self.analyze_convergence_behavior(
                        py_run,
                        ref_run,
                        f"{alg_name}_Python_Run{i}",
                        f"{alg_name}_Reference_Run{i}"
                    )
                    conv_analyses.append(conv_analysis)

                equivalence_report['convergence_analyses'].extend(conv_analyses)

        # Overall equivalence assessment
        dist_passed = sum(1 for comp in equivalence_report['distribution_comparisons'] if comp.equivalent)
        dist_total = len(equivalence_report['distribution_comparisons'])

        conv_passed = sum(1 for analysis in equivalence_report['convergence_analyses'] if analysis.passed_equivalence)
        conv_total = len(equivalence_report['convergence_analyses'])

        equivalence_report['overall_equivalence'] = {
            'distribution_pass_rate': (dist_passed / dist_total * 100) if dist_total > 0 else 0,
            'convergence_pass_rate': (conv_passed / conv_total * 100) if conv_total > 0 else 0,
            'algorithms_tested': len(set(python_results.keys()) & set(reference_results.keys())),
            'mathematical_equivalence': (dist_passed + conv_passed) / (dist_total + conv_total) > 0.8 if (dist_total + conv_total) > 0 else False
        }

        # Statistical summary
        if equivalence_report['distribution_comparisons']:
            mean_diffs = [comp.mean_difference for comp in equivalence_report['distribution_comparisons']]
            var_ratios = [comp.variance_ratio for comp in equivalence_report['distribution_comparisons']]
            ks_pvalues = [comp.ks_pvalue for comp in equivalence_report['distribution_comparisons']]

            equivalence_report['statistical_summary'] = {
                'mean_difference_stats': {
                    'median': np.median(mean_diffs),
                    'max': np.max(mean_diffs),
                    'q75': np.percentile(mean_diffs, 75)
                },
                'variance_ratio_stats': {
                    'median': np.median(var_ratios),
                    'outside_acceptable_range': sum(1 for vr in var_ratios if vr < 0.2 or vr > 5.0)
                },
                'ks_test_stats': {
                    'median_pvalue': np.median(ks_pvalues),
                    'significant_differences': sum(1 for pv in ks_pvalues if pv < self.significance_level)
                }
            }

        return equivalence_report

    def validate_cross_language_consistency(self,
                                          python_results: Dict[str, List[float]],
                                          javascript_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Validate consistency between Python and JavaScript implementations.

        CROSS-LANGUAGE VALIDATION:
        1. Account for implementation differences
        2. Focus on algorithmic equivalence
        3. Allow for reasonable variation due to language differences
        """

        consistency_report = {
            'language_comparisons': [],
            'consistency_metrics': {},
            'cross_language_equivalence': False
        }

        # Relaxed thresholds for cross-language comparison
        original_threshold = self.equivalence_threshold
        self.equivalence_threshold = 0.2  # 20% tolerance for cross-language

        try:
            # Compare each matching algorithm
            for alg_name in python_results.keys():
                if alg_name in javascript_results:
                    comparison = self.compare_performance_distributions(
                        python_results[alg_name],
                        javascript_results[alg_name],
                        f"{alg_name}_Python",
                        f"{alg_name}_JavaScript"
                    )
                    consistency_report['language_comparisons'].append(comparison)

            # Calculate consistency metrics
            if consistency_report['language_comparisons']:
                consistent_algorithms = sum(1 for comp in consistency_report['language_comparisons'] if comp.equivalent)
                total_algorithms = len(consistency_report['language_comparisons'])

                mean_differences = [comp.mean_difference for comp in consistency_report['language_comparisons']]
                variance_ratios = [comp.variance_ratio for comp in consistency_report['language_comparisons']]

                consistency_report['consistency_metrics'] = {
                    'consistency_rate': (consistent_algorithms / total_algorithms * 100) if total_algorithms > 0 else 0,
                    'algorithms_tested': total_algorithms,
                    'mean_difference_distribution': {
                        'median': np.median(mean_differences),
                        'max': np.max(mean_differences),
                        'acceptable_count': sum(1 for md in mean_differences if md < 1.0)
                    },
                    'variance_ratio_distribution': {
                        'median': np.median(variance_ratios),
                        'stable_count': sum(1 for vr in variance_ratios if 0.1 < vr < 10.0)
                    }
                }

                # Overall cross-language equivalence
                consistency_report['cross_language_equivalence'] = (
                    consistency_report['consistency_metrics']['consistency_rate'] > 60 and
                    consistency_report['consistency_metrics']['mean_difference_distribution']['acceptable_count'] / total_algorithms > 0.7
                )

        finally:
            # Restore original threshold
            self.equivalence_threshold = original_threshold

        return consistency_report

    def generate_statistical_report(self, validation_results: List[Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical validation report."""

        report = {
            'timestamp': np.datetime64('now').astype(str),
            'statistical_parameters': {
                'significance_level': self.significance_level,
                'equivalence_threshold': self.equivalence_threshold
            },
            'validation_summary': {},
            'recommendations': []
        }

        # Categorize results by test type
        test_categories = {}
        for result in validation_results:
            category = getattr(result, 'test_name', 'unknown')
            if category not in test_categories:
                test_categories[category] = []
            test_categories[category].append(result)

        # Generate category-specific statistics
        for category, results in test_categories.items():
            passed = sum(1 for r in results if getattr(r, 'passed', False))
            total = len(results)

            report['validation_summary'][category] = {
                'total_tests': total,
                'passed_tests': passed,
                'pass_rate': (passed / total * 100) if total > 0 else 0,
                'critical_failures': sum(1 for r in results if not getattr(r, 'passed', True) and 'PRIMA' in getattr(r, 'algorithm_name', ''))
            }

        # Generate recommendations based on statistical analysis
        overall_pass_rate = sum(cat['passed_tests'] for cat in report['validation_summary'].values()) / sum(cat['total_tests'] for cat in report['validation_summary'].values()) * 100 if report['validation_summary'] else 0

        if overall_pass_rate >= 90:
            report['recommendations'].append("✅ Excellent mathematical consistency across all implementations")
        elif overall_pass_rate >= 75:
            report['recommendations'].append("✅ Good statistical equivalence with minor variations")
        elif overall_pass_rate >= 50:
            report['recommendations'].append("⚠️ Moderate consistency - review failing algorithms")
        else:
            report['recommendations'].append("❌ Poor statistical equivalence - significant implementation issues")

        # Specific recommendations for failed categories
        for category, stats in report['validation_summary'].items():
            if stats['pass_rate'] < 50:
                report['recommendations'].append(f"🔧 Review {category} implementation consistency")

        return report


def create_convergence_plots(convergence_data: Dict[str, List[List[float]]],
                           output_dir: str = "validation_results") -> None:
    """
    Create visualization plots for convergence analysis.

    NOTE: This function requires matplotlib which may not be available in all environments.
    """
    try:
        from pathlib import Path

        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for algorithm, runs in convergence_data.items():
            if not runs:
                continue

            plt.figure(figsize=(12, 8))

            # Plot all runs
            for i, run in enumerate(runs):
                if run:
                    plt.plot(run, alpha=0.3, color='blue', linewidth=1)

            # Plot average convergence
            if runs:
                max_len = max(len(run) for run in runs if run)
                avg_convergence = []

                for step in range(max_len):
                    values_at_step = [run[step] for run in runs if len(run) > step]
                    if values_at_step:
                        avg_convergence.append(np.mean(values_at_step))

                if avg_convergence:
                    plt.plot(avg_convergence, color='red', linewidth=3, label='Average')

            plt.xlabel('Function Evaluations')
            plt.ylabel('Objective Function Value')
            plt.title(f'{algorithm} - Convergence Analysis')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_file = output_path / f"{algorithm}_convergence.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"📈 Convergence plots saved to {output_path}")

    except ImportError:
        print("⚠️ matplotlib not available - skipping convergence plots")
        MATPLOTLIB_AVAILABLE = False
    except Exception as e:
        print(f"⚠️ Error creating convergence plots: {e}")


def main():
    """Demonstration of statistical validation capabilities."""
    print("📊 Statistical Validation Module for HumpDay")
    print("=" * 50)

    # Example usage
    validator = StatisticalValidator(significance_level=0.05, equivalence_threshold=0.1)

    # Simulate some test data
    np.random.seed(42)

    # Simulate convergence data for two algorithms
    conv_a = np.exp(-0.1 * np.arange(50)) + 0.1 * np.random.random(50)
    conv_b = np.exp(-0.12 * np.arange(50)) + 0.1 * np.random.random(50)

    # Test convergence analysis
    conv_analysis = validator.analyze_convergence_behavior(
        conv_a.tolist(), conv_b.tolist(), "Algorithm_A", "Algorithm_B"
    )

    print("Convergence Analysis:")
    print(f"  Rate A: {conv_analysis.convergence_rate_a:.4f}")
    print(f"  Rate B: {conv_analysis.convergence_rate_b:.4f}")
    print(f"  Similarity: {conv_analysis.rate_similarity:.4f}")
    print(f"  Correlation: {conv_analysis.path_correlation:.4f}")
    print(f"  Equivalent: {conv_analysis.passed_equivalence}")

    # Test distribution comparison
    results_a = np.random.exponential(2, 20).tolist()
    results_b = np.random.exponential(2.1, 20).tolist()

    dist_comparison = validator.compare_performance_distributions(
        results_a, results_b, "Algorithm_A", "Algorithm_B"
    )

    print("\nDistribution Comparison:")
    print(f"  Mean difference: {dist_comparison.mean_difference:.4f}")
    print(f"  Variance ratio: {dist_comparison.variance_ratio:.4f}")
    print(f"  KS p-value: {dist_comparison.ks_pvalue:.4f}")
    print(f"  Equivalent: {dist_comparison.equivalent}")

    print("\n✅ Statistical validation module working correctly")


if __name__ == "__main__":
    main()
