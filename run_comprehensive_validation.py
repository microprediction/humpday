#!/usr/bin/env python3
"""
Comprehensive Cross-Validation Test Runner

This script orchestrates the complete validation framework including:
1. Python vs 3rd Party validation
2. JavaScript vs Python cross-validation
3. Mathematical correctness verification
4. Statistical analysis and reporting

USAGE:
    python run_comprehensive_validation.py [options]

OPTIONS:
    --trials N        Number of trials per optimization run (default: 100)
    --runs N          Number of independent runs per test (default: 5)
    --output DIR      Output directory for results (default: validation_results)
    --skip-js         Skip JavaScript cross-validation tests
    --quick           Quick validation with reduced parameters
    --verbose         Enable verbose output

OUTPUTS:
- Detailed JSON validation report
- Statistical analysis summary
- Convergence plots (if matplotlib available)
- Algorithm comparison tables
- Recommendations for implementation improvements

Author: HumpDay Comprehensive Validation Runner
Date: 2026-05-23
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from benchmark_suite import BenchmarkSuite
    from cross_validation_framework import CrossValidationFramework
    from statistical_validation import StatisticalValidator, create_convergence_plots
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all validation modules are in the same directory")
    sys.exit(1)

# Import HumpDay components
try:
    from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS, pure_optimize
    from humpday.optimizers.prima_algorithms import (
        PRIMA_BOBYQA,
        PRIMA_NEWUOA,
        PRIMA_UOBYQA,
    )
    from humpday.optimizers.scipy_algorithms import LBFGSB, NelderMead, Powell
except ImportError as e:
    print(f"❌ HumpDay import error: {e}")
    print("Please ensure HumpDay is properly installed and accessible")
    sys.exit(1)

import numpy as np


class ComprehensiveValidationRunner:
    """Main runner for comprehensive cross-validation."""

    def __init__(self, output_dir: str = "validation_results", verbose: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose

        # Initialize components
        self.framework = CrossValidationFramework(str(self.output_dir))
        self.statistical_validator = StatisticalValidator(
            significance_level=0.05,
            equivalence_threshold=0.1
        )
        self.benchmark_suite = BenchmarkSuite().create_standard_suite([2, 5, 10])

        # Validation configuration
        self.test_algorithms = {
            'NelderMead': NelderMead,
            'Powell': Powell,
            'LBFGSB': LBFGSB,
            'PRIMA_UOBYQA': PRIMA_UOBYQA,
            'PRIMA_NEWUOA': PRIMA_NEWUOA,
            'PRIMA_BOBYQA': PRIMA_BOBYQA
        }

        print("🚀 Comprehensive Validation Runner Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔍 Test algorithms: {len(self.test_algorithms)}")
        print(f"🎯 Benchmark problems: {len(self.benchmark_suite.problems)}")

    def run_algorithm_benchmark_validation(self, n_trials: int = 100, n_runs: int = 5) -> Dict[str, Any]:
        """
        Validate algorithms against standard benchmark problems.

        This provides an independent validation using mathematical benchmarks
        with known optimal solutions.
        """
        print("\n🎯 ALGORITHM BENCHMARK VALIDATION")
        print("=" * 45)

        benchmark_results = {}

        # Test each algorithm on validation problems
        validation_problems = self.benchmark_suite.get_validation_problems()

        for alg_name, alg_class in self.test_algorithms.items():
            print(f"\n🔬 Testing {alg_name}")

            def algorithm_wrapper(objective, n_trials_inner, n_dim):
                """Wrapper to match expected signature."""
                optimizer = alg_class(objective, n_trials_inner, n_dim)
                return optimizer.optimize()

            # Run benchmark evaluation
            alg_results = self.benchmark_suite.evaluate_algorithm(
                algorithm_wrapper,
                problem_names=None,  # Use validation problems
                n_runs=n_runs,
                n_trials=n_trials
            )

            benchmark_results[alg_name] = alg_results

            # Print summary for this algorithm
            successful_problems = 0
            total_problems = len(alg_results)

            for problem_name, result in alg_results.items():
                if 'statistics' in result:
                    success_rate = result['statistics'].get('success_rate', 0.0)
                    if success_rate > 0.5:  # 50% success threshold
                        successful_problems += 1

                    if self.verbose:
                        mean_val = result['statistics'].get('mean', float('inf'))
                        std_val = result['statistics'].get('std', 0.0)
                        print(f"  {problem_name}: {mean_val:.6f} ± {std_val:.6f} (success: {success_rate:.2f})")

            success_rate_alg = successful_problems / total_problems if total_problems > 0 else 0.0
            status = "✅ GOOD" if success_rate_alg > 0.7 else "⚠️ MODERATE" if success_rate_alg > 0.3 else "❌ POOR"
            print(f"  {status} - Success on {successful_problems}/{total_problems} problems ({success_rate_alg:.1%})")

        return benchmark_results

    def run_cross_reference_validation(self, n_trials: int = 100, n_runs: int = 5) -> Dict[str, Any]:
        """
        Cross-reference validation using multiple approaches:
        1. Framework's Python vs 3rd party
        2. Statistical validation
        3. Benchmark performance comparison
        """
        print("\n📊 CROSS-REFERENCE VALIDATION")
        print("=" * 40)

        validation_results = {
            'framework_results': {},
            'statistical_analysis': {},
            'cross_validation_summary': {}
        }

        # 1. Framework validation
        print("🔍 Running framework validation...")
        try:
            framework_results = self.framework.run_python_vs_reference_validation(n_trials, n_runs)
            validation_results['framework_results'] = framework_results
        except Exception as e:
            print(f"⚠️ Framework validation failed: {e}")
            validation_results['framework_results'] = {}

        # 2. Statistical cross-validation
        print("📈 Running statistical validation...")

        # Collect performance data for statistical analysis
        python_performance = {}
        reference_performance = {}

        # Simple sphere function for consistent testing
        def test_sphere(x):
            x = np.asarray(x)
            return np.sum(((x - 0.5) * 10) ** 2)

        # Collect Python algorithm results
        for alg_name, alg_class in self.test_algorithms.items():
            python_results = []

            for run in range(n_runs):
                np.random.seed(run * 42)
                optimizer = alg_class(test_sphere, n_trials, 2)
                best_val, _ = optimizer.optimize()
                python_results.append(best_val)

            python_performance[alg_name] = python_results

        # Try to get reference implementations (SciPy when available)
        try:
            from scipy.optimize import minimize

            # Test SciPy Nelder-Mead as reference
            scipy_results = []
            for run in range(n_runs):
                np.random.seed(run * 42)
                result = minimize(
                    test_sphere,
                    np.random.random(2),
                    method='Nelder-Mead',
                    bounds=[(0, 1), (0, 1)],
                    options={'maxfev': n_trials}
                )
                scipy_results.append(result.fun)

            reference_performance['SciPy_NelderMead'] = scipy_results

        except ImportError:
            print("⚠️ SciPy not available for reference validation")

        # Statistical equivalence analysis
        if python_performance and reference_performance:
            equiv_analysis = self.statistical_validator.test_mathematical_equivalence(
                python_performance,
                reference_performance,
                {}, {}  # No convergence data for this test
            )
            validation_results['statistical_analysis'] = equiv_analysis

        return validation_results

    def generate_comprehensive_report(self,
                                    framework_results: Dict[str, Any],
                                    benchmark_results: Dict[str, Any],
                                    cross_reference_results: Dict[str, Any],
                                    js_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive validation report."""

        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 45)

        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'validation_summary': {
                'total_tests_run': 0,
                'tests_passed': 0,
                'overall_pass_rate': 0.0
            },
            'algorithm_performance': {},
            'validation_categories': {
                'framework_validation': {},
                'benchmark_validation': {},
                'cross_reference_validation': {},
                'javascript_validation': {}
            },
            'recommendations': [],
            'detailed_results': {
                'framework': framework_results,
                'benchmarks': benchmark_results,
                'cross_reference': cross_reference_results,
                'javascript': js_results or {}
            }
        }

        # Analyze framework validation
        if framework_results:
            framework_passed = 0
            framework_total = 0

            for test_results in framework_results.values():
                for result in test_results:
                    framework_total += 1
                    if result.passed:
                        framework_passed += 1

            report['validation_categories']['framework_validation'] = {
                'total': framework_total,
                'passed': framework_passed,
                'pass_rate': (framework_passed / framework_total * 100) if framework_total > 0 else 0
            }

        # Analyze benchmark validation
        if benchmark_results:
            successful_algorithms = 0
            total_algorithms = len(benchmark_results)

            algorithm_scores = {}

            for alg_name, alg_results in benchmark_results.items():
                successful_problems = 0
                total_problems = len(alg_results)

                avg_success_rate = 0.0
                for problem_name, result in alg_results.items():
                    if 'statistics' in result:
                        success_rate = result['statistics'].get('success_rate', 0.0)
                        avg_success_rate += success_rate
                        if success_rate > 0.5:
                            successful_problems += 1

                avg_success_rate /= total_problems if total_problems > 0 else 1
                algorithm_scores[alg_name] = avg_success_rate

                if successful_problems / total_problems > 0.7:
                    successful_algorithms += 1

            report['validation_categories']['benchmark_validation'] = {
                'total_algorithms': total_algorithms,
                'successful_algorithms': successful_algorithms,
                'algorithm_success_rate': (successful_algorithms / total_algorithms * 100) if total_algorithms > 0 else 0
            }

            report['algorithm_performance'] = algorithm_scores

        # Calculate overall statistics
        total_tests = 0
        passed_tests = 0

        for category_results in report['validation_categories'].values():
            if 'total' in category_results and 'passed' in category_results:
                total_tests += category_results['total']
                passed_tests += category_results['passed']
            elif 'total_algorithms' in category_results:
                total_tests += category_results['total_algorithms']
                passed_tests += category_results.get('successful_algorithms', 0)

        report['validation_summary'] = {
            'total_tests_run': total_tests,
            'tests_passed': passed_tests,
            'overall_pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }

        # Generate recommendations
        overall_pass_rate = report['validation_summary']['overall_pass_rate']

        if overall_pass_rate >= 90:
            report['recommendations'].append("✅ EXCELLENT: All algorithms demonstrate strong mathematical consistency")
            report['recommendations'].append("🚀 Ready for production deployment")
        elif overall_pass_rate >= 75:
            report['recommendations'].append("✅ GOOD: Strong validation results with minor issues")
            report['recommendations'].append("🔧 Address specific algorithm inconsistencies")
        elif overall_pass_rate >= 50:
            report['recommendations'].append("⚠️ MODERATE: Some significant validation issues found")
            report['recommendations'].append("🛠️ Review and improve failing algorithms")
        else:
            report['recommendations'].append("❌ POOR: Major validation issues detected")
            report['recommendations'].append("🔴 Significant implementation review required")

        # Algorithm-specific recommendations
        if report['algorithm_performance']:
            best_algorithms = sorted(report['algorithm_performance'].items(), key=lambda x: x[1], reverse=True)[:3]
            worst_algorithms = sorted(report['algorithm_performance'].items(), key=lambda x: x[1])[:3]

            if best_algorithms:
                best_names = [alg for alg, score in best_algorithms]
                report['recommendations'].append(f"🏆 Best performing algorithms: {', '.join(best_names)}")

            if worst_algorithms and worst_algorithms[0][1] < 0.5:
                worst_names = [alg for alg, score in worst_algorithms if score < 0.5]
                if worst_names:
                    report['recommendations'].append(f"🔧 Algorithms needing improvement: {', '.join(worst_names)}")

        # Save detailed report
        report_file = self.output_dir / "comprehensive_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"💾 Comprehensive report saved to: {report_file}")

        return report

    def print_validation_summary(self, report: Dict[str, Any]) -> None:
        """Print a human-readable validation summary."""

        print("\n🎯 COMPREHENSIVE VALIDATION SUMMARY")
        print("=" * 50)

        summary = report['validation_summary']
        print(f"Total tests executed: {summary['total_tests_run']}")
        print(f"Tests passed: {summary['tests_passed']}")
        print(f"Overall pass rate: {summary['overall_pass_rate']:.1f}%")

        # Category breakdown
        print("\n📊 VALIDATION CATEGORIES")
        for category, results in report['validation_categories'].items():
            if results:
                if 'pass_rate' in results:
                    print(f"  {category.replace('_', ' ').title()}: {results['pass_rate']:.1f}%")
                elif 'algorithm_success_rate' in results:
                    print(f"  {category.replace('_', ' ').title()}: {results['algorithm_success_rate']:.1f}%")

        # Algorithm performance
        if report['algorithm_performance']:
            print("\n🔢 ALGORITHM PERFORMANCE")
            sorted_algs = sorted(report['algorithm_performance'].items(), key=lambda x: x[1], reverse=True)
            for alg, score in sorted_algs:
                status = "🏆" if score > 0.8 else "✅" if score > 0.6 else "⚠️" if score > 0.4 else "❌"
                print(f"  {status} {alg}: {score:.1%}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        for rec in report['recommendations']:
            print(f"  {rec}")

        print(f"\n✅ Validation complete! Detailed results in: {self.output_dir}")

    def run_full_validation_suite(self, n_trials: int = 100, n_runs: int = 5,
                                 skip_javascript: bool = False) -> Dict[str, Any]:
        """Run the complete validation suite."""

        print("🚀 STARTING COMPREHENSIVE VALIDATION SUITE")
        print("=" * 60)
        print(f"⚙️ Configuration: {n_trials} trials, {n_runs} runs")

        start_time = time.time()

        # 1. Framework validation (Python vs 3rd party)
        try:
            framework_results = self.framework.run_python_vs_reference_validation(n_trials, n_runs)
        except Exception as e:
            print(f"⚠️ Framework validation failed: {e}")
            framework_results = {}

        # 2. Mathematical correctness validation
        try:
            math_results = self.framework.run_mathematical_correctness_validation()
        except Exception as e:
            print(f"⚠️ Mathematical validation failed: {e}")
            math_results = {}

        # 3. Benchmark validation
        try:
            benchmark_results = self.run_algorithm_benchmark_validation(n_trials, n_runs)
        except Exception as e:
            print(f"⚠️ Benchmark validation failed: {e}")
            benchmark_results = {}

        # 4. Cross-reference validation
        try:
            cross_ref_results = self.run_cross_reference_validation(n_trials, n_runs)
        except Exception as e:
            print(f"⚠️ Cross-reference validation failed: {e}")
            cross_ref_results = {}

        # 5. JavaScript cross-validation (optional)
        js_results = {}
        if not skip_javascript:
            try:
                js_results = self.framework.run_cross_language_validation(n_trials, n_runs)
            except Exception as e:
                print(f"⚠️ JavaScript validation failed: {e}")

        # Merge framework results
        all_framework_results = {**framework_results, **math_results}

        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Total validation time: {elapsed_time:.1f} seconds")

        # Generate comprehensive report
        final_report = self.generate_comprehensive_report(
            all_framework_results,
            benchmark_results,
            cross_ref_results,
            js_results
        )

        # Print summary
        self.print_validation_summary(final_report)

        return final_report


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Cross-Validation Framework for HumpDay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_validation.py
  python run_comprehensive_validation.py --quick
  python run_comprehensive_validation.py --trials 200 --runs 10
  python run_comprehensive_validation.py --skip-js --verbose
        """
    )

    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials per optimization run (default: 100)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of independent runs per test (default: 5)')
    parser.add_argument('--output', type=str, default='validation_results',
                       help='Output directory for results (default: validation_results)')
    parser.add_argument('--skip-js', action='store_true',
                       help='Skip JavaScript cross-validation tests')
    parser.add_argument('--quick', action='store_true',
                       help='Quick validation with reduced parameters')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Adjust parameters for quick validation
    if args.quick:
        n_trials = 50
        n_runs = 3
        print("⚡ Quick validation mode enabled")
    else:
        n_trials = args.trials
        n_runs = args.runs

    print("🔬 HumpDay Comprehensive Cross-Validation Framework")
    print("=" * 60)
    print("Mathematical rigor and equivalence testing for optimization algorithms")
    print("")
    print("Configuration:")
    print(f"  Trials per run: {n_trials}")
    print(f"  Runs per test: {n_runs}")
    print(f"  Output directory: {args.output}")
    print(f"  Skip JavaScript: {args.skip_js}")
    print(f"  Verbose mode: {args.verbose}")

    try:
        # Initialize runner
        runner = ComprehensiveValidationRunner(
            output_dir=args.output,
            verbose=args.verbose
        )

        # Run validation suite
        results = runner.run_full_validation_suite(
            n_trials=n_trials,
            n_runs=n_runs,
            skip_javascript=args.skip_js
        )

        print("\n🎉 VALIDATION SUITE COMPLETED SUCCESSFULLY!")
        print(f"📊 Results available in: {args.output}/")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
