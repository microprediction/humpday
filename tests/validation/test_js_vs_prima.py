#!/usr/bin/env python3
"""
JavaScript vs PRIMA Comparison Tests

Uses PRIMA as a testing dependency to validate JavaScript optimizer implementations
against the actual reference implementations.

Install: pip install prima numpy
"""

import numpy as np
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple, Any

try:
    from pdfo import uobyqa, newuoa, bobyqa
    PRIMA_AVAILABLE = True
    print("Using PDFO algorithms directly")
except ImportError:
    try:
        from prima import minimize
        PRIMA_AVAILABLE = True
        print("Using PRIMA wrapper")
    except ImportError:
        print("WARNING: PRIMA/PDFO not available. Install with: pip install prima or pip install pdfo")
        PRIMA_AVAILABLE = False


class JSPrimaComparator:
    """Compare JavaScript optimizer implementations with PRIMA reference"""

    def __init__(self):
        self.test_functions = {
            'sphere2d': {
                'name': '2D Sphere',
                'python_func': lambda x: x[0]**2 + x[1]**2,
                'js_func': 'x => x[0]*x[0] + x[1]*x[1]',
                'optimum': [0, 0],
                'optimum_value': 0,
                'dimensions': 2
            },
            'rosenbrock2d': {
                'name': '2D Rosenbrock',
                'python_func': lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2,
                'js_func': 'x => { const a = 1, b = 100; return (a - x[0])**2 + b * (x[1] - x[0]**2)**2; }',
                'optimum': [1, 1],
                'optimum_value': 0,
                'dimensions': 2
            },
            'sphere3d': {
                'name': '3D Sphere',
                'python_func': lambda x: np.sum(x**2),
                'js_func': 'x => x[0]*x[0] + x[1]*x[1] + x[2]*x[2]',
                'optimum': [0, 0, 0],
                'optimum_value': 0,
                'dimensions': 3
            }
        }

        self.algorithms_to_test = [
            'uobyqa', 'newuoa', 'bobyqa'  # PRIMA algorithms
        ]

        self.js_algorithm_mapping = {
            'uobyqa': 'PRIMA_UOBYQA',
            'newuoa': 'PRIMA_NEWUOA',
            'bobyqa': 'PRIMA_BOBYQA'
        }

    def run_prima_optimization(self, algorithm: str, func_name: str, max_evals: int = 200) -> Dict[str, Any]:
        """Run PRIMA optimization on test function"""
        if not PRIMA_AVAILABLE:
            return {'error': 'PRIMA not available'}

        test_func = self.test_functions[func_name]
        n_dim = test_func['dimensions']

        # Random starting point in [0,1]
        np.random.seed(42)  # Fixed seed for reproducibility
        x0 = np.random.uniform(0, 1, n_dim)

        # Set bounds for all PRIMA algorithms to [0,1]
        bounds = [(0, 1) for _ in range(n_dim)]

        try:
            # Use specific PDFO functions
            if algorithm == 'uobyqa':
                # UOBYQA doesn't support bounds
                result = uobyqa(
                    test_func['python_func'],
                    x0,
                    options={'maxfev': max_evals, 'rhobeg': 0.1, 'rhoend': 1e-6}
                )
            elif algorithm == 'newuoa':
                # NEWUOA doesn't support bounds
                result = newuoa(
                    test_func['python_func'],
                    x0,
                    options={'maxfev': max_evals, 'rhobeg': 0.1, 'rhoend': 1e-6}
                )
            elif algorithm == 'bobyqa':
                # BOBYQA supports bounds
                result = bobyqa(
                    test_func['python_func'],
                    x0,
                    bounds=bounds,
                    options={'maxfev': max_evals, 'rhobeg': 0.1, 'rhoend': 1e-6}
                )
            else:
                return {'error': f'Unknown algorithm: {algorithm}'}

            return {
                'success': result.success,
                'x': result.x.tolist(),
                'fun': float(result.fun),
                'nfev': int(result.nfev),
                'message': result.message,
                'algorithm': algorithm
            }
        except Exception as e:
            return {'error': str(e)}

    def run_js_optimization(self, algorithm: str, func_name: str, max_evals: int = 200) -> Dict[str, Any]:
        """Run JavaScript optimization using Node.js"""
        test_func = self.test_functions[func_name]
        js_algorithm = self.js_algorithm_mapping[algorithm]

        # Create temporary JavaScript test file
        js_code = f"""
// Load the optimizer implementations
const fs = require('fs');
const optimizerCode = fs.readFileSync('./docs/js/optimizers.js', 'utf8');
// Make OptimizerFactory global for Node.js
eval('(function(){{' + optimizerCode + '; if (typeof OptimizerFactory !== \"undefined\") global.OptimizerFactory = OptimizerFactory; }})()');
const OptimizerFactory = global.OptimizerFactory;

// Set fixed seed for reproducibility
Math.seedrandom = function(seed) {{
    let m = 0x80000000;
    let a = 1103515245;
    let c = 12345;
    let state = seed ? seed : Math.floor(Math.random() * (m - 1));

    Math.random = function() {{
        state = (a * state + c) % m;
        return state / (m - 1);
    }};
}}
Math.seedrandom(42);

// Test function
const testFunc = {test_func['js_func']};

// Run optimization
try {{
    const optimizer = OptimizerFactory.create('{js_algorithm}', testFunc, {max_evals}, {test_func['dimensions']});
    const result = optimizer.optimize();

    console.log(JSON.stringify({{
        success: true,
        x: result.bestX,
        fun: result.bestValue,
        nfev: result.evaluations,
        algorithm: '{js_algorithm}'
    }}));
}} catch (error) {{
    console.log(JSON.stringify({{
        error: error.message
    }}));
}}
"""

        # Write temporary file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(js_code)
            temp_file = f.name

        try:
            # Run Node.js
            result = subprocess.run(
                ['node', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return json.loads(result.stdout.strip())
            else:
                return {'error': f'JavaScript execution failed: {result.stderr}'}

        except subprocess.TimeoutExpired:
            return {'error': 'JavaScript execution timed out'}
        except Exception as e:
            return {'error': str(e)}
        finally:
            os.unlink(temp_file)

    def compare_results(self, prima_result: Dict, js_result: Dict, func_name: str) -> Dict[str, Any]:
        """Compare PRIMA and JavaScript results"""
        test_func = self.test_functions[func_name]

        comparison = {
            'function': test_func['name'],
            'prima_success': prima_result.get('success', False),
            'js_success': js_result.get('success', False),
            'prima_error': prima_result.get('error'),
            'js_error': js_result.get('error')
        }

        if comparison['prima_success'] and comparison['js_success']:
            prima_x = np.array(prima_result['x'])
            js_x = np.array(js_result['x'])

            # Distance to true optimum
            true_opt = np.array(test_func['optimum'])
            prima_dist = np.linalg.norm(prima_x - true_opt)
            js_dist = np.linalg.norm(js_x - true_opt)

            # Function value accuracy
            true_fval = test_func['optimum_value']
            prima_fval_error = abs(prima_result['fun'] - true_fval)
            js_fval_error = abs(js_result['fun'] - true_fval)

            comparison.update({
                'prima_final_value': prima_result['fun'],
                'js_final_value': js_result['fun'],
                'prima_evaluations': prima_result['nfev'],
                'js_evaluations': js_result['nfev'],
                'prima_distance_to_optimum': prima_dist,
                'js_distance_to_optimum': js_dist,
                'prima_fval_error': prima_fval_error,
                'js_fval_error': js_fval_error,
                'solution_similarity': np.linalg.norm(prima_x - js_x),
                'js_matches_prima': {
                    'converged_to_same_region': np.linalg.norm(prima_x - js_x) < 0.1,
                    'similar_function_value': abs(prima_result['fun'] - js_result['fun']) < 0.01,
                    'reasonable_evaluations': abs(prima_result['nfev'] - js_result['nfev']) < prima_result['nfev'] * 0.5
                }
            })

        return comparison

    def run_comparison_suite(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run full comparison suite between JavaScript and PRIMA"""
        results = {}

        for algorithm in self.algorithms_to_test:
            results[algorithm] = []

            print(f"\\nTesting {algorithm.upper()}:")
            print("-" * 50)

            for func_name in self.test_functions:
                print(f"  Testing on {self.test_functions[func_name]['name']}...")

                # Run PRIMA optimization
                prima_result = self.run_prima_optimization(algorithm, func_name)

                # Run JavaScript optimization
                js_result = self.run_js_optimization(algorithm, func_name)

                # Compare results
                comparison = self.compare_results(prima_result, js_result, func_name)

                results[algorithm].append(comparison)

                # Print immediate feedback
                if comparison['prima_success'] and comparison['js_success']:
                    matches = comparison['js_matches_prima']
                    status = "✅" if all(matches.values()) else "⚠️"
                    print(f"    {status} PRIMA: {prima_result['fun']:.6f} | JS: {js_result['fun']:.6f}")
                else:
                    print(f"    ❌ Errors - PRIMA: {comparison['prima_error']} | JS: {comparison['js_error']}")

        return results

    def generate_report(self, results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate detailed comparison report"""
        report = ["# JavaScript vs PRIMA Comparison Report\\n"]

        for algorithm, test_results in results.items():
            report.append(f"## {algorithm.upper()} Results\\n")

            total_tests = len(test_results)
            successful_comparisons = sum(1 for r in test_results
                                       if r['prima_success'] and r['js_success'])

            report.append(f"**Success Rate:** {successful_comparisons}/{total_tests} tests\\n")

            for result in test_results:
                func_name = result['function']
                report.append(f"### {func_name}\\n")

                if result['prima_success'] and result['js_success']:
                    matches = result['js_matches_prima']

                    report.append(f"- **PRIMA Result:** f = {result['prima_final_value']:.6f}, evals = {result['prima_evaluations']}")
                    report.append(f"- **JavaScript Result:** f = {result['js_final_value']:.6f}, evals = {result['js_evaluations']}")
                    report.append(f"- **Distance to True Optimum:** PRIMA = {result['prima_distance_to_optimum']:.6f}, JS = {result['js_distance_to_optimum']:.6f}")
                    report.append(f"- **Solution Similarity:** {result['solution_similarity']:.6f}")

                    status = "✅ MATCH" if all(matches.values()) else "⚠️ DIFFERS"
                    report.append(f"- **Overall Match:** {status}\\n")

                    if not all(matches.values()):
                        report.append("  **Issues:**")
                        if not matches['converged_to_same_region']:
                            report.append("  - Solutions in different regions")
                        if not matches['similar_function_value']:
                            report.append("  - Significantly different function values")
                        if not matches['reasonable_evaluations']:
                            report.append("  - Very different evaluation counts")
                        report.append("")
                else:
                    report.append(f"- **PRIMA Error:** {result['prima_error']}")
                    report.append(f"- **JavaScript Error:** {result['js_error']}\\n")

        return "\\n".join(report)


def main():
    """Run the comparison suite"""
    if not PRIMA_AVAILABLE:
        print("PRIMA/PDFO package required for testing. Install with:")
        print("pip install prima numpy  # or pip install pdfo")
        return

    print("🧪 JavaScript vs PRIMA Optimization Comparison")
    print("=" * 60)

    comparator = JSPrimaComparator()
    results = comparator.run_comparison_suite()

    # Generate and save report
    report = comparator.generate_report(results)

    with open('js_prima_comparison_report.md', 'w') as f:
        f.write(report)

    print(f"\\n📊 Full report saved to: js_prima_comparison_report.md")

    # Save raw results as JSON
    with open('js_prima_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"📊 Raw results saved to: js_prima_results.json")


if __name__ == '__main__':
    main()