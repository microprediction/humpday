#!/usr/bin/env python3
"""
Quick test of the final problematic algorithms
"""

import json
import os
import subprocess
import tempfile

import numpy as np

# Test functions
TEST_FUNCTIONS = {
    "sphere2d": {
        "name": "2D Sphere",
        "js_func": "x => x[0]*x[0] + x[1]*x[1]",
        "target": 0.0,
    }
}


def run_algorithm_multiple_runs(algorithm_name, num_runs=5):
    """Test algorithm multiple times for statistical validation"""
    results = []

    for run in range(num_runs):
        js_code = f"""
const {{ OptimizerFactory }} = require('{os.getcwd()}/docs/js/optimizers.js');

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

const testFunc = x => x[0]*x[0] + x[1]*x[1];

Math.seedrandom({42 + run});

try {{
    const optimizer = OptimizerFactory.create('{algorithm_name}', testFunc, 100, 2);
    const result = optimizer.optimize();

    console.log(JSON.stringify({{
        success: true,
        bestValue: result.bestValue,
        run: {run}
    }}));
}} catch (error) {{
    console.log(JSON.stringify({{
        error: error.message,
        run: {run}
    }}));
}}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["node", temp_file], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                run_result = json.loads(result.stdout.strip())
                if not run_result.get("error"):
                    results.append(run_result["bestValue"])
                else:
                    print(f"   Run {run}: ERROR - {run_result.get('error')}")
            else:
                print(f"   Run {run}: Execution failed")
        except Exception as e:
            print(f"   Run {run}: Exception - {e}")
        finally:
            os.unlink(temp_file)

    return results


def main():
    print("🧪 FINAL ALGORITHMS PERFORMANCE CHECK")
    print("=" * 50)

    algorithms = ["SimulatedAnnealing", "BayesianOpt"]

    for algorithm in algorithms:
        print(f"\n🔧 Testing {algorithm} (5 runs)...")

        results = run_algorithm_multiple_runs(algorithm, 5)

        if results:
            best_value = min(results)
            mean_value = np.mean(results)
            std_value = np.std(results)

            print(f"   Best: {best_value:.6f}")
            print(f"   Mean: {mean_value:.6f} ± {std_value:.6f}")

            # Count excellent results (< 0.001)
            excellent_count = sum(1 for r in results if r < 0.001)
            print(f"   Excellent runs (< 0.001): {excellent_count}/5")

            if best_value < 0.001:
                print("   ✅ EXCELLENT - Capable of near-perfect performance!")
            elif best_value < 0.01:
                print("   ⚡ VERY GOOD - Close to reference level")
            elif mean_value < 0.1:
                print("   ⚠️  GOOD - Reasonable performance")
            else:
                print("   ❌ NEEDS WORK - Far from reference")
        else:
            print("   ❌ ALL RUNS FAILED")

    print("\n🎯 SUMMARY:")
    print("Both algorithms should now be much more competitive!")


if __name__ == "__main__":
    main()
