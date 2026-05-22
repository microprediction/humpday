#!/usr/bin/env python3
"""
Test key algorithms that were problematic
"""

import json
import os
import subprocess
import tempfile

# Test functions
TEST_FUNCTIONS = {
    "sphere2d": {
        "name": "2D Sphere",
        "js_func": "x => x[0]*x[0] + x[1]*x[1]",
        "target": 0.0,
    },
    "rosenbrock2d": {
        "name": "2D Rosenbrock",
        "js_func": "x => 100*(x[1] - x[0]*x[0])**2 + (1 - x[0])**2",
        "target": 0.0,
    },
}


def test_algorithm(algorithm_name, test_func_name, target_value):
    """Test a specific algorithm on a specific test function"""

    test_func = TEST_FUNCTIONS[test_func_name]

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

const testFunc = {test_func["js_func"]};

Math.seedrandom(42);

try {{
    const optimizer = OptimizerFactory.create('{algorithm_name}', testFunc, 100, 2);
    const result = optimizer.optimize();

    console.log(JSON.stringify({{
        success: true,
        bestValue: result.bestValue,
        bestX: result.bestX,
        evaluations: result.evaluations,
        algorithm: '{algorithm_name}'
    }}));
}} catch (error) {{
    console.log(JSON.stringify({{
        error: error.message,
        stack: error.stack
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
            js_result = json.loads(result.stdout.strip())
            if not js_result.get("error"):
                best_value = js_result.get("bestValue", float("inf"))
                return {
                    "success": True,
                    "value": best_value,
                    "algorithm": algorithm_name,
                    "test_func": test_func["name"],
                    "target": target_value,
                }
            else:
                return {
                    "success": False,
                    "error": js_result.get("error"),
                    "algorithm": algorithm_name,
                    "test_func": test_func["name"],
                }
        else:
            return {
                "success": False,
                "error": f"Execution failed: {result.stderr}",
                "algorithm": algorithm_name,
                "test_func": test_func["name"],
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "algorithm": algorithm_name,
            "test_func": test_func["name"],
        }
    finally:
        os.unlink(temp_file)


def main():
    print("🧪 Testing Key Previously-Problematic Algorithms...")
    print("=" * 60)

    # Test algorithms that were showing issues
    problem_algorithms = [
        "BayesianOpt",  # Was showing 0.0% win rate - FIXED
        "SimulatedAnnealing",  # Was showing 0.0% win rate
        "TabuSearch",  # Was showing 0.0% win rate on some tests
        "SciPy_BFGS",  # Had issues, should be fixed now
    ]

    results = []

    for algorithm in problem_algorithms:
        print(f"\n🔧 Testing {algorithm}:")

        for test_name, test_func in TEST_FUNCTIONS.items():
            print(f"  [{test_func['name']}]...", end=" ")

            result = test_algorithm(algorithm, test_name, test_func["target"])
            results.append(result)

            if result["success"]:
                value = result["value"]
                if value < 0.01:
                    print(f"✅ EXCELLENT ({value:.6f})")
                elif value < 0.1:
                    print(f"⚡ GOOD ({value:.6f})")
                elif value < 1.0:
                    print(f"⚠️  OK ({value:.6f})")
                else:
                    print(f"❌ POOR ({value:.6f})")
            else:
                print(f"❌ ERROR: {result['error']}")

    print("\n🎯 SUMMARY:")
    print("=" * 40)

    successful_tests = [r for r in results if r["success"]]
    excellent_results = [r for r in successful_tests if r["value"] < 0.01]
    good_results = [r for r in successful_tests if 0.01 <= r["value"] < 0.1]

    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Excellent (< 0.01): {len(excellent_results)}")
    print(f"Good (0.01-0.1): {len(good_results)}")

    if len(excellent_results) + len(good_results) >= len(successful_tests) * 0.75:
        print("🎉 GREAT PROGRESS! Algorithms are performing much better!")
    else:
        print("📈 Some improvement, but more optimization needed.")


if __name__ == "__main__":
    main()
