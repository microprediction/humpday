#!/usr/bin/env python3
"""
Quick test for SimulatedAnnealing
"""

import json
import os
import subprocess
import tempfile


def test_simulated_annealing():
    """Test SimulatedAnnealing on 2D sphere function"""

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

// 2D Sphere function: f(x) = x[0]^2 + x[1]^2
const sphereFunc = x => x[0]*x[0] + x[1]*x[1];

Math.seedrandom(42);

try {{
    const optimizer = OptimizerFactory.create('SimulatedAnnealing', sphereFunc, 100, 2);
    const result = optimizer.optimize();

    console.log(JSON.stringify({{
        success: true,
        bestValue: result.bestValue,
        bestX: result.bestX,
        evaluations: result.evaluations,
        algorithm: 'SimulatedAnnealing'
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
                print("🧪 SimulatedAnnealing Test Result:")
                print(f"   Best Value: {best_value:.6f} (target: ~0.0)")
                print(f"   Best X: {js_result.get('bestX', [])}")
                print(f"   Evaluations: {js_result.get('evaluations', 0)}")

                # Test against reference (SciPy simulated annealing typically gets ~0.0)
                if best_value < 0.01:
                    print("   ✅ EXCELLENT - Matching reference performance!")
                elif best_value < 0.1:
                    print("   ⚠️  GOOD - Close to reference")
                elif best_value < 1.0:
                    print("   ⚠️  DECENT - Reasonable but could be better")
                else:
                    print("   ❌ POOR - Far from reference performance")

                return js_result
            else:
                print(f"❌ JavaScript Error: {js_result.get('error')}")
                return None
        else:
            print(f"❌ Execution failed: {result.stderr}")
            return None

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    print("🚀 Testing SimulatedAnnealing...")
    result = test_simulated_annealing()
    if result:
        print("\n🎯 Result: Performance needs improvement for reference matching")
