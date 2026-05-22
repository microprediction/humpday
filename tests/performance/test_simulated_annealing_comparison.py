#!/usr/bin/env python3
"""
Compare SimulatedAnnealing JS vs scipy reference
"""

import json
import os
import subprocess
import tempfile


def test_js_simulated_annealing():
    """Test JavaScript SimulatedAnnealing"""
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

const sphereFunc = x => x[0]*x[0] + x[1]*x[1];

Math.seedrandom(42);

try {{
    const optimizer = OptimizerFactory.create('SimulatedAnnealing', sphereFunc, 100, 2);
    const result = optimizer.optimize();

    console.log(JSON.stringify({{
        success: true,
        bestValue: result.bestValue,
        bestX: result.bestX,
        evaluations: result.evaluations
    }}));
}} catch (error) {{
    console.log(JSON.stringify({{
        error: error.message
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
            return json.loads(result.stdout.strip())
        else:
            return {"error": f"Execution failed: {result.stderr}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.unlink(temp_file)


def test_scipy_simulated_annealing():
    """Test scipy simulated annealing reference"""
    try:
        from scipy.optimize import dual_annealing

        def sphere_func(x):
            return x[0] ** 2 + x[1] ** 2

        bounds = [(0, 1), (0, 1)]

        # Use same random seed for reproducibility
        result = dual_annealing(sphere_func, bounds, maxfun=100, seed=42)

        return {
            "success": True,
            "bestValue": float(result.fun),
            "bestX": result.x.tolist(),
            "evaluations": result.nfev,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("🧪 SIMULATED ANNEALING COMPARISON")
    print("=" * 50)

    print("\n🔧 Testing JavaScript SimulatedAnnealing...")
    js_result = test_js_simulated_annealing()

    print("🔧 Testing scipy dual_annealing reference...")
    ref_result = test_scipy_simulated_annealing()

    print("\n📊 RESULTS:")
    print("-" * 30)

    if not js_result.get("error"):
        js_value = js_result.get("bestValue", float("inf"))
        print(f"JavaScript SimulatedAnnealing: {js_value:.6f}")
    else:
        print(f"JavaScript SimulatedAnnealing: ERROR - {js_result.get('error')}")

    if not ref_result.get("error"):
        ref_value = ref_result.get("bestValue", float("inf"))
        print(f"scipy dual_annealing:          {ref_value:.6f}")
    else:
        print(f"scipy dual_annealing:          ERROR - {ref_result.get('error')}")

    if not js_result.get("error") and not ref_result.get("error"):
        js_value = js_result.get("bestValue", float("inf"))
        ref_value = ref_result.get("bestValue", float("inf"))

        if ref_value > 0:
            ratio = js_value / ref_value
            print(f"Performance ratio:             {ratio:.2f}x (lower is better)")
        else:
            print("Reference achieved perfect score!")

        if js_value < ref_value * 2:
            print("✅ JavaScript is competitive!")
        elif js_value < ref_value * 10:
            print("⚠️  JavaScript is decent but could be better")
        else:
            print("❌ JavaScript needs significant improvement")


if __name__ == "__main__":
    main()
