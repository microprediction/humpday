#!/usr/bin/env python3
"""
Test improved ParticleSwarm on Rosenbrock function
"""

import json
import os
import subprocess
import tempfile

import numpy as np


def test_particle_swarm_rosenbrock():
    """Test ParticleSwarm on challenging Rosenbrock function"""

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

// Rosenbrock function: f(x) = 100*(x[1] - x[0]^2)^2 + (1 - x[0])^2
const rosenbrockFunc = x => 100 * Math.pow(x[1] - x[0]*x[0], 2) + Math.pow(1 - x[0], 2);

// Test multiple runs with different seeds
const results = [];

for (let run = 0; run < 5; run++) {{
    Math.seedrandom(42 + run);

    try {{
        const optimizer = OptimizerFactory.create('ParticleSwarm', rosenbrockFunc, 200, 2);
        const result = optimizer.optimize();

        results.push({{
            success: true,
            bestValue: result.bestValue,
            bestX: result.bestX,
            evaluations: result.evaluations,
            run: run
        }});
    }} catch (error) {{
        results.push({{
            error: error.message,
            run: run
        }});
    }}
}}

console.log(JSON.stringify(results));
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(js_code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ["node", temp_file], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        else:
            return {"error": f"Execution failed: {result.stderr}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.unlink(temp_file)


def main():
    print("🧪 IMPROVED PARTICLESWARM ROSENBROCK TEST")
    print("=" * 50)

    results = test_particle_swarm_rosenbrock()

    if isinstance(results, list):
        successful_runs = [r for r in results if not r.get("error")]

        if successful_runs:
            values = [r["bestValue"] for r in successful_runs]
            best_value = min(values)
            mean_value = np.mean(values)

            print(f"📊 Results from {len(successful_runs)} successful runs:")
            print(f"   Best: {best_value:.6f} (target: ~0.0)")
            print(f"   Mean: {mean_value:.6f}")

            # Count excellent results
            excellent_count = sum(1 for v in values if v < 0.01)
            good_count = sum(1 for v in values if v < 0.1) - excellent_count

            print(f"   Excellent (< 0.01): {excellent_count}/{len(values)}")
            print(f"   Good (< 0.1): {good_count}/{len(values)}")

            if best_value < 0.001:
                print("   ✅ EXCELLENT - Near-perfect Rosenbrock performance!")
            elif best_value < 0.01:
                print("   ⚡ VERY GOOD - Competitive with reference!")
            elif best_value < 0.1:
                print("   ⚠️  GOOD - Much improved!")
            else:
                print("   📈 IMPROVED - Better than before")

            # Show best solution
            best_run = min(successful_runs, key=lambda r: r["bestValue"])
            print(f"   Best X: {best_run['bestX']} (target: [1.0, 1.0])")
        else:
            print("❌ All runs failed!")
    else:
        print(f"❌ Test failed: {results.get('error')}")


if __name__ == "__main__":
    main()
