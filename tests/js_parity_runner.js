// Node runner for JavaScript optimization algorithms.
//
// Usage:
//   node js_parity_runner.js <algorithm> <n_trials> <n_dim> <function_id> [<n_runs>]
//
// With n_runs > 1, the algorithm is run `n_runs` times back-to-back in
// the same Node process and one JSON line is emitted per run:
//   {"best_value": <float>, "best_x": [...], "evaluations": <int>}
//
// Batching avoids paying Node startup cost per trial — the win-rate
// test in test_js_parity.py needs ~10 runs per algorithm.

const path = require("path");
const modules = require(path.resolve(__dirname, "../docs/js/modules/index.js"));

const algorithm = process.argv[2];
const nTrials = parseInt(process.argv[3], 10);
const nDim = parseInt(process.argv[4], 10);
const funcId = process.argv[5];
const nRuns = process.argv[6] ? parseInt(process.argv[6], 10) : 1;

// Match the Python test's `OBJECTIVES`. Keep these in sync.
const OBJECTIVES = {
    // Sphere centred at 0.5 — minimum 0 at x = [0.5, 0.5, ...]
    sphere_at_half: (x) => x.reduce((a, v) => a + (v - 0.5) * (v - 0.5), 0),
    // Quadratic centred at 0.7 — minimum 0 at x = [0.7, 0.7, ...]
    quad_at_0_7: (x) => x.reduce((a, v) => a + (v - 0.7) * (v - 0.7), 0),
    // 2-D Rosenbrock mapped to [0,1]^2 via x_real = 4*xi - 2, so the
    // minimum sits at xi = 0.75 (giving x_real = 1). Ill-conditioned,
    // discriminating between algorithms.
    rosenbrock_unit: (x) => {
        const a = 4 * x[0] - 2;
        const b = 4 * x[1] - 2;
        return Math.pow(1 - a, 2) + 100 * Math.pow(b - a * a, 2);
    },
};

const f = OBJECTIVES[funcId];
if (!f) {
    console.error(JSON.stringify({ error: `unknown function_id ${funcId}` }));
    process.exit(2);
}

const Cls = modules.algorithms[algorithm];
if (!Cls) {
    console.error(JSON.stringify({ error: `unknown algorithm ${algorithm}` }));
    process.exit(2);
}

for (let i = 0; i < nRuns; i++) {
    try {
        const opt = new Cls(f, nTrials, nDim);
        opt.optimize();
        process.stdout.write(
            JSON.stringify({
                best_value: opt.bestValue,
                best_x: opt.bestX,
                evaluations: opt.evaluations,
            }) + "\n",
        );
    } catch (e) {
        process.stdout.write(
            JSON.stringify({ error: String((e && e.message) || e) }) + "\n",
        );
    }
}
