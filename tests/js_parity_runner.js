// Node runner for JavaScript optimization algorithms.
//
// Usage (called from the Python parity test):
//   node js_parity_runner.js <algorithm> <n_trials> <n_dim> <function_id>
//
// Outputs a single line of JSON: {"best_value": <float>, "best_x": [...]}
//
// <function_id> picks a Python-equivalent test objective, all defined on
// the unit hypercube [0,1]^n with a known minimum, so the Python and JS
// tests can run the same callable.

const path = require("path");
const modules = require(path.resolve(__dirname, "../docs/js/modules/index.js"));

const algorithm = process.argv[2];
const nTrials = parseInt(process.argv[3], 10);
const nDim = parseInt(process.argv[4], 10);
const funcId = process.argv[5];

// Match the Python test's `OBJECTIVES`. Keep these in sync.
const OBJECTIVES = {
    // Sphere centred at 0.5 — minimum 0 at x = [0.5, 0.5, ...]
    sphere_at_half: (x) => x.reduce((a, v) => a + (v - 0.5) * (v - 0.5), 0),
    // Quadratic centred at 0.7 — minimum 0 at x = [0.7, 0.7, ...]
    quad_at_0_7: (x) => x.reduce((a, v) => a + (v - 0.7) * (v - 0.7), 0),
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

try {
    const opt = new Cls(f, nTrials, nDim);
    opt.optimize();
    process.stdout.write(JSON.stringify({
        best_value: opt.bestValue,
        best_x: opt.bestX,
        evaluations: opt.evaluations,
    }));
} catch (e) {
    console.error(JSON.stringify({ error: String(e && e.message || e) }));
    process.exit(1);
}
