// Drive a JavaScript port on the same objectives the Python reference harness uses, so the JS
// side can be compared against scipy and friends directly rather than only against the Python
// port (#164).
//
// Usage: node js_reference_runner.js <algorithm> <problem> <n_trials> <n_dim> <seed>
// Prints: {"best_value": <float>, "evaluations": <int>}
//
// The objectives here must stay identical to PROBLEMS in tests/test_reference_alignment.py.
// Their optima sit at (0.4127, 0.6831) rather than at the centre of the cube or on a grid node,
// because an optimum an optimizer can arrive at without searching measures aim (#387).

const path = require("path");
const modules = require(path.resolve(__dirname, "../docs/js/modules/index.js"));
const { usePortableRng } = require(path.resolve(
    __dirname,
    "../docs/js/modules/base-optimizer.js",
));

const OPT = [0.4127, 0.6831];

const PROBLEMS = {
    sphere: (x) => x.reduce((a, v, i) => a + (v - OPT[i % 2]) * (v - OPT[i % 2]), 0),
    rosenbrock: (x) => {
        const a = 4 * (x[0] - OPT[0] + 0.5) - 2 + 1.0;
        const b = 4 * (x[1] - OPT[1] + 0.5) - 2 + 1.0;
        return Math.pow(1 - a, 2) + 100 * Math.pow(b - a * a, 2);
    },
    ackley: (x) => {
        const n = x.length;
        const s = x.map((v, i) => 10 * (v - OPT[i % 2]));
        const sumsq = s.reduce((a, v) => a + v * v, 0) / n;
        const sumcos = s.reduce((a, v) => a + Math.cos(2 * Math.PI * v), 0) / n;
        return -20 * Math.exp(-0.2 * Math.sqrt(sumsq)) - Math.exp(sumcos) + 20 + Math.E;
    },
};

const algorithm = process.argv[2];
const problem = process.argv[3];
const nTrials = parseInt(process.argv[4], 10);
const nDim = parseInt(process.argv[5], 10);
const seed = parseInt(process.argv[6], 10);

const f = PROBLEMS[problem];
const Cls = modules.algorithms[algorithm];
if (!f) {
    console.error(JSON.stringify({ error: `unknown problem ${problem}` }));
    process.exit(2);
}
if (!Cls) {
    console.error(JSON.stringify({ error: `unknown algorithm ${algorithm}` }));
    process.exit(2);
}

// The portable stream, so a run is reproducible and so the seed means the same thing it means
// on the Python side.
usePortableRng(seed, 0);
const opt = new Cls(f, nTrials, nDim);
opt.optimize();
process.stdout.write(
    JSON.stringify({ best_value: opt.bestValue, evaluations: opt.evaluations }) + "\n",
);
