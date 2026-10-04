// Evaluates the JavaScript BayesianOpt surrogate on a fixed design, for
// tests/test_js_bayesopt_gp.py.
//
// Usage: node js_bayesopt_gp_runner.js   (reads one JSON object on stdin)
//
// Input:  {"n_dim": d, "X": [[...], ...], "y": [...], "queries": [[...], ...]}
// Output: {"mu": [...], "sigma": [...], "ei": [...]}, one entry per query.
// JSON.stringify writes the shortest string that round-trips a double, so
// the Python side reads back the exact values JavaScript computed.

const path = require("path");
const { algorithms } = require(path.resolve(__dirname, "../docs/js/modules/index.js"));

const spec = JSON.parse(require("fs").readFileSync(0, "utf8"));
const b = new algorithms.BayesianOpt(() => 0, 50, spec.n_dim);
b.XObserved = spec.X.map((r) => [...r]);
b.yObserved = [...spec.y];

const out = { mu: [], sigma: [], ei: [] };
for (const q of spec.queries) {
    const [mu, sigma] = b._gpPredict(q);
    out.mu.push(mu);
    out.sigma.push(sigma);
    out.ei.push(b.acquisitionFunction(q));
}
process.stdout.write(JSON.stringify(out));
