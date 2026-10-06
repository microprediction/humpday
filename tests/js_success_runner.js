// Runs every roster optimizer on a zero budget, an all-NaN objective, an
// all-infinite objective and a finite one, and reports what each result says.
// tests/test_js_success.py asserts success means a finite value was observed
// and that a successful result's bestX is a point the objective was called at (#396).
const path = require("path");

const modules = require(path.resolve(__dirname, "../docs/js/modules/index.js"));
const { Optimizer } = require(path.resolve(__dirname, "../docs/js/modules/base-optimizer.js"));

const CASES = [
    ["zero", () => 1, 0],
    ["nan", () => NaN, 12],
    ["infinity", () => Infinity, 12],
    ["finite", (x) => x.reduce((s, v) => s + (v - 0.3) * (v - 0.3), 0), 40],
];
const DIM = 3;
const out = [];

const roster = Object.keys(modules).filter(
    (k) => typeof modules[k] === "function" && modules[k].prototype instanceof Optimizer
);

function summarise(name, mode, label, budget, calls, seen, r) {
    const key = JSON.stringify(r.bestX);
    out.push({
        name, mode, label, budget, calls,
        evaluations: r.evaluations,
        success: r.success,
        message: r.message,
        bestValue: Number.isFinite(r.bestValue) ? r.bestValue : String(r.bestValue),
        bestXEvaluated: seen.has(key),
        valueAtBestX: seen.has(key) ? seen.get(key) : null,
    });
}

for (const name of roster) {
    for (const [label, f, budget] of CASES) {
        let calls = 0;
        let seen = new Map();
        const objective = (x) => {
            calls += 1;
            const v = f(x);
            seen.set(JSON.stringify(x), v);
            return v;
        };
        let opt = new modules[name](objective, budget, DIM);
        const result = opt.optimize();
        summarise(name, "optimize", label, budget, calls, seen, result);

        opt = new modules[name](objective, budget, DIM);
        if (typeof opt._run !== "function") continue; // legacy loop-owning port: no ask/tell
        calls = 0;
        seen = new Map();
        for (;;) {
            const x = opt.suggestNext();
            if (x === null) break;
            opt.receiveUpdate(objective(x));
        }
        summarise(name, "asktell", label, budget, calls, seen, opt._result());
    }
}
process.stdout.write(JSON.stringify(out));
