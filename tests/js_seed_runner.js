// Run every registered JavaScript optimizer under usePortableRng and report what it did, so the
// Python side can check that a seed actually determines the run (#401).
//
// Usage: node js_seed_runner.js <n_trials> <n_dim> <seedA> <seedB>
// Prints: {"<algorithm>": {"a1": trace, "a2": trace, "b": trace, "error": null}, ...}
// where a trace is the list of every point evaluated plus the best value, as exact strings.
//
// While the portable stream is active Math.random is replaced by a function that throws, so a
// port that bypasses MathUtils fails loudly here instead of quietly drawing an unseeded number.

const path = require("path");
const modules = require(path.resolve(__dirname, "../docs/js/modules/index.js"));
const { usePortableRng, useLegacyRng } = require(path.resolve(
    __dirname,
    "../docs/js/modules/base-optimizer.js",
));

const nTrials = parseInt(process.argv[2], 10);
const nDim = parseInt(process.argv[3], 10);
const seedA = parseInt(process.argv[4], 10);
const seedB = parseInt(process.argv[5], 10);

// Shifted, curved and multimodal enough that no optimizer stops after a handful of points.
const OPT = [0.4127, 0.6831, 0.2219];
const objective = (x) => {
    let s = 0;
    for (let i = 0; i < x.length; i++) {
        const d = x[i] - OPT[i % OPT.length];
        s += d * d + 0.1 * (1 - Math.cos(12 * d));
    }
    return s;
};

const realRandom = Math.random;

function run(Cls, seed) {
    const trace = [];
    const f = (x) => {
        trace.push(Array.from(x, (v) => v.toPrecision(17)).join(","));
        return objective(x);
    };
    usePortableRng(seed, 0);
    Math.random = () => {
        throw new Error("Math.random called while the portable stream is active");
    };
    try {
        const opt = new Cls(f, nTrials, nDim);
        opt.optimize();
        return { best: String(opt.bestValue), points: trace };
    } finally {
        Math.random = realRandom;
        useLegacyRng();
    }
}

const out = {};
const seen = new Set();
for (const [name, Cls] of Object.entries(modules.algorithms)) {
    if (seen.has(Cls)) continue; // aliases such as AdaptiveRandomSearch
    seen.add(Cls);
    try {
        out[name] = { a1: run(Cls, seedA), a2: run(Cls, seedA), b: run(Cls, seedB), error: null };
    } catch (e) {
        out[name] = { error: String(e && e.stack ? e.stack : e) };
    }
}
process.stdout.write(JSON.stringify(out) + "\n");
