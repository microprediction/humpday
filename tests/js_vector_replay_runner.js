// Replays parity/transition_vectors.json against the JS optimizer roster.
//
// Usage: node js_vector_replay_runner.js <Algo1,Algo2,...>
//
// For each vector case whose optimizer is in the given list, seeds the
// portable PCG32 stream, drives the JS optimizer via suggestNext /
// receiveUpdate, and compares every point and value as IEEE-754 bit
// patterns. Emits one JSON verdict array on stdout.

const fs = require("fs");
const path = require("path");

const modules = require(path.resolve(__dirname, "../docs/js/modules/index.js"));
const { usePortableRng } = require(path.resolve(__dirname, "../docs/js/modules/base-optimizer.js"));

const VECTORS = JSON.parse(
    fs.readFileSync(path.resolve(__dirname, "../parity/transition_vectors.json"), "utf8")
);

const ALGOS = new Set(process.argv[2].split(","));

function bits(x) {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setFloat64(0, x, false);
    return Array.from(new Uint8Array(buf))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
}

const OBJECTIVES = {
    sphere03: (x) => {
        let s = 0.0;
        for (const v of x) {
            const d = v - 0.3;
            s = s + d * d;
        }
        return s;
    },
    rosen01: (x) => {
        let s = 0.0;
        for (let i = 0; i < x.length - 1; i++) {
            const a = x[i + 1] - x[i] * x[i];
            const b = 1.0 - x[i];
            s = s + 100.0 * (a * a) + b * b;
        }
        return s;
    },
};

const verdicts = [];
for (const c of VECTORS.cases) {
    if (!ALGOS.has(c.optimizer)) continue;
    const id = `${c.optimizer}-${c.objective}-d${c.n_dim}`;
    const objective = OBJECTIVES[c.objective];
    usePortableRng(BigInt(c.seed), BigInt(c.seq));
    const opt = new modules[c.optimizer](objective, c.n_trials, c.n_dim);
    let ok = true;
    let detail = "";
    let i = 0;
    for (;;) {
        const x = opt.suggestNext();
        if (x === null) break;
        if (i >= c.x.length) {
            ok = false;
            detail = `extra transition ${i}`;
            break;
        }
        const got = x.map(bits);
        if (JSON.stringify(got) !== JSON.stringify(c.x[i])) {
            ok = false;
            detail = `point ${i}: ${got} != ${c.x[i]}`;
            break;
        }
        const v = objective(x);
        if (bits(v) !== c.f[i]) {
            ok = false;
            detail = `value ${i}: ${bits(v)} != ${c.f[i]}`;
            break;
        }
        opt.receiveUpdate(v);
        i++;
    }
    if (ok && i !== c.x.length) {
        ok = false;
        detail = `ended early: ${i} < ${c.x.length}`;
    }
    verdicts.push({ id, ok, detail });
}

process.stdout.write(JSON.stringify(verdicts));
