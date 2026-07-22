// Node runner for the portable-PRNG parity test.
//
// Usage: node js_prng_runner.js <seed> <seq> <n>
//
// Emits one JSON object with parallel streams drawn from independent
// PCG32 instances (one per stream, all seeded with <seed>/<seq>).
// Doubles are reported as big-endian IEEE-754 bit patterns (16 hex
// chars) so the Python side can compare exact bits, not decimal
// round-trips.

const path = require("path");
const { PCG32 } = require(path.resolve(__dirname, "../docs/js/modules/prng.js"));

const seed = BigInt(process.argv[2]);
const seq = BigInt(process.argv[3]);
const n = parseInt(process.argv[4], 10);

function bits(x) {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setFloat64(0, x, false);
    return Array.from(new Uint8Array(buf))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
}

const out = {};

let g = new PCG32(seed, seq);
out.u32 = Array.from({ length: n }, () => g.nextU32());

g = new PCG32(seed, seq);
out.random = Array.from({ length: n }, () => bits(g.random()));

g = new PCG32(seed, seq);
out.gauss = Array.from({ length: n }, () => bits(g.gauss()));

g = new PCG32(seed, seq);
out.uniform = Array.from({ length: n }, () => bits(g.uniform(-3.5, 11.25)));

g = new PCG32(seed, seq);
out.randbelow = [];
for (let i = 0; i < n; i++) {
    const m = [1, 2, 3, 7, 10, 100, 1000, 4294967295][i % 8];
    out.randbelow.push(g.randbelow(m));
}

g = new PCG32(seed, seq);
const arr = Array.from({ length: 30 }, (_, i) => i);
g.shuffle(arr);
out.shuffle = arr;

g = new PCG32(seed, seq);
out.sample = g.sample(Array.from({ length: 30 }, (_, i) => i), 12);

process.stdout.write(JSON.stringify(out));
