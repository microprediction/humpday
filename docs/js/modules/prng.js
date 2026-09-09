/**
 * Portable PRNG: the JavaScript twin of humpday/_prng.py.
 *
 * PCG32 core (O'Neill 2014) with portable distributions. The Python
 * module is the reference; tests/test_prng_parity.py verifies these
 * streams are bit-for-bit identical. State arithmetic uses BigInt
 * (exact 64-bit); doubles are built from two 32-bit draws, and normals
 * use the Marsaglia polar method with portableLog (fixed IEEE-754
 * operation order) instead of Math.log, whose last-ulp behaviour is
 * runtime-specific. See the Python docstring for the full spec.
 */

const MASK64 = (1n << 64n) - 1n;
const PCG_MULT = 6364136223846793005n;

const LN2 = 0.6931471805599453;
const LN2_HI = 6.93147180369123816490e-01;
const LN2_LO = 1.90821492927058770002e-10;
const SQRT2 = 1.4142135623730951;
const INV_SQRT2 = 0.7071067811865476;
const LOG_TERMS = 25;

function portableLog(x) {
    if (x <= 0.0) {
        throw new Error("portableLog requires x > 0");
    }
    let m = x;
    let e = 0;
    while (m >= SQRT2) {
        m = m / 2.0;
        e += 1;
    }
    while (m < INV_SQRT2) {
        m = m * 2.0;
        e -= 1;
    }
    const t = (m - 1.0) / (m + 1.0);
    const t2 = t * t;
    let s = 0.0;
    let p = t;
    let k = 0;
    while (k < LOG_TERMS) {
        s = s + p / (2.0 * k + 1.0);
        p = p * t2;
        k += 1;
    }
    return 2.0 * s + e * LN2;
}

function portableExp(x) {
    if (Number.isNaN(x)) return x;
    if (x > 710.0) return Infinity;
    if (x < -745.0) return 0.0;
    let k = Math.floor(x / LN2 + 0.5);
    const r = (x - k * LN2_HI) - k * LN2_LO;
    let term = 1.0;
    let s = 1.0;
    for (let i = 1; i < 26; i++) {
        term = term * r / i;
        s = s + term;
    }
    while (k > 0) { s = s * 2.0; k -= 1; }
    while (k < 0) { s = s * 0.5; k += 1; }
    return s;
}

class PCG32 {
    constructor(seed, seq = 0) {
        this.inc = ((BigInt(seq) << 1n) | 1n) & MASK64;
        this.state = 0n;
        this._step();
        this.state = (this.state + (BigInt(seed) & MASK64)) & MASK64;
        this._step();
        this._cachedGauss = 0.0;
        this._hasCachedGauss = false;
    }

    _step() {
        this.state = (this.state * PCG_MULT + this.inc) & MASK64;
    }

    nextU32() {
        const old = this.state;
        this._step();
        const xorshifted = (((old >> 18n) ^ old) >> 27n) & 0xFFFFFFFFn;
        const rot = old >> 59n;
        const out = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31n))) & 0xFFFFFFFFn;
        return Number(out);
    }

    random() {
        const hi = this.nextU32() >>> 5;
        const lo = this.nextU32() >>> 6;
        return (hi * 67108864.0 + lo) / 9007199254740992.0;
    }

    uniform(a, b) {
        return a + (b - a) * this.random();
    }

    randbelow(n) {
        if (n <= 0) {
            throw new Error("randbelow requires n >= 1");
        }
        const threshold = Number((1n << 32n) % BigInt(n));
        for (;;) {
            const r = this.nextU32();
            if (r >= threshold) {
                return r % n;
            }
        }
    }

    randrange(low, high = null) {
        if (high === null) {
            return this.randbelow(low);
        }
        return low + this.randbelow(high - low);
    }

    gauss() {
        if (this._hasCachedGauss) {
            this._hasCachedGauss = false;
            return this._cachedGauss;
        }
        let u, v, s;
        for (;;) {
            u = 2.0 * this.random() - 1.0;
            v = 2.0 * this.random() - 1.0;
            s = u * u + v * v;
            if (s > 0.0 && s < 1.0) {
                break;
            }
        }
        const f = Math.sqrt(-2.0 * portableLog(s) / s);
        this._cachedGauss = v * f;
        this._hasCachedGauss = true;
        return u * f;
    }

    shuffle(seq) {
        for (let i = seq.length - 1; i >= 1; i--) {
            const j = this.randbelow(i + 1);
            const tmp = seq[i];
            seq[i] = seq[j];
            seq[j] = tmp;
        }
    }

    choice(seq) {
        return seq[this.randbelow(seq.length)];
    }

    sample(seq, k) {
        const pool = Array.from(seq);
        const n = pool.length;
        if (k < 0 || k > n) {
            throw new Error("sample size out of range");
        }
        for (let i = 0; i < k; i++) {
            const j = i + this.randbelow(n - i);
            const tmp = pool[i];
            pool[i] = pool[j];
            pool[j] = tmp;
        }
        return pool.slice(0, k);
    }
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { PCG32, portableLog, portableExp };
} else {
    window.PCG32 = PCG32;
    window.portableLog = portableLog;
    window.portableExp = portableExp;
}
