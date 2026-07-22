"""Portable PRNG: bit-exact random streams across languages.

This module is the *reference implementation* of humpday's portable
random number generator. The JavaScript twin is
`docs/js/modules/prng.js`; Rust/Julia/R ports must reproduce these
streams bit-for-bit (verified by `tests/test_prng_parity.py`). Every
operation below is specified so that a faithful implementation in any
language with IEEE-754 doubles produces identical outputs:

- The core is PCG32 (O'Neill 2014, `pcg32_random_r`): 64-bit LCG state,
  32-bit XSH-RR output. Integer arithmetic is exact in every language
  (Python ints, JS BigInt, Rust u64, Julia UInt64; R needs a two-limb
  emulation).
- Doubles in [0, 1) are built from exactly two 32-bit draws giving a
  53-bit mantissa: (hi27 * 2**26 + lo26) / 2**53. Integer-valued doubles
  below 2**53 and division by a power of two are exact in IEEE-754.
- Bounded integers use the unbiased threshold-rejection method
  (OpenBSD arc4random_uniform), consuming a variable but deterministic
  number of 32-bit draws.
- Normals use the Marsaglia polar method with sqrt (correctly rounded
  by IEEE-754, hence portable) and `portable_log` below — NOT the
  platform libm log, whose last-ulp behaviour differs between runtimes.
  The second normal of each polar pair is cached and returned by the
  next call.
- `portable_log` is an atanh-series evaluation with a fixed operation
  order: only +, -, *, / on doubles, so it is bit-exact everywhere. It
  agrees with libm log to ~1 ulp, but its exact value is defined by the
  algorithm, not by any libm.

Nothing here is cryptographic; the goal is cross-language determinism
for optimizer trajectories and parity vectors.
"""

_MASK64 = (1 << 64) - 1
_PCG_MULT = 6364136223846793005

# Fixed double constants (their exact IEEE-754 values are part of the spec).
_LN2 = 0.6931471805599453
_SQRT2 = 1.4142135623730951
_INV_SQRT2 = 0.7071067811865476
_LOG_TERMS = 25


def portable_log(x):
    """Natural log of a positive finite double, computed with a fixed
    sequence of IEEE-754 +,-,*,/ operations so every language gets the
    same bits. Range-reduce x = m * 2^e with m in [sqrt(1/2), sqrt(2)),
    then ln m = 2 * atanh((m-1)/(m+1)) by series."""
    if x <= 0.0:
        raise ValueError("portable_log requires x > 0")
    m = x
    e = 0
    while m >= _SQRT2:
        m = m / 2.0
        e += 1
    while m < _INV_SQRT2:
        m = m * 2.0
        e -= 1
    t = (m - 1.0) / (m + 1.0)
    t2 = t * t
    s = 0.0
    p = t
    k = 0
    while k < _LOG_TERMS:
        s = s + p / (2.0 * k + 1.0)
        p = p * t2
        k += 1
    return 2.0 * s + e * _LN2


class PCG32:
    """PCG32 with portable distributions on top. Seeding follows
    O'Neill's pcg32_srandom_r: given (initstate, initseq), set
    state = 0, inc = (initseq << 1) | 1, step, add initstate, step."""

    def __init__(self, seed, seq=0):
        self.inc = ((int(seq) << 1) | 1) & _MASK64
        self.state = 0
        self._step()
        self.state = (self.state + (int(seed) & _MASK64)) & _MASK64
        self._step()
        self._cached_gauss = None
        self._has_cached_gauss = False

    def _step(self):
        self.state = (self.state * _PCG_MULT + self.inc) & _MASK64

    def next_u32(self):
        """One 32-bit output (XSH-RR on the pre-step state)."""
        old = self.state
        self._step()
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def random(self):
        """Uniform double in [0, 1) with 53 random bits (two u32 draws:
        high word first)."""
        hi = self.next_u32() >> 5
        lo = self.next_u32() >> 6
        return (hi * 67108864.0 + lo) / 9007199254740992.0

    def uniform(self, a, b):
        """Uniform double in [a, b): a + (b - a) * random(), in that order."""
        return a + (b - a) * self.random()

    def randbelow(self, n):
        """Unbiased integer in [0, n) (threshold rejection; n in [1, 2^32])."""
        if n <= 0:
            raise ValueError("randbelow requires n >= 1")
        threshold = (1 << 32) % n
        while True:
            r = self.next_u32()
            if r >= threshold:
                return r % n

    def randrange(self, low, high=None):
        """Integer in [low, high); randrange(n) means [0, n)."""
        if high is None:
            return self.randbelow(low)
        return low + self.randbelow(high - low)

    def gauss(self):
        """Standard normal via the Marsaglia polar method. Each accepted
        pair (u, v) consumes exactly four u32 draws (two doubles) plus
        rejections; the v-normal is cached for the next call."""
        if self._has_cached_gauss:
            self._has_cached_gauss = False
            return self._cached_gauss
        while True:
            u = 2.0 * self.random() - 1.0
            v = 2.0 * self.random() - 1.0
            s = u * u + v * v
            if 0.0 < s < 1.0:
                break
        f = (-2.0 * portable_log(s) / s) ** 0.5
        self._cached_gauss = v * f
        self._has_cached_gauss = True
        return u * f

    def shuffle(self, seq):
        """In-place Fisher-Yates, descending: for i = len-1 .. 1,
        j = randbelow(i + 1), swap seq[i], seq[j]."""
        i = len(seq) - 1
        while i >= 1:
            j = self.randbelow(i + 1)
            seq[i], seq[j] = seq[j], seq[i]
            i -= 1

    def choice(self, seq):
        """One element: seq[randbelow(len(seq))]."""
        return seq[self.randbelow(len(seq))]

    def sample(self, seq, k):
        """k distinct elements by partial front Fisher-Yates: copy seq;
        for i = 0 .. k-1, j = i + randbelow(len - i), swap; return the
        first k."""
        pool = list(seq)
        n = len(pool)
        if not 0 <= k <= n:
            raise ValueError("sample size out of range")
        i = 0
        while i < k:
            j = i + self.randbelow(n - i)
            pool[i], pool[j] = pool[j], pool[i]
            i += 1
        return pool[:k]
