"""
Random cube->cube diffeomorphisms — disguise the location of an optimum.

When optimizers (or their meta-tuning) are developed against a *fixed* benchmark
objective, they can quietly memorise *where* the optimum sits in cube
coordinates instead of learning to search. To prevent that, wrap each objective
in a seeded random diffeomorphism of the unit cube:

    g(u) = f(phi_seed(u))

`phi_seed: [0,1]^n -> [0,1]^n` is a smooth bijection, so `g` has exactly the same
landscape as `f` (same number and type of critical points, same optimal value)
but its optimum is relocated to `phi_seed^{-1}(u*)` — an unpredictable spot that
changes with the seed. Evaluate many seeds and an algorithm cannot win by
remembering a location; it has to actually optimise.

The default disguise is **separable** — a per-axis reflection, a per-axis
smooth monotone warp (Kumaraswamy CDF), and an axis permutation. It relocates
and rescales the optimum on every axis while preserving the problem's separability
structure and its multimodality. An optional Gaussian-space **rotation**
(`rotate=True`) additionally mixes the axes (probit -> orthogonal map -> normal
CDF), which disguises more aggressively but turns a separable problem
non-separable — use it deliberately.

Pure Python (math only); a small LCG makes every map reproducible from its seed.
"""

from __future__ import annotations

import math

_EPS = 1e-12


class _LCG:
    """Tiny reproducible PRNG (no global state, no `random` import)."""

    def __init__(self, seed):
        self.state = (seed * 2654435761 + 12345) & 0x7FFFFFFF

    def next(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state + 1) / 0x80000000  # in (0,1)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.next()


def _clip01(x):
    return min(1.0 - _EPS, max(_EPS, x))


def _kumaraswamy(x, a, b):
    """Kumaraswamy CDF: a smooth monotone bijection [0,1]->[0,1]."""
    return 1.0 - (1.0 - _clip01(x) ** a) ** b


def _kumaraswamy_inv(y, a, b):
    return (1.0 - (1.0 - _clip01(y)) ** (1.0 / b)) ** (1.0 / a)


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p):
    # Acklam's rational approximation to the inverse normal CDF.
    p = min(1.0 - 1e-12, max(1e-12, p))
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def _random_orthogonal(n, rng):
    """Random orthogonal n x n matrix via Gram-Schmidt on Gaussian columns."""
    cols = []
    for _ in range(n):
        v = [_norm_ppf(rng.next()) for _ in range(n)]
        for q in cols:
            dot = sum(v[i] * q[i] for i in range(n))
            v = [v[i] - dot * q[i] for i in range(n)]
        norm = math.sqrt(sum(c * c for c in v)) or 1.0
        cols.append([c / norm for c in v])
    # rows of the matrix
    return [[cols[j][i] for j in range(n)] for i in range(n)]


class CubeDisguise:
    """A seeded diffeomorphism of [0,1]^n and its inverse."""

    def __init__(self, n_dim, seed, rotate=False):
        self.n = n_dim
        self.rotate = rotate
        rng = _LCG(seed)
        self.reflect = [rng.next() < 0.5 for _ in range(n_dim)]
        self.a = [rng.uniform(0.7, 1.5) for _ in range(n_dim)]
        self.b = [rng.uniform(0.7, 1.5) for _ in range(n_dim)]
        # Fisher–Yates permutation
        perm = list(range(n_dim))
        for i in range(n_dim - 1, 0, -1):
            j = int(rng.next() * (i + 1))
            perm[i], perm[j] = perm[j], perm[i]
        self.perm = perm
        self.Q = _random_orthogonal(n_dim, rng) if rotate else None

    def forward(self, u):
        """Map a disguised-cube point to the true-objective cube point."""
        x = list(u)
        if self.rotate:
            z = [_norm_ppf(_clip01(x[i])) for i in range(self.n)]
            z = [sum(self.Q[i][j] * z[j] for j in range(self.n)) for i in range(self.n)]
            x = [_norm_cdf(zi) for zi in z]
        x = [(1.0 - x[i]) if self.reflect[i] else x[i] for i in range(self.n)]
        x = [_kumaraswamy(x[i], self.a[i], self.b[i]) for i in range(self.n)]
        return [x[self.perm[i]] for i in range(self.n)]

    def inverse(self, p):
        """Map a true-objective cube point back to disguised-cube coordinates
        (e.g. to locate where a known optimum was moved to)."""
        x = [0.0] * self.n
        for i in range(self.n):  # undo permutation
            x[self.perm[i]] = p[i]
        x = [_kumaraswamy_inv(x[i], self.a[i], self.b[i]) for i in range(self.n)]
        x = [(1.0 - x[i]) if self.reflect[i] else x[i] for i in range(self.n)]
        if self.rotate:
            z = [_norm_ppf(_clip01(x[i])) for i in range(self.n)]
            # inverse of an orthogonal map is its transpose
            z = [sum(self.Q[j][i] * z[j] for j in range(self.n)) for i in range(self.n)]
            x = [_norm_cdf(zi) for zi in z]
        return x


def disguise(objective, n_dim, seed, rotate=False):
    """Wrap `objective` ([0,1]^n -> cost) in a seeded cube->cube diffeomorphism.

    Returns a new objective with the identical landscape but its optimum
    relocated to an unpredictable, seed-dependent point — so an optimizer cannot
    score well by memorising a location."""
    d = CubeDisguise(n_dim, seed, rotate=rotate)

    def disguised(u):
        return objective(d.forward(u))

    disguised.disguise = d  # expose for optimum relocation / inversion
    return disguised
