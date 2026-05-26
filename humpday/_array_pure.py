"""
Pure-Python backend for `humpday._array`.

Provides the same API as `humpday._array_numpy` using only the standard
library. Vectors are represented by `_Vec`, a thin `list` subclass that
implements elementwise arithmetic so optimizer code can keep using
`x + y`, `2 * x`, `x ** 2` regardless of whether numpy is installed.

Design notes
------------
* `_Vec` is a `list` subclass on purpose. Iteration, indexing, `len`,
  slicing, equality vs lists, and JSON-serialisation all "just work"
  without extra code. Only the arithmetic operators are overridden.

* Operators promote a Python scalar against a vector (numpy broadcasting),
  but only the 1-D case — there is no broadcasting between vectors of
  different lengths. Algorithms in humpday only need 1-D semantics today.

* `linalg` operations heavier than `norm` (eig, solve, inv) are deliberately
  not in this first cut; they are needed only by CMA-ES, BayesianOpt, and
  the PRIMA trio, which can be ported in follow-up PRs.

* Performance is intentionally not micro-optimised. The point of this
  backend is *correctness without numpy* — install-anywhere ergonomics,
  not speed. For speed, install numpy.
"""

from __future__ import annotations

import builtins as _builtins
import math
import random as _random
from typing import Iterable, Sequence, Union

Number = Union[int, float]


# ---------------------------------------------------------------------------
# Core value type
# ---------------------------------------------------------------------------


class _Vec(list):
    """List subclass with numpy-like elementwise arithmetic.

    Construction accepts any iterable. All arithmetic operators return a new
    `_Vec`; the operands are never mutated. Mixing a `_Vec` with a plain list
    or tuple of equal length is fine.
    """

    __slots__ = ()

    # Slicing returns a _Vec, not a plain list, so chained ops keep their
    # operator overloads.
    def __getitem__(self, key):  # type: ignore[override]
        result = list.__getitem__(self, key)
        if isinstance(key, slice):
            return _Vec(result)
        return result

    def copy(self):
        """Return an independent `_Vec` with the same elements. Overriding
        `list.copy` (which returns a plain `list`) keeps the type stable
        across `current = vec.copy()` round-trips, so arithmetic operators
        survive."""
        return _Vec(self)

    # ---- operator helpers ----

    def _binop(self, other, op):
        if isinstance(other, (int, float)):
            return _Vec(op(a, other) for a in self)
        if len(other) != len(self):
            raise ValueError(f"vector length mismatch: {len(self)} vs {len(other)}")
        return _Vec(op(a, b) for a, b in zip(self, other))

    def _rbinop(self, other, op):
        # `other` is always a scalar here — list/tuple `+ _Vec` would have
        # dispatched to the list's __add__ (concatenation) first; users
        # should put the _Vec on the left, or wrap with _Vec().
        return _Vec(op(other, a) for a in self)

    # ---- arithmetic ----

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b)

    __radd__ = lambda self, other: self._rbinop(other, lambda a, b: a + b)  # noqa: E731

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._rbinop(other, lambda a, b: a - b)

    def __mul__(self, other):
        # list's __mul__ does repetition; we want elementwise.
        return self._binop(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._rbinop(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._rbinop(other, lambda a, b: a / b)

    def __neg__(self):
        return _Vec(-a for a in self)

    def __pow__(self, other):
        return self._binop(other, lambda a, b: a**b)

    def __repr__(self):
        return f"_Vec({list.__repr__(self)})"


def _as_vec(x) -> _Vec:
    """Coerce input into a `_Vec`. Scalars become length-1 vectors? No — we
    only ever wrap iterables; scalars are passed through as scalars elsewhere."""
    if isinstance(x, _Vec):
        return x
    if isinstance(x, (list, tuple)):
        return _Vec(x)
    if hasattr(x, "__iter__"):
        return _Vec(x)
    raise TypeError(f"cannot convert {type(x).__name__} to _Vec")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def asarray(seq: Iterable) -> _Vec:
    """Coerce an iterable into a `_Vec`. Already-vec inputs pass through."""
    return _as_vec(seq)


def zeros(n: int) -> _Vec:
    return _Vec([0.0] * int(n))


def ones(n: int) -> _Vec:
    return _Vec([1.0] * int(n))


def full(n: int, value: Number) -> _Vec:
    return _Vec([value] * int(n))


def arange(n: int) -> _Vec:
    return _Vec(float(i) for i in range(int(n)))


# ---------------------------------------------------------------------------
# Elementwise math
# ---------------------------------------------------------------------------


def _elementwise(fn, x):
    if isinstance(x, (int, float)):
        return fn(x)
    return _Vec(fn(a) for a in x)


def cos(x):
    return _elementwise(math.cos, x)


def sin(x):
    return _elementwise(math.sin, x)


def exp(x):
    return _elementwise(math.exp, x)


def log(x):
    return _elementwise(math.log, x)


def sqrt(x):
    return _elementwise(math.sqrt, x)


def abs(x):  # noqa: A001 — match numpy API
    return _elementwise(_builtins.abs, x)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def sum(x):  # noqa: A001
    return _builtins.sum(x)


def mean(x):
    n = len(x)
    if n == 0:
        raise ValueError("mean of empty vector")
    return sum(x) / n


def min(x):  # noqa: A001
    return _builtins.min(x)


def max(x):  # noqa: A001
    return _builtins.max(x)


def argmin(x: Sequence) -> int:
    """Index of the smallest element. Numpy returns a numpy.intp; we return int."""
    if len(x) == 0:
        raise ValueError("argmin of empty vector")
    best_i = 0
    best_v = x[0]
    for i in range(1, len(x)):
        if x[i] < best_v:
            best_v = x[i]
            best_i = i
    return best_i


def argmax(x: Sequence) -> int:
    if len(x) == 0:
        raise ValueError("argmax of empty vector")
    best_i = 0
    best_v = x[0]
    for i in range(1, len(x)):
        if x[i] > best_v:
            best_v = x[i]
            best_i = i
    return best_i


# ---------------------------------------------------------------------------
# Vector ops
# ---------------------------------------------------------------------------


def dot(a: Sequence, b: Sequence) -> float:
    if len(a) != len(b):
        raise ValueError(f"dot: length mismatch {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def norm(x: Sequence) -> float:
    """Euclidean (L2) norm."""
    return math.sqrt(sum(v * v for v in x))


def clip(x, lo: Number, hi: Number):
    """Elementwise clamp to [lo, hi]. Scalar `lo`/`hi` only — that is what
    every existing humpday optimizer uses."""
    if isinstance(x, (int, float)):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x
    return _Vec(lo if v < lo else (hi if v > hi else v) for v in x)


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

# Module-level RNG so `seed()` affects subsequent draws, mirroring
# numpy.random's global-state semantics. Algorithm code that needs
# independent streams should instantiate its own random.Random later.
_rng = _random.Random()


def random_uniform(n: int) -> _Vec:
    return _Vec(_rng.random() for _ in range(int(n)))


def random_normal(n: int) -> _Vec:
    # Box-Muller via gauss(); single call per element is fine here.
    return _Vec(_rng.gauss(0.0, 1.0) for _ in range(int(n)))


def random_scalar() -> float:
    """A single uniform [0, 1) sample. Shares RNG state with `random_uniform`."""
    return _rng.random()


def random_int(low: int, high: int = None) -> int:
    """Random integer in `[low, high)`. If `high` is None, range is `[0, low)`.
    Matches `numpy.random.randint`'s default semantics."""
    if high is None:
        low, high = 0, low
    return _rng.randrange(int(low), int(high))


def random_choice(seq, k=None, replace=True):
    """Sample from `seq`. If `seq` is an int N, sample from `range(N)`.
    `k=None` returns a single item; otherwise returns a length-`k` list,
    with or without replacement to match `numpy.random.choice`."""
    if isinstance(seq, int):
        seq = list(range(seq))
    if k is None:
        return _rng.choice(seq)
    if replace:
        return [_rng.choice(seq) for _ in range(k)]
    return _rng.sample(seq, k)


def seed(s):
    _rng.seed(s)


__all__ = [
    # Construction
    "asarray",
    "zeros",
    "ones",
    "full",
    "arange",
    # Elementwise math
    "cos",
    "sin",
    "exp",
    "log",
    "sqrt",
    "abs",
    # Reductions
    "sum",
    "mean",
    "min",
    "max",
    "argmin",
    "argmax",
    # Vector ops
    "dot",
    "clip",
    "norm",
    # Random
    "random_uniform",
    "random_normal",
    "random_scalar",
    "random_int",
    "random_choice",
    "seed",
]
