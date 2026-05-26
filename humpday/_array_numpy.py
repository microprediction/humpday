"""
Numpy backend for `humpday._array`.

This is intentionally thin — for the most part it just re-exports numpy's
own names so the indirection is free at runtime. The point is API parity
with the pure backend (`humpday._array_pure`), not a wrapper.

Any function added here must also be added to `_array_pure.py` with
equivalent semantics, and tested in `tests/test_array_shim.py` under both
backends.
"""

from __future__ import annotations

import numpy as _np

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

asarray = _np.asarray
zeros = _np.zeros
ones = _np.ones
full = _np.full
arange = _np.arange

# ---------------------------------------------------------------------------
# Elementwise math
# ---------------------------------------------------------------------------

cos = _np.cos
sin = _np.sin
exp = _np.exp
log = _np.log
sqrt = _np.sqrt
abs = _np.abs  # noqa: A001 — intentional shadowing to match numpy's API

# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

sum = _np.sum  # noqa: A001
mean = _np.mean
min = _np.min  # noqa: A001
max = _np.max  # noqa: A001
argmin = _np.argmin
argmax = _np.argmax

# ---------------------------------------------------------------------------
# Vector ops
# ---------------------------------------------------------------------------

dot = _np.dot
clip = _np.clip


def norm(x):
    """Euclidean (L2) norm. Matches `numpy.linalg.norm(x)` for 1-D inputs."""
    return _np.linalg.norm(x)


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


def random_uniform(n):
    """Length-`n` vector of independent uniform [0, 1) samples."""
    return _np.random.random(n)


def random_normal(n):
    """Length-`n` vector of independent standard-normal samples."""
    return _np.random.randn(n)


def seed(s):
    """Seed the global RNG used by `random_uniform` / `random_normal`."""
    _np.random.seed(s)


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
    "seed",
]
