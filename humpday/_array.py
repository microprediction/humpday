"""
Tiny array shim for humpday.

Why this exists
---------------
humpday's optimizer code uses a small set of array primitives: construction,
elementwise math, reductions, a few linear-algebra calls. Today every
implementation calls into `numpy` directly. That makes numpy a hard runtime
dependency, which means humpday cannot ship as truly "pure Python" — the
install footprint, the Pyodide story, and the embedded-Python story all
inherit numpy's constraints.

This module is the abstraction layer. Optimizer code calls into
`humpday._array` (or its eventual `linalg` submodule) instead of importing
`numpy as np`. The shim selects a backend at import time:

  - If `numpy` is installed, the numpy backend transparently re-exports it.
    No performance penalty — the indirection is one module reference.
  - If `numpy` is not installed, the pure backend provides equivalent
    operations on plain Python lists wrapped in a small `_Vec` class.

This lets a single codebase ship two install modes:

  pip install humpday           -> pure-Python, ~87 KB wheel, no deps
  pip install humpday[fast]     -> numpy-backed, full-speed for n > ~20

Backend selection
-----------------
By default the shim picks numpy when available. Setting the environment
variable `HUMPDAY_FORCE_PURE_ARRAY=1` forces the pure backend even when
numpy is installed; this lets the test suite exercise both code paths in
the same CI run.

The active backend is exposed as `humpday._array.BACKEND` ('numpy' or 'pure').
"""

from __future__ import annotations

import os

_FORCE_PURE = os.environ.get("HUMPDAY_FORCE_PURE_ARRAY", "") == "1"

if _FORCE_PURE:
    from . import _array_pure as _impl
    from . import _array_pure_linalg as linalg
    from ._array_pure import *  # noqa: F401,F403

    BACKEND = "pure"
else:
    try:
        import numpy as _np  # noqa: F401

        from . import _array_numpy as _impl
        from . import _array_numpy_linalg as linalg
        from ._array_numpy import *  # noqa: F401,F403

        BACKEND = "numpy"
    except ImportError:
        from . import _array_pure as _impl
        from . import _array_pure_linalg as linalg
        from ._array_pure import *  # noqa: F401,F403

        BACKEND = "pure"


# Re-export the canonical name list from whichever backend is live so callers
# can introspect what's available without caring which backend it is.
# `linalg` is the submodule namespace; `numpy.linalg`-shaped API lives there.
__all__ = ["BACKEND", "linalg", *_impl.__all__]


# ---------------------------------------------------------------------------
# Portable RNG mode
# ---------------------------------------------------------------------------
# By default (legacy mode) the random primitives below delegate to the live
# backend (numpy's global RNG or the pure backend's random.Random), and the
# rng_* facade delegates to the stdlib global `random` module — preserving
# every historical seeded trajectory. After use_portable_rng(seed), ALL of
# them draw from one PCG32 stream (humpday._prng), which is bit-identical
# across languages — the basis for cross-language optimizer parity vectors.

import random as _stdlib_random  # noqa: E402

_backend_random_uniform = random_uniform  # noqa: F405
_backend_random_normal = random_normal  # noqa: F405
_backend_random_scalar = random_scalar  # noqa: F405
_backend_random_int = random_int  # noqa: F405
_backend_random_choice = random_choice  # noqa: F405
_backend_seed = seed  # noqa: F405

_portable = None


def use_portable_rng(s, seq=0):
    """Switch all humpday randomness to a portable PCG32 stream."""
    global _portable
    from ._prng import PCG32

    _portable = PCG32(s, seq)
    return _portable


def use_legacy_rng():
    """Return to the historical backend/stdlib random sources."""
    global _portable
    _portable = None


def random_uniform(n):
    if _portable is None:
        return _backend_random_uniform(n)
    return asarray([_portable.random() for _ in range(int(n))])  # noqa: F405


def random_normal(n):
    if _portable is None:
        return _backend_random_normal(n)
    return asarray([_portable.gauss() for _ in range(int(n))])  # noqa: F405


def random_scalar():
    if _portable is None:
        return _backend_random_scalar()
    return _portable.random()


def random_int(low, high=None):
    if _portable is None:
        return _backend_random_int(low, high)
    return _portable.randrange(int(low), None if high is None else int(high))


def random_choice(seq, k=None, replace=True):
    if _portable is None:
        return _backend_random_choice(seq, k=k, replace=replace)
    if isinstance(seq, int):
        seq = list(range(seq))
    if k is None:
        return _portable.choice(seq)
    if replace:
        return [_portable.choice(seq) for _ in range(k)]
    return _portable.sample(seq, k)


def seed(s):
    if _portable is None:
        return _backend_seed(s)
    return use_portable_rng(s)


# Stdlib-shaped facade for optimizers that historically called the global
# `random` module directly (Alloy, HarmonySearch, NEWUOA/BOBYQA restarts).
# Legacy mode delegates to the SAME stdlib global instance, so rewiring an
# optimizer from `random.x(...)` to `_A.rng_x(...)` leaves its seeded
# trajectory bit-for-bit unchanged.


def rng_random():
    return _portable.random() if _portable is not None else _stdlib_random.random()


def rng_gauss():
    return (
        _portable.gauss() if _portable is not None else _stdlib_random.gauss(0.0, 1.0)
    )


def rng_uniform(a, b):
    return (
        _portable.uniform(a, b)
        if _portable is not None
        else _stdlib_random.uniform(a, b)
    )


def rng_choice(seq):
    return (
        _portable.choice(seq) if _portable is not None else _stdlib_random.choice(seq)
    )


def rng_shuffle(seq):
    if _portable is not None:
        _portable.shuffle(seq)
    else:
        _stdlib_random.shuffle(seq)


def rng_randrange(n):
    return (
        _portable.randrange(n) if _portable is not None else _stdlib_random.randrange(n)
    )


__all__ += [
    "use_portable_rng",
    "use_legacy_rng",
    "rng_random",
    "rng_gauss",
    "rng_uniform",
    "rng_choice",
    "rng_shuffle",
    "rng_randrange",
]
