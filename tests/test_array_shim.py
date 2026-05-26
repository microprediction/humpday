"""
Tests for the `humpday._array` shim.

Strategy
--------
Every test runs against BOTH backends — the numpy-backed implementation
(via `humpday._array_numpy`) and the pure-Python implementation (via
`humpday._array_pure`). The dispatch in `humpday._array` itself is
exercised by a separate handful of tests.

We import the backends *directly* rather than juggling environment
variables, because:
  * tests run in the same process and import caches don't unload cleanly;
  * direct import keeps the test failure message pointing at the right
    backend file.

The pure backend's `_Vec` and the numpy backend's `ndarray` are different
types, so we always compare via `_close(a, b)` which iterates and
tolerates floating-point noise.
"""

from __future__ import annotations

import importlib
import math

import pytest

from humpday import _array_numpy as A_np
from humpday import _array_pure as A_pure

BACKENDS = [
    pytest.param(A_pure, id="pure"),
    pytest.param(A_np, id="numpy"),
]


def _close(a, b, tol=1e-9):
    """Compare scalar or iterable values for near-equality."""
    if hasattr(a, "__iter__") and not isinstance(a, str):
        a_list = list(a)
        b_list = list(b)
        assert len(a_list) == len(b_list), (
            f"length mismatch: {len(a_list)} vs {len(b_list)}"
        )
        for ai, bi in zip(a_list, b_list):
            assert math.isclose(float(ai), float(bi), abs_tol=tol, rel_tol=tol), (
                f"element mismatch: {ai} vs {bi}"
            )
    else:
        assert math.isclose(float(a), float(b), abs_tol=tol, rel_tol=tol), (
            f"mismatch: {a} vs {b}"
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("A", BACKENDS)
def test_zeros_ones_full(A):
    _close(A.zeros(4), [0.0, 0.0, 0.0, 0.0])
    _close(A.ones(3), [1.0, 1.0, 1.0])
    _close(A.full(3, 2.5), [2.5, 2.5, 2.5])
    assert len(A.zeros(7)) == 7


@pytest.mark.parametrize("A", BACKENDS)
def test_asarray_and_arange(A):
    _close(A.asarray([1, 2, 3]), [1, 2, 3])
    _close(A.arange(4), [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Elementwise math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("A", BACKENDS)
def test_elementwise_math(A):
    x = A.asarray([0.0, math.pi / 2, math.pi])
    _close(A.cos(x), [1.0, 0.0, -1.0], tol=1e-12)
    _close(A.sin(x), [0.0, 1.0, 0.0], tol=1e-12)
    _close(A.exp(A.asarray([0.0, 1.0])), [1.0, math.e])
    _close(A.log(A.asarray([1.0, math.e])), [0.0, 1.0])
    _close(A.sqrt(A.asarray([0.0, 1.0, 4.0])), [0.0, 1.0, 2.0])
    _close(A.abs(A.asarray([-1.0, 0.0, 3.0])), [1.0, 0.0, 3.0])


@pytest.mark.parametrize("A", BACKENDS)
def test_elementwise_math_on_scalar(A):
    _close(A.cos(0.0), 1.0)
    _close(A.sqrt(4.0), 2.0)
    _close(A.abs(-7.5), 7.5)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("A", BACKENDS)
def test_reductions(A):
    x = A.asarray([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    _close(A.sum(x), 31.0)
    _close(A.mean(x), 31.0 / 8.0)
    _close(A.min(x), 1.0)
    _close(A.max(x), 9.0)
    assert int(A.argmin(x)) == 1
    assert int(A.argmax(x)) == 5


def test_pure_mean_empty_raises():
    # Documented divergence from numpy: the pure backend raises on
    # mean-of-empty rather than silently returning nan. Numpy returns
    # nan with a RuntimeWarning, which is its long-standing behaviour
    # and beyond this shim's scope to police.
    with pytest.raises((ValueError, ZeroDivisionError)):
        A_pure.mean(A_pure.asarray([]))


# ---------------------------------------------------------------------------
# Vector ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("A", BACKENDS)
def test_dot_and_norm(A):
    a = A.asarray([1.0, 2.0, 3.0])
    b = A.asarray([4.0, -5.0, 6.0])
    _close(A.dot(a, b), 1 * 4 + 2 * -5 + 3 * 6)  # = 12
    _close(A.norm(A.asarray([3.0, 4.0])), 5.0)


@pytest.mark.parametrize("A", BACKENDS)
def test_clip(A):
    x = A.asarray([-1.0, 0.5, 2.0, 3.5])
    _close(A.clip(x, 0.0, 2.0), [0.0, 0.5, 2.0, 2.0])
    _close(A.clip(1.5, 0.0, 1.0), 1.0)  # scalar input


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("A", BACKENDS)
def test_random_uniform_shape_and_range(A):
    A.seed(0)
    x = A.random_uniform(50)
    assert len(x) == 50
    for v in x:
        assert 0.0 <= v < 1.0


@pytest.mark.parametrize("A", BACKENDS)
def test_random_normal_shape(A):
    A.seed(0)
    x = A.random_normal(64)
    assert len(x) == 64
    # Very loose sanity: a sample of 64 standard-normals has |mean| < ~1.0
    # with overwhelming probability.
    assert abs(A.mean(x)) < 1.0


@pytest.mark.parametrize("A", BACKENDS)
def test_random_scalar_shape_and_range(A):
    A.seed(0)
    for _ in range(50):
        v = A.random_scalar()
        assert isinstance(v, float)
        assert 0.0 <= v < 1.0


@pytest.mark.parametrize("A", BACKENDS)
def test_random_scalar_shares_state_with_random_uniform(A):
    """random_scalar() consuming one draw must shift the next random_uniform
    output. Confirms both helpers share the same RNG (the seed() contract)."""
    A.seed(7)
    a = list(A.random_uniform(3))
    A.seed(7)
    _ = A.random_scalar()
    b = list(A.random_uniform(3))
    # `a` is the first three uniforms after the seed; `b` is the second,
    # third, fourth (because random_scalar() consumed the first). They
    # cannot be equal — that would mean random_scalar() didn't advance the
    # RNG, which would break reproducibility.
    assert a != b


@pytest.mark.parametrize("A", BACKENDS)
def test_seed_is_reproducible(A):
    A.seed(42)
    a = list(A.random_uniform(10))
    A.seed(42)
    b = list(A.random_uniform(10))
    _close(a, b, tol=0.0)


# ---------------------------------------------------------------------------
# Arithmetic (pure backend's _Vec specifically — numpy's ndarray is its own
# well-tested surface and doesn't need re-validating here)
# ---------------------------------------------------------------------------


def test_pure_vec_elementwise_arithmetic():
    a = A_pure.asarray([1.0, 2.0, 3.0])
    b = A_pure.asarray([10.0, 20.0, 30.0])
    _close(a + b, [11.0, 22.0, 33.0])
    _close(b - a, [9.0, 18.0, 27.0])
    _close(a * b, [10.0, 40.0, 90.0])
    _close(b / a, [10.0, 10.0, 10.0])
    _close(-a, [-1.0, -2.0, -3.0])
    _close(a**2, [1.0, 4.0, 9.0])


def test_pure_vec_scalar_broadcast():
    a = A_pure.asarray([1.0, 2.0, 3.0])
    _close(a + 1.0, [2.0, 3.0, 4.0])
    _close(2.0 * a, [2.0, 4.0, 6.0])  # __rmul__
    _close(a - 1.0, [0.0, 1.0, 2.0])
    _close(10.0 - a, [9.0, 8.0, 7.0])  # __rsub__
    _close(a / 2.0, [0.5, 1.0, 1.5])
    _close(6.0 / a, [6.0, 3.0, 2.0])  # __rtruediv__


def test_pure_vec_length_mismatch_raises():
    a = A_pure.asarray([1.0, 2.0, 3.0])
    b = A_pure.asarray([1.0, 2.0])
    with pytest.raises(ValueError, match="length mismatch"):
        _ = a + b


def test_pure_vec_slice_returns_vec():
    a = A_pure.asarray([1.0, 2.0, 3.0, 4.0])
    s = a[1:3]
    assert isinstance(s, A_pure._Vec)
    _close(s, [2.0, 3.0])
    # Sliced result still supports elementwise ops:
    _close(s + 10.0, [12.0, 13.0])


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def test_dispatch_picks_numpy_when_available():
    """In an env that has numpy installed (which every CI runner does),
    the dispatch module should select the numpy backend."""
    from humpday import _array

    importlib.reload(_array)  # respect any env changes a prior test made
    # On any normal dev box / CI runner, numpy is importable, so we should
    # land on the numpy backend.
    assert _array.BACKEND == "numpy"


def test_dispatch_force_pure(monkeypatch):
    """Setting HUMPDAY_FORCE_PURE_ARRAY=1 should pin the pure backend even
    when numpy is installed."""
    monkeypatch.setenv("HUMPDAY_FORCE_PURE_ARRAY", "1")
    from humpday import _array

    importlib.reload(_array)
    try:
        assert _array.BACKEND == "pure"
        # And the re-exported `zeros` should come from the pure backend.
        assert isinstance(_array.zeros(3), A_pure._Vec)
    finally:
        # Restore the default backend for subsequent tests.
        monkeypatch.delenv("HUMPDAY_FORCE_PURE_ARRAY", raising=False)
        importlib.reload(_array)


def test_both_backends_export_the_same_names():
    """API parity: every name in the pure backend's __all__ also exists in
    the numpy backend's __all__, and vice versa. This is the guarantee that
    optimizer code calling `humpday._array.X` works under either backend."""
    pure = set(A_pure.__all__)
    npy = set(A_np.__all__)
    extra_in_pure = pure - npy
    extra_in_npy = npy - pure
    assert not extra_in_pure, f"pure backend has extra names: {extra_in_pure}"
    assert not extra_in_npy, f"numpy backend has extra names: {extra_in_npy}"
