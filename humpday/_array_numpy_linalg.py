"""
Numpy backend for `humpday._array.linalg`.

Thin re-exports from numpy. Every name here must also exist with equivalent
semantics in `_array_pure_linalg.py`, and be tested under both backends.

Matrix convention
-----------------
The shim treats matrices as 2-D arrays / lists-of-rows. The pure backend
returns a `list[list[float]]`; the numpy backend returns `np.ndarray`. The
algorithm code that uses these should never `isinstance`-check the result.
"""

from __future__ import annotations

import numpy as _np

# ---- 2-D construction --------------------------------------------------------

eye = _np.eye


def matrix_zeros(rows, cols):
    return _np.zeros((rows, cols))


def outer(a, b):
    """Outer product of two 1-D vectors. Returns shape (len(a), len(b))."""
    return _np.outer(a, b)


def diag(vec):
    """Diagonal matrix built from a 1-D vector. Returns shape (n, n)."""
    return _np.diag(vec)


def diagonal(mat):
    """Extract the diagonal of a square 2-D matrix as a 1-D vector."""
    return _np.diag(mat)


# ---- Matrix-matrix / matrix-vector ------------------------------------------


def matmul(A, B):
    """2-D matrix x 2-D matrix."""
    return _np.matmul(A, B)


def matvec(A, x):
    """2-D matrix x 1-D vector."""
    return _np.matmul(A, x)


def transpose(A):
    return _np.transpose(A)


# ---- Linear algebra primitives ----------------------------------------------

solve = _np.linalg.solve
inv = _np.linalg.inv
cholesky = _np.linalg.cholesky
pinv = _np.linalg.pinv


def eigh(A):
    """Symmetric eigendecomposition. Returns (eigenvalues_asc, eigenvectors)
    where eigenvectors is a 2-D array whose columns are the unit eigenvectors,
    matching `numpy.linalg.eigh`."""
    return _np.linalg.eigh(A)


def svd(A, full_matrices=False):
    """Singular value decomposition. Returns (U, s, Vt) such that
    `A ≈ U @ diag(s) @ Vt`. Matches `numpy.linalg.svd(A, full_matrices=False)`
    by default — the thin / reduced form, which is what every humpday
    caller wants."""
    return _np.linalg.svd(A, full_matrices=full_matrices)


def qr(A):
    """QR factorisation. Returns (Q, R) such that `A = Q @ R`,
    matching `numpy.linalg.qr(A, mode='reduced')`."""
    return _np.linalg.qr(A)


__all__ = [
    "eye",
    "matrix_zeros",
    "outer",
    "diag",
    "diagonal",
    "matmul",
    "matvec",
    "transpose",
    "solve",
    "inv",
    "pinv",
    "cholesky",
    "eigh",
    "svd",
    "qr",
]
