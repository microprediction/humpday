"""
Tests for `humpday._array.linalg` — the linear-algebra primitives.

Every test runs against both backends. The numpy backend is the reference;
the pure-Python backend must agree element-wise within floating-point
tolerance.

Matrices are passed in as plain `list[list[float]]`. The numpy backend
accepts these (numpy will coerce); the pure backend uses them directly.
Results are compared after `list`-coercing so the type difference between
`ndarray` and `list` doesn't trip equality.
"""

from __future__ import annotations

import math

import pytest

from humpday import _array_numpy_linalg as L_np
from humpday import _array_pure_linalg as L_pure

BACKENDS = [
    pytest.param(L_pure, id="pure"),
    pytest.param(L_np, id="numpy"),
]


def _close_scalar(a, b, tol=1e-9):
    assert math.isclose(float(a), float(b), abs_tol=tol, rel_tol=tol), (
        f"scalar mismatch: {a} vs {b}"
    )


def _close_vec(a, b, tol=1e-9):
    a_list = [float(x) for x in a]
    b_list = [float(x) for x in b]
    assert len(a_list) == len(b_list)
    for x, y in zip(a_list, b_list):
        _close_scalar(x, y, tol=tol)


def _close_matrix(A, B, tol=1e-9):
    A_list = [list(row) for row in A]
    B_list = [list(row) for row in B]
    assert len(A_list) == len(B_list), (
        f"row count mismatch: {len(A_list)} vs {len(B_list)}"
    )
    for ra, rb in zip(A_list, B_list):
        _close_vec(ra, rb, tol=tol)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", BACKENDS)
def test_eye(L):
    _close_matrix(L.eye(3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    _close_matrix(L.eye(1), [[1.0]])


@pytest.mark.parametrize("L", BACKENDS)
def test_matrix_zeros(L):
    Z = L.matrix_zeros(2, 3)
    _close_matrix(Z, [[0, 0, 0], [0, 0, 0]])


# ---------------------------------------------------------------------------
# Matrix-matrix / matrix-vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", BACKENDS)
def test_matmul_2x2_3x3(L):
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    # AB = [[19, 22], [43, 50]]
    _close_matrix(L.matmul(A, B), [[19, 22], [43, 50]])

    # Multiplying by identity is a no-op.
    I = L.eye(2)
    _close_matrix(L.matmul(A, I), A)
    _close_matrix(L.matmul(I, A), A)


@pytest.mark.parametrize("L", BACKENDS)
def test_matvec(L):
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = [1.0, 0.0, -1.0]
    # Ax = [1*1 + 2*0 + 3*-1, 4*1 + 5*0 + 6*-1] = [-2, -2]
    _close_vec(L.matvec(A, x), [-2.0, -2.0])


@pytest.mark.parametrize("L", BACKENDS)
def test_transpose(L):
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    _close_matrix(L.transpose(A), [[1, 4], [2, 5], [3, 6]])
    # Double transpose returns the original.
    _close_matrix(L.transpose(L.transpose(A)), A)


# ---------------------------------------------------------------------------
# Linear solve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", BACKENDS)
def test_solve_2x2(L):
    # [[2, 1], [5, 7]] x = [11, 13]  ->  x = [7.11..., -3.22...]
    A = [[2.0, 1.0], [5.0, 7.0]]
    b = [11.0, 13.0]
    x = L.solve(A, b)
    # Verify by reconstruction: A @ x should equal b.
    Ax = L.matvec(A, list(x))
    _close_vec(Ax, b, tol=1e-12)


@pytest.mark.parametrize("L", BACKENDS)
def test_solve_3x3_random(L):
    # A deliberately well-conditioned 3x3 system.
    A = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]
    b = [7.5, 4.5, 7.5]
    x = L.solve(A, b)
    _close_vec(L.matvec(A, list(x)), b, tol=1e-10)


@pytest.mark.parametrize("L", BACKENDS)
def test_solve_singular_raises(L):
    # Row 2 is twice row 1 — singular.
    A = [[1.0, 2.0], [2.0, 4.0]]
    b = [3.0, 6.0]
    # Both backends should raise — the type differs (numpy raises LinAlgError,
    # pure raises ValueError) so we accept the union via base Exception.
    with pytest.raises(Exception, match="(?i)singular|matrix"):
        L.solve(A, b)


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", BACKENDS)
def test_inv_identity(L):
    I = L.eye(4)
    _close_matrix(L.inv(I), [[1 if i == j else 0 for j in range(4)] for i in range(4)])


@pytest.mark.parametrize("L", BACKENDS)
def test_inv_roundtrip(L):
    A = [[4.0, 7.0], [2.0, 6.0]]
    Ainv = L.inv(A)
    # A @ A^-1 should be the identity.
    _close_matrix(L.matmul(A, Ainv), [[1, 0], [0, 1]], tol=1e-12)
    # And A^-1 @ A.
    _close_matrix(L.matmul(Ainv, A), [[1, 0], [0, 1]], tol=1e-12)


# ---------------------------------------------------------------------------
# Cholesky
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", BACKENDS)
def test_cholesky_identity(L):
    I = L.eye(3)
    Lmat = L.cholesky(I)
    _close_matrix(Lmat, I)


@pytest.mark.parametrize("L", BACKENDS)
def test_cholesky_roundtrip(L):
    # SPD matrix: start with M @ M.T for some lower-triangular M.
    A = [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
    Lmat = L.cholesky(A)
    # Lower-triangular by construction — check L @ L.T == A.
    LT = L.transpose(Lmat)
    _close_matrix(L.matmul(Lmat, LT), A, tol=1e-10)


@pytest.mark.parametrize("L", BACKENDS)
def test_cholesky_non_spd_raises(L):
    # Negative diagonal — not positive definite.
    A = [[1.0, 2.0], [2.0, 1.0]]  # eigenvalues 3 and -1
    with pytest.raises(Exception):
        L.cholesky(A)


# ---------------------------------------------------------------------------
# Symmetric eigendecomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", BACKENDS)
def test_eigh_diagonal(L):
    # Diagonal matrix: eigenvalues are the diagonal entries (sorted ascending).
    A = [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]
    vals, vecs = L.eigh(A)
    _close_vec(vals, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("L", BACKENDS)
def test_eigh_simple_2x2(L):
    # A = [[2, 1], [1, 2]]  ->  eigenvalues 1, 3 ; eigenvectors [(-1,1)/√2, (1,1)/√2]
    A = [[2.0, 1.0], [1.0, 2.0]]
    vals, vecs = L.eigh(A)
    _close_vec(vals, [1.0, 3.0], tol=1e-10)


@pytest.mark.parametrize("L", BACKENDS)
def test_eigh_reconstruction(L):
    # For any symmetric A, A @ v_k = lambda_k * v_k for each eigenpair.
    A = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]
    vals, vecs = L.eigh(A)
    n = len(A)
    for k in range(n):
        v_k = [vecs[i][k] for i in range(n)]
        Av = L.matvec(A, v_k)
        lam_v = [vals[k] * v_k[i] for i in range(n)]
        _close_vec(Av, lam_v, tol=1e-8)


def test_pure_eigh_non_symmetric_raises():
    # The numpy backend silently symmetrises (it reads only the lower
    # triangle by default), so this guarantee is pure-only.
    A = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="symmetric"):
        L_pure.eigh(A)
