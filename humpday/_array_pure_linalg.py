"""
Pure-Python linear algebra for `humpday._array.linalg`.

Provides the operations that humpday's heavier algorithms (CMA-ES,
BayesianOpt, PRIMA trio, LBFGSB) need from `numpy.linalg`, without
requiring numpy.

Matrix representation
---------------------
A matrix is a `list[list[float]]` — i.e. a list of row lists. Shape is
`(len(M), len(M[0]))`. We intentionally do *not* introduce a Matrix class:
keeping the data type as plain nested lists means JSON-serialises for free,
indexes naturally (`M[i][j]`), and never surprises algorithm code with an
unexpected operator overload.

Numerical scope
---------------
Robust enough for small-to-moderate dimensions (n ≤ a few hundred). The
algorithms in humpday that exercise these primitives generally run with
n ≤ 50; correctness matters more than micro-optimisation. For speed on
large matrices, install numpy and the dispatch in `humpday._array` selects
the numpy backend automatically.

Algorithms used
---------------
* `solve`    — Gaussian elimination with partial pivoting.
* `inv`      — Solve `A x = e_k` for each standard basis vector.
* `cholesky` — Standard Cholesky-Banachiewicz, raises if not SPD.
* `eigh`     — Jacobi rotations on the off-diagonal pairs of a symmetric
               matrix. Converges quadratically once entries are small.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ._array_pure import _Vec

# Tolerance below which we treat a pivot as zero.
_PIVOT_TOL = 1e-14


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------


def eye(n: int) -> List[List[float]]:
    """`n x n` identity matrix."""
    n = int(n)
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matrix_zeros(rows: int, cols: int) -> List[List[float]]:
    rows, cols = int(rows), int(cols)
    return [[0.0] * cols for _ in range(rows)]


# ---------------------------------------------------------------------------
# Matrix-matrix / matrix-vector
# ---------------------------------------------------------------------------


def matmul(A, B):
    """2-D `A` times 2-D `B`. No broadcasting — both must be matrices."""
    n_rows = len(A)
    n_inner = len(A[0])
    n_cols = len(B[0])
    if len(B) != n_inner:
        raise ValueError(
            f"matmul: incompatible shapes ({n_rows}x{n_inner}) x ({len(B)}x{n_cols})"
        )
    out = matrix_zeros(n_rows, n_cols)
    for i in range(n_rows):
        Ai = A[i]
        out_i = out[i]
        for k in range(n_inner):
            a_ik = Ai[k]
            Bk = B[k]
            for j in range(n_cols):
                out_i[j] += a_ik * Bk[j]
    return out


def matvec(A, x: Sequence) -> _Vec:
    """2-D `A` times 1-D `x`. Returns a `_Vec`."""
    n_rows = len(A)
    n_cols = len(A[0])
    if len(x) != n_cols:
        raise ValueError(
            f"matvec: incompatible shapes ({n_rows}x{n_cols}) x ({len(x)},)"
        )
    out = [0.0] * n_rows
    for i in range(n_rows):
        Ai = A[i]
        s = 0.0
        for k in range(n_cols):
            s += Ai[k] * x[k]
        out[i] = s
    return _Vec(out)


def transpose(A):
    n_rows = len(A)
    n_cols = len(A[0])
    return [[A[i][j] for i in range(n_rows)] for j in range(n_cols)]


# ---------------------------------------------------------------------------
# Linear solve
# ---------------------------------------------------------------------------


def solve(A, b: Sequence) -> _Vec:
    """Solve `A x = b` for a square `A`. Gaussian elimination with partial
    pivoting. `A` and `b` are not mutated."""
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("solve: A must be square")
    if len(b) != n:
        raise ValueError(f"solve: dimension mismatch, A is {n}x{n} but b has length {len(b)}")

    # Working copy. Use list of lists; flatten for cache-friendly traversal
    # is overkill at the sizes we target.
    M = [list(row) for row in A]
    rhs = list(b)

    for k in range(n):
        # Partial pivot — find the row with the largest |M[i][k]| at or below k.
        pivot_row = k
        pivot_val = abs(M[k][k])
        for i in range(k + 1, n):
            v = abs(M[i][k])
            if v > pivot_val:
                pivot_val = v
                pivot_row = i
        if pivot_val < _PIVOT_TOL:
            raise ValueError("solve: singular matrix (pivot below tolerance)")
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
            rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]
        # Eliminate below the pivot.
        pivot = M[k][k]
        for i in range(k + 1, n):
            factor = M[i][k] / pivot
            if factor == 0.0:
                continue
            Mi = M[i]
            Mk = M[k]
            Mi[k] = 0.0
            for j in range(k + 1, n):
                Mi[j] -= factor * Mk[j]
            rhs[i] -= factor * rhs[k]

    # Back-substitute.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        Mi = M[i]
        for j in range(i + 1, n):
            s -= Mi[j] * x[j]
        x[i] = s / Mi[i]
    return _Vec(x)


def inv(A):
    """Inverse of square `A`, computed column-by-column via `solve`."""
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("inv: A must be square")
    cols = []
    for k in range(n):
        e_k = [1.0 if i == k else 0.0 for i in range(n)]
        cols.append(solve(A, e_k))
    # Re-assemble: cols[k] is the k-th column of A^-1.
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Cholesky
# ---------------------------------------------------------------------------


def cholesky(A) -> List[List[float]]:
    """Cholesky-Banachiewicz: lower-triangular `L` with `A = L @ L.T`,
    for symmetric positive-definite `A`. Raises `ValueError` if the matrix
    is not SPD (a non-positive value appears on the diagonal during the
    factorisation)."""
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("cholesky: A must be square")
    L = matrix_zeros(n, n)
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            s = 0.0
            Lj = L[j]
            for k in range(j):
                s += Li[k] * Lj[k]
            if i == j:
                d = A[i][i] - s
                if d <= 0.0:
                    raise ValueError("cholesky: matrix is not positive definite")
                Li[j] = math.sqrt(d)
            else:
                Li[j] = (A[i][j] - s) / Lj[j]
    return L


# ---------------------------------------------------------------------------
# Symmetric eigendecomposition (Jacobi rotations)
# ---------------------------------------------------------------------------


def _jacobi_rotate(A, V, p: int, q: int) -> None:
    """In-place Jacobi rotation that zeroes the (p, q) and (q, p) entries of
    the symmetric matrix `A`, accumulating the rotation into the eigenvector
    matrix `V`."""
    app = A[p][p]
    aqq = A[q][q]
    apq = A[p][q]
    if apq == 0.0:
        return

    # Choose the rotation angle (Press et al., NR, eq. 11.1.8-9).
    theta = (aqq - app) / (2.0 * apq)
    if theta >= 0:
        t = 1.0 / (theta + math.sqrt(1.0 + theta * theta))
    else:
        t = 1.0 / (theta - math.sqrt(1.0 + theta * theta))
    c = 1.0 / math.sqrt(1.0 + t * t)
    s = t * c

    # Update the diagonals and zero the (p, q) entry.
    A[p][p] = app - t * apq
    A[q][q] = aqq + t * apq
    A[p][q] = 0.0
    A[q][p] = 0.0

    # Rotate the other entries of rows / columns p and q.
    n = len(A)
    for i in range(n):
        if i == p or i == q:
            continue
        a_ip = A[i][p]
        a_iq = A[i][q]
        A[i][p] = c * a_ip - s * a_iq
        A[p][i] = A[i][p]
        A[i][q] = s * a_ip + c * a_iq
        A[q][i] = A[i][q]

    # Rotate the eigenvector matrix.
    for i in range(n):
        v_ip = V[i][p]
        v_iq = V[i][q]
        V[i][p] = c * v_ip - s * v_iq
        V[i][q] = s * v_ip + c * v_iq


def eigh(A, tol: float = 1e-12, max_sweeps: int = 100) -> Tuple[_Vec, List[List[float]]]:
    """Symmetric eigendecomposition of `A` via cyclic Jacobi rotations.

    Returns `(eigenvalues, eigenvectors)` with eigenvalues sorted ascending,
    matching `numpy.linalg.eigh`. The columns of `eigenvectors` are the
    unit-norm eigenvectors; column `k` corresponds to eigenvalue `k`.

    Assumes `A` is symmetric. The input is not mutated.
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("eigh: A must be square")

    # Symmetry check (cheap, catches the most common caller bug).
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol * (1.0 + abs(A[i][j]) + abs(A[j][i])):
                raise ValueError("eigh: A is not symmetric")

    # Working copies.
    M = [list(row) for row in A]
    V = eye(n)

    for _sweep in range(max_sweeps):
        # Sum of squared off-diagonal entries — convergence criterion.
        off = 0.0
        for p in range(n):
            Mp = M[p]
            for q in range(p + 1, n):
                off += Mp[q] * Mp[q]
        if off < tol * tol:
            break
        # One sweep over all off-diagonal positions.
        for p in range(n):
            for q in range(p + 1, n):
                if abs(M[p][q]) > tol:
                    _jacobi_rotate(M, V, p, q)

    # Diagonal of M is the eigenvalues; columns of V are eigenvectors.
    eigvals = [M[i][i] for i in range(n)]
    # Sort ascending, carrying eigenvectors along.
    order = sorted(range(n), key=lambda i: eigvals[i])
    sorted_vals = _Vec(eigvals[i] for i in order)
    sorted_vecs = [[V[i][order[j]] for j in range(n)] for i in range(n)]
    return sorted_vals, sorted_vecs


__all__ = [
    "eye",
    "matrix_zeros",
    "matmul",
    "matvec",
    "transpose",
    "solve",
    "inv",
    "cholesky",
    "eigh",
]
