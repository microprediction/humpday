"""The pure SVD does not invent rank, and its U is orthonormal (#391).

Singular values here come from diagonalising A^T A, which squares the condition number: an
exactly zero singular value surfaces as sqrt(roundoff), about 1e-8 relative to the largest, not
1e-16. The rank floor was 1e-14, so that noise was accepted as a singular value, `A v / sigma`
divided by it, and the resulting U column was neither unit nor orthogonal to the others.

On a 3x3 whose third column is the sum of the first two, that gave ||U^T U - I|| = 1.0 and a
pseudo-inverse 331 times wrong. On a 4x3 with a null vector, 2.3 million times.
"""

from __future__ import annotations

import pytest

from humpday import _array_pure_linalg as L

np = pytest.importorskip("numpy")

# Exactly singular matrices, with the dependency stated so a reader can check it by eye.
SINGULAR = {
    "3x3, col3 = col1 + col2": [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 4.0]],
    "4x3, 8*col1 + col2 + col3 = 0": [
        [3.0, -3.0, -21.0],
        [-4.0, 28.0, 4.0],
        [-1.0, 13.0, -5.0],
        [1.0, -16.0, 8.0],
    ],
    "3x3, rank 1": [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
    "2x2, zero row": [[1.0, 2.0], [0.0, 0.0]],
}

WELL_CONDITIONED = {
    "identity": [[1.0, 0.0], [0.0, 1.0]],
    "scaled": [[3.0, 1.0], [1.0, 2.0]],
    "tall": [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
}


@pytest.mark.parametrize("name", sorted(SINGULAR))
def test_a_rank_deficient_matrix_reports_a_zero_singular_value(name):
    a = SINGULAR[name]
    _, s, _ = L.svd(a, full_matrices=True)
    values = [float(v) for v in s]
    expected_rank = int(np.linalg.matrix_rank(np.array(a)))
    nonzero = sum(1 for v in values if v > 0.0)
    assert nonzero == expected_rank, (
        f"{name}: reported {values}, rank is {expected_rank}"
    )


@pytest.mark.parametrize("name", sorted({**SINGULAR, **WELL_CONDITIONED}))
def test_u_is_orthonormal(name):
    a = {**SINGULAR, **WELL_CONDITIONED}[name]
    u, _, _ = L.svd(a, full_matrices=True)
    U = np.array(u)
    err = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))
    assert err < 1e-12, f"{name}: ||U^T U - I|| = {err:.3e}"


@pytest.mark.parametrize("name", sorted({**SINGULAR, **WELL_CONDITIONED}))
def test_the_pseudo_inverse_matches_numpy(name):
    a = {**SINGULAR, **WELL_CONDITIONED}[name]
    got = np.array(L.pinv(a))
    want = np.linalg.pinv(np.array(a))
    rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-300)
    assert rel < 1e-9, f"{name}: relative error {rel:.3e}"


@pytest.mark.parametrize("name", sorted({**SINGULAR, **WELL_CONDITIONED}))
def test_the_pseudo_inverse_satisfies_its_own_definition(name):
    """pinv(A) A is symmetric, and A pinv(A) A is A. Checked directly rather than against numpy,
    so the test says something even if both implementations were wrong the same way."""
    A = np.array({**SINGULAR, **WELL_CONDITIONED}[name])
    P = np.array(L.pinv(A.tolist()))
    scale = max(np.linalg.norm(A), 1.0)
    assert np.linalg.norm(P @ A - (P @ A).T) / scale < 1e-9
    assert np.linalg.norm(A @ P @ A - A) / scale < 1e-9


@pytest.mark.parametrize("name", sorted({**SINGULAR, **WELL_CONDITIONED}))
def test_the_factorisation_reconstructs_the_matrix(name):
    a = {**SINGULAR, **WELL_CONDITIONED}[name]
    u, s, vt = L.svd(a, full_matrices=False)
    U, S, Vt = np.array(u), np.array([float(v) for v in s]), np.array(vt)
    rec = U @ np.diag(S) @ Vt
    A = np.array(a)
    assert np.linalg.norm(rec - A) / max(np.linalg.norm(A), 1.0) < 1e-12
