"""The minimum-Frobenius quadratic model is built, not silently replaced (#343).

On the pure backend the model builder used to raise NotImplementedError from
svd(full_matrices=True), and the optimizers caught it with a bare `except
Exception` and ran a finite-difference gradient with an identity Hessian
instead. Every test here runs on whichever backend is live; CI runs the suite
with HUMPDAY_FORCE_PURE_ARRAY=1 as well, which is where these used to fail.
"""

import itertools

import pytest

from humpday import _array as _A
from humpday.optimizers.prima_algorithms import (
    PRIMA_BOBYQA,
    PRIMA_NEWUOA,
    _build_min_frobenius_quadratic,
)


def _quadratic(c, g, H):
    def f(x):
        n = len(x)
        v = c + sum(g[i] * x[i] for i in range(n))
        v += 0.5 * sum(H[i][j] * x[i] * x[j] for i in range(n) for j in range(n))
        return v

    return f


def _axis_points(n, h=0.2):
    pts = [[0.0] * n]
    for i in range(n):
        for sign in (1.0, -1.0):
            p = [0.0] * n
            p[i] = sign * h
            pts.append(p)
    return pts


def _full_points(n, h=0.2):
    pts = _axis_points(n, h)
    for i, j in itertools.combinations(range(n), 2):
        p = [0.0] * n
        p[i] = h
        p[j] = h
        pts.append(p)
    return pts


def _close_matrix(M, want, tol=1e-8):
    for i in range(len(want)):
        for j in range(len(want[0])):
            assert abs(float(M[i][j]) - want[i][j]) <= tol, (i, j, M[i][j], want[i][j])


@pytest.mark.parametrize("n", [2, 3, 5])
def test_diagonal_quadratic_is_recovered_from_2n_plus_1_points(n):
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        H[i][i] = 2.0 * (i + 1)
    g = [0.1 * (i + 1) for i in range(n)]
    f = _quadratic(0.3, g, H)
    XPT = [_A.asarray(p) for p in _axis_points(n)]
    FVAL = [f(p) for p in _axis_points(n)]
    H_prev = _A.linalg.matrix_zeros(n, n)
    c, g_hat, H_hat = _build_min_frobenius_quadratic(XPT, FVAL, H_prev, n)
    assert abs(c - 0.3) <= 1e-8
    for i in range(n):
        assert abs(float(g_hat[i]) - g[i]) <= 1e-8
    _close_matrix(H_hat, H)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_full_quadratic_with_cross_terms_is_recovered_from_a_complete_set(n):
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            H[i][j] = 3.0 if i == j else 0.5 * (i + j + 1)
    g = [0.2 * (i + 1) for i in range(n)]
    f = _quadratic(-1.0, g, H)
    pts = _full_points(n)
    assert len(pts) == (n + 1) * (n + 2) // 2
    XPT = [_A.asarray(p) for p in pts]
    FVAL = [f(p) for p in pts]
    H_prev = _A.linalg.matrix_zeros(n, n)
    c, g_hat, H_hat = _build_min_frobenius_quadratic(XPT, FVAL, H_prev, n)
    assert abs(c + 1.0) <= 1e-8
    for i in range(n):
        assert abs(float(g_hat[i]) - g[i]) <= 1e-8
    _close_matrix(H_hat, H)


def test_a_shrunk_interpolation_set_builds_the_same_model():
    # Trust-region radii shrink; the model must not degrade with them (#350).
    n = 3
    H = [[2.0, 0.5, 0.0], [0.5, 4.0, 0.25], [0.0, 0.25, 6.0]]
    g = [0.1, -0.2, 0.3]
    f = _quadratic(0.0, g, H)
    for h in [0.2, 1e-3, 1e-5]:
        pts = _full_points(n, h)
        XPT = [_A.asarray(p) for p in pts]
        FVAL = [f(p) for p in pts]
        _c, g_hat, H_hat = _build_min_frobenius_quadratic(
            XPT, FVAL, _A.linalg.matrix_zeros(n, n), n
        )
        for i in range(n):
            assert abs(float(g_hat[i]) - g[i]) <= 1e-6
        _close_matrix(H_hat, H, tol=1e-5)


@pytest.mark.parametrize("cls", [PRIMA_NEWUOA, PRIMA_BOBYQA])
def test_the_optimizers_use_the_model_they_advertise(cls):
    """Not the identity-Hessian fallback: the built model carries curvature."""
    n = 2
    H = [[2.0, 0.0], [0.0, 4.0]]
    f = _quadratic(0.0, [0.0, 0.0], H)
    opt = cls(f, 50, n)
    pts = _axis_points(n)
    XPT = [_A.asarray(p) for p in pts]
    FVAL = [f(p) for p in pts]
    if cls is PRIMA_NEWUOA:
        _g, H_hat = opt._build_newuoa_model(XPT, FVAL, None, None, 0, n)
    else:
        _g, H_hat = opt._build_bobyqa_model(XPT, FVAL, 0, n)
    _close_matrix(H_hat, H)


def test_capability_errors_are_not_disguised_as_rank_deficiency(monkeypatch):
    from humpday.optimizers import prima_algorithms as P

    def broken(*_a, **_k):
        raise NotImplementedError("backend cannot do this")

    monkeypatch.setattr(P, "_build_min_frobenius_quadratic", broken)
    opt = PRIMA_NEWUOA(lambda x: 0.0, 50, 2)
    XPT = [_A.asarray(p) for p in _axis_points(2)]
    with pytest.raises(NotImplementedError):
        opt._build_newuoa_model(XPT, [0.0] * len(XPT), None, None, 0, 2)


def test_rank_deficiency_still_falls_back():
    # Every point on one line: A_l is rank-deficient, and that IS the case
    # the finite-difference fallback exists for.
    n = 2
    pts = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]
    XPT = [_A.asarray(p) for p in pts]
    FVAL = [p[0] ** 2 for p in pts]
    opt = PRIMA_NEWUOA(lambda x: 0.0, 50, n)
    _g, H_hat = opt._build_newuoa_model(XPT, FVAL, None, None, 0, n)
    _close_matrix(H_hat, [[1.0, 0.0], [0.0, 1.0]])
