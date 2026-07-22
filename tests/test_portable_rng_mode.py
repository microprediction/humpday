"""The shim's portable-RNG mode: same optimizer, same seed, same trajectory
regardless of backend or platform — and legacy mode untouched by default."""

import random

import pytest

from humpday import _array as _A
from humpday.optimizers.evolutionary_algorithms import DifferentialEvolution
from humpday.optimizers.scipy_algorithms import NelderMead

try:
    import numpy as np
except ImportError:
    np = None


@pytest.fixture(autouse=True)
def _restore_legacy():
    yield
    _A.use_legacy_rng()


def sphere(x):
    return sum((float(v) - 0.3) ** 2 for v in x)


def _run(cls, n_trials=80, n_dim=3):
    opt = cls(sphere, n_trials, n_dim)
    best_value, best_x = opt.optimize()
    return float(best_value), [float(v) for v in best_x], opt.evaluations


@pytest.mark.parametrize("cls", [DifferentialEvolution, NelderMead])
def test_portable_mode_is_reproducible(cls):
    _A.use_portable_rng(42, 54)
    a = _run(cls)
    _A.use_portable_rng(42, 54)
    b = _run(cls)
    assert a == b


@pytest.mark.parametrize("cls", [DifferentialEvolution, NelderMead])
def test_portable_mode_ignores_stdlib_and_numpy_state(cls):
    _A.use_portable_rng(7)
    random.seed(0)
    if np is not None:
        np.random.seed(0)
    a = _run(cls)
    _A.use_portable_rng(7)
    random.seed(999)
    if np is not None:
        np.random.seed(999)
    b = _run(cls)
    assert a == b


def test_legacy_facade_matches_stdlib_exactly():
    random.seed(123)
    expected = [
        random.random(),
        random.gauss(0.0, 1.0),
        random.uniform(-2.0, 5.0),
        random.choice([10, 20, 30]),
        random.randrange(17),
    ]
    random.seed(123)
    got = [
        _A.rng_random(),
        _A.rng_gauss(),
        _A.rng_uniform(-2.0, 5.0),
        _A.rng_choice([10, 20, 30]),
        _A.rng_randrange(17),
    ]
    assert got == expected
    random.seed(5)
    a = list(range(9))
    random.shuffle(a)
    random.seed(5)
    b = list(range(9))
    _A.rng_shuffle(b)
    assert a == b


def test_seed_reseeds_portable_stream():
    _A.use_portable_rng(1)
    _A.seed(42)
    xs = [_A.random_scalar() for _ in range(3)]
    _A.seed(42)
    assert xs == [_A.random_scalar() for _ in range(3)]


def test_portable_primitives_shapes():
    _A.use_portable_rng(3)
    v = _A.random_uniform(5)
    assert len(v) == 5 and all(0.0 <= float(x) < 1.0 for x in v)
    assert len(_A.random_normal(4)) == 4
    assert 0 <= _A.random_int(10) < 10
    assert 3 <= _A.random_int(3, 6) < 6
    assert _A.random_choice([1, 2, 3]) in (1, 2, 3)
    pick = _A.random_choice(10, k=4, replace=False)
    assert len(set(pick)) == 4


def test_entire_roster_reproducible_under_portable_mode():
    """Every optimizer, run twice from the same portable seed with
    scrambled stdlib/numpy state in between, reproduces its run exactly.
    This is what makes cross-language transition vectors possible."""
    from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

    for cls in PURE_OPTIMIZERS.values():
        _A.use_portable_rng(11, 3)
        a = _run(cls, n_trials=60, n_dim=2)
        random.seed(777)
        if np is not None:
            np.random.seed(777)
        _A.use_portable_rng(11, 3)
        b = _run(cls, n_trials=60, n_dim=2)
        assert a == b, f"{cls.__name__} not reproducible under portable RNG"
