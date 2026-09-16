"""A benchmark problem is one landscape (#355), the same in every process (#354).

Several generators redrew the landscape inside the objective: Ackley's
coefficients on every evaluation, the sphere/Rosenbrock variants' shifts and
scales on every evaluation, and the rotation matrix (a QR factorisation) on
every evaluation. Observation noise is a separate, explicit control. And the
rotation was seeded from Python's process-salted hash(), so a "fixed"
instance was a different landscape in every worker.
"""

import os
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

from humpday import _array as _A
from humpday.objectives.stochastic_surfaces import StochasticSurfaceGenerator
from humpday.optimizers.adaptive_optimizer import (
    rosenbrock_variants_generator,
    sphere_variants_generator,
)

SURFACES = [
    "stochastic_sphere",
    "stochastic_rastrigin",
    "stochastic_rosenbrock",
    "stochastic_ackley",
    "stochastic_griewank",
]


def _generator(seed=42, rotation=True):
    g = StochasticSurfaceGenerator(seed)
    g.noise_level = 0
    g.use_rotation = rotation
    return g


@pytest.mark.parametrize("name", SURFACES)
@pytest.mark.parametrize("n_dim", [2, 3, 10])
def test_zero_noise_evaluations_agree(name, n_dim):
    f = getattr(_generator(), name)("fixed")
    x = [0.31] * n_dim
    values = [f(x) for _ in range(3)]
    assert values[0] == values[1] == values[2]


@pytest.mark.parametrize("name", SURFACES)
def test_distinct_instances_are_distinct_landscapes(name):
    g = _generator()
    f1 = getattr(g, name)("instance-1")
    f2 = getattr(g, name)("instance-2")
    assert f1([0.31, 0.42, 0.53]) != f2([0.31, 0.42, 0.53])


def test_rotation_is_factorised_once():
    f = _generator().stochastic_rastrigin("fixed")
    real_qr = np.linalg.qr
    with mock.patch("numpy.linalg.qr", side_effect=real_qr) as qr:
        for _ in range(3):
            f([0.1] * 10)
    assert qr.call_count == 1


def test_evaluating_does_not_touch_the_global_rngs():
    # Optimizers draw from these; an objective that consumed them would make
    # a trajectory depend on how many times the objective was called.
    f = _generator().stochastic_ackley("fixed")
    np.random.seed(7)
    _A.seed(7)
    before = np.random.get_state()[1].tolist()
    for _ in range(5):
        f([0.2, 0.9, 0.4])
    assert np.random.get_state()[1].tolist() == before


def test_noise_is_the_separate_control():
    g = StochasticSurfaceGenerator(3)
    g.noise_level = 0.05
    f = g.stochastic_sphere("fixed")
    assert f([0.3, 0.4]) != f([0.3, 0.4])


def test_building_a_suite_does_not_reseed_the_global_rng():
    np.random.seed(11)
    expected = np.random.get_state()[1].tolist()
    np.random.seed(11)
    StochasticSurfaceGenerator(99).get_random_function_suite(4)
    assert np.random.get_state()[1].tolist() == expected


def test_seeded_suites_are_reproducible():
    a = StochasticSurfaceGenerator(5).get_random_function_suite(4)
    b = StochasticSurfaceGenerator(5).get_random_function_suite(4)
    assert list(a) == list(b)
    for name in a:
        assert a[name]([0.3, 0.6]) == b[name]([0.3, 0.6])


_SNIPPET = (
    "from humpday.objectives.stochastic_surfaces import StochasticSurfaceGenerator as G;"
    "g=G(2); g.noise_level=0; g.use_rotation=True;"
    "print(repr(g.stochastic_rastrigin('fixed')([.31,.42,.53])))"
)


def test_fixed_instance_is_the_same_landscape_in_every_process():
    outs = []
    for hashseed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        outs.append(
            subprocess.check_output(
                [sys.executable, "-c", _SNIPPET], env=env, text=True
            )
        )
    assert outs[0] == outs[1], outs


@pytest.mark.parametrize(
    "make", [sphere_variants_generator, rosenbrock_variants_generator]
)
def test_variant_generators_yield_fixed_problems(make):
    _A.seed(1)
    gen = make(2)
    seen = {}
    for _ in range(30):
        f = next(gen)
        assert f([0.3, 0.4]) == f([0.3, 0.4])
        seen.setdefault(f.__name__, []).append(f([0.3, 0.4]))
    assert {"sphere_pure", "shifted_sphere", "scaled_sphere"} == set(seen) or {
        "rosenbrock_pure",
        "scaled_rosenbrock",
        "shifted_rosenbrock",
    } == set(seen)
    # Different instances of a shifted/scaled variant are different problems.
    for name, values in seen.items():
        if not name.endswith("_pure") and len(values) > 1:
            assert len(set(values)) > 1, name
