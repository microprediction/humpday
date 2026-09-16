"""The *_combo* surfaces blend their components instead of echoing one (#336).

They summed components whose scales on the cube differed by up to thirteen
orders of magnitude, then divided by one constant, so five of them were
rank-identical (Spearman 1.000) to their largest component. Each component
is now scaled to its own empirical range before the blend.
"""

import random

import pytest

from humpday.objectives import classic as C

COMBOS = {
    C.deap_combo1_on_cube: [C.schwefel_on_cube, C.griewank_on_cube, C.shekel_on_cube],
    C.deap_combo2_on_cube: [C.shaffer_on_cube, C.shekel_on_cube],
    C.deap_combo3_on_cube: [
        C.rosenbrock_on_cube,
        C.bohachevsky_on_cube,
        C.shekel_on_cube,
    ],
    C.landscapes_combo1_on_cube: [C.qing_on_cube, C.michaelewicz_on_cube],
    C.landscapes_combo2_on_cube: [C.rotated_hyper_ellipsoid_on_cube, C.salomon_on_cube],
    C.landscapes_combo3_on_cube: [C.zakharov_on_cube, C.styblinski_tang_on_cube],
}


def _spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0] * len(v)
        for i, idx in enumerate(order):
            r[idx] = i
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


@pytest.mark.parametrize("combo", list(COMBOS), ids=lambda f: f.__name__)
@pytest.mark.parametrize("n_dim", [2, 5, 20])
def test_every_component_contributes(combo, n_dim):
    rnd = random.Random(0)
    points = [[rnd.random() for _ in range(n_dim)] for _ in range(120)]
    values = [float(combo(p)) for p in points]
    for component in COMBOS[combo]:
        rho = _spearman(values, [float(component(p)) for p in points])
        # Not a monotone rescaling of any one component ...
        assert abs(rho) < 0.999, (combo.__name__, component.__name__, rho)


@pytest.mark.parametrize("combo", list(COMBOS), ids=lambda f: f.__name__)
def test_the_blend_is_not_dominated_by_the_largest_component(combo):
    # ... and the component that used to dwarf the others is now one voice
    # among several: every component moves the blend in a way no other does.
    rnd = random.Random(1)
    points = [[rnd.random() for _ in range(5)] for _ in range(120)]
    values = [float(combo(p)) for p in points]
    rhos = [_spearman(values, [float(c(p)) for p in points]) for c in COMBOS[combo]]
    assert max(rhos) < 0.999
    assert min(rhos) > -0.999


def test_shekel_is_one_deterministic_landscape():
    # It drew its 15 peaks from np.random on every call, and mapped the cube
    # to [-400, 400]^n while the peaks sat in [0, 10]^n, so it was a constant
    # plus 1e-6 of noise.
    rnd = random.Random(2)
    pts = [[rnd.random() for _ in range(5)] for _ in range(40)]
    values = [C.shekel_on_cube(p) for p in pts]
    assert [C.shekel_on_cube(p) for p in pts] == values
    assert max(values) - min(values) > 0.05
