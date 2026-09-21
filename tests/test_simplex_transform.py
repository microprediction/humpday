"""The cube-to-simplex map reaches the allocations it says it reaches.

The README lifts a box-domain objective onto the probability simplex through this map, so what
the map cannot express, no optimizer can find, whatever its budget. What it could express was
decided by a clip constant rather than by the arithmetic.
"""

from __future__ import annotations

import math

import pytest

from humpday.transforms.cubetosimplex import (
    cube_to_simplex,
    lift_to_cube,
    simplex_to_cube,
)
from humpday.transforms.thurstone_transform import attainable_weights

# The scale the dfo_recommender outline tunes to; small enough to put a concentrated optimum
# somewhere an optimizer sampling the cube evenly will actually look.
TUNED_SCALE = 0.24


def test_a_concentrated_allocation_survives_the_round_trip():
    """#358. `[0.9999, 0.0001]` came back as `[0.998275, 0.001724]`.

    Every cube coordinate was clipped to [1e-10, 1 - 1e-10] before the probit, which held the
    latent logit inside +-6.36 and so confined a two-component simplex to [0.00172, 0.99828].
    A portfolio wanting a component below a sixth of a percent could not be written down, let
    alone optimized to.
    """
    p = [0.9999, 0.0001]
    back = cube_to_simplex(simplex_to_cube(p))
    assert back == pytest.approx(p, abs=1e-12)
    assert back[1] < 0.00172, "still inside the old clip's attainable region"


@pytest.mark.parametrize("n_dim", [1, 2, 5, 12])
def test_the_cube_round_trips_through_the_simplex(n_dim):
    import random

    rng = random.Random(n_dim)
    for _ in range(50):
        u = [rng.random() for _ in range(n_dim)]
        back = simplex_to_cube(cube_to_simplex(u))
        assert back == pytest.approx(u, abs=1e-12)


@pytest.mark.parametrize(
    "p",
    [
        [0.5, 0.5],
        [0.7, 0.2, 0.09, 0.01],
        [0.999999999999, 1e-12],
        [0.9, 0.0999999999, 1e-10],
        [0.25] * 4,
    ],
)
def test_allocations_inside_the_image_round_trip(p):
    assert math.isclose(math.fsum(p), 1.0, abs_tol=1e-12), "premise: a simplex point"
    back = cube_to_simplex(simplex_to_cube(p))
    assert back == pytest.approx(p, abs=1e-9)


def test_the_attainable_region_is_reported_and_is_the_truth():
    """A caller with a concentrated allocation in mind can ask what the map can express, instead
    of discovering a silent clip in the answer."""
    for n_dim in (1, 2, 7):
        bounds = attainable_weights(n_dim)

        # A component is at its smallest with every other coordinate pushed the other way, so the
        # extremes of the image are the mixed corners of the cube and not its diagonal.
        smallest = cube_to_simplex([0.0] + [1.0] * (n_dim - 1))[1]
        largest = cube_to_simplex([1.0] + [0.0] * (n_dim - 1))[1]
        assert smallest == pytest.approx(bounds["non_reference"][0], rel=1e-9)
        assert largest == pytest.approx(bounds["non_reference"][1], rel=1e-9)

        assert cube_to_simplex([0.0] * n_dim)[0] == pytest.approx(
            bounds["reference"][1], rel=1e-9
        )
        assert cube_to_simplex([1.0] * n_dim)[0] == pytest.approx(
            bounds["reference"][0], rel=1e-9
        )

        import random

        rng = random.Random(n_dim)
        for _ in range(200):
            p = cube_to_simplex([rng.random() for _ in range(n_dim)])
            assert bounds["reference"][0] <= p[0] <= bounds["reference"][1]
            for w in p[1:]:
                assert bounds["non_reference"][0] <= w <= bounds["non_reference"][1]


def test_the_reachable_region_is_wider_than_the_clip_it_replaced():
    """The old clip's floor, in the two-component case, was 0.00172407."""
    bounds = attainable_weights(1)
    assert bounds["non_reference"][0] < 1e-16, bounds
    assert bounds["reference"][0] < 3e-4, bounds
    assert bounds["non_reference"][1] > 0.9997, bounds


def test_a_smaller_scale_reaches_further_out():
    wide = attainable_weights(1, TUNED_SCALE)
    default = attainable_weights(1)
    assert wide["non_reference"][0] < default["non_reference"][0]
    assert wide["reference"][0] < default["reference"][0]
    assert wide["non_reference"][1] > default["non_reference"][1]


def test_the_scale_cancels_in_the_round_trip():
    """Any positive scale is a bijection onto its own image; it is the image that differs."""
    import random

    rng = random.Random(7)
    for scale in (0.05, TUNED_SCALE, 1.0, 5.0):
        u = [rng.random() for _ in range(4)]
        back = simplex_to_cube(cube_to_simplex(u, scale), scale)
        assert back == pytest.approx(u, abs=1e-10), scale


def test_a_small_scale_does_not_overflow():
    """exp(8.21 / 0.05) is 1e71 and exp of the lower end underflows; the softmax is taken about
    its largest logit so that neither is inf/inf."""
    p = cube_to_simplex([1.0, 0.0, 0.5], 0.05)
    assert math.isclose(math.fsum(p), 1.0, rel_tol=1e-12)
    assert all(math.isfinite(w) and w >= 0.0 for w in p)


def test_the_corners_of_the_cube_are_accepted():
    """Optimizers propose the closed cube: bound-constrained methods sit on 0.0 and 1.0 routinely,
    and an exception or a nan there would be charged to the optimizer."""
    for u in ([0.0], [1.0], [0.0, 1.0], [1.0, 1.0, 0.0]):
        p = cube_to_simplex(u)
        assert math.isclose(math.fsum(p), 1.0, rel_tol=1e-12)
        assert all(math.isfinite(w) for w in p)


def test_a_vertex_maps_to_the_most_concentrated_point_available():
    """The open simplex is the image, so a zero weight is read as "as small as this map goes"
    rather than rejected or floored at 1e-10."""
    u = simplex_to_cube([1.0, 0.0])
    p = cube_to_simplex(u)
    assert p[1] == pytest.approx(attainable_weights(1)["non_reference"][0], rel=1e-9)


@pytest.mark.parametrize(
    "bad",
    [
        [0.5],  # not a simplex point at all
        [0.5, 0.4],  # does not sum to one
        [1.5, -0.5],  # negative weight
        [float("nan"), 1.0],
        [float("inf"), 1.0],
    ],
)
def test_an_invalid_probability_vector_is_rejected(bad):
    with pytest.raises(ValueError):
        simplex_to_cube(bad)


def test_a_nan_cube_coordinate_is_rejected():
    with pytest.raises(ValueError):
        cube_to_simplex([0.5, float("nan")])


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_a_scale_that_is_not_a_positive_number_is_rejected(scale):
    with pytest.raises(ValueError):
        cube_to_simplex([0.5], scale)
    with pytest.raises(ValueError):
        simplex_to_cube([0.5, 0.5], scale)


def test_an_optimizer_can_find_a_concentrated_optimum_through_the_lift():
    """Reachable is not the same as findable, which is what the scale is for.

    The optimum here puts a ten-thousandth of the allocation on one component. At the default
    scale its pre-image is at u = 1.6e-20, a region no optimizer sampling the cube evenly will
    ever propose; at the tuned scale it sits at u = 0.014, and an
    ordinary local method walks to it. Under the old clip it was not reachable at either scale:
    0.0001 was outside the image entirely.
    """
    from humpday import _array as _A
    from humpday.optimizers.alloptimizers import pure_optimize

    target = 1e-4

    def objective(p):
        # On a log scale, so the landscape is not a spike on a plateau.
        return (math.log(max(p[1], 1e-300)) - math.log(target)) ** 2

    _A.seed(0)
    lifted = lift_to_cube(objective, scale=TUNED_SCALE)
    _, u_best = pure_optimize(lifted, "NelderMead", 200, 1)
    found = cube_to_simplex(u_best, TUNED_SCALE)[1]

    assert found == pytest.approx(target, rel=0.1), found
    assert found < 0.00172, "the old clip could not represent this allocation at all"
    assert target > attainable_weights(1, TUNED_SCALE)["non_reference"][0]
