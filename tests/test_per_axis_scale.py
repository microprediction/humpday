"""A per-coordinate unbounded scale works on both array backends, and a bad one fails at entry (#394)"""

import os
import subprocess
import sys
import textwrap

import pytest

from humpday import minimize, unbounded_to_unit_cube, unit_cube_to_unbounded

SCRIPT = textwrap.dedent(
    """
    import math
    from humpday import minimize, unbounded_to_unit_cube, unit_cube_to_unbounded

    for scale in ([2.0, 3.0], (2.0, 3.0)):
        x = [5.0, -7.0]
        u = unbounded_to_unit_cube(x, scale)
        assert all(0 < v < 1 for v in u)
        back = unit_cube_to_unbounded(u, scale)
        assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(back, x)), back
        # The scale acts per axis: the same unit point lands 2 and 3 times as far out.
        far = unit_cube_to_unbounded([0.75, 0.75], scale)
        assert math.isclose(far[0], 2.0) and math.isclose(far[1], 3.0), far

    calls = []

    def f(x):
        calls.append(list(x))
        return sum(v * v for v in x)

    r = minimize(f, x0=[0.0, 0.0], scale=[2.0, 3.0], method="RandomSearch",
                 options={"maxiter": 3})
    assert r.nfev == len(calls) == 3, (r.nfev, len(calls))
    print("ok")
    """
)


@pytest.mark.parametrize("pure", ["1", "0"])
def test_per_axis_scale_on_each_backend(pure):
    if pure == "0":
        pytest.importorskip("numpy")
    env = dict(os.environ, HUMPDAY_FORCE_PURE_ARRAY=pure)
    if pure == "0":
        env.pop("HUMPDAY_FORCE_PURE_ARRAY")
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


@pytest.mark.parametrize(
    "scale",
    [
        0,
        -1.0,
        float("nan"),
        float("inf"),
        [1.0, 0.0],
        [1.0, -2.0],
        [1.0],
        [1, 2, 3],
        "ab",
    ],
)
def test_bad_scale_fails_before_the_objective_is_called(scale):
    calls = []

    def f(x):
        calls.append(x)
        return sum(v * v for v in x)

    with pytest.raises(ValueError):
        minimize(f, x0=[0.0, 0.0], scale=scale, options={"maxiter": 5})
    assert calls == []


def test_bad_scale_fails_in_the_public_transforms():
    with pytest.raises(ValueError):
        unbounded_to_unit_cube([1.0, 2.0], scale=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        unit_cube_to_unbounded([0.5, 0.5], scale=0)


def test_scalar_points_still_map():
    assert unbounded_to_unit_cube(0.0, 5.0) == 0.5
    assert unit_cube_to_unbounded(0.75, 5.0) == pytest.approx(5.0)
