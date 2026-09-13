"""The two lists must stay raceable. Screening them is the only thing that keeps them honest.

Both failure modes here were found in the wild, and neither is visible by reading the source. An
objective can accept a vector of any length and read only its first two coordinates, so it passes
every naive check while posing a 2-d problem at d=100. And one slow objective is enough to make a
tournament impractical once it is multiplied by trials, problems and optimizers.
"""

import pathlib
import random
import time

import pytest

from humpday.objectives import SURFACES, morphed_surfaces, physics_objectives

DIM = 100
MAX_SECONDS_PER_EVAL = 0.01


def _probe(fn, n_dim, seed):
    rnd = random.Random(seed)
    worst_live, slowest = n_dim, 0.0
    for _ in range(
        3
    ):  # several points: a coordinate can be flat at one of them by chance
        base = [rnd.random() for _ in range(n_dim)]
        started = time.time()
        value = float(fn(base))
        slowest = max(slowest, time.time() - started)
        live = sum(
            1
            for i in range(n_dim)
            if abs(float(fn([*base[:i], 1.0 - base[i], *base[i + 1 :]])) - value)
            > 1e-12
        )
        worst_live = min(worst_live, live)
    return worst_live, slowest


@pytest.mark.parametrize("fn", SURFACES, ids=lambda f: f.__name__)
def test_surface_is_raceable_in_high_dimension(fn):
    live, slowest = _probe(fn, DIM, hash(fn.__name__) & 0xFFFF)
    assert live >= DIM * 0.9, (
        f"{fn.__name__} reads only {live} of {DIM} coordinates: it is a fixed-dimension surface "
        f"wearing a general one's signature. Add it to FIXED_DIMENSION."
    )
    assert slowest <= MAX_SECONDS_PER_EVAL, (
        f"{fn.__name__} takes {slowest * 1000:.0f}ms per evaluation, too slow to race. Add it to SLOW."
    )


def test_surfaces_are_unique():
    names = [f.__name__ for f in SURFACES]
    assert len(names) == len(set(names)), (
        "the curated list must not repeat an objective"
    )


def test_physics_objectives_load_and_evaluate():
    demos = physics_objectives()
    assert len(demos) > 50
    for name, fn, n_dim in demos:
        assert n_dim >= 1, name
        value = float(fn([0.5] * n_dim))
        assert value == value, f"{name} returned nan at the centre of its cube"


def test_morphed_surfaces_differ_between_runs():
    # The point of morphing: a fixed landscape can be learned, so two runs must not coincide.
    a, b = morphed_surfaces(4, seed=1), morphed_surfaces(4, seed=2)
    x = [0.4] * 8
    assert [float(f(x)) for f in a] != [float(f(x)) for f in b]


def test_importing_the_package_does_not_require_numpy():
    """humpday declares no dependencies. Importing this package must not create one.

    The surface modules import numpy at module scope, so a module-level `from ... import
    CLASSIC_OBJECTIVES` here turns numpy into a hard requirement for anyone who imports
    `humpday.objectives` at all. Built lazily instead.
    """
    import subprocess
    import sys

    probe = (
        "import sys, builtins;"
        "real=builtins.__import__;"
        "builtins.__import__=lambda n,*a,**k: (_ for _ in ()).throw(ImportError(n))"
        " if n.split('.')[0]=='numpy' else real(n,*a,**k);"
        "import humpday.objectives;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, (
        f"importing humpday.objectives pulled in numpy: {out.stderr[-400:]}"
    )


def test_physics_objectives_is_empty_rather_than_raising_off_repository():
    """The demos are a repository asset and are not in the wheel.

    Resolving them relative to the installed package used to raise FileNotFoundError from
    site-packages, which is not a useful answer to "what engineering problems do you have".
    """
    import humpday.objectives as mod

    original, mod._PHYSICS_CACHE = mod._PHYSICS_CACHE, None
    real_is_dir = pathlib.Path.is_dir
    try:
        pathlib.Path.is_dir = lambda self: (
            False
        )  # simulate an installed, repo-less layout
        assert mod.physics_objectives() == []
    finally:
        pathlib.Path.is_dir = real_is_dir
        mod._PHYSICS_CACHE = original
