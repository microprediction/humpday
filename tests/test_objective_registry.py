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
    # One untimed call first: a surface may do one-off setup for a dimension
    # (the combos measure their components' ranges), which a tournament pays
    # once per cell, not per evaluation.
    fn([rnd.random() for _ in range(n_dim)])
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


def test_the_ratings_table_is_package_data_not_a_repository_asset():
    """The counterpart of the test above, and the opposite answer.

    The demos are deliberately left out of the wheel; the ratings table must be in it, or a
    recommendation needs a benchmark run. What makes that work is resolving it relative to the
    package rather than the repository, which is precisely what `physics_objectives` got wrong.
    """
    import humpday
    import humpday.ratings

    table = pathlib.Path(humpday.__file__).parent / "data" / "ratings.json"
    assert table.is_file(), "the ratings table must live inside the package directory"
    assert humpday.ratings.cells(), "the shipped table must not be empty"


def test_scalable_engineering_objectives_are_raceable():
    """The fixed demos stop at 24 dimensions, so high-dimensional physics comes from these.

    They are screened exactly like the shipped suites. The packing objective is the reason:
    written with a hard `min` over clearances, as the real demo is, it reads zero of fifty
    coordinates at a random point, because the binding constraint is a single pair and moving
    any other circle changes nothing. That is a valid objective and a useless tournament.
    """
    import itertools

    from benchmarks.record_ratings import _scalable as scalable_physics

    for n_dim in (50, 100):
        for fn in itertools.islice(scalable_physics(n_dim, seed=7), 3):
            live, slowest = _probe(fn, n_dim, hash(fn.__name__) & 0xFFFF)
            assert live >= n_dim * 0.9, (
                f"{fn.__name__} at d={n_dim} reads only {live} of {n_dim} coordinates"
            )
            assert slowest <= MAX_SECONDS_PER_EVAL, f"{fn.__name__} is too slow to race"


def test_rating_aggregation_penalises_a_specialist():
    """A specialist that wins one suite and sinks in the other must not lead the robust order."""
    from humpday import ratings
    from humpday.eligibility import robust_order

    for n_dim in ratings.recorded_dimensions():
        suites = {s: c["ratings"] for s, c in ratings.cells_at(n_dim, 100).items()}
        if len(suites) < 2:
            continue
        order = robust_order(n_dim)
        assert order, f"d={n_dim} has two suites but no robust order"
        leader = order[0]
        leader_worst = max(
            sorted(r, key=lambda n: -r[n]).index(leader) + 1 for r in suites.values()
        )
        for other in order[1:]:
            other_best = min(
                sorted(r, key=lambda n: -r[n]).index(other) + 1 for r in suites.values()
            )
            # Nothing ranked below the leader may be better than it in *every* suite.
            if other_best < leader_worst:
                other_worst = max(
                    sorted(r, key=lambda n: -r[n]).index(other) + 1
                    for r in suites.values()
                )
                assert other_worst >= leader_worst, (
                    f"d={n_dim}: {other} beats {leader} in every suite yet ranks below it"
                )


def test_no_two_surfaces_are_the_same_function():
    """Six entries in SURFACES were `sum(xi**2)` wearing different names.

    `classic.py` imported them from `landscapes` when installed and silently substituted the
    sphere when not — and it is not a declared dependency, so the substitution was always in
    effect. The `_on_cube` wrappers still applied their individual shifts and scalings, which
    preserves rank order and hid it completely.

    Rank correlation is the right screen: a monotone rescaling of the same function gives exactly
    1.0, while two genuinely different functions that happen to be similarly bowl-shaped do not.
    """
    import itertools
    import random

    from humpday.objectives import SURFACES

    rnd = random.Random(7)
    n_dim = 5
    points = [[rnd.random() for _ in range(n_dim)] for _ in range(120)]

    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        out = [0] * len(values)
        for position, index in enumerate(order):
            out[index] = position
        return out

    def spearman(a, b):
        ra, rb = ranks(a), ranks(b)
        n = len(a)
        ma, mb = sum(ra) / n, sum(rb) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
        return num / den if den else 0.0

    evaluated = {}
    for fn in SURFACES:
        try:
            evaluated[fn.__name__] = [float(fn(p)) for p in points]
        except Exception:
            continue

    # The `*_combo*` surfaces used to sum components of wildly different scale after one shared
    # divisor, which made five of them rank-identical to their dominant component (#336). Each
    # component is now scaled to its own empirical range before the blend, so nothing is tolerated
    # here; the set stays so a future exception has somewhere explicit to go.
    KNOWN_DOMINATED_COMBOS = set()

    duplicates = []
    for (na, va), (nb, vb) in itertools.combinations(evaluated.items(), 2):
        if (na, nb) in KNOWN_DOMINATED_COMBOS or (nb, na) in KNOWN_DOMINATED_COMBOS:
            continue
        if len(set(va)) < 2 or len(set(vb)) < 2:
            continue
        if abs(spearman(va, vb)) > 0.999:
            duplicates.append((na, nb))

    assert not duplicates, (
        "these entries are the same function under different names: "
        + ", ".join(f"{a} == {b}" for a, b in duplicates)
    )
