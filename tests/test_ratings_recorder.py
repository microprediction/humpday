"""The recorder and the reader agree, and the reader is the only way in.

`benchmarks/record_ratings.py` writes the table and `humpday.ratings` reads it. They are in
different trees and nothing but convention keeps the key format aligned, which is how the library
previously ended up with three tables that no single consumer could read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks import record_ratings as R  # noqa: E402
from humpday import ratings  # noqa: E402


def test_merged_keys_are_what_the_reader_parses():
    shard = {"n_dim": 8, "budget": 200, "suite": "surfaces"}
    key = f"{shard['n_dim']}/{shard['budget']}/{shard['suite']}"
    n_dim, budget, suite = key.split("/")
    assert (int(n_dim), int(budget), suite) == (8, 200, "surfaces")
    assert suite in ratings.SUITES


def test_the_recorder_only_writes_suites_the_reader_knows():
    assert set(R.GENERATORS) == set(ratings.SUITES)


def test_every_shipped_cell_parses_and_names_a_known_suite():
    for key, cell in ratings.cells().items():
        n_dim, budget, suite = key.split("/")
        assert int(n_dim) > 0 and int(budget) > 0
        assert suite in ratings.SUITES, f"{key} names a suite the reader cannot use"
        assert isinstance(cell.get("ratings", {}), dict)
        assert isinstance(cell.get("timed_out", []), list)


def test_a_finished_cell_is_skipped_rather_than_re_raced(tmp_path, monkeypatch):
    # Re-running the recorder must be cheap, or nobody re-runs it after changing an optimizer.
    monkeypatch.setattr(R, "SHARDS", tmp_path)
    path = R._shard_path(3, 50, "surfaces")
    R._write_shard(
        path,
        {"n_dim": 3, "budget": 50, "suite": "surfaces", "problems": 30, "ratings": {}},
    )

    def explode(*_a, **_k):
        raise AssertionError("a cell already at target depth must not be re-raced")

    monkeypatch.setattr(R, "GENERATORS", {"surfaces": explode, "engineering": explode})
    assert R.run_cell(3, 50, "surfaces", 30, 0, 25.0, 60.0)["problems"] == 30


def test_a_shard_write_cannot_truncate_the_record(tmp_path, monkeypatch):
    # Written through a temporary file and renamed: an interrupted run leaves the previous
    # complete shard, not half a JSON document that no later run can read.
    monkeypatch.setattr(R, "SHARDS", tmp_path)
    path = R._shard_path(4, 50, "surfaces")
    R._write_shard(path, {"n_dim": 4, "budget": 50, "suite": "surfaces", "problems": 1})
    before = path.read_text()

    real_replace = Path.replace

    def fail_replace(self, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(KeyboardInterrupt):
        R._write_shard(
            path, {"n_dim": 4, "budget": 50, "suite": "surfaces", "problems": 2}
        )
    monkeypatch.setattr(Path, "replace", real_replace)

    assert path.read_text() == before
    assert json.loads(path.read_text())["problems"] == 1


def test_the_deadline_is_not_catchable_as_an_ordinary_failure():
    # Twenty-five `except Exception` handlers sit between the timer and the running optimizer. The
    # first version of this cap inherited from Exception, was swallowed by one of them, and a
    # ten-second allowance ran for seven minutes.
    assert not issubclass(R._Overran, Exception)
    assert issubclass(R._Overran, BaseException)


def test_the_deadline_fires_and_keeps_firing():
    """Defence in depth: `BaseException` is the fix, the repeating timer is the backstop.

    `setitimer` is one-shot by default, so a handler anywhere in the stack that does swallow the
    abort disarms the deadline permanently. Here one is swallowed deliberately, and a second must
    still arrive, because the run has to end whatever the optimizer does with the first.
    """
    swallowed = 0
    fired_again = False
    try:
        with R._deadline(0.05):
            while True:
                try:
                    for _ in range(10**7):
                        pass
                except BaseException:  # deliberately worse than any real handler
                    swallowed += 1
                    if swallowed >= 1:
                        break
            deadline = time.time() + 5.0
            while time.time() < deadline:
                pass
    except R._Overran:
        fired_again = True

    assert swallowed == 1, (
        "the first alarm should have been swallowed by the test's own handler"
    )
    assert fired_again, (
        "a swallowed alarm must not disarm the deadline for the rest of the run"
    )


def test_the_allowance_follows_the_objective_not_the_clock():
    # A flat cap in seconds disqualifies an optimizer for the objective's cost. The floor is per
    # evaluation, so a five-thousand-evaluation run is allowed proportionally more than a fifty.
    assert R.SECONDS_PER_EVAL * 5000 > R.SECONDS_PER_EVAL * 50
    assert R.MIN_SECONDS <= R.SECONDS_PER_EVAL * 1000 <= R.MAX_SECONDS


def test_the_recorder_races_exactly_what_recommend_can_return():
    """Otherwise the table holds ratings for optimizers no caller can be given, and spends the
    wall clock producing them. The two filters are one rule and must not drift apart."""
    from humpday.eligibility import passes_dim, passes_trials
    from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

    for n_dim, budget in ((2, 50), (12, 1000), (100, 5000)):
        expected = {
            n
            for n in PURE_OPTIMIZERS
            if passes_dim(n, n_dim) and passes_trials(n, n_dim, budget)
        }
        assert R.REFERENCE in expected, (
            "the timing reference must survive its own filter"
        )
        for suite in ratings.SUITES:
            cell = ratings.cells().get(f"{n_dim}/{budget}/{suite}")
            if not cell:
                continue
            raced = set(cell.get("ratings", {})) | set(cell.get("timed_out", []))
            assert raced <= expected, (
                f"raced the ineligible: {sorted(raced - expected)}"
            )
            assert set(cell.get("ineligible", {})).isdisjoint(raced)


def test_an_absent_optimizer_says_why_it_is_absent():
    from humpday import ratings as Rd

    for n_dim in Rd.recorded_dimensions():
        for suite in Rd.SUITES:
            cell = Rd.cell_for(n_dim, suite, 5000)
            if not cell:
                continue
            for name, reason in Rd.ineligible(n_dim, suite, 5000).items():
                assert reason, f"{name} excluded at d={n_dim} with no reason recorded"


def test_no_optimizer_is_rated_without_playing():
    """EloRatingSystem seeds every optimizer at 1500 on construction.

    An optimizer that never completed a rated match therefore leaves an untouched 1500 in
    `elo.ratings`, which lands mid-table and outranks anything that actually lost. It is a
    fabricated number of exactly the kind `suggest()` used to return.

    Two ways to never play, and the first version of this test only checked one of them, so
    fifty seeded entries shipped in v0.24.0: filtered out before the tournament (`ineligible`),
    or struck out on the first two problems (`timed_out`). The second is the one that got
    through, and `elo_by_dimension()` ranked a seeded 1500.0 above a PatternSearch that had
    earned 1499.4 by losing.

    What the shipped table can be checked for is that no rating belongs to an optimizer the cell
    itself says it never raced. The participation rule behind the filter is checked directly
    against `merge`, below, because the merged table records the ratings that survived it and
    not the counts they were judged on.
    """
    for key, cell in ratings.cells().items():
        rated = set(cell.get("ratings", {}))
        assert rated.isdisjoint(set(cell.get("ineligible", {}))), (
            f"{key} rates optimizers it never raced: "
            f"{sorted(rated & set(cell.get('ineligible', {})))}"
        )


def _merged_cell(tmp_path, monkeypatch, shard: dict) -> dict:
    monkeypatch.setattr(R, "SHARDS", tmp_path)
    monkeypatch.setattr(R, "OUT", tmp_path / "ratings.json")
    R._write_shard(
        R._shard_path(shard["n_dim"], shard["budget"], shard["suite"]), shard
    )
    key = f"{shard['n_dim']}/{shard['budget']}/{shard['suite']}"
    return R.merge()["cells"][key]


def test_merge_drops_every_rating_that_was_never_earned(tmp_path, monkeypatch):
    """The three ways to hold a 1500 without having played for it.

    `Filtered` never entered the tournament, `StruckOut` was disqualified before returning a
    value, and `OneStrike` was still one overrun short of disqualification when the cell ran out
    of wall clock -- so it is named nowhere, and only its match count gives it away.
    """
    cell = _merged_cell(
        tmp_path,
        monkeypatch,
        {
            "n_dim": 3,
            "budget": 50,
            "suite": "surfaces",
            "problems": 4,
            "ratings": {
                "Filtered": 1500.0,
                "StruckOut": 1500.0,
                "OneStrike": 1500.0,
                "Played": 1488.0,
            },
            "played": {"Played": 6},
            "timed_out": {"StruckOut": 12.0},
            "ineligible": {"Filtered": "needs more trials than the budget"},
        },
    )
    assert set(cell["ratings"]) == {"Played"}


def test_merge_keeps_a_rating_that_played_its_way_back_to_the_seed(
    tmp_path, monkeypatch
):
    """#356. An optimizer that draws every match it plays stays on exactly 1500.

    That is not a corner case in this tournament. Several optimizers finding the same optimum of
    an easy surface to the last bit is a round of exact ties, and a cell of such rounds leaves
    every one of them on the construction seed, having played every problem.

    Testing the value rather than the participation deleted those ratings, silently withdrawing
    optimizers that raced the whole cell -- from the cell, and from the recommendation grid built
    on it. The number is the same as the construction seed; the evidence behind it is not.
    """
    from humpday.optimizers.adaptive_optimizer import EloRatingSystem

    elo = EloRatingSystem()
    for _ in range(4):
        elo.update_ratings("Even", "Other", 0.5)
    assert elo.get_rating("Even") == pytest.approx(R.INITIAL_RATING), (
        "the premise: a drawn match between equals moves neither rating"
    )

    cell = _merged_cell(
        tmp_path,
        monkeypatch,
        {
            "n_dim": 3,
            "budget": 50,
            "suite": "surfaces",
            "problems": 4,
            "ratings": {
                "Even": elo.get_rating("Even"),
                "Other": elo.get_rating("Other"),
            },
            "played": {"Even": 4, "Other": 4},
        },
    )
    assert cell["ratings"]["Even"] == pytest.approx(R.INITIAL_RATING)


def test_merge_falls_back_to_the_value_test_for_a_shard_without_counts(
    tmp_path, monkeypatch
):
    """A shard recorded before participation was counted is still readable; it just cannot
    distinguish the earned 1500 from the seeded one, which is what the counts were added for."""
    cell = _merged_cell(
        tmp_path,
        monkeypatch,
        {
            "n_dim": 3,
            "budget": 50,
            "suite": "surfaces",
            "problems": 4,
            "ratings": {"Seeded": 1500.0, "Played": 1488.0},
        },
    )
    assert set(cell["ratings"]) == {"Played"}


def test_a_disqualified_optimizer_keeps_a_rating_it_did_earn():
    """The converse, so the fix above cannot be over-applied.

    An optimizer disqualified late in a cell played most of its matches, and that rating is real
    even though it ranks last. It is also what the TIMEOUT_REVOLT branch falls back on when a
    cell's allowance turns out to have been mis-calibrated, so discarding every timed-out
    optimizer's rating would empty exactly the cells that need one.
    """
    partial = 0
    for cell in ratings.cells().values():
        for name in cell.get("timed_out", []):
            if name in cell.get("ratings", {}):
                partial += 1
    assert partial, (
        "no timed-out optimizer retains a rating anywhere; the never-played filter is "
        "discarding earned ratings as well as seeded ones"
    )


def test_the_recommendation_grid_is_inside_the_package():
    """It was resolved as `Path(__file__).parent.parent`, which is the repository.

    An installed humpday therefore looked for `site-packages/benchmarks/recommendation_grid.json`,
    found nothing, and silently fell through to the rule-based ranking — recommending a different
    optimizer than the repo, the README and every test describe. At d=30 with 100 evaluations the
    repo said PRIMA_BOBYQA and a `pip install humpday` said DifferentialEvolution.

    Third instance of this defect, after `physics_objectives` and the `humpday.objectives` numpy
    import, so it is pinned rather than merely fixed. `parent.parent` escapes the package; only
    `parent` stays inside it.
    """
    import humpday
    from humpday import eligibility

    package = Path(humpday.__file__).parent
    grid = eligibility._GRID_PATH_DEFAULT
    assert grid.is_file(), f"the grid must ship with the package; looked at {grid}"
    assert package in grid.parents, (
        f"{grid} resolves outside the package directory {package}, so an installed humpday "
        "cannot read it"
    )
    assert eligibility._load_grid(grid), (
        "the shipped grid must be readable and non-empty"
    )


def test_evolution_strategy_adapts_its_step_size():
    """Closes #329. Without adaptation this is a fixed-radius sampler, not an evolution strategy.

    Asserted as a property of the run rather than of the source, so a future edit that reinstates a
    constant sigma fails here. Both directions must fire: a rule that only grows is as stuck as a
    constant.
    """
    from humpday.optimizers.evolutionary_algorithms import EvolutionStrategy

    seen = []
    original = EvolutionStrategy._run

    def spy(self):
        gen = original(self)
        try:
            value = None
            while True:
                point = gen.send(value)
                seen.append(gen.gi_frame.f_locals.get("sigma"))
                value = yield point
        except StopIteration:
            return

    EvolutionStrategy._run = spy
    try:

        def sphere(u):
            return sum((x - 0.4) ** 2 for x in u)

        best = EvolutionStrategy(objective=sphere, n_trials=2000, n_dim=4).optimize()[0]
    finally:
        EvolutionStrategy._run = original

    sigmas = [s for s in seen if s is not None]
    assert len(set(sigmas)) > 5, f"sigma took only {len(set(sigmas))} distinct values"
    assert any(b > a for a, b in zip(sigmas, sigmas[1:])), "sigma never grew"
    assert any(b < a for a, b in zip(sigmas, sigmas[1:])), "sigma never shrank"
    assert best < 1e-8, (
        f"best {best:.3e} on a sphere with 2,000 evaluations; a fixed radius stalls near 1e-3"
    )


def _stub_race(monkeypatch, too_costly=(), value_of=None, failing=()):
    """Drive `run_cell` over a recording generator, with the optimizers stubbed out.

    Returns the list the stub appends an entry to for every problem actually raced, so a test can
    say which problems a resumed cell worked on rather than how many. Problems named in
    `too_costly` overrun on the timing reference, which is how a real cell skips a problem
    without rating it, and optimizers named in `failing` raise, which is how one loses a problem
    it entered. `value_of(name, index)` supplies the objective values when the ordering matters.
    """
    import zlib

    import humpday.optimizers.alloptimizers as A

    raced: list[tuple[str, int]] = []

    def generator(n_dim, seed):
        index = 0
        while True:

            def objective(x, _index=index):
                return float(_index)

            yield objective
            index += 1

    def pure_optimize(objective, name, budget, n_dim):
        index = int(objective([0.5] * n_dim))
        if name == R.REFERENCE and index in too_costly:
            raise R._Overran()
        if name in failing:
            raise RuntimeError(f"{name} cannot run this problem")
        raced.append((name, index))
        if value_of is not None:
            return value_of(name, index), [0.5] * n_dim
        # Distinct per optimizer, so the round is rated rather than drawn throughout.
        return float(zlib.crc32(name.encode()) % 1000) + index, [0.5] * n_dim

    monkeypatch.setattr(
        R, "GENERATORS", {"surfaces": generator, "engineering": generator}
    )
    monkeypatch.setattr(A, "pure_optimize", pure_optimize)
    return raced


def test_a_resumed_cell_continues_the_problem_stream(tmp_path, monkeypatch):
    """#354. The shard counted the problems it recorded, and a resumed cell fast-forwarded the
    generator by that count -- so every problem skipped as too costly shifted the stream, and the
    resumed half of a cell re-raced problems the first half had already rated.

    A cell is meant to be one tournament over distinct problems. Rating the same landscape twice
    and half the intended sample once is a different, quieter tournament.
    """
    # Problem 1 costs more than the ceiling allows, so it is drawn and skipped: two problems are
    # rated out of the three drawn, which is exactly the case the old bookkeeping lost.
    raced = _stub_race(monkeypatch, too_costly={1})
    monkeypatch.setattr(R, "SHARDS", tmp_path)

    first = R.run_cell(3, 50, "surfaces", 2, 7, 25.0, 60.0)
    assert first["problems"] == 2
    assert (first["drawn"], first["skipped"]) == (3, 1)
    assert {index for _, index in raced} == {0, 2}

    raced.clear()
    second = R.run_cell(3, 50, "surfaces", 4, 7, 25.0, 60.0)
    assert second["problems"] == 4
    assert {index for _, index in raced} == {3, 4}, (
        "a resumed cell re-raced problems the first half of the cell already rated"
    )


def test_a_resumed_cell_refuses_to_continue_a_different_recording(
    tmp_path, monkeypatch
):
    """Two halves recorded under different seeds are two tournaments, and one Elo table over both
    is a table over a sample nobody chose. Refuse rather than splice."""
    _stub_race(monkeypatch)
    monkeypatch.setattr(R, "SHARDS", tmp_path)

    R.run_cell(3, 50, "surfaces", 2, 7, 25.0, 60.0)
    with pytest.raises(ValueError, match="seed"):
        R.run_cell(3, 50, "surfaces", 4, 99, 25.0, 60.0)


def test_a_cell_records_what_it_was_recorded_with(tmp_path, monkeypatch):
    _stub_race(monkeypatch)
    monkeypatch.setattr(R, "SHARDS", tmp_path)

    from humpday import _array as _A

    shard = R.run_cell(3, 50, "surfaces", 1, 7, 25.0, 60.0)
    assert shard["provenance"]["seed"] == 7
    assert shard["provenance"]["backend"] == _A.BACKEND
    assert shard["provenance"]["git"]


def test_a_run_is_seeded_by_its_coordinates_not_by_what_ran_before():
    """#354. An optimizer's draws used to depend on whatever the worker process had done before
    it, so the same cell recorded on twenty-seven workers was not the cell recorded on one, and
    `--seed` reproduced the problems without reproducing the tournament."""
    import random

    from humpday import _array as _A

    def sample(draw: int, name: str) -> list:
        R._seed_run(7, 3, 50, "surfaces", draw, name)
        return [_A.rng_random(), _A.random_scalar(), random.random()]

    first = sample(2, "RandomSearch")
    for _ in range(11):  # whatever the worker happened to do in between
        _A.rng_random()
        random.random()
    assert sample(2, "RandomSearch") == first

    assert sample(3, "RandomSearch") != first, "a different problem, same draws"
    assert sample(2, "NelderMead") != first, "a different optimizer, same draws"


def test_one_failed_optimizer_does_not_reorder_the_round(tmp_path, monkeypatch):
    """#344, on the recorder's own rating path.

    Normalising a round containing an inf made every score NaN. Both NaN comparisons are false, so
    the tie test failed and `scores[a] > scores[b]` failed, and every pair was recorded as a win
    for whichever optimizer the roster listed second: a round in which the objective values played
    no part at all, and whose ranking was exactly the roster reversed.

    So the values here run with the roster -- the first contender returns the best value, the last
    one fails -- and the recorded ranking must run with them. Under the old rule it came out
    backwards, led by the optimizer that never returned a value.

    Six identical rounds rather than one: Elo is updated pair by pair, so after a single round the
    optimizers rated early hold a point or two over equally-placed ones rated later, and the
    order is not yet exactly the order of the values. It settles by the fifth round.
    """
    from humpday.eligibility import passes_dim, passes_trials
    from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

    n_dim, budget = 3, 50
    contenders = [
        n
        for n in PURE_OPTIMIZERS
        if passes_dim(n, n_dim) and passes_trials(n, n_dim, budget)
    ]
    loser = contenders[-1]
    assert loser != R.REFERENCE, "the timing reference has to return a value"
    position = {name: float(i) for i, name in enumerate(contenders)}

    _stub_race(
        monkeypatch,
        value_of=lambda name, index: position[name],
        failing=(loser,),
    )
    monkeypatch.setattr(R, "SHARDS", tmp_path)

    shard = R.run_cell(n_dim, budget, "surfaces", 6, 7, 25.0, 120.0)
    ranked = sorted(shard["ratings"], key=lambda n: -shard["ratings"][n])

    assert ranked == contenders, (
        "the round was not ranked by the values the optimizers returned"
    )
    assert ranked[-1] == loser, "a failure must finish last, not first"
    assert shard["failures"][loser] == 6, "every failure is counted, not just the first"
    assert "RuntimeError" in shard["failure_reasons"][loser][0]
    assert len(shard["failure_reasons"][loser]) <= R.MAX_FAILURE_REASONS


def test_a_failure_is_not_counted_as_a_match_against_another_failure(
    tmp_path, monkeypatch
):
    """Two optimizers that both failed have shown nothing about each other, so neither may be
    credited with a match against the other. Rating that pair would hand one of them a win for
    being listed later, which is the same defect in a smaller place."""
    from humpday.eligibility import passes_dim, passes_trials
    from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

    n_dim, budget = 3, 50
    contenders = [
        n
        for n in PURE_OPTIMIZERS
        if passes_dim(n, n_dim) and passes_trials(n, n_dim, budget)
    ]
    both = (contenders[-2], contenders[-1])

    _stub_race(monkeypatch, value_of=lambda name, index: 1.0, failing=both)
    monkeypatch.setattr(R, "SHARDS", tmp_path)

    shard = R.run_cell(n_dim, budget, "surfaces", 1, 7, 25.0, 60.0)
    finished = [n for n in shard["ratings"] if n not in both]
    for name in both:
        assert shard["played"][name] == len(finished), (
            f"{name} played {shard['played'][name]} matches against {len(finished)} optimizers "
            "that returned a value, so it was also rated against the other failure"
        )
        assert shard["ratings"][name] < min(shard["ratings"][n] for n in finished)
