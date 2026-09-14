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

    An optimizer filtered out before the tournament therefore leaves an untouched 1500 in
    `elo.ratings`, which lands mid-table and outranks anything that actually lost. It is a
    fabricated number of exactly the kind `suggest()` used to return.
    """
    for key, cell in ratings.cells().items():
        excluded = set(cell.get("ineligible", {}))
        rated = set(cell.get("ratings", {}))
        assert rated.isdisjoint(excluded), (
            f"{key} rates optimizers it never raced: {sorted(rated & excluded)}"
        )
