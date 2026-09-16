"""adaptive_optimize spends at most its budget and reports what it did (#353);
save_ratings saves to a bare filename and never announces a save that did not
happen (#362).
"""

import builtins
import json
import os
import warnings
from unittest import mock

import pytest

from humpday.optimizers.adaptive_optimizer import EloRatingSystem, adaptive_optimize
from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

N_ALG = len(PURE_OPTIMIZERS)
TRIALS = 5
ROUND = N_ALG * TRIALS


def _problems(k):
    def make(i):
        return lambda x: sum((float(v) - 0.1 * i) ** 2 for v in x)

    return iter([make(i) for i in range(k)])


def _run(gen, budget, **kw):
    calls = {"n": 0}

    def counting(gen):
        for f in gen:

            def g(x, _f=f):
                calls["n"] += 1
                return _f(x)

            yield g

    r = adaptive_optimize(
        counting(gen),
        trials_budget=budget,
        n_dim=2,
        trials_per_warmup=TRIALS,
        verbose=False,
        **kw,
    )
    return r, calls["n"]


def test_budget_smaller_than_one_round_makes_no_calls():
    r, calls = _run(_problems(10), budget=1, n_warmup_problems=1)
    assert calls == 0
    assert r["total_problems_solved"] == 0
    assert r["total_evaluations"] == 0


def test_empty_generator_reports_zero_problems():
    r, calls = _run(iter([]), budget=10 * ROUND, n_warmup_problems=2)
    assert calls == 0
    assert r["total_problems_solved"] == 0


@pytest.mark.parametrize("budget", [ROUND, ROUND + 3, 2 * ROUND - 1, 3 * ROUND])
def test_never_exceeds_the_budget(budget):
    r, calls = _run(_problems(20), budget=budget, n_warmup_problems=5)
    assert calls <= budget
    assert r["total_evaluations"] == calls
    # Warmup problems scheduled are exactly what the budget can pay for.
    assert r["total_problems_solved"] >= budget // ROUND


def test_remainder_smaller_than_an_adaptive_round_is_not_spent():
    # One full warmup round fits; what is left cannot pay for 8 x TRIALS.
    r, calls = _run(_problems(20), budget=ROUND + 8 * TRIALS - 1, n_warmup_problems=1)
    assert calls <= ROUND
    assert r["total_problems_solved"] == 1


def test_short_generator_counts_only_consumed_problems():
    r, calls = _run(_problems(2), budget=10 * ROUND, n_warmup_problems=5)
    assert r["total_problems_solved"] == 2
    assert calls <= 10 * ROUND


# --- #362 ------------------------------------------------------------------


def test_save_to_bare_filename_and_nested_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    elo = EloRatingSystem()
    elo.update_ratings("Alloy", "NelderMead", 1.0)
    assert elo.save_ratings("my_ratings.json") is True
    assert os.path.exists("my_ratings.json")
    assert elo.save_ratings(str(tmp_path / "deep" / "er" / "ratings.json")) is True
    assert EloRatingSystem().load_ratings("my_ratings.json") is True
    with open("my_ratings.json") as f:
        assert "Alloy" in json.load(f)["ratings"]


def test_injected_write_failure_is_reported_not_swallowed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def denied(path, *a, **k):
        if str(path).endswith("ratings.json"):
            raise PermissionError("injected")
        return real_open(path, *a, **k)

    elo = EloRatingSystem()
    with mock.patch("builtins.open", side_effect=denied):
        assert elo.save_ratings("ratings.json") is False
    assert isinstance(elo.last_error, PermissionError)
    assert not os.path.exists("ratings.json")


def test_malformed_ratings_file_is_a_reported_failure(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[1, 2, 3]")
    elo = EloRatingSystem()
    assert elo.load_ratings(str(p)) is False
    assert isinstance(elo.last_error, ValueError)


def test_adaptive_optimize_does_not_announce_a_failed_save(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def denied(path, *a, **k):
        if str(path).endswith("learned.json"):
            raise PermissionError("injected")
        return real_open(path, *a, **k)

    with mock.patch("builtins.open", side_effect=denied):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = adaptive_optimize(
                _problems(3),
                trials_budget=ROUND,
                n_dim=2,
                n_warmup_problems=1,
                trials_per_warmup=TRIALS,
                elo_ratings_file="learned.json",
                verbose=True,
            )
    out = capsys.readouterr().out
    assert "Saved updated Elo ratings" not in out
    assert r["ratings_saved"] is False
    assert any("could not save Elo ratings" in str(w.message) for w in caught)


def test_adaptive_optimize_reports_a_real_save(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = adaptive_optimize(
        _problems(3),
        trials_budget=ROUND,
        n_dim=2,
        n_warmup_problems=1,
        trials_per_warmup=TRIALS,
        elo_ratings_file="learned.json",
        verbose=False,
    )
    assert r["ratings_saved"] is True
    assert os.path.exists("learned.json")
