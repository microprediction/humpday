"""The reference gates refuse anything that is not a result (#405).

`test_reference_alignment` computed its statistics from whatever came back. A reference that
returned `inf` made the ratio zero; `NaN` made every comparison false, and false is a pass; a
HumpDay value of -1 on a problem whose minimum is 0 was accepted. #405 called the real gate with
only the adapters replaced and every one of these went green:

    HumpDay  Reference  calls in stub  gate
    100      +inf       0              PASS
    100      NaN        0              PASS
    NaN      1e-6       0              PASS
    -1       1e-6       0              PASS

These tests call the same gate the same way. They run in the default suite: nothing here needs
a reference library, because the point is what the gate does with what it is handed.
"""

import json
import math
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from tests import test_reference_alignment as G

INF, NAN = math.inf, math.nan


def _stub(value, evals, **extra):
    return lambda *a, **k: {"best_value": value, "evals": evals, **extra}


def _patched(module, **attrs):
    stack = ExitStack()
    for name, value in attrs.items():
        stack.enter_context(patch.object(module, name, value))
    return stack


def _run_gate(tmp_path: Path, hd, ref):
    """The real gate on one pair, with only the two sides replaced."""
    with _patched(
        G,
        REPO_ROOT=tmp_path,
        REFERENCES={"NelderMead": ("stub reference", ref, [])},
        PROBLEMS={"sphere": G.PROBLEMS["sphere"]},
        _run_humpday=hd,
    ):
        G.test_reference_alignment()


def _snapshot(tmp_path: Path):
    def refuse(token):
        raise AssertionError(f"snapshot holds the non-JSON token {token}")

    text = (tmp_path / "benchmarks" / "reference_alignment.json").read_text()
    return json.loads(text, parse_constant=refuse)


# The issue's table, exactly: no calls stand behind any of these.
ISSUE_TABLE = [
    (100.0, INF, "reference result invalid"),
    (100.0, NAN, "reference result invalid"),
    (NAN, 1e-6, "humpday result invalid"),
    (-1.0, 1e-6, "humpday result invalid"),
]


@pytest.mark.parametrize("hd_value, ref_value, heading", ISSUE_TABLE)
def test_the_issue_table_now_fails(tmp_path, hd_value, ref_value, heading):
    with pytest.raises(AssertionError) as caught:
        _run_gate(tmp_path, _stub(hd_value, 0), _stub(ref_value, 0))
    assert heading in str(caught.value)
    _snapshot(tmp_path)  # and the table it was judged on is still strict JSON


# The same values with honest call counts, so it is the value that is refused and not only the
# missing observation behind it.
@pytest.mark.parametrize("hd_value, ref_value, heading", ISSUE_TABLE)
def test_the_values_are_refused_even_with_calls_behind_them(
    tmp_path, hd_value, ref_value, heading
):
    with pytest.raises(AssertionError) as caught:
        _run_gate(tmp_path, _stub(hd_value, 50), _stub(ref_value, 50))
    message = str(caught.value)
    assert heading in message
    expected = "below the known minimum" if hd_value == -1.0 else "not a finite number"
    assert expected in message
    _snapshot(tmp_path)


def test_a_pair_of_real_results_still_passes(tmp_path):
    """The control: the refusals above are about the values, not the stubbing."""
    _run_gate(tmp_path, _stub(1e-6, 50), _stub(1e-6, 50))
    (row,) = _snapshot(tmp_path)["rows"]
    assert row["head_to_head_lost"] == 0.5


def test_one_bad_reference_run_among_good_ones_fails_and_stays_out_of_the_statistics(
    tmp_path,
):
    seeds = iter([INF] + [1e-3] * (G.N_RUNS - 1))
    ref = lambda *a, **k: {"best_value": next(seeds), "evals": 50}  # noqa: E731
    with pytest.raises(AssertionError) as caught:
        _run_gate(tmp_path, _stub(1e-3, 50), ref)
    assert "seed 0" in str(caught.value)
    (row,) = _snapshot(tmp_path)["rows"]
    assert row["reference_valid_runs"] == G.N_RUNS - 1
    assert row["head_to_head_lost"] == 0.5  # not 0: the inf run did not count as a loss


class TestValidateRun:
    P = {"opt": 0.0}

    def test_a_result(self):
        assert G.validate_run({"best_value": 1e-9, "evals": 10}, self.P, 10) == []

    def test_no_calls(self):
        assert G.validate_run({"best_value": 1e-9, "evals": 0}, self.P, 10)

    def test_over_budget(self):
        (issue,) = G.validate_run({"best_value": 1e-9, "evals": 11}, self.P, 10)
        assert "budget of 10" in issue

    def test_rounding_at_the_optimum_is_not_below_it(self):
        assert G.validate_run({"best_value": -4.4e-16, "evals": 10}, self.P, 10) == []

    def test_a_value_the_objective_never_returned(self):
        run = {"best_value": 1e-9, "evals": 10, "observed_best": 0.5}
        (issue,) = G.validate_run(run, self.P, 10)
        assert "lowest value the objective returned" in issue

    def test_a_call_count_that_disagrees_with_the_calls(self):
        run = {"best_value": 1e-9, "evals": 10, "reported_evals": 7}
        (issue,) = G.validate_run(run, self.P, 10)
        assert "reports 7 evaluations" in issue

    def test_a_library_reporting_below_what_it_evaluated(self):
        run = {"best_value": 1e-3, "evals": 10, "reported_value": 1e-6}
        (issue,) = G.validate_run(run, self.P, 10)
        assert "below anything it evaluated" in issue

    def test_a_library_reporting_its_last_point_rather_than_its_best_is_fine(self):
        run = {"best_value": 1e-6, "evals": 10, "reported_value": 1e-3}
        assert G.validate_run(run, self.P, 10) == []

    def test_points_outside_the_cube(self):
        run = {"best_value": 1e-6, "evals": 10, "outside": 3}
        (issue,) = G.validate_run(run, self.P, 10)
        assert "outside the unit cube" in issue


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_head_to_head_refuses_what_is_not_a_number(bad):
    with pytest.raises(ValueError):
        G.head_to_head([1.0, bad], [1.0])
    with pytest.raises(ValueError):
        G.head_to_head([1.0], [bad])


def test_the_real_humpday_runner_reports_what_it_observed():
    run = G._run_humpday("NelderMead", G.PROBLEMS["sphere"]["func"], 60, 2, seed=0)
    assert run["evals"] == run["reported_evals"] <= 60
    assert run["best_value"] == run["observed_best"]
    assert G.validate_run(run, G.PROBLEMS["sphere"], 60) == []


class TestTheJavaScriptGateHasTheSameContract:
    """#405 asked for the same contract on #399's JavaScript gate, where a returned `+inf`
    reference also made a zero ratio."""

    def _gate(self, js_run, ref_run):
        from tests import test_js_reference_alignment as J

        with _patched(
            J,
            REFERENCES={"NelderMead": ("stub reference", ref_run, [])},
            _all_installed=lambda mods: True,
            _run_js=js_run,
        ):
            J.test_the_javascript_port_tracks_its_reference("NelderMead", "sphere")

    @pytest.mark.parametrize(
        "js_value, ref_value",
        [(100.0, INF), (100.0, NAN), (NAN, 1e-6), (-1.0, 1e-6)],
    )
    def test_it_refuses_the_issue_table(self, js_value, ref_value):
        with pytest.raises(AssertionError, match="not results"):
            self._gate(_stub(js_value, 50), _stub(ref_value, 50))

    def test_and_passes_a_real_pair(self):
        self._gate(_stub(1e-6, 50), _stub(1e-6, 50))
