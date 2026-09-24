"""JavaScript success means a finite objective value was observed (#396).

Python's minimize has reported it that way since #347. The JavaScript base
returned success: true unconditionally, so a zero budget or an objective that
never returned a finite number came back as a successful run whose bestX was
the random placeholder drawn in the constructor.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
RUNNER = Path(__file__).parent / "js_success_runner.js"


@pytest.fixture(scope="module")
def rows():
    if not NODE:
        pytest.skip("node not on PATH")
    result = subprocess.run(
        [NODE, str(RUNNER)], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert len({r["name"] for r in rows}) >= 20
    return rows


def test_no_finite_observation_is_not_success(rows):
    wrong = [
        (r["name"], r["mode"], r["label"])
        for r in rows
        if r["label"] != "finite" and r["success"] is not False
    ]
    assert not wrong, wrong


def test_the_failure_says_why(rows):
    for r in rows:
        if r["label"] == "zero":
            assert r["message"] == "no evaluations were made", r
        elif r["label"] in ("nan", "infinity"):
            assert r["message"] == "no finite objective value was observed", r


def test_finite_run_succeeds_at_a_point_it_evaluated(rows):
    wrong = [
        r
        for r in rows
        if r["label"] == "finite"
        and not (
            r["success"] is True
            and r["bestXEvaluated"]
            and r["valueAtBestX"] == r["bestValue"]
        )
    ]
    assert not wrong, wrong


def test_call_counts_are_accurate(rows):
    wrong = [r for r in rows if r["calls"] != r["evaluations"]]
    assert not wrong, wrong
