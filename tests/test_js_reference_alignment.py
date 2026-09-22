"""The JavaScript ports against the implementations they were written from (#164).

The only correctness check the JS side had was `test_js_parity.py`, which compares it with the
Python port. That catches one port drifting from the other, and misses both drifting together --
a shared simplification passes parity and fails nothing. `test_reference_alignment.py` compares
Python against scipy and friends; nothing compared JavaScript against anything but Python.

This closes the triangle: JS against the same references, on the same objectives, judged the
same way. The issue asked for a characterisation harness that always passes and leaves a
snapshot; it gets a gate instead, because #326 made the Python one a gate for the reason that a
test which cannot fail was cited as evidence it could not provide.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_reference_alignment import (  # noqa: E402
    CONVERGED_GAP,
    N_RUNS,
    PROBLEMS,
    REFERENCES,
    _all_installed,
    _run_humpday,
)

NODE = shutil.which("node")
RUNNER = Path(__file__).parent / "js_reference_runner.js"
REPO_ROOT = Path(__file__).parent.parent

# The ports where alignment matters most and where a reference exists on both sides. PRIMA_BOBYQA
# is in the issue's list but its reference (Py-BOBYQA) is an optional install, so it joins the
# others only when that is present -- `_all_installed` decides, as it does for the Python gate.
JS_ALGORITHMS = ["NelderMead", "Powell", "LBFGSB", "PRIMA_BOBYQA"]

N_DIM = 2
N_TRIALS = 200

# How far a JavaScript port may sit behind the reference its Python twin was written from.
# Measured, like the Python table, and for the same reason: these are what the ports do, not
# what they should do. A JS port is a behavioural twin rather than a bit-exact one for every
# algorithm here, so the bar is looser than the Python gate's 3.0.
DEFAULT_RATIO_CEILING = 10.0

# What the ports do today, measured with both languages on the same objectives -- which they
# were not, the first time these numbers were taken. The runner defined its objectives in
# JavaScript, the harness defined them in Python, and the shifted ones from #387 reached only
# one side, so every ratio below was computed across two different problems (#406). The
# identity tests above exist so that cannot recur.
#
# Re-measured, the picture changes in one place and holds in the others:
#
#   Powell on the sphere is genuinely 8e+10 behind scipy -- 8.06e-05 against 3.08e-33 -- because
#   its JavaScript line search tries four fixed step sizes where Python uses Brent (#78).
#
#   Powell on Ackley was recorded as 3.09e+09 and is actually 0.18: the JavaScript port is
#   better than scipy's Powell there. That number was an artifact of the mismatch.
#
#   LBFGSB falls behind on Rosenbrock, where the curvature is what the memory is for, by the
#   same 1,318 as before -- that pair was unaffected.
#
#   PRIMA_BOBYQA falls behind on Rosenbrock and Ackley.
#
# All of these are #78's territory: ports that agree with Python on behaviour but not on quality.
RATIO_CEILING = {
    ("Powell", "sphere"): 1.7e11,  # measured 80609000013.37
    ("Powell", "rosenbrock"): 17.0,  # measured 7.99
    ("LBFGSB", "rosenbrock"): 2700.0,  # measured 1318.23
    ("PRIMA_BOBYQA", "ackley"): 160.0,  # measured 71.51
    # This one varies run to run because the JavaScript PRIMA ports call Math.random() directly
    # rather than the portable stream, so the seed the runner sets does not reach them (#401).
    # The ceiling has room for that spread until the ports are seeded properly.
    ("PRIMA_BOBYQA", "rosenbrock"): 2.0e10,  # measured 7.1e+08 to 4.9e+09
}

# Ports whose result differs from their Python twin by more than six orders of magnitude on the
# sphere. Listed rather than tolerated: the test below asserts the divergence is still there, so
# repairing one fails this file and asks for the entry to be removed.
KNOWN_PORT_DIVERGENCE = {
    "Powell": "the JS line search is four fixed steps where Python uses Brent (#78)",
}


def _ratio_ceiling(algorithm: str, problem: str) -> float:
    return RATIO_CEILING.get((algorithm, problem), DEFAULT_RATIO_CEILING)


# Points the two implementations must agree on before any optimizer runs: the optimum, the
# centre of the cube, two interior points and both bounds. Chosen to catch a shift (the optimum
# and the centre disagree), a scale (the interior points disagree) and a domain mapping (the
# bounds disagree).
IDENTITY_POINTS = [
    [0.4127, 0.6831],
    [0.5, 0.5],
    [0.13, 0.87],
    [0.62, 0.31],
    [0.0, 0.0],
    [1.0, 1.0],
]


def _probe_js(problem: str, point: list) -> float:
    result = subprocess.run(
        [NODE, str(RUNNER), "--probe", problem, ",".join(repr(v) for v in point)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return float(json.loads(result.stdout)["value"])


@pytest.mark.skipif(not NODE, reason="node not on PATH")
@pytest.mark.parametrize("problem_id", sorted(PROBLEMS))
def test_both_languages_optimize_the_same_function(problem_id):
    """The check that was missing, and without which the rest of this file means nothing.

    The JS runner defines its objectives in JavaScript and the Python harness defines them in
    Python; nothing tied the two together. They drifted -- the runner carried the shifted
    objectives from #387 while the Python side still had the old ones -- and every ratio in this
    file was computed across two different problems. The optimizers all passed, because an
    optimizer solves whichever sphere it is given (#406).

    Comparing final values cannot catch that. Comparing the functions can.
    """
    python_f = PROBLEMS[problem_id]["func"]
    for point in IDENTITY_POINTS:
        py = float(python_f(point))
        js = _probe_js(problem_id, point)
        assert js == pytest.approx(py, rel=1e-12, abs=1e-12), (
            f"{problem_id} differs between the languages at {point}: "
            f"Python {py!r}, JavaScript {js!r}"
        )


@pytest.mark.skipif(not NODE, reason="node not on PATH")
@pytest.mark.parametrize("problem_id", sorted(PROBLEMS))
def test_the_optimum_is_where_both_languages_say_it_is(problem_id):
    """And it is a minimum on both sides, not merely the same number."""
    problem = PROBLEMS[problem_id]
    x_opt = problem["x_opt"]
    assert _probe_js(problem_id, x_opt) == pytest.approx(problem["opt"], abs=1e-12)
    for offset in (0.05, -0.05):
        moved = [min(1.0, max(0.0, v + offset)) for v in x_opt]
        assert _probe_js(problem_id, moved) > problem["opt"], (
            f"{problem_id} is not minimised at {x_opt} in JavaScript"
        )


def _run_js(algorithm: str, problem: str, seed: int) -> float:
    result = subprocess.run(
        [NODE, str(RUNNER), algorithm, problem, str(N_TRIALS), str(N_DIM), str(seed)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return float(json.loads(result.stdout)["best_value"])


@pytest.mark.reference
@pytest.mark.skipif(not NODE, reason="node not on PATH")
@pytest.mark.parametrize("algorithm", JS_ALGORITHMS)
@pytest.mark.parametrize("problem_id", sorted(PROBLEMS))
def test_the_javascript_port_tracks_its_reference(algorithm, problem_id):
    ref_label, ref_fn, mods = REFERENCES[algorithm]
    if not _all_installed(mods):
        pytest.skip(f"{', '.join(mods)} not installed")

    problem = PROBLEMS[problem_id]
    func, opt_value = problem["func"], problem["opt"]

    js_vals = [_run_js(algorithm, problem_id, seed) for seed in range(N_RUNS)]
    ref_vals = []
    for seed in range(N_RUNS):
        ref_vals.append(ref_fn(func, N_TRIALS, N_DIM, seed=seed)["best_value"])

    js_med = sorted(js_vals)[N_RUNS // 2]
    ref_med = sorted(ref_vals)[N_RUNS // 2]
    js_gap = js_med - opt_value
    ref_gap = ref_med - opt_value
    ratio = (js_gap + 1e-15) / (ref_gap + 1e-15)

    print(
        f"  {algorithm:<14} {problem_id:<11} js={js_med:>11.4g}  {ref_label}={ref_med:>11.4g}"
        f"  js/ref={ratio:>9.2f}"
    )

    if js_gap <= CONVERGED_GAP:
        return  # solved it; the ratio is then the epsilon guard dividing itself
    ceiling = _ratio_ceiling(algorithm, problem_id)
    assert ratio <= ceiling, (
        f"{algorithm} in JavaScript is {ratio:.2f}x the {ref_label} gap on {problem_id}, "
        f"over its ceiling of {ceiling:g}"
    )


@pytest.mark.reference
@pytest.mark.skipif(not NODE, reason="node not on PATH")
@pytest.mark.parametrize("algorithm", JS_ALGORITHMS)
def test_the_two_ports_agree_about_the_same_problem(algorithm):
    """The JS port and its Python twin, on one objective, sanity-checked against each other.

    Not a parity test -- test_js_parity.py does that properly -- but a guard that this file is
    driving the same algorithm on both sides rather than comparing two different things.
    """
    if not _all_installed(REFERENCES[algorithm][2]):
        pytest.skip("reference not installed")
    js = _run_js(algorithm, "sphere", 0)
    py = _run_humpday(algorithm, PROBLEMS["sphere"]["func"], N_TRIALS, N_DIM, seed=0)[
        "best_value"
    ]
    assert math.isfinite(js) and math.isfinite(py)
    both_converged = js <= CONVERGED_GAP and py <= CONVERGED_GAP
    decades = abs(math.log10(max(js, 1e-300)) - math.log10(max(py, 1e-300)))
    close = both_converged or decades < 6

    if algorithm in KNOWN_PORT_DIVERGENCE:
        assert not close, (
            f"{algorithm}'s ports now agree ({js:.4g} vs {py:.4g}). That is the good news; "
            f"remove it from KNOWN_PORT_DIVERGENCE, which says: "
            f"{KNOWN_PORT_DIVERGENCE[algorithm]}"
        )
        return

    assert close, (
        f"{algorithm}: JavaScript reached {js:.4g} and Python {py:.4g} on the same problem"
    )
