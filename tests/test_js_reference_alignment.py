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

# What the ports do today.
#
#   LBFGSB's JavaScript port tracks its reference on the sphere and Ackley and falls behind on
#   Rosenbrock, where the curvature is what the memory is for.
#
#   PRIMA_BOBYQA falls behind on Rosenbrock and Ackley.
#
# Both are #78's territory -- ports that agree with Python on behaviour but not on quality.
# Recorded so they cannot get worse, and so that fixing one shows up as a failing ceiling that
# wants lowering. Powell's three entries were removed when its line search was ported; that is
# what this table is for.
RATIO_CEILING = {
    ("LBFGSB", "rosenbrock"): 2700.0,  # measured 1318.61
    # PRIMA_BOBYQA's reference is Py-BOBYQA, an optional install.
    ("PRIMA_BOBYQA", "ackley"): 140.0,  # measured 65.27
    # This one varies run to run -- 3.3e+08, 1.7e+09, 2.7e+09, 4.9e+09 -- because the JavaScript
    # PRIMA ports call Math.random() directly rather than the portable stream, so the seed the
    # runner sets does not reach them (#401). The ceiling has room for that spread until the
    # ports are seeded properly.
    ("PRIMA_BOBYQA", "rosenbrock"): 2.0e10,  # measured 3.3e+08 to 4.9e+09
}

# Ports whose result differs from their Python twin by more than six orders of magnitude on the
# sphere. Listed rather than tolerated: the test below asserts the divergence is still there, so
# repairing one fails this file and asks for the entry to be removed. Powell was the only entry
# and it has been removed, which is how the mechanism is supposed to end.
KNOWN_PORT_DIVERGENCE: dict = {}


def _ratio_ceiling(algorithm: str, problem: str) -> float:
    return RATIO_CEILING.get((algorithm, problem), DEFAULT_RATIO_CEILING)


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
