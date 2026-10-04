"""A seed passed to usePortableRng determines every JavaScript optimizer's run (#401).

Thirteen ports drew from MathUtils and so from the portable PCG32 stream; the other ten called
Math.random() directly, so seeding a JavaScript run made it reproducible only for those
thirteen. These tests hold all of them to it: two runs with the same seed evaluate the same
points in the same order, a different seed changes the run for every algorithm that draws at
all, and no optimizer source calls Math.random() outside the legacy fallback in MathUtils.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
RUNNER = Path(__file__).parent / "js_seed_runner.js"
MODULES = Path(__file__).parent.parent / "docs" / "js" / "modules"

# Large enough that the PRIMA ports reach their interpolation-set jitter, which is where they
# draw; at small budgets they finish without touching the stream and look deterministic.
N_TRIALS = 200
N_DIM = 2

# Algorithms that make no random draws, so a different seed is expected to change nothing.
# Listed so that one quietly becoming deterministic, or this one starting to draw, is noticed.
DETERMINISTIC = {"GridSearch"}

# The only file allowed to call Math.random: MathUtils' fallback for when no portable stream is
# active, which is what keeps unseeded demos working.
FALLBACK_FILE = "base-optimizer.js"


@pytest.fixture(scope="module")
def runs():
    result = subprocess.run(
        [NODE, str(RUNNER), str(N_TRIALS), str(N_DIM), "1", "2"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_every_registered_optimizer_ran_on_the_portable_stream(runs):
    # The runner makes Math.random throw while the stream is active, so a port that bypasses
    # MathUtils shows up here as an error rather than as a run that merely varies.
    assert len(runs) >= 23
    errors = {name: r["error"] for name, r in runs.items() if r["error"]}
    assert not errors, errors


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_the_same_seed_gives_the_same_run(runs):
    differing = [n for n, r in runs.items() if not r["error"] and r["a1"] != r["a2"]]
    assert not differing, f"same seed, different run: {differing}"


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_a_different_seed_gives_a_different_run(runs):
    unmoved = {n for n, r in runs.items() if not r["error"] and r["a1"] == r["b"]}
    assert unmoved == DETERMINISTIC, (
        f"seed had no effect on {sorted(unmoved - DETERMINISTIC)}; "
        f"expected to be deterministic but varied: {sorted(DETERMINISTIC - unmoved)}"
    )


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def test_no_optimizer_source_calls_math_random_directly():
    offenders = {}
    for path in sorted(MODULES.glob("*.js")):
        if path.name == FALLBACK_FILE:
            continue
        count = len(re.findall(r"Math\.random\s*\(", _strip_comments(path.read_text())))
        if count:
            offenders[path.name] = count
    assert not offenders, (
        f"Math.random() bypasses usePortableRng; use MathUtils.randomScalar() and friends: "
        f"{offenders}"
    )


def test_math_random_in_the_fallback_file_is_only_the_fallback():
    # Every direct call in base-optimizer.js must sit on the legacy branch of MathUtils, i.e.
    # inside the object literal, which ends before the portable transcendentals are attached.
    source = (MODULES / FALLBACK_FILE).read_text()
    start = source.index("const MathUtils = {")
    end = source.index("MathUtils.portableLog =")
    code = _strip_comments(source)
    code_start = len(_strip_comments(source[:start]))
    code_end = len(_strip_comments(source[:end]))
    for m in re.finditer(r"Math\.random\s*\(", code):
        assert code_start <= m.start() < code_end, "Math.random() outside MathUtils"
    # And MathUtils.random itself goes through the stream rather than around it.
    assert re.search(r"random:\s*\(\)\s*=>\s*MathUtils\.randomScalar\(\)", source)
