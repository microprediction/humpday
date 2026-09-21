"""The documentation pages load the modules their optimizers need, in the order they need them.

All 83 pages that load `base-optimizer.js` omitted `prng.js`. The base module reads window globals
in a browser and `require('./prng.js')` under Node, so it captured nulls for portableLog and
portableExp on every page while every Node parity test passed, and SimulatedAnnealing,
HillClimbing and FireflyAlgorithm threw on their first call across the whole site.

Loading prng.js afterwards would not have helped: the capture happens when base-optimizer.js runs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
REPO_ROOT = Path(__file__).parent.parent
DOCS = REPO_ROOT / "docs"
RUNNER = Path(__file__).parent / "js_script_tag_runner.js"

# The three the issue caught -- the ones that call into the portable math -- and the module each
# is declared in, since not every page loads every module.
NEEDS_PORTABLE_MATH = {
    "SimulatedAnnealing": "evolutionary-algorithms.js",
    "FireflyAlgorithm": "evolutionary-algorithms.js",
    "HillClimbing": "search-algorithms.js",
}


def _pages() -> list:
    return sorted(
        p
        for p in DOCS.rglob("*.html")
        if "base-optimizer.js" in p.read_text(encoding="utf-8")
    )


def test_there_are_pages_to_check():
    assert len(_pages()) > 50, "the sweep is not finding the documentation pages"


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_prng_is_loaded_before_the_module_that_captures_it(page):
    text = page.read_text(encoding="utf-8")
    prng = text.find("js/modules/prng.js")
    base = text.find("js/modules/base-optimizer.js")
    assert prng != -1, f"{page.relative_to(DOCS)} never loads prng.js"
    assert prng < base, (
        f"{page.relative_to(DOCS)} loads prng.js after base-optimizer.js, which has already "
        "captured nulls by then"
    )


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_every_page_runs_every_optimizer_it_loads():
    """Runs them, rather than constructing them, in a context with `window` and no `module` --
    the browser configuration, and the only one in which this defect is visible."""
    pages = [str(p.relative_to(REPO_ROOT)) for p in _pages()]
    result = subprocess.run(
        [NODE, str(RUNNER), "60", "2", *pages],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert len(report) == len(pages)

    broken = {r["page"]: r["failures"] for r in report if r["failures"]}
    assert not broken, json.dumps(broken, indent=2)[:2000]

    for entry in report:
        assert entry["scripts"], f"{entry['page']} loads no modules"
        assert len(entry["ran"]) >= 15, f"{entry['page']} ran only {entry['ran']}"
        for name, module in NEEDS_PORTABLE_MATH.items():
            if any(module in src for src in entry["scripts"]):
                assert name in entry["ran"], f"{entry['page']} did not run {name}"

    # And the three were exercised somewhere, so a page set that stopped loading those modules
    # would not quietly turn this test into a formality.
    for name in NEEDS_PORTABLE_MATH:
        pages_running = sum(1 for entry in report if name in entry["ran"])
        assert pages_running > 50, f"{name} ran on only {pages_running} pages"
