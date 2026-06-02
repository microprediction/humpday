"""
Parity test for humpday.eligibility (Python) ↔ docs/js/modules/eligibility.js (JS).

For every grid cell (n_dim, n_trials) the recommendation grid covers, run
`Eligibility.recommend(n_dim, n_trials, eval_time)` on both sides at each
overhead-tier threshold and assert they pick the same algorithm.

This test runs in a Node subprocess. If `node` isn't on PATH it skips
cleanly so Python-only CI runs aren't broken.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from humpday import eligibility as E

REPO_ROOT = Path(__file__).parent.parent
NODE = shutil.which("node")

# Sample eval_times spanning every tier boundary.
EVAL_TIMES = [1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1.0]

JS_HARNESS = r"""
const E = require("REPO/docs/js/modules/eligibility.js");
const cells = CELLS_JSON;
const evalTimes = ETS_JSON;
const out = [];
for (const [d, t] of cells) {
    for (const et of evalTimes) {
        out.push({ d, t, et, pick: E.recommend(d, t, et) });
    }
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_js_recommend_matches_python_on_every_grid_cell():
    grid_path = REPO_ROOT / "benchmarks" / "recommendation_grid.json"
    grid = json.loads(grid_path.read_text())
    cells = []
    for key in grid["cells"]:
        d, t = key.split("/")
        cells.append([int(d), int(t)])
    cells.sort()

    js_src = (
        JS_HARNESS.replace("REPO", str(REPO_ROOT))
        .replace("CELLS_JSON", json.dumps(cells))
        .replace("ETS_JSON", json.dumps(EVAL_TIMES))
    )
    result = subprocess.run(
        [NODE, "-e", js_src],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"node harness exited {result.returncode}\n"
            f"stderr: {result.stderr}\n"
            f"stdout: {result.stdout[:400]}"
        )

    js_picks = json.loads(result.stdout)
    E._clear_grid_cache()

    mismatches: list[tuple[int, int, float, str, str]] = []
    for row in js_picks:
        d, t, et, js_pick = row["d"], row["t"], row["et"], row["pick"]
        py_pick = E.recommend(n_dim=d, n_trials=t, eval_time=et, grid_path=grid_path)
        if js_pick != py_pick:
            mismatches.append((d, t, et, py_pick, js_pick))

    assert not mismatches, (
        f"JS and Python recommenders diverged on {len(mismatches)} of "
        f"{len(js_picks)} (cell, eval_time) tuples:\n"
        + "\n".join(
            f"  n_dim={d}, n_trials={t}, eval_time={et:.0e}: python={py!r} js={js!r}"
            for d, t, et, py, js in mismatches[:10]
        )
    )
