"""The browser demo and Python mirror see the same earthquake (#454)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from example_applications.tuned_mass_damper import problem

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")
# Exercise all six attachment floors, cube endpoints and the manual default.
# Half-integer floor decoding is a separate Python/JS rounding convention.
POINTS = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    [(0.02 - 0.003) / 0.037, (0.95 - 0.4) / 1.2, 0.08 / 0.4, 1.0],
    *[[0.7, 0.45, 0.2, floor / 5] for floor in range(6)],
]


@pytest.fixture(scope="module")
def browser():
    # Execute the actual page's model and its actual linalg dependency, not a
    # duplicate implementation. Stop before the canvas/rendering portion.
    script = r"""
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const root = process.argv[1];
const html = fs.readFileSync(path.join(root, 'docs/applications/tuned-mass-damper.html'), 'utf8');
const start = html.indexOf('const N = 6;');
const end = html.indexOf('const BASE_Y =', start);
if (start < 0 || end < 0) throw new Error('model extraction markers missing');
const source = html.slice(start, end);
const context = { Linalg: require(path.join(root, 'docs/js/modules/linalg.js')) };
vm.createContext(context);
vm.runInContext(source, context);
const points = JSON.parse(process.argv[2]);
const result = vm.runInContext(`({
    ground: GM.ag, baseline: D0,
    values: ${JSON.stringify(points)}.map(u => objective(u))
})`, context);
process.stdout.write(JSON.stringify(result));
"""
    result = subprocess.run(
        [NODE, "-e", script, str(ROOT), json.dumps(POINTS)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout)


def test_complete_ground_motion_matches(browser):
    assert len(browser["ground"]) == problem.L == 1600
    np.testing.assert_allclose(browser["ground"], problem.AG, rtol=1e-10, atol=1e-12)


def test_bare_building_matches(browser):
    assert browser["baseline"] == pytest.approx(problem._D0, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("index,point", list(enumerate(POINTS)))
def test_objective_matches(index, point, browser):
    assert browser["values"][index] == pytest.approx(
        problem.objective(point), rel=1e-8, abs=1e-8
    )
