"""
Opt-in regression tests wrapping the cross-validation framework.

These wrap `cross_validation_framework.py` (and its sibling files at
repo root: `statistical_validation.py`, `benchmark_suite.py`,
`run_comprehensive_validation.py`) so the framework actually runs
under pytest, with a real pass/fail outcome and a regression threshold.

The framework is slow (seconds to minutes per category) and noisy,
so the tests are gated behind the `@pytest.mark.validation` marker.
Default `pytest` does NOT run them. To run on demand:

    pytest -m validation
    pytest -m validation -v tests/test_validation_framework.py

The thresholds below are calibrated against the current measured
pass rates (recorded 2026-05-27). When an algorithm's pass rate
improves, tighten its floor — that turns the test into a ratchet
that prevents future regression.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# Make the repo-root validation modules importable.
sys.path.insert(0, str(REPO_ROOT))

# Skip the whole module if the framework isn't importable for any reason
# (e.g. scipy missing on a minimal install).
cross_validation_framework = pytest.importorskip(
    "cross_validation_framework",
    reason="cross_validation_framework module not importable",
)
CrossValidationFramework = cross_validation_framework.CrossValidationFramework


def _aggregate(results: dict) -> tuple[int, int, dict[str, tuple[int, int]]]:
    """Sum passed/total over every ValidationResult in the framework's
    return dict, and break down per algorithm."""
    total = 0
    passed = 0
    by_alg: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [passed, total]
    for _category, items in results.items():
        for r in items:
            # r is a ValidationResult dataclass instance.
            total += 1
            by_alg[r.algorithm_name][1] += 1
            if bool(r.passed):
                passed += 1
                by_alg[r.algorithm_name][0] += 1
    return passed, total, {k: (v[0], v[1]) for k, v in by_alg.items()}


@pytest.mark.validation
def test_mathematical_correctness_pass_rate():
    """Math-correctness suite: simplex validity, trust-region behaviour,
    bounds handling, parameter consistency, monotone convergence on
    unimodal targets.

    Asserts the OVERALL pass rate stays above a calibrated floor. When
    the failing algorithms (Powell, LBFGSB, NelderMead per the current
    sweep) are fixed and their per-algorithm rates climb, tighten the
    floor here so a future regression is caught."""
    framework = CrossValidationFramework()
    results = framework.run_mathematical_correctness_validation()

    passed, total, by_alg = _aggregate(results)

    print(f"\nMathematical correctness: {passed}/{total} = {passed / total:.1%}")
    for alg in sorted(by_alg):
        p, t = by_alg[alg]
        print(f"  {alg:20s}  {p:3d}/{t:3d}  ({p / t:.0%})")

    # Calibrated 2026-05-27: framework reports ~91% on the math suite.
    # Set floor at 70% to leave headroom for stochastic variance while
    # still catching a real regression (e.g. half the tests failing).
    pass_rate = passed / total
    assert pass_rate >= 0.70, (
        f"Mathematical correctness regressed: {passed}/{total} = "
        f"{pass_rate:.1%} (floor 70%)"
    )


@pytest.mark.validation
@pytest.mark.slow
def test_comprehensive_runner_pass_rate():
    """End-to-end smoke test: invoke `run_comprehensive_validation.py
    --quick --skip-js` as a subprocess and check the overall pass rate
    parsed from its output."""
    runner = REPO_ROOT / "run_comprehensive_validation.py"
    assert runner.exists(), f"missing {runner}"

    proc = subprocess.run(
        [sys.executable, str(runner), "--quick", "--skip-js"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.returncode == 0, (
        f"runner exited non-zero ({proc.returncode}); stderr tail:\n"
        + "\n".join(proc.stderr.splitlines()[-10:])
    )

    m = re.search(r"Overall pass rate:\s*([\d.]+)%", proc.stdout)
    assert m, (
        "Could not parse 'Overall pass rate' from runner output. "
        "Last 5 lines:\n" + "\n".join(proc.stdout.splitlines()[-5:])
    )
    pass_rate = float(m.group(1))
    print(f"\nComprehensive runner overall pass rate: {pass_rate:.1f}%")

    # Calibrated 2026-05-27: --quick --skip-js gives ~60% overall.
    # Floor at 45% so a real regression trips before noise.
    assert pass_rate >= 45.0, (
        f"Comprehensive runner regressed: {pass_rate:.1f}% < 45% floor"
    )
