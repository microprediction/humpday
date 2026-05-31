"""
Headless-browser smoke tests for the static site under `docs/`.

These tests load three representative pages — `test-modular.html`,
`contest.html`, and an algorithm visualizer page — in headless
Chromium and assert that page-level JavaScript loads cleanly. They
exist specifically to catch the class of regression that broke the
live site in PR #73: a JS module change that worked in Node but
threw `"Identifier 'Optimizer' has already been declared"` in the
browser, taking down every page that loads the per-family modules
as separate `<script>` tags.

Marked `@pytest.mark.browser` and excluded from the default `pytest`
run because Playwright + Chromium aren't in the minimal install
profile. To run them:

    pip install playwright
    playwright install chromium
    pytest -m browser

The fixture spins up `python3 -m http.server` against `docs/` on a
random port for the duration of the module.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Skip the module if Playwright isn't available.
playwright = pytest.importorskip(
    "playwright",
    reason="Install with: pip install playwright && playwright install chromium",
)
from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
DOCS = REPO_ROOT / "docs"


def _free_port() -> int:
    """Bind to port 0 to get a free local port from the OS."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_server():
    """Serve `docs/` over HTTP for the duration of the module."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--directory", str(DOCS)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait until the server is actually accepting connections.
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.terminate()
        raise RuntimeError(f"http.server didn't start on port {port}")

    yield f"http://127.0.0.1:{port}"

    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def browser():
    """Headless Chromium for the duration of the module."""
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"Couldn't launch Chromium: {e}")
        yield b
        b.close()


def _open(browser, url: str, wait_ms: int = 2500):
    """Open a page, wait for network idle plus a settle period,
    and return (page, list-of-pageerror-strings)."""
    page = browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url, wait_until="networkidle", timeout=30_000)
    page.wait_for_timeout(wait_ms)
    return page, errors


@pytest.mark.browser
def test_test_modular_shows_22_of_22(http_server, browser):
    """`test-modular.html` should report `21/21 algorithms working`.

    Regression target: PR #73's `const Optimizer = ...` at module scope
    re-declared the same identifier across multiple per-family
    `<script>` tags, breaking module loading in the browser. The
    factory then couldn't find any algorithm and the page reported
    `0/21 algorithms working` (per the user's screenshot 2026-05-27).
    """
    page, errors = _open(browser, f"{http_server}/test-modular.html")
    body = page.locator("body").inner_text()
    page.close()

    assert not errors, "Page-level JS errors:\n  " + "\n  ".join(errors)
    assert "21/21 algorithms working" in body, (
        f"test-modular.html did not report 21/21 — body excerpt: {body[:400]}"
    )


@pytest.mark.browser
def test_contest_page_loads_optimizer_factory(http_server, browser):
    """`contest.html` should expose a working `OptimizerFactory.create()`.

    Same regression class as `test-modular`: per-family modules
    failing to load makes the factory's algorithm registry empty.
    """
    page, errors = _open(browser, f"{http_server}/contest.html")

    probe = page.evaluate(
        """
        () => {
            if (typeof window.OptimizerFactory === 'undefined') return { error: 'OptimizerFactory missing' };
            try {
                const f = (x) => x[0] * x[0] + x[1] * x[1];
                const opt = window.OptimizerFactory.create('NelderMead', f, 50, 2);
                return { name: opt.constructor.name };
            } catch (e) { return { error: String(e.message || e) }; }
        }
        """
    )
    page.close()

    assert not errors, "Page-level JS errors:\n  " + "\n  ".join(errors)
    assert probe.get("name") == "NelderMead", f"OptimizerFactory probe failed: {probe}"


@pytest.mark.browser
def test_visualizer_page_loads_and_clears_overlay(http_server, browser):
    """`algorithms/uobyqa.html` should load Three.js + the algorithm
    modules + the visualizer, clear the loading overlay, and expose
    `PRIMA_UOBYQA` and `AlgorithmVisualizer` as globals.

    Catches the per-family module load break (PR #73 regression) and
    the earlier `toggleMore()` brace-balance break (PR #65 root cause)
    on every visualizer-bearing page at once.
    """
    page, errors = _open(browser, f"{http_server}/algorithms/uobyqa.html", wait_ms=4000)
    state = page.evaluate(
        """
        () => ({
            primaLoaded: typeof PRIMA_UOBYQA !== 'undefined',
            visualizerLoaded: typeof AlgorithmVisualizer !== 'undefined',
            loadingDisplay: (document.getElementById('loadingMessage') || {}).style?.display ?? 'NO_ELEM',
        })
        """
    )
    page.close()

    assert not errors, "Page-level JS errors:\n  " + "\n  ".join(errors)
    assert state["primaLoaded"], "PRIMA_UOBYQA class not on window"
    assert state["visualizerLoaded"], "AlgorithmVisualizer not on window"
    assert state["loadingDisplay"] == "none", (
        f"3D visualizer never finished loading — overlay still shown "
        f"(display={state['loadingDisplay']!r})"
    )
