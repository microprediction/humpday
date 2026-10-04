"""The reference gates fail when the comparisons they exist for did not run (#410).

`test_reference_alignment` asserted only that its table was nonempty. Six of its references are
inline baselines with no dependencies, so with every third-party library missing -- or present
and broken -- it still went green on those six, and a passing "Ports against their references"
said nothing about any third-party implementation. The CI job installed `.[dev,reference]`,
which did not include PDFO, so the PRIMA_NEWUOA and PRIMA_UOBYQA comparisons were skipped in
the one environment configured to run them; installed, PDFO then failed only when called.

These tests call the real gate with stubbed adapters. Nothing here needs a reference library.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from tests import test_reference_alignment as G

REPO_ROOT = Path(__file__).parent.parent


def _env(strict):
    return patch.dict("os.environ", {G.STRICT_ENV: "1" if strict else ""})


def _ok(*a, **k):
    return {"best_value": 1e-6, "evals": 50}


def _gate(tmp_path, references, statuses, strict=False):
    """The real gate on the sphere, with stub adapters and stubbed dependency status."""
    stack = ExitStack()
    for name, value in {
        "REPO_ROOT": tmp_path,
        "REFERENCES": references,
        "PROBLEMS": {"sphere": G.PROBLEMS["sphere"]},
        "_run_humpday": _ok,
        "module_status": lambda m: statuses.get(m, (G.OK, "")),
        # A stubbed experiment is not the calibrated one, and that is not what these test.
        "experiment_digest": lambda spec=None: G.CALIBRATED_FOR,
    }.items():
        stack.enter_context(patch.object(G, name, value))
    stack.enter_context(_env(strict))
    with stack:
        G.test_reference_alignment()


def _coverage(tmp_path):
    text = (tmp_path / "benchmarks" / "reference_alignment.json").read_text()
    return json.loads(text)["coverage"]


INLINE = {"RandomSearch": ("inline baseline", _ok, [])}
SCIPY = {"NelderMead": ("stub scipy", _ok, ["scipy"])}
PDFO = {"PRIMA_NEWUOA": ("stub pdfo", _ok, ["pdfo"])}


def test_only_the_inline_baselines_running_is_a_failure(tmp_path):
    """The issue's reproduction: every optional dependency unavailable."""
    absent = {"scipy": (G.ABSENT, "scipy is not installed")}
    with pytest.raises(AssertionError, match="no third-party comparison ran"):
        _gate(tmp_path, {**INLINE, **SCIPY}, absent)
    assert _coverage(tmp_path)["absent"] == {"NelderMead": "scipy is not installed"}


def test_a_broken_install_is_not_reported_as_a_missing_one(tmp_path):
    broken = {"scipy": (G.PROBE_FAILED, "scipy: RuntimeError: compiled for NumPy 1.x")}
    with pytest.raises(AssertionError, match="no third-party comparison ran"):
        _gate(tmp_path, {**INLINE, **SCIPY}, broken)
    coverage = _coverage(tmp_path)
    assert coverage["probe_failed"] == {
        "NelderMead": "scipy: RuntimeError: compiled for NumPy 1.x"
    }
    assert coverage["absent"] == {}


def test_locally_a_missing_library_is_a_skip_that_is_reported(tmp_path):
    _gate(tmp_path, {**SCIPY, **PDFO}, {"pdfo": (G.ABSENT, "pdfo is not installed")})
    coverage = _coverage(tmp_path)
    assert coverage["compared"] == ["NelderMead"]
    assert coverage["absent"] == {"PRIMA_NEWUOA": "pdfo is not installed"}
    assert coverage["strict"] is False


def test_in_ci_a_missing_library_fails(tmp_path):
    with pytest.raises(AssertionError) as caught:
        _gate(
            tmp_path,
            {**SCIPY, **PDFO},
            {"pdfo": (G.ABSENT, "pdfo is not installed")},
            strict=True,
        )
    assert "comparison expected in CI did not run" in str(caught.value)
    assert "PRIMA_NEWUOA: absent: pdfo is not installed" in str(caught.value)


def test_in_ci_a_failed_probe_fails_and_says_so(tmp_path):
    with pytest.raises(AssertionError) as caught:
        _gate(
            tmp_path,
            {**SCIPY, **PDFO},
            {"pdfo": (G.PROBE_FAILED, "pdfo: ImportError: numpy 1.x")},
            strict=True,
        )
    assert "PRIMA_NEWUOA: probe failed: pdfo: ImportError: numpy 1.x" in str(
        caught.value
    )


def test_in_ci_full_coverage_passes(tmp_path):
    _gate(tmp_path, {**INLINE, **SCIPY, **PDFO}, {}, strict=True)
    assert _coverage(tmp_path)["compared"] == [
        "RandomSearch",
        "NelderMead",
        "PRIMA_NEWUOA",
    ]


class TestModuleStatus:
    def test_a_module_that_is_not_there_is_absent(self):
        G._STATUS.pop("humpday_no_such_module", None)
        status, _ = G.module_status("humpday_no_such_module")
        assert status == G.ABSENT

    def test_a_module_whose_own_dependency_is_missing_is_broken_not_absent(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "humpday_half_installed.py").write_text(
            "import humpday_its_missing_dependency\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        G._STATUS.pop("humpday_half_installed", None)
        status, detail = G.module_status("humpday_half_installed")
        assert status == G.PROBE_FAILED
        assert "humpday_its_missing_dependency" in detail

    def test_a_module_that_raises_on_import_is_broken(self, tmp_path, monkeypatch):
        (tmp_path / "humpday_raises_on_import.py").write_text(
            "raise RuntimeError('compiled against NumPy 1.x')\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        G._STATUS.pop("humpday_raises_on_import", None)
        status, detail = G.module_status("humpday_raises_on_import")
        assert status == G.PROBE_FAILED
        assert "NumPy 1.x" in detail

    def test_a_module_that_imports_but_fails_its_probe_is_broken(self, monkeypatch):
        def probe():
            raise ImportError("gethuge was built for NumPy 1.x")

        monkeypatch.setitem(G._PROBES, "json", probe)
        G._STATUS.pop("json", None)
        try:
            status, detail = G.module_status("json")
        finally:
            G._STATUS.pop("json", None)
        assert status == G.PROBE_FAILED
        assert "gethuge" in detail

    def test_the_status_is_cached_rather_than_reprobed(self):
        assert G.module_status("json") is G.module_status("json")


class TestTheCIJobIsConfiguredToNotice:
    def test_the_reference_extra_brings_pdfo(self):
        tomllib = pytest.importorskip("tomllib")
        extra = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]["reference"]
        assert any(req.lower().startswith("pdfo") for req in extra)
        # PDFO 2.2.0, the latest, is compiled against NumPy 1.x.
        assert any(req.replace(" ", "").startswith("numpy<2") for req in extra)

    def test_the_workflow_runs_strict_with_node(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "reference.yml").read_text()
        assert f'{G.STRICT_ENV}: "1"' in workflow
        assert "actions/setup-node" in workflow
        assert '".[dev,reference]"' in workflow


class TestTheJavaScriptGateFailsClosedInCI:
    """#410 asked the same of #399's gate: missing Node or references fail in its CI job."""

    @pytest.mark.parametrize("strict, outcome", [(False, "Skipped"), (True, "Failed")])
    def test_missing_node(self, strict, outcome):
        from tests import test_js_reference_alignment as J

        with patch.object(J, "NODE", None), _env(strict):
            with pytest.raises(BaseException) as caught:
                J.test_the_javascript_port_tracks_its_reference("NelderMead", "sphere")
        assert type(caught.value).__name__ == outcome

    @pytest.mark.parametrize("strict, outcome", [(False, "Skipped"), (True, "Failed")])
    def test_missing_reference(self, strict, outcome):
        from tests import test_js_reference_alignment as J

        absent = lambda m: (G.ABSENT, f"{m} is not installed")  # noqa: E731
        with patch.object(J, "NODE", "node"), patch.object(G, "module_status", absent):
            with _env(strict), pytest.raises(BaseException) as caught:
                J.test_the_javascript_port_tracks_its_reference("NelderMead", "sphere")
        assert type(caught.value).__name__ == outcome
        assert "scipy is not installed" in str(caught.value)
