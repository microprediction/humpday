"""The console command pyproject.toml declares exists and runs.

`humpday = "humpday.__main__:main"` shipped in the metadata of every release while there was no
such module, so an installed wheel put a command on the PATH that raised ModuleNotFoundError on
every invocation. Importing the package in the source tree cannot catch that; resolving the
declared target can.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _declared_scripts() -> dict:
    """The [project.scripts] table, read without assuming a TOML parser on 3.9/3.10."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - only on Python < 3.11
        pytest.skip("no tomllib on this interpreter")
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle).get("project", {}).get("scripts", {})


def test_every_declared_console_script_resolves():
    import importlib

    scripts = _declared_scripts()
    assert scripts, (
        "pyproject declares no console scripts; this test is watching nothing"
    )
    for command, target in scripts.items():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        entry = getattr(module, attribute, None)
        assert callable(entry), f"{command} points at {target}, which is not callable"


def test_the_command_runs_from_outside_the_checkout(tmp_path):
    """`python -m humpday` from another directory, which is where a user runs it."""
    result = subprocess.run(
        [sys.executable, "-m", "humpday", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_no_arguments_prints_help_rather_than_failing(capsys):
    from humpday.__main__ import main

    assert main([]) == 0
    assert "suggest" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["suggest", "--dim", "8", "--trials", "200", "-n", "3"],
        ["suggest", "--dim", "8", "--trials", "200", "--smooth"],
        ["suggest", "--dim", "8", "--trials", "200", "--rugged"],
        ["recommend", "--dim", "8", "--trials", "200"],
        ["recommend", "--dim", "50", "--trials", "100", "--cost-weight", "auto"],
        ["ratings", "--dim", "8"],
        ["optimizers"],
        ["optimizers", "--dim", "100", "--trials", "50"],
    ],
)
def test_each_subcommand_prints_something(argv, capsys):
    from humpday.__main__ import main

    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize(
    "argv",
    [
        ["suggest", "--dim", "8", "--trials", "200", "-n", "3"],
        ["recommend", "--dim", "8", "--trials", "200"],
        ["ratings", "--dim", "8"],
        ["optimizers", "--dim", "8"],
    ],
)
def test_json_output_is_json(argv, capsys):
    from humpday.__main__ import main

    assert main(["--json", *argv]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict) and payload


def test_suggest_agrees_with_the_library():
    from humpday import suggest
    from humpday.__main__ import main

    expected = [name for _, _, name in suggest(n_dim=8, n_trials=200)][:5]

    from contextlib import redirect_stdout
    from io import StringIO

    out = StringIO()
    with redirect_stdout(out):
        main(["--json", "suggest", "--dim", "8", "--trials", "200", "-n", "5"])
    assert json.loads(out.getvalue())["order"] == expected


def test_a_dimension_nothing_was_recorded_at_still_answers(capsys):
    """suggest falls back to the hand-written ordering, and ratings says there is nothing."""
    from humpday.__main__ import main

    assert main(["suggest", "--dim", "7", "--trials", "100", "-n", "3"]) == 0
    assert capsys.readouterr().out.strip()
    assert main(["ratings", "--dim", "7"]) == 0
    assert "nothing recorded" in capsys.readouterr().out


def test_a_bad_cost_weight_is_refused():
    from humpday.__main__ import main

    with pytest.raises(SystemExit):
        main(["recommend", "--dim", "8", "--trials", "200", "--cost-weight", "cheap"])
