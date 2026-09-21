"""The documented paths work when copied out of the document.

The package used to be a set of wrappers around third-party optimizers, and the documentation
still led with modules that had been deleted: `humpday.optimizers.pysotcube`,
`humpday.optimizers.nloptcube`, `humpday.comparisons.eloratings`, a `scipy_minimize` that never
existed. Every one of those was a copy-paste path that failed on its first line, and nothing in
the suite looked at documentation, so none of it was noticed.

The rule this file enforces: a ```python fence in a maintained document runs as written, from a
clean interpreter and a directory that is not the repository. A snippet that is a signature
sketch, or that calls a function the reader is expected to supply, is marked ```text instead --
it is then documentation rather than a promise.
"""

from __future__ import annotations

import os
import re
import runpy
import sys
import urllib.parse
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
DOCS = REPO_ROOT / "docs"

MAINTAINED_DOCS = [
    "README.md",
    "README_CLAUDE.md",
    "humpday/optimizers/README.md",
    "docs/scipy-interface.md",
    "docs/adaptive-optimization.md",
]

# Examples are run as scripts, not imported, so a module-level import of something deleted is
# caught the same way a user would catch it. markowitz.py is excluded from execution only because
# it optimizes for a couple of seconds; its imports are still checked.
EXAMPLES = sorted(
    p for p in (REPO_ROOT / "examples").glob("*.py") if p.name != "__init__.py"
)


def _fences(path: Path, language: str = "python") -> list:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    return re.findall(rf"```{language}\n(.*?)```", text, re.S)


def _run_isolated(source: str, name: str, tmp_path: Path) -> None:
    """Run a snippet the way a reader would: fresh namespace, and not in the repository.

    The working directory matters. A snippet that writes `my_ratings.json` should not write it
    into the checkout, and one that only works because the repository is the current directory
    is not a snippet that works.
    """
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        exec(compile(source, name, "exec"), {"__name__": "__doc__"})
    finally:
        os.chdir(previous)


@pytest.mark.parametrize("doc", MAINTAINED_DOCS)
def test_the_document_has_snippets_to_check(doc):
    assert _fences(Path(doc)), f"{doc} has no python fences; is it still maintained?"


@pytest.mark.parametrize(
    "doc,index",
    [(doc, i) for doc in MAINTAINED_DOCS for i in range(len(_fences(Path(doc))))],
    ids=lambda v: str(v),
)
def test_every_python_fence_runs_as_written(doc, index, tmp_path):
    _run_isolated(_fences(Path(doc))[index], f"{doc}:{index}", tmp_path)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_example_imports_what_it_says_it_imports(example):
    """The import block alone, which is where markowitz.py named a module that was deleted."""
    source = example.read_text(encoding="utf-8")
    imports = re.findall(
        r"^(?:from\s+\S+\s+import\s+\([^)]*\)|from\s+\S+\s+import\s+[^\n]+|import\s+[^\n]+)$",
        source,
        re.M,
    )
    assert imports, f"{example.name} imports nothing at all"
    exec(compile("\n".join(imports), str(example), "exec"), {})


@pytest.mark.parametrize(
    "example",
    [p for p in EXAMPLES if p.name != "markowitz.py"],
    ids=lambda p: p.name,
)
def test_every_example_runs(example, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(example)])
    runpy.run_path(str(example), run_name="__main__")


def _relative_targets(page: Path):
    text = page.read_text(encoding="utf-8", errors="ignore")
    for match in re.finditer(r'(?:href|src)="([^"]+)"', text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "#", "mailto:", "data:", "//")):
            continue
        # algorithm-template.html is a template: {{PAPER_URL}} and friends are filled in by
        # whoever copies it, and are not links yet.
        if "{{" in target:
            continue
        path = urllib.parse.urlparse(target).path
        if path:
            yield target, (page.parent / path)


@pytest.mark.parametrize(
    "page",
    sorted(DOCS.rglob("*.html")),
    ids=lambda p: str(p.relative_to(DOCS)),
)
def test_site_links_point_at_files_that_exist(page):
    missing = [
        target for target, resolved in _relative_targets(page) if not resolved.exists()
    ]
    assert not missing, f"{page.relative_to(DOCS)} links to {missing}"
