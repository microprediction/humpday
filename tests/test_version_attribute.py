"""
Regression tests for the `humpday.__version__` attribute.

The publish workflow (.github/workflows/publish.yml, `test-install` job)
runs:

    python -c "import humpday; print(f'humpday {humpday.__version__} \
installed successfully')"

If `__version__` is missing or unparseable, the publish workflow fails
*after* the wheel has already been built and uploaded as an artifact —
a noisy, hard-to-diagnose failure mode. These tests catch it locally.
"""

from __future__ import annotations

import re

import humpday


def test_version_attribute_exists():
    """humpday.__version__ must be a non-empty string."""
    assert hasattr(humpday, "__version__"), "humpday.__version__ is not defined"
    assert isinstance(humpday.__version__, str), (
        f"__version__ must be str, got {type(humpday.__version__).__name__}"
    )
    assert humpday.__version__, "__version__ must not be empty"


def test_version_looks_like_pep_440():
    """When running against an installed wheel, the version should look like
    a PEP 440 release version (e.g. '0.9.0'). The source-checkout fallback
    is '0.0.0+source', which is also a valid PEP 440 local version segment."""
    pep440_loose = re.compile(
        r"^\d+(\.\d+)*"  # release segment
        r"((a|b|rc)\d+)?"  # pre-release
        r"(\.post\d+)?"  # post-release
        r"(\.dev\d+)?"  # dev release
        r"(\+[A-Za-z0-9.]+)?$"  # local version (e.g. '+source')
    )
    assert pep440_loose.match(humpday.__version__), (
        f"__version__={humpday.__version__!r} does not look like PEP 440"
    )


def test_version_in_dunder_all():
    """__version__ is part of the documented public surface."""
    assert "__version__" in humpday.__all__
