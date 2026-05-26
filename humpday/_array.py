"""
Tiny array shim for humpday.

Why this exists
---------------
humpday's optimizer code uses a small set of array primitives: construction,
elementwise math, reductions, a few linear-algebra calls. Today every
implementation calls into `numpy` directly. That makes numpy a hard runtime
dependency, which means humpday cannot ship as truly "pure Python" — the
install footprint, the Pyodide story, and the embedded-Python story all
inherit numpy's constraints.

This module is the abstraction layer. Optimizer code calls into
`humpday._array` (or its eventual `linalg` submodule) instead of importing
`numpy as np`. The shim selects a backend at import time:

  - If `numpy` is installed, the numpy backend transparently re-exports it.
    No performance penalty — the indirection is one module reference.
  - If `numpy` is not installed, the pure backend provides equivalent
    operations on plain Python lists wrapped in a small `_Vec` class.

This lets a single codebase ship two install modes:

  pip install humpday           -> pure-Python, ~87 KB wheel, no deps
  pip install humpday[fast]     -> numpy-backed, full-speed for n > ~20

Backend selection
-----------------
By default the shim picks numpy when available. Setting the environment
variable `HUMPDAY_FORCE_PURE_ARRAY=1` forces the pure backend even when
numpy is installed; this lets the test suite exercise both code paths in
the same CI run.

The active backend is exposed as `humpday._array.BACKEND` ('numpy' or 'pure').
"""

from __future__ import annotations

import os

_FORCE_PURE = os.environ.get("HUMPDAY_FORCE_PURE_ARRAY", "") == "1"

if _FORCE_PURE:
    from . import _array_pure as _impl
    from . import _array_pure_linalg as linalg
    from ._array_pure import *  # noqa: F401,F403

    BACKEND = "pure"
else:
    try:
        import numpy as _np  # noqa: F401

        from . import _array_numpy as _impl
        from . import _array_numpy_linalg as linalg
        from ._array_numpy import *  # noqa: F401,F403

        BACKEND = "numpy"
    except ImportError:
        from . import _array_pure as _impl
        from . import _array_pure_linalg as linalg
        from ._array_pure import *  # noqa: F401,F403

        BACKEND = "pure"


# Re-export the canonical name list from whichever backend is live so callers
# can introspect what's available without caring which backend it is.
# `linalg` is the submodule namespace; `numpy.linalg`-shaped API lives there.
__all__ = ["BACKEND", "linalg", *_impl.__all__]
