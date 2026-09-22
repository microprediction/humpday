"""The package works on the install its own pyproject advertises: no dependencies at all.

numpy is the `fast` extra. `humpday._array` dispatches to it when present and to a pure-Python
backend when not, so the optimizers never need it -- but several modules imported it at module
scope, which took them, and anything importing them, down on a real install (#377). The failure
was invisible here because the development environment has numpy.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


class _NoNumpy:
    """Import hook that makes numpy absent, the way it is absent on a plain install."""

    def find_spec(self, name, path=None, target=None):
        if name == "numpy" or name.startswith("numpy."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


@pytest.fixture
def without_numpy(monkeypatch):
    for mod in [m for m in list(sys.modules) if m == "numpy" or m.startswith("numpy.")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    for mod in [m for m in list(sys.modules) if m.startswith("humpday")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_NoNumpy(), *sys.meta_path])
    monkeypatch.setenv("HUMPDAY_FORCE_PURE_ARRAY", "1")
    yield
    for mod in [m for m in list(sys.modules) if m.startswith("humpday")]:
        sys.modules.pop(mod, None)


IMPORTABLE = [
    "humpday",
    "humpday.ratings",
    "humpday.eligibility",
    "humpday.optimizers.alloptimizers",
    "humpday.transforms.cubetosimplex",
    "humpday.transforms.thurstone_transform",
    "humpday.objectives.classic",
    "humpday.objectives.horse",
    "humpday.objectives.portfolio",
    "humpday.objectives.chatgptobjectives",
    "humpday.objectives.allobjectives",
]


@pytest.mark.parametrize("module", IMPORTABLE)
def test_the_module_imports_without_numpy(module, without_numpy):
    assert importlib.import_module(module) is not None


def test_the_optimizers_run_without_numpy(without_numpy):
    from humpday import _array, pure_optimize

    assert _array.BACKEND == "pure"
    value, point = pure_optimize(lambda x: sum(v * v for v in x), "NelderMead", 50, 3)
    assert value >= 0.0 and len(point) == 3


def test_the_classic_objectives_evaluate_without_numpy(without_numpy):
    from humpday.objectives import classic

    names = [n for n in dir(classic) if n.endswith("_on_cube")]
    assert len(names) > 20
    for name in names:
        value = float(getattr(classic, name)([0.3, 0.6, 0.45, 0.7]))
        assert value == value, f"{name} returned nan"


def test_objectives_that_still_need_numpy_say_so(without_numpy):
    """portfolio and chatgptobjectives keep numpy; what they must not do is fail at import.

    A multivariate normal and a covariance estimate are not a few lines of shim, and the
    chatgpt objectives are eighteen published landscapes written in array operations. They are
    importable, and they explain themselves when called.
    """
    from humpday.objectives import chatgptobjectives, portfolio

    with pytest.raises(ImportError, match="fast"):
        portfolio.make_sigma_matrix()
    with pytest.raises(ImportError, match="fast"):
        chatgptobjectives.chat_0([0.3, 0.6])
