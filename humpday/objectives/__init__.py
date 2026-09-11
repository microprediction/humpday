"""Objective functions for benchmarking optimizers.

There are exactly two lists, and everything else in this package feeds one of them.

``SURFACES``
    Classic formulaic surfaces: sphere, Rastrigin, Ackley, Griewank and the rest, each mapped onto
    the unit cube and defined for **any** dimension. This is the list to race optimizers on when you
    want a landscape whose shape you know.

``physics_objectives()``
    The engineering and physics demos under ``example_applications`` -- a brachistochrone descent, a
    truss, a cantilever, a cart-pole policy, antenna spacing. Each has a fixed, natural dimension,
    so unlike ``SURFACES`` you take the dimension the problem comes with. Loaded on demand, because
    importing eighty-odd demo modules is not free.

``morphed_surfaces()``
    Draws from ``SURFACES`` with a fresh random shift, rotation, scale, noise level and modal
    frequency every time. Use this for tournaments: a fixed landscape can be learned, and an
    optimizer tuned on one is not evidence about anything.

Two things are deliberately excluded from ``SURFACES``, both of which look fine until they are not:

* Fixed-dimension surfaces. Several accept a vector of any length and then read only the first two
  or three coordinates, so at d=100 they silently pose a 2-d problem. They are listed in
  ``FIXED_DIMENSION`` and kept out.
* Slow ones. Anything above roughly 10ms per evaluation makes a tournament impractical, and a
  tournament is what these exist for.
"""

from __future__ import annotations

import importlib
import pathlib
from typing import Callable, List, Tuple

from humpday.objectives.chatgptobjectives import CHATGPT_OBJECTIVES
from humpday.objectives.classic import (
    CLASSIC_OBJECTIVES,
    DEAP_OBJECTIVES,
    LANDSCAPES_OBJECTIVES,
    MISC_OBJECTIVES,
    SWARM_OBJECTIVES,
)

# Read only their leading coordinates, whatever length of vector you hand them. Measured, not
# assumed: perturbing coordinate i of a d=100 point leaves the value unchanged for most i.
FIXED_DIMENSION = frozenset(
    {
        "damavandi_on_cube",
        "chat_10",
        "chat_11",
        "chat_12",
        "chat_15",
        "booth_on_cube",
        "powers_on_cube",
        "schwefel_on_cube",
        "rosenbrock_modified_on_cube",
        "paviani_on_cube",
        "shekel_on_cube",
    }
)

# PORTFOLIO_OBJECTIVES is excluded wholesale rather than by name: every member runs 20-60ms per
# evaluation, which makes a tournament impractical, and a Markowitz objective is a portfolio problem
# rather than a formulaic surface. Import it from humpday.objectives.portfolio when you want it.
# 12ms per evaluation, which is 1.2s per problem at a 100-trial budget, times every optimizer.
SLOW = frozenset({"deap_combo1_on_cube"})


def _name(f) -> str:
    return getattr(f, "__name__", repr(f))


def _curate(*groups) -> List[Callable]:
    """One list, deduplicated by name, excluding what cannot be raced."""
    seen, out = set(), []
    for group in groups:
        for f in group:
            n = _name(f)
            if n in seen or n in FIXED_DIMENSION or n in SLOW:
                continue
            seen.add(n)
            out.append(f)
    return out


SURFACES: List[Callable] = _curate(
    CLASSIC_OBJECTIVES,
    DEAP_OBJECTIVES,
    LANDSCAPES_OBJECTIVES,
    SWARM_OBJECTIVES,
    MISC_OBJECTIVES,
    CHATGPT_OBJECTIVES,
)

# Retired from this list but still importable from their own modules:
#   horse.HORSE_OBJECTIVES        one racing-derived objective, not a formulaic surface
#   bbob_inspired_suite           hard-coded to two dimensions; raises above d=2
#   enhanced_surfaces             a thin wrapper over `opfunu`, which is not a dependency and is
#                                 not installed, so every call silently took the fallback path
#   deapobjectives                raw DEAP functions returning fitness *tuples* rather than floats;
#                                 classic.DEAP_OBJECTIVES is the usable wrapper and is included
#   portfolio.PORTFOLIO_OBJECTIVES  20-60ms per evaluation, and portfolio problems rather than
#                                 formulaic surfaces

_PHYSICS_CACHE: List[Tuple[str, Callable, int]] | None = None
# trebuchet returns the same value everywhere; chess_piece_values takes ~5s per evaluation.
_PHYSICS_EXCLUDE = frozenset({"trebuchet", "chess_piece_values"})


def physics_objectives() -> List[Tuple[str, Callable, int]]:
    """Return ``(name, objective, n_dim)`` for each usable engineering demo.

    Each demo owns its dimension, so these are raced at the size the problem actually is.
    """
    global _PHYSICS_CACHE
    if _PHYSICS_CACHE is not None:
        return _PHYSICS_CACHE
    root = pathlib.Path(__file__).resolve().parents[2] / "example_applications"
    found: List[Tuple[str, Callable, int]] = []
    for demo in sorted(p for p in root.iterdir() if (p / "problem.py").exists()):
        if demo.name in _PHYSICS_EXCLUDE:
            continue
        try:
            mod = importlib.import_module(f"example_applications.{demo.name}.problem")
            objective, n_dim = mod.objective, int(mod.N_DIM)
        except Exception:  # pragma: no cover - a demo that will not import is simply skipped
            continue
        found.append((demo.name, objective, n_dim))
    _PHYSICS_CACHE = found
    return found


def morphed_surfaces(n_functions: int = 6, seed: int | None = None):
    """Randomly shifted, rotated, rescaled and noised instances. See ``stochastic_surfaces``."""
    from humpday.objectives.stochastic_surfaces import create_fair_benchmark_run

    produced = create_fair_benchmark_run(n_functions=n_functions, seed=seed)
    suite = produced[0] if isinstance(produced, tuple) else produced
    return list(suite.values()) if isinstance(suite, dict) else list(suite)


__all__ = [
    "SURFACES",
    "physics_objectives",
    "morphed_surfaces",
    "FIXED_DIMENSION",
    "SLOW",
]
