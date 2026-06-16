"""
Adapter that turns every `example_applications/<name>/problem.py` port
into a `Demo(name, n_dim, suggested_n_trials, objective)` for the
recommender paper's § 5 pipeline.

This supersedes the small hand-curated list in `demos.py`. Each
`problem.py` exposes:
  - `N_DIM: int`
  - `objective(u: list[float]) -> float` (sometimes with an optional
    `seed_offset` kwarg — we always call it positionally with `u`).
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

import example_applications

DEFAULT_N_TRIALS = 200

# Demos we deliberately exclude. chess_piece_values plays several depth-2 chess
# games per evaluation; at 200 trials × ~21 algos × 3 seeds it runs for ~14 hours.
# Excluded until we add a budget knob to the run-demos loop.
SKIP: frozenset[str] = frozenset({"chess_piece_values"})


@dataclass
class Demo:
    name: str
    n_dim: int
    suggested_n_trials: int
    objective: Callable[[list[float]], float]


def _collect_example_demos() -> list[Demo]:
    out: list[Demo] = []
    for _, name, ispkg in pkgutil.iter_modules(example_applications.__path__):
        if not ispkg or name in SKIP:
            continue
        try:
            problem = importlib.import_module(f"example_applications.{name}.problem")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {name}: import failed ({e})")
            continue
        if not hasattr(problem, "N_DIM") or not hasattr(problem, "objective"):
            continue
        out.append(
            Demo(
                name=name,
                n_dim=int(problem.N_DIM),
                suggested_n_trials=DEFAULT_N_TRIALS,
                objective=problem.objective,
            )
        )
    return out


DEMOS: list[Demo] = _collect_example_demos()
