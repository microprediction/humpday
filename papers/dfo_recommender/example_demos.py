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
from humpday.transforms.cube_disguise import disguise

DEFAULT_N_TRIALS = 200

# Default number of disguised instances generated per problem. Each instance
# applies a different seeded cube->cube diffeomorphism, relocating the optimum
# so an optimizer (or its meta-tuning) cannot score by memorising a location.
DISGUISE_SEEDS = (0, 1, 2, 3, 4)

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


def _collect_scaled_demos() -> list[Demo]:
    """Faithful high-dimensional variants. A `problem.py` opts in by exposing
    `SCALABLE_DIMS: list[int]` plus `make_objective(n_dim) -> objective`; we emit
    one `Demo` per requested dimension (skipping the module's native N_DIM, which
    `_collect_example_demos` already provides). These populate the n>=16 regime
    where synthetic-vs-real benchmark divergence is studied."""
    out: list[Demo] = []
    for _, name, ispkg in pkgutil.iter_modules(example_applications.__path__):
        if not ispkg or name in SKIP:
            continue
        try:
            problem = importlib.import_module(f"example_applications.{name}.problem")
        except Exception:  # noqa: BLE001
            continue
        dims = getattr(problem, "SCALABLE_DIMS", None)
        make = getattr(problem, "make_objective", None)
        if not dims or not callable(make):
            continue
        base = int(getattr(problem, "N_DIM", -1))
        for n in dims:
            if int(n) == base:
                continue
            out.append(
                Demo(
                    name=f"{name}_{int(n)}d",
                    n_dim=int(n),
                    suggested_n_trials=DEFAULT_N_TRIALS,
                    objective=make(int(n)),
                )
            )
    return out


DEMOS: list[Demo] = _collect_example_demos() + _collect_scaled_demos()


# --------------------------------------------------------------------------
# Disguised instances: wrap each objective in a seeded cube->cube diffeomorphism
# so the optimum is relocated to an unpredictable, instance-specific point. The
# landscape (difficulty, critical-point structure, optimal value) is unchanged,
# but no algorithm can win by remembering WHERE the optimum is — it must search.
# Use this when developing/meta-tuning optimizers against the suite.
# --------------------------------------------------------------------------


def disguise_demo(demo: Demo, seed: int, rotate: bool = False) -> Demo:
    """Return a disguised copy of `demo`: same n_dim / budget, objective wrapped
    in the seed's cube->cube diffeomorphism. Name is suffixed `#<seed>`."""
    return Demo(
        name=f"{demo.name}#{seed}",
        n_dim=demo.n_dim,
        suggested_n_trials=demo.suggested_n_trials,
        objective=disguise(demo.objective, demo.n_dim, seed, rotate=rotate),
    )


def disguised_demos(
    demos: list[Demo] | None = None,
    seeds=DISGUISE_SEEDS,
    rotate: bool = False,
) -> list[Demo]:
    """Expand `demos` (default: all DEMOS) into len(seeds) disguised instances
    each — a benchmark on which optimum locations cannot be memorised."""
    demos = DEMOS if demos is None else demos
    return [disguise_demo(d, s, rotate) for d in demos for s in seeds]


if __name__ == "__main__":
    import random

    print(f"{len(DEMOS)} base demos")
    print(
        f"{len(disguised_demos())} disguised instances ({len(DISGUISE_SEEDS)} seeds each)\n"
    )

    # Verify the disguise is a faithful relocation: for a wrapped objective g and
    # any cube point p, g(phi^{-1}(p)) must equal the raw objective f(p) — i.e. the
    # value at p simply moved to phi^{-1}(p). Also show the optimum moves per seed.
    base = {d.name: d for d in DEMOS}
    probe = base.get("multi_exponential_fit") or DEMOS[0]
    f = probe.objective
    print(f"verifying relocation on '{probe.name}' (n_dim={probe.n_dim}):")
    for seed in DISGUISE_SEEDS:
        g = disguise_demo(probe, seed)
        dis = g.objective.disguise
        random.seed(seed)
        worst = 0.0
        for _ in range(200):
            p = [random.random() for _ in range(probe.n_dim)]
            worst = max(worst, abs(g.objective(dis.inverse(p)) - f(p)))
        # where a fixed reference point's value now lives in disguised coords
        ref = [0.5] * probe.n_dim
        moved = dis.inverse(ref)
        print(
            f"  seed {seed}:  max |g(phi^-1 p) - f(p)| = {worst:.2e}   "
            f"u=½ relocated to {[round(x, 3) for x in moved[:4]]}"
        )
