"""
Build (or extend) the recommendation grid that humpday.eligibility consults
at minimize() time when picking a default algorithm.

The grid maps each ``(n_dim, n_trials)`` cell to, for every algorithm that
passed structural eligibility, the median best-value over a fixed seed
bouquet and a small set of standard benchmark surfaces. At query time
``eligibility.recommend`` snaps the user's (n_dim, n_trials) onto the
nearest cell and picks the eligible algorithm with the smallest median_best.

Two properties drive the design:

  * **Parallelism.** Each (cell, algorithm, objective, seed) tuple is an
    independent unit of work. We dispatch them across a process pool so
    long sweeps scale linearly with cores on a beefy machine.

  * **Incremental.** Re-running the script reads any existing grid file,
    skips work that already has a recorded result, and only runs the
    missing tuples. So you can run a quick sweep, push it, and a teammate
    can later run a bigger sweep that extends rather than overwrites.

Usage:
    python benchmarks/build_recommendation_grid.py
    python benchmarks/build_recommendation_grid.py --quick
    python benchmarks/build_recommendation_grid.py --workers 16 --seeds 5
    python benchmarks/build_recommendation_grid.py --dims 2,5 --trials 50,200
    python benchmarks/build_recommendation_grid.py --include-skipped

Storage shape (so external tools can read it without importing humpday):

    {
      "meta": {
        "objectives": ["sphere", "rosenbrock", ...],
        "dims": [2, 5, 10, ...],
        "trials": [50, 200, 1000],
        "first_built": "2026-06-01T08:43:00Z",
        "last_updated": "2026-06-01T11:12:00Z",
        "per_run_budget_seconds": 30.0
      },
      "cells": {
        "5/200": {
          "DifferentialEvolution": {
            "runs": {
              "sphere/0":     {"best": 1.2e-9,  "wall": 0.011},
              "sphere/1":     {"best": 2.7e-9,  "wall": 0.012},
              "rosenbrock/0": {"best": 4.5e-4,  "wall": 0.013},
              ...
            },
            "median_best":   1.4e-9,
            "mean_wall":     0.012,
            "n_runs":        15,
            "n_failures":    0,
            "skipped_too_slow": false
          }
        }
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Callable

# Pin BLAS to 1 thread per process BEFORE importing numpy. Without this,
# every worker spawns its own thread-pool inside numpy/MKL/OpenBLAS, and
# `--workers N` × ~K BLAS threads each saturates the CPU and slows every
# task. The combinatorics matter most for BayesianOpt's O(n_obs^3) GP fits.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from humpday import eligibility as E  # noqa: E402
from humpday.objectives.classic import (  # noqa: E402
    griewank_on_cube,
    rastrigin_on_cube,
    rosenbrock_on_cube,
    salomon_on_cube,
)
from humpday.optimizers.alloptimizers import (  # noqa: E402
    PURE_OPTIMIZERS,
    pure_optimize,
)


# Module-level so worker processes can pickle it. (A lambda inside a list
# literal would fail to ship to a subprocess.)
def _sphere_on_cube(u):
    return float(sum((x - 0.5) ** 2 for x in u))


# Objectives are unit-cube benchmarks. They cover separable convex (sphere),
# banana (rosenbrock), rugged multimodal (rastrigin, salomon), and nearly-
# separable-with-global-structure (griewank).
TEST_FUNCTIONS: list[tuple[str, Callable]] = [
    ("sphere", _sphere_on_cube),
    ("rosenbrock", rosenbrock_on_cube),
    ("rastrigin", rastrigin_on_cube),
    ("griewank", griewank_on_cube),
    ("salomon", salomon_on_cube),
]


DEFAULT_DIMS = [2, 5, 10, 20, 50, 100]
DEFAULT_TRIALS = [50, 200, 1000]
DEFAULT_SEEDS = 3
QUICK_DIMS = [2, 10, 50]
QUICK_TRIALS = [50, 200]
QUICK_SEEDS = 2

# Hard wall-clock budget per single optimize() call. An algorithm whose
# first probe blows the budget gets marked "skipped_too_slow" and is not
# tried again on later seeds/objectives in the same cell, unless the user
# passes --include-skipped (handy when re-running on a faster machine).
PER_RUN_BUDGET_SECONDS = 30.0


# -----------------------------------------------------------------------------
# Worker (process pool)
# -----------------------------------------------------------------------------

def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _run_one_in_worker(args) -> tuple:
    """Run a single (cell, algorithm, objective, seed) tuple.

    The function lives at module scope and takes a single tuple so it
    pickles cleanly across processes. Returns a tuple of all the keys the
    parent needs to merge the result back into the grid.
    """
    (cell_key, algorithm, objective_name, seed, n_trials, n_dim) = args
    # Look up the objective callable by name — worker process inherits the
    # module's TEST_FUNCTIONS dict, so this stays inside the worker.
    obj = dict(TEST_FUNCTIONS)[objective_name]
    _set_global_seed(seed)
    t0 = time.perf_counter()
    try:
        best_value, _ = pure_optimize(obj, algorithm, n_trials, n_dim)
        wall = time.perf_counter() - t0
        return (cell_key, algorithm, objective_name, seed, float(best_value), wall, None)
    except Exception as exc:  # noqa: BLE001
        wall = time.perf_counter() - t0
        return (cell_key, algorithm, objective_name, seed, float("inf"), wall, str(exc))


# -----------------------------------------------------------------------------
# Grid management — load, store, aggregate
# -----------------------------------------------------------------------------

def _empty_grid(dims: list[int], trials: list[int], n_seeds: int) -> dict:
    return {
        "meta": {
            "objectives": [name for name, _ in TEST_FUNCTIONS],
            "dims": dims,
            "trials": trials,
            "n_seeds": n_seeds,
            "first_built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "per_run_budget_seconds": PER_RUN_BUDGET_SECONDS,
        },
        "cells": {},
    }


def _load_or_init(path: Path, dims, trials, n_seeds) -> dict:
    if not path.exists():
        return _empty_grid(dims, trials, n_seeds)
    try:
        with path.open() as fh:
            grid = json.load(fh)
    except (OSError, json.JSONDecodeError):
        print(
            f"  (existing {path} unreadable — starting fresh)", file=sys.stderr
        )
        return _empty_grid(dims, trials, n_seeds)
    # Extend meta with any newly-requested dims/trials.
    grid.setdefault("meta", {})
    grid["meta"].setdefault("objectives", [name for name, _ in TEST_FUNCTIONS])
    grid["meta"]["dims"] = sorted(set(grid["meta"].get("dims", [])) | set(dims))
    grid["meta"]["trials"] = sorted(set(grid["meta"].get("trials", [])) | set(trials))
    grid["meta"]["n_seeds"] = max(grid["meta"].get("n_seeds", 0), n_seeds)
    grid["meta"]["per_run_budget_seconds"] = PER_RUN_BUDGET_SECONDS
    grid["meta"].setdefault("first_built", grid["meta"].get(
        "first_built",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    ))
    grid.setdefault("cells", {})
    return grid


def _aggregate(entry: dict) -> None:
    """Recompute median_best, mean_wall, n_runs, n_failures from entry['runs']
    in-place so the on-disk format always carries the summary that
    eligibility.recommend reads."""
    runs = entry.get("runs", {})
    values = [r["best"] for r in runs.values() if r["best"] != float("inf")]
    walls = [r["wall"] for r in runs.values()]
    entry["median_best"] = median(values) if values else float("inf")
    entry["mean_wall"] = sum(walls) / len(walls) if walls else 0.0
    entry["n_runs"] = len(runs)
    entry["n_failures"] = len(runs) - len(values)


def _save_atomic(path: Path, grid: dict) -> None:
    """Write via a sibling temp file + rename so a kill mid-write can't leave
    the grid file truncated or syntactically broken."""
    grid["meta"]["last_updated"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(grid, indent=2, sort_keys=True))
    tmp.replace(path)


def _existing_run_keys(entry: dict | None) -> set[str]:
    if not entry:
        return set()
    return set(entry.get("runs", {}).keys())


def _is_skipped_too_slow(entry: dict | None) -> bool:
    return bool(entry and entry.get("skipped_too_slow"))


# -----------------------------------------------------------------------------
# Task graph
# -----------------------------------------------------------------------------

def _build_tasks(
    grid: dict,
    dims: list[int],
    trials_list: list[int],
    n_seeds: int,
    include_skipped: bool,
) -> tuple[list[tuple], list[tuple[str, str, float]]]:
    """Walk every (cell, algorithm, objective, seed) and emit the ones that
    still need to be run. Returns (tasks_to_run, slow_skips_carried_over).
    """
    tasks: list[tuple] = []
    skipped: list[tuple[str, str, float]] = []

    for n_dim in dims:
        for n_trials in trials_list:
            cell_key = f"{n_dim}/{n_trials}"
            cell = grid["cells"].setdefault(cell_key, {})

            candidates = E.eligible(
                PURE_OPTIMIZERS.keys(),
                n_dim=n_dim,
                n_trials=n_trials,
                eval_time=None,
            )
            for algorithm in candidates:
                entry = cell.setdefault(algorithm, {"runs": {}})
                if _is_skipped_too_slow(entry) and not include_skipped:
                    skipped.append((cell_key, algorithm, entry["mean_wall"]))
                    continue
                done = _existing_run_keys(entry)
                for obj_name, _ in TEST_FUNCTIONS:
                    for seed in range(n_seeds):
                        run_key = f"{obj_name}/{seed}"
                        if run_key in done:
                            continue
                        tasks.append(
                            (cell_key, algorithm, obj_name, seed, n_trials, n_dim)
                        )
    return tasks, skipped


def _probe_first(
    tasks: list[tuple],
    grid: dict,
    save_every: int,
    output: Path,
) -> tuple[list[tuple], int]:
    """Single-process probe pass: run *one* tuple (seed=0, first objective)
    for each (cell, algorithm) pair, in serial, so we can detect
    over-budget algorithms and prune the rest of their tasks before paying
    for them in parallel. Returns (remaining_tasks_after_pruning, n_probed).

    Only probes algorithms that have zero recorded runs yet — already-known
    fast algorithms don't get re-probed.
    """
    by_algo: dict[tuple[str, str], list[int]] = {}
    for i, t in enumerate(tasks):
        cell_key, algorithm, *_ = t
        by_algo.setdefault((cell_key, algorithm), []).append(i)

    drop_idx: set[int] = set()
    n_probed = 0
    for (cell_key, algorithm), idxs in by_algo.items():
        entry = grid["cells"][cell_key].setdefault(algorithm, {"runs": {}})
        if entry.get("runs"):
            continue  # already have data; trust it
        # First task in the group is sphere/seed=0 (TEST_FUNCTIONS order).
        probe = tasks[idxs[0]]
        cell_key, algo, obj_name, seed, n_trials, n_dim = probe
        n_probed += 1
        result = _run_one_in_worker(probe)
        _ck, _algo, _obj, _seed, best, wall, err = result
        entry.setdefault("runs", {})[f"{_obj}/{_seed}"] = {"best": best, "wall": wall}
        if err:
            entry.setdefault("errors", []).append(err)

        if wall > PER_RUN_BUDGET_SECONDS:
            print(
                f"  ! {cell_key} {algo}: probe took {wall:.1f}s "
                f"(>{PER_RUN_BUDGET_SECONDS:.0f}s budget) — skipping cell",
                file=sys.stderr,
            )
            entry["skipped_too_slow"] = True
            _aggregate(entry)
            # Drop every other task for this (cell, algorithm) pair.
            drop_idx.update(idxs[1:])
        else:
            # We already ran the first task; drop it from the remaining list.
            drop_idx.add(idxs[0])
            _aggregate(entry)

        if n_probed % save_every == 0:
            _save_atomic(output, grid)

    remaining = [t for i, t in enumerate(tasks) if i not in drop_idx]
    return remaining, n_probed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dims",
        type=str,
        default=None,
        help="Comma-separated n_dim values (default: 2,5,10,20,50,100)",
    )
    parser.add_argument(
        "--trials",
        type=str,
        default=None,
        help="Comma-separated n_trials values (default: 50,200,1000)",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Process-pool size for parallel runs (default: cpu_count-1)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller grid for fast iteration during development.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Retry algorithms previously marked too slow (e.g. on a faster machine).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save the grid every N completed tasks (atomic write).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "recommendation_grid.json",
    )
    args = parser.parse_args()

    if args.quick:
        dims = QUICK_DIMS
        trials_list = QUICK_TRIALS
        n_seeds = QUICK_SEEDS
    else:
        dims = (
            [int(d) for d in args.dims.split(",")] if args.dims else DEFAULT_DIMS
        )
        trials_list = (
            [int(t) for t in args.trials.split(",")]
            if args.trials
            else DEFAULT_TRIALS
        )
        n_seeds = args.seeds

    print(
        f"Building recommendation grid: dims={dims}, trials={trials_list}, "
        f"seeds={n_seeds}, objectives={len(TEST_FUNCTIONS)}, "
        f"workers={args.workers}, output={args.output}"
    )

    grid = _load_or_init(args.output, dims, trials_list, n_seeds)
    tasks, skipped = _build_tasks(
        grid, dims, trials_list, n_seeds, args.include_skipped
    )

    if skipped:
        print(
            f"  Carrying over {len(skipped)} (cell, algorithm) entries previously "
            f"marked too slow — pass --include-skipped to retry them."
        )

    if not tasks:
        # Nothing new to do — still re-aggregate (cheap) and write so any
        # meta-only updates (dims/trials added) are persisted.
        for cell in grid["cells"].values():
            for entry in cell.values():
                _aggregate(entry)
        _save_atomic(args.output, grid)
        print("Nothing to run — grid is already complete for this configuration.")
        return 0

    # Probe pass (serial, fast) — prunes too-slow algorithms before we burn
    # parallel work on them.
    print("  Probe pass: timing first run per (cell, algorithm) pair ...")
    t0 = time.perf_counter()
    tasks, n_probed = _probe_first(tasks, grid, args.save_every, args.output)
    print(
        f"  Probed {n_probed} (cell, algorithm) pairs in "
        f"{time.perf_counter() - t0:.1f}s. {len(tasks)} parallel tasks remain."
    )
    _save_atomic(args.output, grid)

    # Parallel pass.
    print(f"  Parallel pass: {len(tasks)} tasks across {args.workers} workers")
    completed = 0
    t0 = time.perf_counter()
    if args.workers <= 1:
        # Serial — useful for debugging and reproducible-order updates.
        for task in tasks:
            result = _run_one_in_worker(task)
            _merge_result(grid, result)
            completed += 1
            if completed % args.save_every == 0:
                _save_atomic(args.output, grid)
                _report_progress(completed, len(tasks), t0)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_one_in_worker, t) for t in tasks]
            for fut in as_completed(futures):
                result = fut.result()
                _merge_result(grid, result)
                completed += 1
                if completed % args.save_every == 0:
                    _save_atomic(args.output, grid)
                    _report_progress(completed, len(tasks), t0)

    _save_atomic(args.output, grid)
    print(
        f"Done in {time.perf_counter() - t0:.1f}s. "
        f"Wrote {args.output} ({completed} runs)."
    )
    return 0


def _merge_result(grid: dict, result: tuple) -> None:
    cell_key, algorithm, obj_name, seed, best, wall, err = result
    entry = grid["cells"][cell_key].setdefault(algorithm, {"runs": {}})
    entry.setdefault("runs", {})[f"{obj_name}/{seed}"] = {
        "best": best,
        "wall": wall,
    }
    if err:
        entry.setdefault("errors", []).append(err)
    _aggregate(entry)


def _report_progress(done: int, total: int, t0: float) -> None:
    rate = done / max(time.perf_counter() - t0, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    print(f"  ... {done}/{total} ({rate:.1f} runs/s, eta {eta:.0f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
