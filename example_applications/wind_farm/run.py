"""
Wind-farm layout across a representative slice of HumpDay's optimisers.

    python -m example_applications.wind_farm.run

Place 8 turbines to maximise expected power over a 12-sector wind rose (Jensen
wake model). The optimiser must spread turbines out of each other's wakes while
respecting a minimum spacing.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "DifferentialEvolution",
    "CMAEvolutionStrategy",
    "ParticleSwarm",
    "PRIMA_BOBYQA",
    "NelderMead",
    "RandomSearch",
]
N_TRIALS = 250


def main():
    print(
        f"Wind farm layout  —  n_dim={problem.N_DIM} ({problem.N_TURBINES} turbines), "
        f"n_trials={N_TRIALS}"
    )
    print(f"{'algorithm':<24}  {'score':>7}  {'power frac':>10}  {'spacing pen':>11}")
    print("-" * 60)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        score, pf, pen = problem.evaluate_layout(opt.best_x)
        rows.append((name, score, pf, pen))
    for name, score, pf, pen in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{score:>7.1f}  {pf * 100:>9.1f}%  {pen:>11.3f}")
    print()
    print("Higher = more expected power. Wake coupling makes it non-separable;")
    print("spreading turbines apart and out of the prevailing wakes is the game.")


if __name__ == "__main__":
    main()
