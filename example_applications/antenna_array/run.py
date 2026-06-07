"""
Antenna element placement across a representative slice of HumpDay's optimisers.

    python -m example_applications.antenna_array.run

A human spaces the 7 elements evenly (~8 dBi); the optimisers find an irregular
spacing that beats it by a couple of decibels of forward gain.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "CMAEvolutionStrategy",
    "DifferentialEvolution",
    "ParticleSwarm",
    "PRIMA_BOBYQA",
    "NelderMead",
    "Powell",
    "RandomSearch",
]
N_TRIALS = 200


def main():
    print(
        f"Antenna array  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}, "
        f"span={problem.SPAN}λ"
    )
    print(f"  uniform (evenly spaced) baseline: {problem.UNIFORM_DBI:.2f} dBi")
    print(f"{'algorithm':<24}  {'gain (dBi)':>10}  {'vs uniform':>10}")
    print("-" * 50)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, -float(opt.best_value)))
    for name, dbi in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<22}  {dbi:>10.2f}  {dbi - problem.UNIFORM_DBI:>+9.2f}")
    print()
    print("Higher dBi = a tighter forward beam. The optimisers beat the even array.")


if __name__ == "__main__":
    main()
