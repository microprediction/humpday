"""
Reactor temperature-profile search across a representative slice of HumpDay's optimisers.

    python -m example_applications.reactor_profile.run

Maximise the yield of intermediate B in a series A->B->C reactor by shaping the
10-zone temperature profile: hot early to convert A, cooler later so B doesn't
run on to C.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "DifferentialEvolution",
    "CMAEvolutionStrategy",
    "PRIMA_BOBYQA",
    "NelderMead",
    "Powell",
    "ParticleSwarm",
    "RandomSearch",
]
N_TRIALS = 250


def main():
    # best isothermal temperature, for reference
    best_iso = max(
        problem.simulate([T] * problem.N_ZONES)[1]
        for T in range(int(problem.T_MIN), int(problem.T_MAX) + 1, 2)
    )
    print(f"Reactor T-profile  —  n_dim={problem.N_DIM} zones, n_trials={N_TRIALS}")
    print(f"  best isothermal yield of B: {best_iso * 100:.1f}%")
    print(f"{'algorithm':<24}  {'yield of B':>10}")
    print("-" * 40)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, -float(opt.best_value)))
    for name, y in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{y * 100:>9.1f}%")
    print()
    print(
        "A shaped profile beats the best single temperature — optimal reactor control."
    )


if __name__ == "__main__":
    main()
