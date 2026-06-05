"""
Circle packing across a representative slice of HumpDay's optimisers.

    python -m example_applications.circle_packing.run

Pack 6 equal circles into a unit square. Score is the achievable radius as a
percentage of the best-known packing (r* ≈ 0.1875); 100 is near-optimal.
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
    "RandomSearch",
]
N_TRIALS = 300


def main():
    print(
        f"Circle packing  —  {problem.N_CIRCLES} circles, n_dim={problem.N_DIM}, "
        f"n_trials={N_TRIALS}"
    )
    print(f"{'algorithm':<24}  {'score':>7}  {'radius':>7}")
    print("-" * 44)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        s = -float(opt.best_value)
        rows.append((name, s, s / 100 * problem.R_REF))
    for name, s, r in sorted(rows, key=lambda x: -x[1]):
        print(f"  {name:<22}  {s:>6.1f}%  {r:>7.4f}")
    print()
    print("Non-smooth 'maximise the minimum' problem; 100% ≈ the optimal pack.")


if __name__ == "__main__":
    main()
