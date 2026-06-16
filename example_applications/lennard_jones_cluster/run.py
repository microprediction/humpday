"""
Run a representative slice of HumpDay's optimisers on the Lennard-Jones
cluster (N = 5) problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.lennard_jones_cluster.run
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

# Pick a representative slice: one local, one trust-region, two
# population-based, one Bayesian. Keep the list short so the output
# table is readable.
ALGORITHMS = [
    "NelderMead",
    "Powell",
    "PRIMA_BOBYQA",
    "DifferentialEvolution",
    "ParticleSwarm",
    "CMAEvolutionStrategy",
    "BayesianOpt",
    "SimulatedAnnealing",
]

N_TRIALS = 800


def main():
    print(f"Lennard-Jones cluster (N=5)  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best energy':>12}")
    print("-" * 40)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        print(f"  {name:<22}  {d['energy']:>12.4f}")

    print()
    print("Reference global minimum  energy ≈ -9.103852  (triangular bipyramid)")


if __name__ == "__main__":
    main()
