"""
Run a representative slice of HumpDay's optimisers on the continuous
p-median facility location problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.facility_location.run
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

N_TRIALS = 500


def main():
    print(f"Facility location  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best cost':>10}  facilities (x, y) x 3")
    print("-" * 88)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        facs = "  ".join(f"({fx:.2f}, {fy:.2f})" for fx, fy in d["facilities"])
        print(f"  {name:<22}  {d['total_cost']:>10.4f}  {facs}")

    print()
    centres = "  ".join(f"({cx:.2f}, {cy:.2f})" for cx, cy in problem.CLUSTER_CENTRES)
    print(f"Demand clusters centred near:  {centres}")


if __name__ == "__main__":
    main()
