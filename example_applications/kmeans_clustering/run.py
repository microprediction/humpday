"""
Run a representative slice of HumpDay's optimisers on the k-means
clustering problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.kmeans_clustering.run
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
    print(f"k-means clustering  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best SSE':>10}   centroids (x, y) ×3")
    print("-" * 88)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        cents = "  ".join(f"({cx:.2f}, {cy:.2f})" for cx, cy in d["centroids"])
        print(f"  {name:<22}  {d['sse']:>10.4f}   {cents}")

    print()
    print(
        "Global optimum places one centroid per blob (low SSE); a 2-1 split "
        "traps the optimiser at a higher SSE."
    )


if __name__ == "__main__":
    main()
