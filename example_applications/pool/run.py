"""Pool (reduced-order) across HumpDay optimisers.
python -m example_applications.pool.run
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
    "Powell",
    "RandomSearch",
]
N_TRIALS = 200


def main():
    print(f"Pool (reduced-order)  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'pot score':>9}")
    print("-" * 38)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, -float(opt.best_value)))
    for name, sc in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{sc:>9.1f}")
    print("\nThe ghost-ball cut angle is a narrow target. Simplified single-cut model")
    print("(the browser demo simulates the full table with Matter.js).")


if __name__ == "__main__":
    main()
