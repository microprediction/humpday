"""
Boids flocking-weight search across a representative slice of HumpDay's optimisers.

    python -m example_applications.boids_flocking.run

The score is the percentage of the swarm that reaches the goal through the
chicane. Start jitter makes it mildly noisy, so 'train' (the seeds the optimiser
saw) is reported alongside a held-out cohort.
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

N_TRIALS = 120


def main():
    print(
        f"Boids flocking  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}, "
        f"{problem.NB} boids"
    )
    print(f"{'algorithm':<24}  {'train %':>8}  {'test mean %':>11}  {'test max %':>10}")
    print("-" * 62)

    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append(
            (name, -float(opt.best_value), problem.evaluate_weights(opt.best_x))
        )

    for name, train, test in sorted(rows, key=lambda r: -r[2]["mean"]):
        print(
            f"  {name:<22}  {train:>8.1f}  {test['mean']:>11.1f}  {test['max']:>10.1f}"
        )

    print()
    print("A flailing swarm crashes into the chicane; a good weighting threads it")
    print("and most boids reach the goal. Nobody told them how to flock.")


if __name__ == "__main__":
    main()
