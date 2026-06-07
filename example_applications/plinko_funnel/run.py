"""
Plinko funnel tuning across a representative slice of HumpDay's optimisers.

    python -m example_applications.plinko_funnel.run

The objective is inherently noisy (every board is 400 random balls), so 'train'
is a single noisy board and 'test' averages a held-out cohort. The optimiser is
steering a stochastic process onto an off-centre target.
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

N_TRIALS = 150


def main():
    print(
        f"Plinko funnel  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}, "
        f"balls/board={problem.N_BALLS}, target bin #{problem.TARGET}"
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
        train = -float(opt.best_value)
        test = problem.evaluate_profile(opt.best_x)
        rows.append((name, train, test))

    for name, train, test in sorted(rows, key=lambda r: -r[2]["mean"]):
        print(
            f"  {name:<22}  {train:>8.1f}  {test['mean']:>11.1f}  {test['max']:>10.1f}"
        )

    print()
    print("With no lean only a few percent reach an off-centre bin; a good")
    print("funnel gets more than half of them there.")


if __name__ == "__main__":
    main()
