"""
Direct policy search for CartPole across a representative slice of
HumpDay's optimisers.

    python -m example_applications.cart_pole_policy.run
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "NelderMead",
    "Powell",
    "PRIMA_BOBYQA",
    "DifferentialEvolution",
    "ParticleSwarm",
    "CMAEvolutionStrategy",
    "GeneticAlgorithm",
    "SimulatedAnnealing",
]

N_TRIALS = 200


def main():
    print(
        f"CartPole direct policy search  —  n_dim={problem.N_DIM}, "
        f"n_trials={N_TRIALS}, episodes_per_eval={problem.N_EPISODES}"
    )
    print(
        f"{'algorithm':<24}  {'train ret.':>11}  {'test mean':>9}  "
        f"{'test med.':>9}  {'min':>5}  {'max':>5}"
    )
    print("-" * 78)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        train_return = -float(opt.best_value)
        test = problem.evaluate_policy(opt.best_x)
        print(
            f"  {name:<22}  {train_return:>11.1f}  {test['mean']:>9.1f}  "
            f"{test['median']:>9.1f}  {test['min']:>5d}  {test['max']:>5d}"
        )

    print()
    print(f"Max possible return per episode: {problem.MAX_STEPS}")
    print("Gap between 'train ret.' and 'test mean' is the noise-overfit tax.")


if __name__ == "__main__":
    main()
