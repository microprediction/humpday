"""
Free-kick optimisation across a representative slice of HumpDay's optimisers.

    python -m example_applications.free_kick.run

Bend the ball over/around the wall and past a diving keeper. A clean goal scores
~100+; wall blocks and saves score low. Curve and loft must be co-ordinated.
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
N_TRIALS = 250


def main():
    print(
        f"Free kick  —  n_dim={problem.N_DIM} (aim/loft/power/curve), n_trials={N_TRIALS}"
    )
    print(f"{'algorithm':<24}  {'score':>7}  {'outcome':>14}")
    print("-" * 50)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        score, result = problem.evaluate_kick(opt.best_x)
        rows.append((name, score, result))
    for name, score, result in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{score:>7.1f}  {result:>14}")
    print()
    print("Score >100 = a goal the keeper couldn't reach. The wall and keeper make")
    print("it multimodal: rise over the wall, dip under the bar, find the corner.")


if __name__ == "__main__":
    main()
