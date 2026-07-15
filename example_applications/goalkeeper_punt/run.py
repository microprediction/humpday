"""
Goalkeeper-punt optimisation across a representative slice of HumpDay's optimisers.

    python -m example_applications.goalkeeper_punt.run

Two run-up steps, one punt: hit the spider-cam wire strung 30 m downfield and
13 m up. A wire strike scores 100+; misses are shaped by closest approach.
Overstep the penalty area and it's a foul.
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
N_TRIALS = 150


def main():
    print(
        f"Goalkeeper punt  —  n_dim={problem.N_DIM} "
        f"(step1/step2/loft/power), n_trials={N_TRIALS}"
    )
    print(f"{'algorithm':<24}  {'score':>7}  {'outcome':>24}")
    print("-" * 60)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        score, result = problem.evaluate_punt(opt.best_x)
        rows.append((name, score, result))
    for name, score, result in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{score:>7.1f}  {result:>24}")
    print()
    print("Score >100 = the ball clips the wire. The best punt (~109.4) takes the")
    print("longest strides the box allows, in accelerating rhythm, and hits the")
    print("wire flat and fast on the way up.")


if __name__ == "__main__":
    main()
