"""Curling (reduced-order) across HumpDay optimisers.
python -m example_applications.curling.run
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
    print(f"Curling (reduced-order)  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'score':>7}  {'dist to button (px)':>20}")
    print("-" * 56)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, -float(opt.best_value), problem.stop_distance(opt.best_x)))
    for name, sc, d in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{sc:>7.1f}  {d:>20.1f}")
    print("\nSimplified slide-with-curl model (the browser demo uses Matter.js).")


if __name__ == "__main__":
    main()
