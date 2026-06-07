"""Mini-golf (reduced-order) across HumpDay optimisers.
python -m example_applications.mini_golf.run
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
    print(f"Mini-golf (reduced-order)  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'score':>7}  {'finish dist (px)':>16}")
    print("-" * 52)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, -float(opt.best_value), problem.finish_distance(opt.best_x)))
    for name, sc, d in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{sc:>7.1f}  {d:>16.1f}")
    print(
        "\nSink (dist < hole radius) scores 100. Simplified roll model (demo uses Matter.js)."
    )


if __name__ == "__main__":
    main()
