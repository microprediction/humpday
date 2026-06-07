"""Trebuchet (reduced-order) across HumpDay optimisers.
python -m example_applications.trebuchet.run
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
    print(
        f"Trebuchet (reduced-order)  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}, target {problem.TARGET_M:.0f} m"
    )
    print(f"{'algorithm':<24}  {'score':>7}  {'range (m)':>9}")
    print("-" * 48)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, -float(opt.best_value), problem.throw_range(opt.best_x)))
    for name, sc, rng in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{sc:>7.1f}  {rng:>9.1f}")
    print("\nSimplified energy/ballistics model (the browser demo uses Matter.js).")


if __name__ == "__main__":
    main()
