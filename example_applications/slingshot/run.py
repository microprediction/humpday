"""Slingshot (reduced-order) across HumpDay optimisers.
python -m example_applications.slingshot.run
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
        f"Slingshot (reduced-order)  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}, {len(problem.BLOCKS)} blocks"
    )
    print(f"{'algorithm':<24}  {'blocks hit':>10}")
    print("-" * 40)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, problem.blocks_hit(opt.best_x)))
    for name, h in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{h:>10d}")
    print("\nTwo stacks across a gap (rake one or loft onto the other) — two basins.")
    print("Simplified ballistic model (the browser demo uses Matter.js).")


if __name__ == "__main__":
    main()
