"""
Battery dispatch across a representative slice of HumpDay's optimisers.

    python -m example_applications.battery_dispatch.run

Arbitrage one day of electricity prices: buy low overnight, sell into the evening
peak, never overfill or drain the pack, and pay the round-trip efficiency tax.
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
N_TRIALS = 300


def main():
    print(f"Battery dispatch  —  n_dim={problem.N_DIM} hours, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'revenue $/day':>13}  {'cycles':>7}")
    print("-" * 50)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rev, soc, cycles = problem.evaluate_schedule(opt.best_x)
        rows.append((name, rev, cycles))
    for name, rev, cycles in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{rev:>13,.0f}  {cycles:>7.2f}")
    print()
    print("Higher revenue is better. State-of-charge limits + efficiency losses")
    print("constrain how aggressively you can chase the price spread.")


if __name__ == "__main__":
    main()
