"""
Run a representative slice of HumpDay's optimisers on the economic
dispatch with valve-point loading problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.economic_dispatch_valve.run
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

# Pick a representative slice: one local, one trust-region, two
# population-based, one Bayesian. Keep the list short so the output
# table is readable.
ALGORITHMS = [
    "NelderMead",
    "Powell",
    "PRIMA_BOBYQA",
    "DifferentialEvolution",
    "ParticleSwarm",
    "CMAEvolutionStrategy",
    "BayesianOpt",
    "SimulatedAnnealing",
]

N_TRIALS = 600


def main():
    print(
        f"Economic dispatch (valve-point)  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}"
    )
    print(
        f"{'algorithm':<24}  {'best cost':>10}  {'mismatch':>9}   powers (P1, P2, P3) MW"
    )
    print("-" * 90)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        print(
            f"  {name:<22}  {d['fuel_cost']:>10.2f}  {d['demand_mismatch']:>+9.3f}   "
            f"({d['P1']:.2f}, {d['P2']:.2f}, {d['P3']:.2f})"
        )

    print()
    print("Reference best-known  cost ≈ 8234.07 $/h  at  P ≈ (300.3, 400.0, 149.7) MW")


if __name__ == "__main__":
    main()
