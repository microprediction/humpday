"""
Run a representative slice of HumpDay's optimisers on the pressure
vessel problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.pressure_vessel.run
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
    print(f"Pressure vessel design  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best cost':>12}  feasible?   design (Ts, Th, R, L)")
    print("-" * 92)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        feas = "✓" if d["feasible"] else f"✗ (max viol {d['max_violation']:.2g})"
        print(
            f"  {name:<22}  {d['cost']:>12.4f}  {feas:<10}  "
            f"({d['Ts']:.4f}, {d['Th']:.4f}, {d['R']:.4f}, {d['L']:.4f})"
        )

    print()
    print(
        "Reference global minimum  cost ≈ 6059.714  at  "
        "(0.8125, 0.4375, 42.0984, 176.6366)"
    )


if __name__ == "__main__":
    main()
