"""
Run a representative slice of HumpDay's optimisers on the tension/
compression spring problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.tension_spring.run
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
    print(f"Tension spring design  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best weight':>11}  feasible?   design (d, D, N)")
    print("-" * 80)

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
            f"  {name:<22}  {d['weight']:>11.5f}  {feas:<10}  "
            f"({d['d']:.5f}, {d['D']:.5f}, {d['N']:.4f})"
        )

    print()
    print("Reference global minimum  weight ≈ 0.012665  at  (0.05169, 0.35672, 11.289)")


if __name__ == "__main__":
    main()
