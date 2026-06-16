"""
Run a representative slice of HumpDay's optimisers on the (s, S)
inventory-control problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.inventory_policy.run
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

N_TRIALS = 400


def main():
    print(f"(s, S) inventory policy  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best cost':>10}  valid?   policy (s, S)")
    print("-" * 72)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        valid = "✓" if d["valid"] else "✗"
        print(
            f"  {name:<22}  {d['cost']:>10.2f}  {valid:<6}  "
            f"(s={d['s']:.2f}, S={d['S']:.2f})"
        )

    print()
    print(
        "Costs are NOISY (demand is random each rollout); a sensible (s, S) with\n"
        "moderate s and S>s clearly beats extreme policies, but no single run is exact."
    )


if __name__ == "__main__":
    main()
