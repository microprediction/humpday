"""
Run a representative slice of HumpDay's optimisers on the non-convex
portfolio allocation problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.portfolio_frontier.run
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

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
    print(f"Portfolio allocation  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"(gamma={problem.GAMMA}, per-holding cost kappa={problem.KAPPA})")
    print(
        f"{'algorithm':<24}  {'objective':>10}  {'ret':>6}  {'vol':>6}  {'#':>2}  top holdings"
    )
    print("-" * 92)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        top = sorted(d["weights"].items(), key=lambda kv: -kv[1])[:3]
        holdings = ", ".join(f"{k} {v:.0%}" for k, v in top)
        print(
            f"  {name:<22}  {d['objective']:>10.5f}  {d['expected_return']:>6.1%}  "
            f"{d['volatility']:>6.1%}  {d['n_holdings']:>2}  {holdings}"
        )

    print()
    print("Lower objective = better risk-adjusted utility net of holding costs.")
    print("The cardinality penalty makes this non-convex: optimisers can land in")
    print("different subset-of-assets basins, so the table should show real spread.")


if __name__ == "__main__":
    main()
