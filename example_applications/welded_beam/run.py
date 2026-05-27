"""
Run a representative slice of HumpDay's optimisers on the welded beam
problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.welded_beam.run
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
    print(f"Welded beam design  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best cost':>10}  feasible?   design (h, l, t, b)")
    print("-" * 88)

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
            f"  {name:<22}  {d['cost']:>10.4f}  {feas:<10}  "
            f"({d['h']:.4f}, {d['l']:.4f}, {d['t']:.4f}, {d['b']:.4f})"
        )

    print()
    print(
        "Reference global minimum  cost ≈ 1.725  at  (0.2057, 3.4705, 9.0366, 0.2057)"
    )


if __name__ == "__main__":
    main()
