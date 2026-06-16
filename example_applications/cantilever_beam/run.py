"""
Run a representative slice of HumpDay's optimisers on the cantilever
beam problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.cantilever_beam.run
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

N_TRIALS = 500


def main():
    print(f"Cantilever beam design  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best weight':>11}  feasible?   design (x1..x5)")
    print("-" * 92)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        feas = "✓" if d["feasible"] else f"✗ (g {d['g']:.2g})"
        xs = ", ".join(f"{xi:.3f}" for xi in d["x"])
        print(f"  {name:<22}  {d['weight']:>11.4f}  {feas:<12}  ({xs})")

    print()
    print(
        "Reference global minimum  weight ≈ 1.33996  at  "
        "(6.016, 5.309, 4.494, 3.502, 2.153)"
    )


if __name__ == "__main__":
    main()
