"""
Run a representative slice of HumpDay's optimisers on the radiation
therapy beam-weight problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.radiation_therapy.run
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
    print(
        f"Radiation therapy beam weights  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}"
    )
    print(f"(prescription Dp={problem.DP}, OAR tolerance Dmax={problem.DMAX})")
    print(
        f"{'algorithm':<24}  {'best score':>10}  "
        f"{'tumor mean':>10}  {'tumor min':>9}  {'tumor max':>9}  {'OAR max':>8}"
    )
    print("-" * 88)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        print(
            f"  {name:<22}  {d['score']:>10.4f}  "
            f"{d['tumor_mean']:>10.4f}  {d['tumor_min']:>9.4f}  "
            f"{d['tumor_max']:>9.4f}  {d['oar_max']:>8.4f}"
        )

    print()
    print(
        "Tumor doses should cluster near Dp=1.0; OAR max should sit at or "
        "below Dmax=0.5 where the trade-off allows."
    )


if __name__ == "__main__":
    main()
