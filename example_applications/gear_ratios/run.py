"""
Run a representative slice of HumpDay's optimisers on the gear-train design
problem and print a comparison table.

    python -m example_applications.gear_ratios.run
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

N_TRIALS = 500


def main():
    print(f"Gear train design  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"Target ratio 1/6.931 = {problem.TARGET:.7f}")
    print(f"{'algorithm':<24}  {'sq error':>12}  ratio       teeth")
    print("-" * 74)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        z = d["teeth"]
        print(
            f"  {name:<22}  {d['sq_error']:>12.3e}  {d['ratio']:.6f}  "
            f"({z[0]},{z[1]},{z[2]},{z[3]})"
        )

    print()
    print("Best-known optimum  f ≈ 2.7e-12  at teeth (19,16,43,49) [ratio 304/2107].")
    print("The objective is flat on each lattice cell — methods that need a gradient")
    print("stall; lattice-hoppers (DE, annealing) find better near-ties.")


if __name__ == "__main__":
    main()
