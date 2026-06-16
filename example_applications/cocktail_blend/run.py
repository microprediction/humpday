"""
Run a representative slice of HumpDay's optimisers on the cocktail-blend
inverse problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.cocktail_blend.run
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

N_TRIALS = 300


def main():
    print(f"Cocktail blend  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"Target flavour: {dict(zip(problem.AXES, problem.TARGET))}")
    print(f"{'algorithm':<24}  {'rms err':>8}   top ingredients")
    print("-" * 78)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        top = sorted(d["recipe"].items(), key=lambda kv: -kv[1])[:3]
        recipe = ", ".join(f"{k} {v:.0%}" for k, v in top)
        print(f"  {name:<22}  {d['rms_error']:>8.4f}   {recipe}")

    print()
    print("Lower RMS error = closer to the target flavour. The optimum is a")
    print("balanced blend, not a single-ingredient vertex of the simplex.")


if __name__ == "__main__":
    main()
