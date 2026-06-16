"""
Run a representative slice of HumpDay's optimisers on the dartboard
aiming problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.darts_aim.run
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
    print(f"Dartboard aiming  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"(SIGMA = {problem.SIGMA:.0f} mm throw noise, an amateur)")
    print(f"{'algorithm':<24}  {'exp. score':>10}   aim (x, y) mm")
    print("-" * 64)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        print(
            f"  {name:<22}  {d['expected_score']:>10.3f}   "
            f"({d['x']:+.1f}, {d['y']:+.1f})"
        )

    print()
    print(
        "Famous result: skilled players aim at treble-20, but as throw noise "
        "grows the\noptimal aim slides down and to the lower-left of centre "
        "(Tibshirani et al., 2011)."
    )


if __name__ == "__main__":
    main()
