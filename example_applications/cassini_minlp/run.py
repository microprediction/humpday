"""
Run a representative slice of HumpDay's optimisers on the Cassini-style
mixed-integer gravity-assist trajectory and print a comparison table.

    python -m example_applications.cassini_minlp.run
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

N_TRIALS = 800


def main():
    print(
        f"Cassini-style MGA trajectory  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}"
    )
    print("(Earth → 4 flybys → Saturn; flyby planets are discrete choices)")
    print(f"{'algorithm':<24}  {'Δv':>8}   flyby sequence")
    print("-" * 74)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        seq = " → ".join(s[:3] for s in d["sequence"])
        print(f"  {name:<22}  {d['delta_v']:>8.3f}   {seq}")

    print()
    print("The flyby planets are discrete (mixed-integer): different sequences reach")
    print("near-tied Δv, so methods disagree on the sequence — the combinatorial trap.")


if __name__ == "__main__":
    main()
