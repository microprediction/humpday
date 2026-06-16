"""
Run a representative slice of HumpDay's optimisers on the interplanetary
transfer-window (porkchop) problem and print a comparison table.

    python -m example_applications.transfer_window.run
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
    print(
        f"Earth→Mars transfer window (porkchop)  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}"
    )
    print(f"{'algorithm':<24}  {'Δv':>8}   departure   time-of-flight")
    print("-" * 70)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        print(
            f"  {name:<22}  {d['delta_v']:>8.4f}   {d['departure']:>9.2f}   "
            f"{d['time_of_flight']:>9.2f}"
        )

    print()
    print("Global best Δv ≈ 0.188 (matches the analytic Hohmann transfer).")
    print("Low-Δv launch windows are DISJOINT islands ~one synodic period apart;")
    print("methods landing in different windows reach the same Δv at different dates —")
    print("local search sees only the island it started in.")


if __name__ == "__main__":
    main()
