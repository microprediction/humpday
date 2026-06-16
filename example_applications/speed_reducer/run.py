"""
Run a representative slice of HumpDay's optimisers on the speed-reducer
(Golinski gearbox) problem and print a comparison table.

    python -m example_applications.speed_reducer.run
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

N_TRIALS = 600


def main():
    print(
        f"Speed reducer (Golinski gearbox)  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}"
    )
    print(f"{'algorithm':<24}  {'weight':>10}  feasible?   teeth")
    print("-" * 70)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        feas = "✓" if d["feasible"] else f"✗ ({d['max_violation']:.2g})"
        print(f"  {name:<22}  {d['weight']:>10.3f}  {feas:<10}  {d['vars']['teeth']}")

    print()
    print("Reference global optimum  weight ≈ 2994.471  at")
    print("(b,m,teeth,l1,l2,d1,d2) = (3.5, 0.7, 17, 7.3, 7.7153, 3.3503, 5.2867).")


if __name__ == "__main__":
    main()
