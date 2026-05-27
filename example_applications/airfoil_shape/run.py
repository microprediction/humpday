"""
Low-budget airfoil shape optimisation. The point of this example is the
budget — n_trials=50, on a 6-D problem — exactly the regime where
Bayesian / surrogate-based optimisers are supposed to beat evolutionary
methods.

    python -m example_applications.airfoil_shape.run
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "BayesianOpt",
    "PRIMA_BOBYQA",
    "NelderMead",
    "Powell",
    "CMAEvolutionStrategy",
    "DifferentialEvolution",
    "ParticleSwarm",
    "GeneticAlgorithm",
]

N_TRIALS = 50  # the whole point of this example


def main():
    print(
        f"Airfoil Hicks-Henne shape optimisation  —  n_dim={problem.N_DIM}, "
        f"n_trials={N_TRIALS} (low budget!)"
    )
    print(
        f"{'algorithm':<24}  {'best drag':>10}  {'max t/c':>8}  "
        f"{'x of max t':>10}  stall?"
    )
    print("-" * 78)

    # Reference: NACA 0012 baseline (no perturbation).
    baseline_u = [0.5] * problem.N_DIM
    baseline = problem.decode(baseline_u)
    print(
        f"  {'NACA-0012 baseline':<22}  {baseline['drag']:>10.6f}  "
        f"{baseline['max_thickness']:>8.4f}  {baseline['max_thickness_x']:>10.3f}  "
        f"{'no' if not baseline['stalls'] else 'YES'}"
    )
    print("  " + "-" * 76)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        print(
            f"  {name:<22}  {d['drag']:>10.6f}  {d['max_thickness']:>8.4f}  "
            f"{d['max_thickness_x']:>10.3f}  {'YES' if d['stalls'] else 'no'}"
        )

    print()
    print(
        "With only 50 evaluations on a 6-D problem, surrogate/Bayesian methods "
        "should do better than population-based ones that can't finish even one\n"
        "generation."
    )


if __name__ == "__main__":
    main()
