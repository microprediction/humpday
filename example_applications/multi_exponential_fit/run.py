"""
Run a representative slice of HumpDay's optimisers on the multi-exponential
fit and print a comparison table.

    python -m example_applications.multi_exponential_fit.run
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
    print(f"Multi-exponential fit  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(
        f"True: A=({problem.TRUE_A[0]}, {problem.TRUE_A[1]}), "
        f"k=({problem.TRUE_K[0]}, {problem.TRUE_K[1]})  "
        f"noise σ={problem.NOISE_SIGMA}"
    )
    print(f"{'algorithm':<24}  {'rms resid':>10}  recovered (A1,k1; A2,k2)")
    print("-" * 86)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        print(
            f"  {name:<22}  {d['rms_residual']:>10.5f}  "
            f"({d['A1']:.2f}, {d['k1']:.2f}; {d['A2']:.2f}, {d['k2']:.2f})"
        )

    print()
    print(
        f"Noise floor ≈ σ = {problem.NOISE_SIGMA:.3f}. Recovered rates vary wildly "
        "between methods"
    )
    print("even at similar residuals — the signature of the ill-conditioned valley.")


if __name__ == "__main__":
    main()
