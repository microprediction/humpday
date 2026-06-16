"""
Run a representative slice of HumpDay's optimisers on the
Michaelis–Menten enzyme-kinetics fit and print a comparison table.

Invoke from the repo root:

    python -m example_applications.enzyme_kinetics.run
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
    print(f"Enzyme kinetics fit  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best SSE':>10}  {'RMS':>8}   fit (Vmax, Km)")
    print("-" * 72)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        print(
            f"  {name:<22}  {d['sse']:>10.5f}  {d['rms']:>8.4f}   "
            f"({d['Vmax']:.4f}, {d['Km']:.4f})"
        )

    print()
    print(
        f"True parameters  Vmax = {problem.TRUE_VMAX}, Km = {problem.TRUE_KM}  "
        f"(noise floor RMS ≈ {problem.NOISE_SIGMA})"
    )


if __name__ == "__main__":
    main()
