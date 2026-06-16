"""
Run a representative slice of HumpDay's optimisers on the Kalman-filter
noise-tuning problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.kalman_tuning.run
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
    print(f"Kalman filter tuning  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best RMSE':>10}  tuned (q, r)")
    print("-" * 64)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        print(f"  {name:<22}  {d['rmse']:>10.4f}  (q={d['q']:.3e}, r={d['r']:.3e})")

    print()
    print(
        f"Raw measurement-noise std ≈ {problem.MEAS_NOISE_STD:.2f}; "
        "a well-tuned filter reaches a position RMSE of roughly 1.0–1.5."
    )


if __name__ == "__main__":
    main()
