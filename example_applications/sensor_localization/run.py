"""
Run a representative slice of HumpDay's optimisers on the sensor-network
localisation problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.sensor_localization.run
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

N_TRIALS = 500


def main():
    # Residual achieved by the true layout: the noise floor good
    # optimisers should approach.
    floor = problem.decode(problem.true_u())["residual"]
    print(f"Sensor localisation  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best residual':>13}  estimated node positions")
    print("-" * 88)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        pts = ", ".join(f"({x:.2f},{y:.2f})" for x, y in d["positions"])
        print(f"  {name:<22}  {d['residual']:>13.5f}  {pts}")

    print()
    print(f"Noise-floor residual at the true layout ≈ {floor:.5f}")
    print(
        "Higher residuals indicate the optimiser is trapped in a "
        "flipped / folded local minimum."
    )


if __name__ == "__main__":
    main()
