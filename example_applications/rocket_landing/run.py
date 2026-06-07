"""
Rocket-landing throttle search across a representative slice of HumpDay's optimisers.

    python -m example_applications.rocket_landing.run

Pick a 12-segment throttle schedule that lands the booster softly with fuel to
spare. Several local optima (gradual descent vs late suicide burn) make it a
real test.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "DifferentialEvolution",
    "CMAEvolutionStrategy",
    "ParticleSwarm",
    "PRIMA_BOBYQA",
    "NelderMead",
    "Powell",
    "RandomSearch",
]
N_TRIALS = 250


def main():
    print(
        f"Rocket landing  —  n_dim={problem.N_DIM} throttle segments, n_trials={N_TRIALS}"
    )
    print(f"{'algorithm':<24}  {'score':>7}  {'landed?':>7}  {'touchdown v':>11}")
    print("-" * 56)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        score, landed, speed = problem.evaluate_schedule(opt.best_x)
        rows.append((name, score, landed, speed))
    for name, score, landed, speed in sorted(rows, key=lambda r: -r[1]):
        print(
            f"  {name:<24.24}{score:>7.1f}  {'yes' if landed else 'no':>7}  {speed:>11.1f}"
        )
    print()
    print("Score >100 = soft landing with fuel left (perfect v≈0 is 100, +10 fuel).")


if __name__ == "__main__":
    main()
