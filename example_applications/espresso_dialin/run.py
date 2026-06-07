"""
Espresso dial-in across a representative slice of HumpDay's optimisers.

    python -m example_applications.espresso_dialin.run

A small budget (a dozen-ish pulls) is the whole point: this is the
expensive-evaluation, sample-efficient regime. Watch interpolation / Bayesian
methods nail the sweet spot while population methods are still warming up.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "PRIMA_BOBYQA",
    "BayesianOpt",
    "NelderMead",
    "Powell",
    "CMAEvolutionStrategy",
    "DifferentialEvolution",
    "RandomSearch",
]

N_TRIALS = 24  # only a couple-dozen pulls


def main():
    print(
        f"Espresso dial-in  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS} pulls "
        f"(noise sd={problem.NOISE_SD})"
    )
    print(f"{'algorithm':<24}  {'true score':>10}  {'extract %':>9}  {'ratio':>6}")
    print("-" * 56)

    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        problem._counter[0] = 0
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, problem.evaluate_recipe(opt.best_x)))

    for name, ev in sorted(rows, key=lambda r: -r[1]["score"]):
        print(
            f"  {name:<22}  {ev['score']:>10.1f}  {ev['extraction']:>9.1f}  {ev['ratio']:>6.2f}"
        )

    print()
    print("'true score' re-evaluates the chosen recipe without noise. The target")
    print("is extraction ~20% and ratio ~1.75 (≈1:2); the sweet spot is small.")


if __name__ == "__main__":
    main()
