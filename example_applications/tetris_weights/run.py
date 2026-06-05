"""
Tetris weight tuning across a representative slice of HumpDay's optimisers.

    python -m example_applications.tetris_weights.run

The objective is noisy (random piece bags), so 'train' (the seeds the optimiser
saw) flatters the weights relative to 'test' (a held-out cohort). The gap is the
noise-overfit tax — and re-validating the best candidate on fresh seeds before
committing is the real lever against it.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "CMAEvolutionStrategy",
    "DifferentialEvolution",
    "ParticleSwarm",
    "PRIMA_BOBYQA",
    "NelderMead",
    "Powell",
    "RandomSearch",
]

N_TRIALS = 150


def main():
    print(
        f"Tetris weight tuning  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}, "
        f"board={problem.W}x{problem.H}, games/eval={problem.N_GAMES}"
    )
    print(
        f"{'algorithm':<24}  {'train lines':>11}  {'test mean':>9}  "
        f"{'test med.':>9}  {'weights (H/L/Ho/B)':>22}"
    )
    print("-" * 86)

    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        train = -float(opt.best_value)
        test = problem.evaluate_weights(opt.best_x)
        w = [2 * v - 1 for v in opt.best_x]
        rows.append((name, train, test, w))

    for name, train, test, w in sorted(rows, key=lambda r: -r[2]["mean"]):
        wt = " ".join(f"{x:+.2f}" for x in w)
        print(
            f"  {name:<22}  {train:>11.1f}  {test['mean']:>9.1f}  "
            f"{test['median']:>9.0f}  {wt:>22}"
        )

    print()
    print("Sorted by held-out mean. 'train' > 'test' is overfitting to the")
    print("training seeds; the optimum is not the textbook 1/+/-/- weighting.")


if __name__ == "__main__":
    main()
