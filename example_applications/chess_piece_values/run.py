"""
Chess evaluation tuning across a representative slice of HumpDay's optimisers.

    python -m example_applications.chess_piece_values.run

NOTE: this is the EXPENSIVE example — every evaluation plays several full depth-2
games, so a run takes a few minutes. The score is your bot's win % vs a textbook
bot; random openings make it noisy, so 'train' (the seeds searched) beats 'test'
(a held-out cohort). The optimiser does NOT rediscover the textbook 1/3/3/5/9
values — it exploits the shallow fixed opponent.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "CMAEvolutionStrategy",
    "DifferentialEvolution",
    "PRIMA_BOBYQA",
    "NelderMead",
    "RandomSearch",
]
N_TRIALS = 12  # kept small: each trial plays several depth-2 games


def main():
    print(
        f"Chess piece values  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS} "
        f"(slow: depth-{problem.DEPTH} games), train seeds={problem.N_TRAIN}"
    )
    print(
        f"{'algorithm':<24}  {'train win%':>10}  {'test win%':>9}  "
        f"{'values N/B/R/Q':>20}"
    )
    print("-" * 72)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        train = -float(opt.best_value)
        test = problem.evaluate_candidate(opt.best_x)
        a = problem.decode(opt.best_x)
        vals = f"{a[0]:.0f}/{a[1]:.0f}/{a[2]:.0f}/{a[3]:.0f}"
        rows.append((name, train, test, vals))
    for name, train, test, vals in sorted(rows, key=lambda r: -r[2]):
        print(f"  {name:<22}  {train:>10.1f}  {test:>9.1f}  {vals:>20}")
    print()
    print("Sorted by held-out win%. 50% is parity with the textbook bot.")
    print("Textbook values are 320/330/500/900; the optimum is rarely those.")


if __name__ == "__main__":
    main()
