"""
Tennis-doubles strategy search across a representative slice of HumpDay's optimisers.

    python -m example_applications.tennis_doubles.run

The win rate over a small training batch flatters the strategy relative to a
large held-out set — the in/out-of-sample overfitting story. A textbook team
wins ~50% against itself; the optimiser pushes your win rate above that.
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
        f"Tennis doubles  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}, "
        f"train pts={problem.N_TRAIN * 2}, held-out pts={len(problem.TEST_SEEDS) * 2}"
    )
    print(
        f"{'algorithm':<24}  {'train win%':>10}  {'test win%':>9}  "
        f"{'strategy (net/depth/poach/cross/risk/lob)':>42}"
    )
    print("-" * 92)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        train = -float(opt.best_value)
        test = problem.evaluate_strategy(opt.best_x)
        d = problem.decode(opt.best_x)
        spec = (
            f"{d['netX']:.0f}/{d['baseDepth']:.0f}/{d['poach']:.2f}/"
            f"{d['aimCross']:.2f}/{d['depthRisk']:.2f}/{d['lob']:.2f}"
        )
        rows.append((name, train, test, spec))
    for name, train, test, spec in sorted(rows, key=lambda r: -r[2]):
        print(f"  {name:<22}  {train:>10.1f}  {test:>9.1f}  {spec:>42}")
    print()
    print("Sorted by held-out win%. 'train' > 'test' is overfitting to the small")
    print("training batch; 50% is parity with the textbook team.")


if __name__ == "__main__":
    main()
