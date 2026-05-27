"""
Walk-forward optimisation of a z-score channel strategy. Tunes
parameters on the first half of a synthetic price series, then
reports both in-sample and out-of-sample Sharpe ratios.

    python -m example_applications.algo_trading.run
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
    "GeneticAlgorithm",
]

N_TRIALS = 200


def main():
    print(
        f"Algorithmic trading walk-forward  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}"
    )
    print(
        f"{'algorithm':<24}  {'IS Sharpe':>9}  {'OOS Sharpe':>10}  "
        f"{'OOS/IS':>7}  parameters (lookback, entry_z, exit_z)"
    )
    print("-" * 100)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        in_sharpe = -float(opt.best_value)
        oos_sharpe = problem.out_of_sample_sharpe(opt.best_x)
        ratio = oos_sharpe / in_sharpe if in_sharpe > 1e-6 else float("nan")
        d = problem.decode(opt.best_x)
        print(
            f"  {name:<22}  {in_sharpe:>9.3f}  {oos_sharpe:>10.3f}  "
            f"{ratio:>7.2f}  ({d['lookback']}, {d['entry_z']:.2f}, {d['exit_z']:.2f})"
        )

    print()
    print("OOS/IS ratio: closer to 1.0 = better generalisation; near 0 or")
    print("negative = the optimiser overfit a regime-specific spurious peak.")


if __name__ == "__main__":
    main()
