"""
Ebola intervention timing across a representative slice of HumpDay's optimisers.

    python -m example_applications.ebola_response.run

The score is the percentage of harm (deaths + control cost) avoided versus doing
nothing. Watch the optimisers converge on the epidemiologist's playbook: little
control early, hard during the growth phase, then ease off.
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

N_TRIALS = 200


def main():
    print(
        f"Ebola response  —  n_dim={problem.N_DIM} control windows, "
        f"n_trials={N_TRIALS}, {problem.T} days"
    )
    print(
        f"{'algorithm':<24}  {'harm avoided':>12}  {'control profile (8 windows)':>30}"
    )
    print("-" * 74)

    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, -float(opt.best_value), list(opt.best_x)))

    for name, score, u in sorted(rows, key=lambda r: -r[1]):
        prof = "".join(
            "#" if v > 0.66 else "+" if v > 0.33 else "." for v in u[: problem.K]
        )
        print(f"  {name:<22}  {score:>11.1f}%  {prof:>30}")

    print()
    print("'.' little / '+' moderate / '#' hard control per window (early -> late).")
    print("The good policies stay low, spike during the growth phase, then relax.")


if __name__ == "__main__":
    main()
