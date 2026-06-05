"""
Bowling throw optimisation across a representative slice of HumpDay's optimisers.

    python -m example_applications.bowling.run

Launch a heavy ball into a 105-pin triangle and maximise the chain reaction.
Tiny changes in entry angle and spin cascade into very different pin counts.
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
    "RandomSearch",
]
N_TRIALS = 150


def main():
    print(
        f"Bowling  —  n_dim={problem.N_DIM} (speed/angle/spin/release), "
        f"n_trials={N_TRIALS}, {problem.TOTAL_PINS} pins"
    )
    print(f"{'algorithm':<24}  {'pins down':>9}  {'params (spd/ang/spin/relX)':>28}")
    print("-" * 68)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        down = problem.evaluate_throw(opt.best_x)
        p = problem.decode(opt.best_x)
        spec = f"{p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f}/{p[3]:.0f}"
        rows.append((name, down, spec))
    for name, down, spec in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{down:>9d}  {spec:>28}")
    print()
    print(f"Max is {problem.TOTAL_PINS}. A rough, sensitive landscape — chain")
    print("reactions mean small entry changes swing the pin count.")


if __name__ == "__main__":
    main()
