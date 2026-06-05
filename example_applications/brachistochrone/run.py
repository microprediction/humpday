"""
Brachistochrone ramp shaping across a representative slice of HumpDay's optimisers.

    python -m example_applications.brachistochrone.run

Find the ramp that gets a marble from top-left to bottom-right fastest. The
straight line is NOT the answer — the optimum dips early to build speed (the
cycloid). Lower descent time is better.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "PRIMA_BOBYQA",
    "NelderMead",
    "Powell",
    "CMAEvolutionStrategy",
    "DifferentialEvolution",
    "ParticleSwarm",
    "RandomSearch",
]
N_TRIALS = 200


def main():
    # straight-line baseline: control heights on the line from START to END.
    x0, y0 = problem.START
    x1, y1 = problem.END
    line_u = [
        (y0 + (y1 - y0) * (x - x0) / (x1 - x0) - problem.Y_MIN)
        / (problem.Y_MAX - problem.Y_MIN)
        for x in problem.CONTROL_XS
    ]
    straight = problem.descent_time(line_u)
    print(
        f"Brachistochrone  —  n_dim={problem.N_DIM} control heights, n_trials={N_TRIALS}"
    )
    print(f"  straight-line ramp baseline: {straight:.2f}")
    print(f"{'algorithm':<24}  {'descent time':>12}")
    print("-" * 40)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rows.append((name, float(opt.best_value)))
    for name, t in sorted(rows, key=lambda r: r[1]):
        print(f"  {name:<22}  {t:>12.2f}")
    print()
    print(
        "Lower time = faster ramp. The optimum is the cycloid, not the straight line."
    )


if __name__ == "__main__":
    main()
