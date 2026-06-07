"""
Bridge-truss member sizing across a representative slice of HumpDay's optimisers.

    python -m example_applications.bridge_truss.run

Size 10 members for the lightest truss that doesn't yield or buckle under a
midspan load. The objective is weight plus a penalty for any unsafe member, so
the optimiser must sit right on the constraint boundary.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "DifferentialEvolution",
    "CMAEvolutionStrategy",
    "PRIMA_BOBYQA",
    "NelderMead",
    "Powell",
    "ParticleSwarm",
    "RandomSearch",
]
N_TRIALS = 300


def main():
    print(f"Bridge truss sizing  —  n_dim={problem.N_DIM} members, n_trials={N_TRIALS}")
    print(
        f"{'algorithm':<24}  {'weight (kg)':>11}  {'feasible?':>9}  {'unsafe members':>14}"
    )
    print("-" * 66)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        total, weight, nv, feasible = problem.evaluate_design(opt.best_x)
        rows.append((name, weight, feasible, nv))
    rows.sort(key=lambda r: (not r[2], r[1]))  # feasible first, then lightest
    for name, weight, feasible, nv in rows:
        w = f"{weight:.1f}" if feasible else "—"
        print(f"  {name:<24.24}{w:>11}  {'yes' if feasible else 'no':>9}  {nv:>14d}")
    print()
    print("Lightest feasible truss wins. The X-braced bay is indeterminate, so")
    print("force flow shifts as members resize — yield + buckling cliffs bound it.")


if __name__ == "__main__":
    main()
