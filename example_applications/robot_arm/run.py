"""
Robot-arm reaching across a representative slice of HumpDay's optimisers.

    python -m example_applications.robot_arm.run

Find six joint angles that put the tip on the target while keeping every link
clear of three obstacles. Score ≈ 100 means the tip touches the target with no
collision; negative means a link is crashing through an obstacle.
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
    print(f"Robot arm reach  —  n_dim={problem.N_DIM} joints, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'score':>7}  {'tip err (px)':>12}  {'collisions':>10}")
    print("-" * 60)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        score, tip_err, coll = problem.evaluate_pose(opt.best_x)
        rows.append((name, score, tip_err, coll))
    for name, score, tip_err, coll in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<24.24}{score:>7.1f}  {tip_err:>12.1f}  {coll:>10d}")
    print()
    print("Score ~100 = tip on target, no collision. Constrained IK with narrow")
    print("collision-free corridors and disjoint elbow-up/down/wrap solutions.")


if __name__ == "__main__":
    main()
