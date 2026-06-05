"""
Walking-creature gait search across a representative slice of HumpDay's optimisers.

    python -m example_applications.walking_creature.run

The score is the distance walked in 8 seconds (body-lengths). Watch the
phase-offset column converge near 0.50 (half a cycle) — the optimiser
rediscovering that legs should alternate — and note that PRIMA_NEWUOA can stall
near its start and never learn to walk.
"""

from __future__ import annotations

import math

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "DifferentialEvolution",
    "ParticleSwarm",
    "CMAEvolutionStrategy",
    "PRIMA_BOBYQA",
    "PRIMA_NEWUOA",
    "NelderMead",
    "RandomSearch",
]

N_TRIALS = 200


def main():
    print(
        f"Walking creature  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}, "
        f"sim={problem.T:.0f}s"
    )
    print(f"{'algorithm':<24}  {'distance':>9}  {'fell?':>5}  {'phase/pi':>8}")
    print("-" * 56)

    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        dist, fell = problem.simulate(opt.best_x)
        phase = opt.best_x[2] * 2.0  # u in [0,1] -> dphi in [0,2pi], report in units of pi
        rows.append((name, dist, fell, phase))

    for name, dist, fell, phase in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<22}  {dist:>9.1f}  {'yes' if fell else 'no':>5}  {phase:>7.2f}π")

    print()
    print("A flailing creature scores near zero; a good alternating gait covers")
    print("tens of body-lengths. The winning gait is ~half a cycle out of phase.")


if __name__ == "__main__":
    main()
