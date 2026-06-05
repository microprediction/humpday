"""
Genetic-art image fit across a representative slice of HumpDay's optimisers.

    python -m example_applications.genetic_art.run

High-dimensional and multimodal: with the default 6 triangles this is already
60-D. Population methods (DE, CMA, Particle Swarm) and PRIMA_BOBYQA all cope,
but Nelder-Mead — which wins the low-D lens_design example — collapses here. Push
N_TRIANGLES toward the 300-D browser scale and even the interpolation methods
fall behind: the No-Free-Lunch theorem in two examples.
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
    "RandomSearch",
]

N_TRIALS = 250


def main():
    print(
        f"Genetic art  —  n_dim={problem.N_DIM} "
        f"({problem.N_TRIANGLES} triangles), n_trials={N_TRIALS}, "
        f"canvas={problem.RES}x{problem.RES}"
    )
    print(f"{'algorithm':<24}  {'RMS error':>10}  {'similarity':>10}")
    print("-" * 50)

    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rms = float(opt.best_value)
        rows.append((name, rms, problem.similarity(opt.best_x)))

    for name, rms, sim in sorted(rows, key=lambda r: r[1]):
        print(f"  {name:<22}  {rms:>10.3f}  {sim:>9.1f}%")

    print()
    print("Lower RMS / higher similarity = closer to the target image.")
    print("Note Nelder-Mead collapsing in this many dimensions, while it WINS the")
    print("low-D lens_design example — the same method, opposite verdict.")


if __name__ == "__main__":
    main()
