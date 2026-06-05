"""
Lens design across a representative slice of HumpDay's optimisers.

    python -m example_applications.lens_design.run

Watch the interpolation / local methods (PRIMA_BOBYQA, Nelder-Mead) find the
sharp-focus needle that Random Search cannot — the reverse of the high-D
genetic_art example, where those same methods come last.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "NelderMead",
    "Powell",
    "PRIMA_BOBYQA",
    "CMAEvolutionStrategy",
    "DifferentialEvolution",
    "ParticleSwarm",
    "RandomSearch",
]

N_TRIALS = 250


def main():
    print(
        f"Lens design (RMS spot size)  —  n_dim={problem.N_DIM}, "
        f"n_trials={N_TRIALS}, rays={problem.N_RAYS}"
    )
    print(f"{'algorithm':<24}  {'spot (RMS)':>11}  {'rays lost':>9}")
    print("-" * 50)

    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        rms, lost = problem.spot(opt.best_x)
        rows.append((name, rms, lost))

    for name, rms, lost in sorted(rows, key=lambda r: r[1]):
        print(f"  {name:<22}  {rms:>11.4f}  {lost:>9d}")

    print()
    print("Smaller spot = sharper focus. A perfect point approaches zero;")
    print("blind random search rarely gets below a few tenths.")


if __name__ == "__main__":
    main()
