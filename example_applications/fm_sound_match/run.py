"""
FM sound-matching across a representative slice of HumpDay's optimisers.

    python -m example_applications.fm_sound_match.run

The objective is the spectral distance to a fixed target timbre; lower is a
closer match. Because the harmonic ratios are fixed the landscape is smooth, so
CMA-ES, Particle Swarm and PRIMA_BOBYQA reliably drive the error to near zero
while Nelder-Mead, Powell and Random Search settle for a rough likeness.
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

ALGORITHMS = [
    "CMAEvolutionStrategy",
    "ParticleSwarm",
    "PRIMA_BOBYQA",
    "DifferentialEvolution",
    "NelderMead",
    "Powell",
    "RandomSearch",
]

N_TRIALS = 200


def main():
    print(f"FM sound match  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'spectral error':>14}  {'knobs (I1/I2/I3/fb)':>22}")
    print("-" * 66)

    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        p = problem._decode(opt.best_x)
        knobs = f"{p['I1']:.1f}/{p['I2']:.1f}/{p['I3']:.1f}/{p['fb']:.2f}"
        rows.append((name, float(opt.best_value), knobs))

    for name, err, knobs in sorted(rows, key=lambda r: r[1]):
        print(f"  {name:<22}  {err:>14.4f}  {knobs:>22}")

    print()
    tp = problem._decode(problem.TARGET_U)
    print(
        f"Target knobs: {tp['I1']:.1f}/{tp['I2']:.1f}/{tp['I3']:.1f}/{tp['fb']:.2f}. "
        "Lower error = closer timbre; the good methods recover the target."
    )


if __name__ == "__main__":
    main()
