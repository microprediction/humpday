"""
Tuned-mass-damper design across a representative slice of HumpDay's optimisers.

    python -m example_applications.tuned_mass_damper.run

A MIXED-INTEGER problem: three continuous damper knobs plus one integer choice
(which floor). Score is the % cut in peak roof sway versus the bare tower under
the same earthquake.
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
N_TRIALS = 120


def main():
    print(
        f"Tuned mass damper  —  n_dim={problem.N_DIM} (3 continuous + 1 integer floor), "
        f"n_trials={N_TRIALS}, {problem.N} storeys"
    )
    print(f"{'algorithm':<24}  {'sway cut':>8}  {'mass% / tune / damp% / floor':>30}")
    print("-" * 70)
    rows = []
    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        cut, d = problem.run_damper(opt.best_x)
        spec = f"{d['mu'] * 100:.1f}% / {d['tuning']:.2f} / {d['zeta'] * 100:.0f}% / {d['floor']}"
        rows.append((name, cut, spec))
    for name, cut, spec in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<22}  {cut:>7.1f}%  {spec:>30}")
    print()
    print("Higher = more sway absorbed. Good designs tune near the first mode and")
    print("sit high on the tower; the integer floor makes it mixed-integer.")


if __name__ == "__main__":
    main()
