"""
Run a representative slice of HumpDay's optimisers on the PID tuning
problem and print a comparison table.

Invoke from the repo root:

    python -m example_applications.pid_tuning.run
"""

from __future__ import annotations

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

from . import problem

# Pick a representative slice: one local, one trust-region, two
# population-based, one Bayesian. Keep the list short so the output
# table is readable.
ALGORITHMS = [
    "NelderMead",
    "Powell",
    "PRIMA_BOBYQA",
    "DifferentialEvolution",
    "ParticleSwarm",
    "CMAEvolutionStrategy",
    "BayesianOpt",
    "SimulatedAnnealing",
]

N_TRIALS = 400


def main():
    print(f"PID controller tuning  —  n_dim={problem.N_DIM}, n_trials={N_TRIALS}")
    print(f"{'algorithm':<24}  {'best ISE':>10}  stable?     gains (Kp, Ki, Kd)")
    print("-" * 80)

    for name in ALGORITHMS:
        cls = PURE_OPTIMIZERS.get(name)
        if cls is None:
            print(f"  {name:<22}  not registered")
            continue
        opt = cls(problem.objective, n_trials=N_TRIALS, n_dim=problem.N_DIM)
        opt.optimize()
        d = problem.decode(opt.best_x)
        stab = "✓" if d["stable"] else "✗ (unstable)"
        print(
            f"  {name:<22}  {d['ISE']:>10.4f}  {stab:<10}  "
            f"({d['Kp']:.4f}, {d['Ki']:.4f}, {d['Kd']:.4f})"
        )

    print()
    print(
        "No closed-form optimum; a good tuning gives ISE ≈ 1–3 with a stable "
        "step response. Unstable gains sit on the flat penalty plateau (1e3)."
    )


if __name__ == "__main__":
    main()
