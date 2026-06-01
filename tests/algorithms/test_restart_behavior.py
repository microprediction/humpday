"""
Focused tests for the global-behavior restart fixes added to
ParticleSwarm (SPSO-2011-style stagnation reseed), NelderMead (Kelley 1999
simplex-collapse restart, also covered in test_scipy_algorithms.py), and
CMAEvolutionStrategy (IPOP/BIPOP — added in a follow-up commit).

These verify the *restart* itself, not the optimizer's general convergence
— that's already covered by the broad-coverage tests. The point of each
test below is to construct a scenario where bare (single-trajectory) NM/
PSO/CMA would stall, and assert that the restart-equipped version
escapes.
"""

import numpy as np

from humpday.optimizers.evolutionary_algorithms import (
    CMAEvolutionStrategy,
    ParticleSwarm,
)

# -----------------------------------------------------------------------------
# ParticleSwarm — SPSO-2011 stagnation reseed
# -----------------------------------------------------------------------------


def _two_basin_1d(x):
    """Misleading shallow basin near 0.2; deeper one at 0.8."""
    x0 = float(x[0])
    shallow = 0.5 * (x0 - 0.2) ** 2 + 0.05
    deep = (x0 - 0.8) ** 2
    return min(shallow, deep)


def test_pso_stagnation_reseed_escapes_shallow_basin():
    """When the swarm initialises near the shallow basin (0.2) and runs
    out of inertia, vanilla PSO would settle there. The SPSO-2011 reseed
    of the worst half should let the swarm rediscover the deep basin
    (around 0.8) over many seeds substantially more often than chance."""
    deep_hits = 0
    for seed in range(20):
        np.random.seed(seed)
        opt = ParticleSwarm(_two_basin_1d, n_trials=400, n_dim=1)
        v, _ = opt.optimize()
        if v < 0.05:
            deep_hits += 1
    # The polish stage refines whichever basin PSO finds, so the only
    # way deep_hits jumps above bare-PSO levels is via the reseed
    # genuinely re-exploring. With a fully random initial swarm and the
    # reseed mechanism, the deep basin should win comfortably.
    assert deep_hits >= 12, (
        f"PSO restart found the deep basin in only {deep_hits}/20 runs"
    )


# -----------------------------------------------------------------------------
# CMAEvolutionStrategy — IPOP restart
# -----------------------------------------------------------------------------


def test_cma_es_ipop_escapes_local_optimum():
    """A 2D Rastrigin-style landscape with many local minima — vanilla
    CMA-ES converges to whichever basin its initial mean lands in. IPOP
    restart with growing λ should find the global minimum (near 0.5)
    substantially more often than a single-trajectory run would."""

    def rastrigin_centered(x):
        # Shift the Rastrigin function so its global minimum is at
        # 0.5 in [0, 1]^n (rather than at 0). Same multimodal structure.
        a = 10.0
        return a * len(x) + sum(
            (xi - 0.5) ** 2 - a * np.cos(2 * np.pi * (xi - 0.5)) for xi in x
        )

    near_global = 0
    for seed in range(15):
        np.random.seed(seed)
        opt = CMAEvolutionStrategy(rastrigin_centered, n_trials=600, n_dim=2)
        v, _ = opt.optimize()
        # The global minimum is 0; the next-nearest local minima sit at
        # ~1 unit of x-distance and have f ≈ 1.0 (one period of the
        # cosine ripple). Anything under 0.5 is the global basin.
        if v < 0.5:
            near_global += 1
    assert near_global >= 9, (
        f"CMA-ES IPOP found the global Rastrigin minimum in only {near_global}/15 runs"
    )


def test_cma_es_ipop_completes_on_smooth_landscape():
    """On a smooth convex landscape, IPOP shouldn't degrade — the first
    restart's converge-and-restart-and-converge sequence still ends up
    polishing into the same basin. Acts as a regression test on the
    enlarged code path."""

    def sphere(x):
        return float(sum((xi - 0.5) ** 2 for xi in x))

    np.random.seed(31)
    opt = CMAEvolutionStrategy(sphere, n_trials=400, n_dim=3)
    v, _ = opt.optimize()
    assert v < 1e-6, f"CMA-ES degraded on sphere after IPOP changes: f={v:.3e}"


def test_pso_reseed_does_not_abort_optimization():
    """The reseed branch evaluates new particles and updates
    personal_best — this test makes sure that bookkeeping doesn't
    accidentally throw or terminate the run. A successful optimize() on
    a smooth landscape with budget large enough to trigger at least one
    reseed proves the new code path is exercised cleanly."""

    def sphere(x):
        return float(sum((xi - 0.5) ** 2 for xi in x))

    np.random.seed(7)
    opt = ParticleSwarm(sphere, n_trials=500, n_dim=4)
    v, x = opt.optimize()
    # Final answer should still be near the minimum — the reseed must
    # not have undone the swarm's progress (the polish stage acts on
    # self.best_x, which is preserved across reseeds).
    assert v < 1e-3, f"PSO degraded after reseed: f={v:.3e}"
    # And the optimizer should have run a meaningful amount of work.
    assert opt.evaluations >= 200
