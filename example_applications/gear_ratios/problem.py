"""
Gear Train Design — a discrete, staircase landscape.

Choose the tooth counts of four gears so the compound train realises a target
ratio as closely as possible. The classic Sandgren (1990) formulation:

    minimise  ( 1/6.931  -  (z1 * z2) / (z3 * z4) )^2

with each tooth count z_i an integer in [12, 60]. The target 1/6.931 ≈ 0.144279
is irrational-ish for integer gears, so the best achievable ratio is a rational
near-miss.

What it stresses — a landscape TYPE the rest of the suite lacks:
  - **Discreteness / plateaus.** The objective depends only on the *rounded*
    integer tooth counts, so it is piecewise-constant: flat over every cell of
    the integer lattice, with no gradient to follow. Local / trust-region
    methods that assume smoothness stall on a plateau; only methods that hop
    around the lattice (population, annealing, random restart) make progress.
  - A vast number of near-ties: many distinct tooth combinations give almost the
    same ratio, so the global optimum is a needle.

Known best-known optimum: z = (19, 16, 43, 49) (and permutations / the swap
z1<->z2, z3<->z4), giving ratio 304/2107 ≈ 0.1442809 and f ≈ 2.7e-12.

Refs: Sandgren (1990), "Nonlinear integer and discrete programming in
mechanical design optimization"; a standard mixed-integer DFO benchmark.
"""

from __future__ import annotations

N_DIM = 4

TEETH_MIN = 12
TEETH_MAX = 60
TARGET = 1.0 / 6.931  # desired gear ratio


def _teeth(u):
    """Map [0,1]^4 to four integer tooth counts in [TEETH_MIN, TEETH_MAX]."""
    span = TEETH_MAX - TEETH_MIN
    z = [int(round(TEETH_MIN + span * ui)) for ui in u]
    return [min(TEETH_MAX, max(TEETH_MIN, zi)) for zi in z]


def objective(u):
    """HumpDay-style objective: `u ∈ [0,1]^4` -> squared error between the
    realised compound gear ratio and the target. Piecewise-constant on the
    integer lattice."""
    z1, z2, z3, z4 = _teeth(u)
    ratio = (z1 * z2) / (z3 * z4)
    return (TARGET - ratio) ** 2


def decode(u):
    """Convenience: tooth counts, realised ratio, and error for a `[0,1]^4` point."""
    z1, z2, z3, z4 = _teeth(u)
    ratio = (z1 * z2) / (z3 * z4)
    return {
        "teeth": (z1, z2, z3, z4),
        "ratio": ratio,
        "target": TARGET,
        "abs_error": abs(TARGET - ratio),
        "sq_error": (TARGET - ratio) ** 2,
    }
