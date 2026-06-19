"""
Lennard-Jones Cluster (N = 5) — a canonical extreme-multimodality benchmark.

Place N = 5 atoms in 3-D space and minimise the total Lennard-Jones
pair potential

    E = sum_{i<j} 4 * (r_ij**-12 - r_ij**-6),    r_ij = |x_i - x_j|.

Variables: the 3N = 15 atom coordinates. HumpDay hands us a point in the
unit hypercube [0,1]^15, which we map coordinate-wise to a cubic box
[0.0, 3.0]^3. So N_DIM = 15.

To keep the objective finite when two atoms nearly coincide, each
pairwise distance is clamped to a minimum of 0.1 before the inverse
powers are taken. Overlapping atoms therefore produce a very large but
finite positive energy rather than an overflow.

Pathology: the number of distinct local minima of the LJ potential grows
roughly exponentially with N, making this the textbook hard multimodal
global-optimisation landscape. There is also gauge freedom — any
translation or rotation of a configuration leaves the energy unchanged,
so every optimum is really a continuum of equivalent optima. Optimisers
must escape an enormous forest of funnels to find the basin of the true
ground state.

Known global minimum (N = 5): the lowest-energy configuration is a
triangular bipyramid with E* ≈ -9.103852. Reaching it within a few
hundred evaluations is genuinely difficult; that difficulty is the point.
"""

from __future__ import annotations

import math

N_ATOMS = 5
N_DIM = 3 * N_ATOMS

# Each coordinate is mapped from [0,1] into this box (in LJ length units).
BOX_LOW = 0.0
BOX_HIGH = 3.0

# Floor on pairwise distance to keep r**-12 finite for overlapping atoms.
MIN_DISTANCE = 0.1


def _scale_unit_to_box(u):
    """Map a point in [0,1]^15 to a list of 5 (x, y, z) atom positions."""
    coords = [BOX_LOW + (BOX_HIGH - BOX_LOW) * v for v in u]
    return [tuple(coords[3 * i : 3 * i + 3]) for i in range(N_ATOMS)]


def _energy(atoms):
    """Total Lennard-Jones potential energy of a list of (x, y, z) atoms."""
    total = 0.0
    for i in range(N_ATOMS):
        xi, yi, zi = atoms[i]
        for j in range(i + 1, N_ATOMS):
            xj, yj, zj = atoms[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < MIN_DISTANCE:
                r = MIN_DISTANCE
            inv6 = r**-6
            total += 4.0 * (inv6 * inv6 - inv6)
    return total


def objective(u):
    """HumpDay-style objective: input is `u ∈ [0,1]^15`, output is the
    Lennard-Jones energy (to minimise). Good clusters give negative
    energies; the N=5 ground state is ≈ -9.103852."""
    atoms = _scale_unit_to_box(u)
    return _energy(atoms)


def decode(u):
    """Convenience: return the 5 atom positions and the cluster energy
    for a `[0,1]^15` point."""
    atoms = _scale_unit_to_box(u)
    return {
        "atoms": atoms,
        "energy": _energy(atoms),
    }


# --- Faithful high-dimensional variants -------------------------------------
# Scaling knob: number of atoms (n_dim = 3 * atoms). The pair potential and the
# distance floor are identical; the cubic box side is grown as (atoms/5)**(1/3)
# so the *number density* is held constant. Without this the box would not hold
# the extra atoms at LJ-equilibrium spacing and the problem would degenerate into
# pure overlap-energy. With it, larger n is a faithful larger-cluster instance —
# and the count of local minima still grows ~exponentially, so it stays the
# textbook hard multimodal landscape.
SCALABLE_DIMS = [30, 60, 90]


def make_objective(n_dim):
    """Return a HumpDay objective for an `n_dim // 3`-atom LJ cluster at the
    same number density as the N=5 base."""
    assert n_dim % 3 == 0, "LJ n_dim must be a multiple of 3"
    n_atoms = n_dim // 3
    box_high = BOX_LOW + (BOX_HIGH - BOX_LOW) * (n_atoms / N_ATOMS) ** (1.0 / 3.0)

    def objective_scaled(u):
        coords = [BOX_LOW + (box_high - BOX_LOW) * v for v in u]
        atoms = [tuple(coords[3 * i : 3 * i + 3]) for i in range(n_atoms)]
        total = 0.0
        for i in range(n_atoms):
            xi, yi, zi = atoms[i]
            for j in range(i + 1, n_atoms):
                xj, yj, zj = atoms[j]
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                r = math.sqrt(dx * dx + dy * dy + dz * dz)
                if r < MIN_DISTANCE:
                    r = MIN_DISTANCE
                inv6 = r**-6
                total += 4.0 * (inv6 * inv6 - inv6)
        return total

    return objective_scaled
