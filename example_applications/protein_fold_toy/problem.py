"""
Toy protein folding: the 2D AB off-lattice model.

A chain of thirteen residues, each hydrophobic (A) or polar (B) in a fixed sequence,
folds in the plane. The shape is set by the bend angle at each interior residue. The
energy has a backbone-bending term plus a Lennard-Jones interaction between every
non-bonded pair, where the well depth depends on the residue types: hydrophobic pairs
attract strongly, polar pairs weakly, and mixed pairs repel. Minimising it collapses the
chain into a compact hydrophobic core, the textbook extreme-multimodality benchmark
(Stillinger's AB model).

The HumpDay objective takes an 11-D point in [0,1]^11 (bend angles, mapped to
[-pi, pi]) and returns the total energy.
"""
from __future__ import annotations

import math

N_BEADS = 13
N_DIM = N_BEADS - 2
# 1 = hydrophobic (A), 0 = polar (B)
SEQ = (1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0)
MIN_DISTANCE = 0.5


def _coeff(a, b):
    # Stillinger AB model: AA attractive (1), BB weak (0.5), AB repulsive (-0.5)
    if a == 1 and b == 1:
        return 1.0
    if a == 0 and b == 0:
        return 0.5
    return -0.5


def decode(u):
    return [(min(1.0, max(0.0, v)) - 0.5) * 2 * math.pi for v in u]


def _positions(angles):
    pos = [(0.0, 0.0), (1.0, 0.0)]
    phi = 0.0
    for k in range(len(angles)):
        phi += angles[k]
        x, y = pos[-1]
        pos.append((x + math.cos(phi), y + math.sin(phi)))
    return pos


def objective(u):
    angles = decode(u)
    pos = _positions(angles)
    energy = 0.0
    for a in angles:
        energy += 0.25 * (1.0 - math.cos(a))   # backbone bending
    for i in range(N_BEADS):
        xi, yi = pos[i]
        for j in range(i + 2, N_BEADS):
            xj, yj = pos[j]
            r = math.hypot(xi - xj, yi - yj)
            if r < MIN_DISTANCE:
                r = MIN_DISTANCE
            inv6 = r ** -6
            energy += 4.0 * (inv6 * inv6 - _coeff(SEQ[i], SEQ[j]) * inv6)
    return energy
