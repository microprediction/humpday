"""
Bridge-truss objective: size the members for the lightest safe bridge.

A pure-Python truss finite-element model. A 6-node, 10-member truss spans two
supports (a pin and a roller) with a midspan load. The HumpDay objective takes a
10-D point in [0,1]^10 — the cross-sectional area of each member, on a log scale —
solves the linear statics for member forces, and returns the **structural weight
plus a penalty** for any member that yields (overstress) or buckles (Euler, in
compression). Minimising it finds the lightest truss that stays safe.

This is the classic constrained structural-optimisation problem: bigger members
are heavier but stronger, the X-braced bay is statically indeterminate (so force
flow shifts as you resize), and the feasible region is bounded by stress and
buckling cliffs the optimiser must descend without falling off. Uses numpy for
the FEM solve.

Mirrors the browser demo docs/applications/bridge-truss.html.
"""

from __future__ import annotations

import math

import numpy as np

NODES = [
    (0.0, 0.0, "pin"),
    (2.0, 0.0, None),
    (4.0, 0.0, "load"),
    (6.0, 0.0, "roller"),
    (2.0, 1.5, None),
    (4.0, 1.5, None),
]
MEMBERS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (0, 4),
    (3, 5),
    (1, 4),
    (2, 5),
    (1, 5),
    (2, 4),
]
MEMBER_COUNT = len(MEMBERS)
N_DIM = MEMBER_COUNT
E = 200e9  # Young's modulus, Pa
RHO = 7850.0  # density, kg/m^3
YIELD = 250e6  # yield stress, Pa
LOAD = 20e3  # midspan load, N
SECTION_K = 0.3  # shape factor for the buckling formula
A_MIN, A_MAX = 50e-6, 3000e-6
PENALTY = 5e3


def decode(u):
    return [A_MIN * (A_MAX / A_MIN) ** ui for ui in u]


def _mem_info():
    info = []
    for a, b in MEMBERS:
        dx = NODES[b][0] - NODES[a][0]
        dy = NODES[b][1] - NODES[a][1]
        L = math.hypot(dx, dy)
        info.append((a, b, L, dx / L, dy / L))
    return info


_INFO = _mem_info()


def _solve(areas):
    """Return member axial forces, or None if the truss is singular."""
    n = len(NODES)
    ndof = 2 * n
    K = np.zeros((ndof, ndof))
    for i, (a, b, L, c, s) in enumerate(_INFO):
        k = E * areas[i] / L
        c2, s2, cs = c * c, s * s, c * s
        ke = k * np.array(
            [
                [c2, cs, -c2, -cs],
                [cs, s2, -cs, -s2],
                [-c2, -cs, c2, cs],
                [-cs, -s2, cs, s2],
            ]
        )
        dofs = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]
        for ii in range(4):
            for jj in range(4):
                K[dofs[ii], dofs[jj]] += ke[ii, jj]
    F = np.zeros(ndof)
    for i, node in enumerate(NODES):
        if node[2] == "load":
            F[2 * i + 1] -= LOAD
    fixed = set()
    for i, node in enumerate(NODES):
        if node[2] == "pin":
            fixed.add(2 * i)
            fixed.add(2 * i + 1)
        elif node[2] == "roller":
            fixed.add(2 * i + 1)
    free = [i for i in range(ndof) if i not in fixed]
    Kff = K[np.ix_(free, free)]
    try:
        uf = np.linalg.solve(Kff, F[free])
    except np.linalg.LinAlgError:
        return None
    u = np.zeros(ndof)
    u[free] = uf
    forces = []
    for i, (a, b, L, c, s) in enumerate(_INFO):
        elong = (u[2 * b] - u[2 * a]) * c + (u[2 * b + 1] - u[2 * a + 1]) * s
        forces.append((E * areas[i] / L) * elong)
    return forces


def evaluate_design(u):
    """Return (total, weight, n_violated, feasible) for member areas in [0,1]^10."""
    areas = decode(u)
    forces = _solve(areas)
    if forces is None or any(not math.isfinite(f) for f in forces):
        return math.inf, math.inf, MEMBER_COUNT, False
    weight = sum(areas[i] * _INFO[i][2] * RHO for i in range(MEMBER_COUNT))
    penalty = 0.0
    n_violated = 0
    for i in range(MEMBER_COUNT):
        f = forces[i]
        sigma = f / areas[i]
        yv = (abs(sigma) - YIELD) / YIELD
        if yv > 0:
            n_violated += 1
            penalty += PENALTY * yv * yv
        if f < 0:  # compression: Euler buckling
            L = _INFO[i][2]
            pc = math.pi**2 * E * SECTION_K * areas[i] ** 2 / (L * L)
            bv = (-f - pc) / pc
            if bv > 0:
                n_violated += 1
                penalty += PENALTY * bv * bv
    return weight + penalty, weight, n_violated, n_violated == 0


def objective(u):
    """HumpDay objective: structural weight + constraint penalty (minimise)."""
    return evaluate_design(u)[0]
