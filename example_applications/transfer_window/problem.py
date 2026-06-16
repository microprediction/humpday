"""
Interplanetary Transfer Window — disjoint feasible islands (the porkchop).

Pick a departure date and a time-of-flight for an Earth→Mars transfer and pay
the total Δv (departure burn + arrival burn). Solving Lambert's problem for the
transfer arc and sweeping (departure, time-of-flight) produces the famous
**porkchop plot**: total Δv as a 2-D landscape with several *disjoint* low-cost
islands — one per launch opportunity (synodic alignment) — separated by tall
ridges of impossible/expensive transfers.

What it stresses — the landscape TYPE the suite under-samples:
  - **Disjoint feasible islands / multi-basin.** The good launch windows are
    isolated valleys with high walls between them. A local or trust-region method
    converges to whichever island it started in and never sees the others; this
    is basin-hopping / CMA / population territory. (This is exactly the structure
    of ESA's GTOP / GTOPX trajectory benchmarks, here in a pure-Python reduced
    form.)

Reduced-order model: Sun-centred, both planets on coplanar circular orbits,
canonical units (μ_sun = 1, distance in AU, so Earth's period = 2π). A
universal-variable Lambert solver (bisection — no dependencies) returns the
transfer velocities; Δv is summed against the circular planet velocities.

Decision variables (mapped from [0,1]^2): departure epoch and time-of-flight.

Refs: ESA GTOP / GTOPX (arXiv 2010.07517); the porkchop plot (mission design).
*Simplified — coplanar circular orbits, single-revolution Lambert.*
"""

from __future__ import annotations

import math

N_DIM = 2

MU = 1.0  # Sun gravitational parameter (canonical units)
R_EARTH = 1.0  # AU
R_MARS = 1.5237  # AU
MARS_PHASE0 = 0.7  # Mars angular position (rad) at epoch 0

# Search ranges (canonical time units; Earth period = 2π ≈ 6.283).
DEP_MIN, DEP_MAX = 0.0, 30.0  # departure epoch (spans ~2 synodic periods)
TOF_MIN, TOF_MAX = 1.5, 9.0  # time of flight

# Returned when Lambert has no valid single-revolution transfer for that
# (departure, time-of-flight). Well above any real Δv (best ≈ 0.19) so those
# regions read as the high "walls" between the porkchop islands, but not so
# extreme that they swamp the landscape into a single flat plateau.
FAIL_DV = 10.0


def _omega(r):
    return math.sqrt(MU / r**3)


def _planet_state(r, phase0, t):
    """Position and (circular) velocity of a planet at time t."""
    th = phase0 + _omega(r) * t
    pos = (r * math.cos(th), r * math.sin(th))
    v = math.sqrt(MU / r)
    vel = (-v * math.sin(th), v * math.cos(th))
    return pos, vel


def _stumpff_C(z):
    if z > 1e-9:
        return (1.0 - math.cos(math.sqrt(z))) / z
    if z < -1e-9:
        return (math.cosh(math.sqrt(-z)) - 1.0) / (-z)
    return 0.5


def _stumpff_S(z):
    if z > 1e-9:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / sz**3
    if z < -1e-9:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / sz**3
    return 1.0 / 6.0


def _lambert(r1v, r2v, tof, prograde=True):
    """Universal-variable Lambert (Curtis Alg. 5.2), solved by bisection on z.
    Returns (v1, v2) or None if no valid single-revolution transfer."""
    r1 = math.hypot(*r1v)
    r2 = math.hypot(*r2v)
    cross_z = r1v[0] * r2v[1] - r1v[1] * r2v[0]
    dot = r1v[0] * r2v[0] + r1v[1] * r2v[1]
    dth = math.acos(max(-1.0, min(1.0, dot / (r1 * r2))))
    if prograde:
        if cross_z < 0:
            dth = 2 * math.pi - dth
    else:
        if cross_z >= 0:
            dth = 2 * math.pi - dth

    A = math.sin(dth) * math.sqrt(r1 * r2 / (1.0 - math.cos(dth)))
    if A == 0.0:
        return None

    def y(z):
        c = _stumpff_C(z)
        return r1 + r2 + A * (z * _stumpff_S(z) - 1.0) / math.sqrt(c)

    def tof_of(z):
        yz = y(z)
        if yz < 0:
            return None
        c = _stumpff_C(z)
        chi = math.sqrt(yz / c)
        return (chi**3 * _stumpff_S(z) + A * math.sqrt(yz)) / math.sqrt(MU)

    # bisection on z in (zlo, zhi): tof_of is monotincreasing in z
    zlo, zhi = -4.0 * math.pi**2, 4.0 * math.pi**2 - 1e-3
    tlo = tof_of(zlo)
    # walk zlo up until y(zlo) is valid
    tries = 0
    while tlo is None and tries < 60:
        zlo += 0.5
        tlo = tof_of(zlo)
        tries += 1
    thi = tof_of(zhi)
    if tlo is None or thi is None or not (tlo <= tof <= thi):
        return None
    for _ in range(100):
        zm = 0.5 * (zlo + zhi)
        tm = tof_of(zm)
        if tm is None:
            zlo = zm
            continue
        if tm < tof:
            zlo = zm
        else:
            zhi = zm
    z = 0.5 * (zlo + zhi)
    yz = y(z)
    if yz < 0:
        return None
    f = 1.0 - yz / r1
    g = A * math.sqrt(yz / MU)
    gdot = 1.0 - yz / r2
    if g == 0.0:
        return None
    v1 = ((r2v[0] - f * r1v[0]) / g, (r2v[1] - f * r1v[1]) / g)
    v2 = ((gdot * r2v[0] - r1v[0]) / g, (gdot * r2v[1] - r1v[1]) / g)
    return v1, v2


def _total_dv(dep, tof):
    p1, vp1 = _planet_state(R_EARTH, 0.0, dep)
    p2, vp2 = _planet_state(R_MARS, MARS_PHASE0, dep + tof)
    sol = _lambert(p1, p2, tof, prograde=True)
    if sol is None:
        return FAIL_DV
    v1, v2 = sol
    dv1 = math.hypot(v1[0] - vp1[0], v1[1] - vp1[1])
    dv2 = math.hypot(v2[0] - vp2[0], v2[1] - vp2[1])
    return dv1 + dv2


def objective(u):
    """HumpDay-style objective: `u ∈ [0,1]^2` -> total Δv (departure + arrival
    burns) for the Earth→Mars transfer at that (departure, time-of-flight)."""
    dep = DEP_MIN + (DEP_MAX - DEP_MIN) * u[0]
    tof = TOF_MIN + (TOF_MAX - TOF_MIN) * u[1]
    return _total_dv(dep, tof)


def decode(u):
    """Convenience: departure, time-of-flight and Δv for a `[0,1]^2` point."""
    dep = DEP_MIN + (DEP_MAX - DEP_MIN) * u[0]
    tof = TOF_MIN + (TOF_MAX - TOF_MIN) * u[1]
    return {"departure": dep, "time_of_flight": tof, "delta_v": _total_dv(dep, tof)}
