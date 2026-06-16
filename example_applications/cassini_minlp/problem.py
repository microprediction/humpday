"""
Cassini-style MGA trajectory — mixed-integer flyby-sequence selection.

Fly from Earth to Saturn using four gravity-assist flybys, paying the total
Δv (launch burn + a powered-flyby correction at each swing-by + Saturn arrival
burn). You choose BOTH:

  - 6 continuous variables: the launch epoch and the five leg times-of-flight;
  - 4 discrete variables: which planet to swing by at each of the four flybys
    (each chosen from a candidate set).

This mirrors ESA's Cassini1-MINLP benchmark (GTOPX): the continuous timing is
multimodal (launch windows recur), and on top of it sits a *combinatorial*
choice of flyby sequence — so the landscape is mixed-integer with many
near-tied sequences. Picking the wrong planet sequence strands you in a costly
basin no amount of timing tweaks can fix.

What it stresses — pathologies the suite under-samples:
  - **Mixed-integer** (4 discrete planet choices + 6 continuous), like
    tuned_mass_damper but combinatorial in several slots.
  - **Multimodal with deceptive near-ties** between distinct flyby sequences:
    different integer choices reach almost the same Δv, so a search can't tell
    which sequence is truly best from a few samples.

Reduced-order model: Sun-centred, all bodies on coplanar circular orbits,
canonical units (μ_sun = 1, AU, Earth period 2π). A universal-variable Lambert
solver (bisection, no deps) links the legs; each flyby is modelled as a powered
swing-by (Δv = relative-speed mismatch + a soft turn-angle penalty).

*Simplified — coplanar circular orbits, single-revolution Lambert, idealised
gravity assists. Not a numeric replica of GTOPX Cassini1.*

Refs: ESA GTOP / GTOPX Cassini1-MINLP (arXiv 2010.07517).
"""

from __future__ import annotations

import math

MU = 1.0  # Sun (canonical units)

# (name, orbit radius in AU, phase at epoch 0 in rad)
BODIES = {
    "Mercury": (0.387, 1.10),
    "Venus": (0.723, 2.40),
    "Earth": (1.000, 0.00),
    "Mars": (1.524, 4.10),
    "Jupiter": (5.203, 0.90),
    "Saturn": (9.537, 3.30),
}

START = "Earth"
TARGET = "Saturn"
# Candidate bodies for each of the four flyby slots (the discrete choices).
FLYBY_CANDIDATES = ("Venus", "Earth", "Mars", "Jupiter")
N_FLYBY = 4
N_LEG = N_FLYBY + 1  # Earth -> f1 -> f2 -> f3 -> f4 -> Saturn

# Continuous-variable ranges (canonical time; Earth period ≈ 6.283).
DEP_MIN, DEP_MAX = 0.0, 13.0
TOF_MIN, TOF_MAX = 0.8, 14.0

N_DIM = 6 + N_FLYBY  # 6 continuous (launch + 5 ToF) + 4 discrete flyby planets

TURN_MAX = 1.3  # max "free" gravity-assist bend (rad); excess is penalised
TURN_PENALTY = 0.6
FAIL_DV = 50.0


def _omega(r):
    return math.sqrt(MU / r**3)


def _state(name, t):
    r, ph0 = BODIES[name]
    th = ph0 + _omega(r) * t
    pos = (r * math.cos(th), r * math.sin(th))
    v = math.sqrt(MU / r)
    vel = (-v * math.sin(th), v * math.cos(th))
    return pos, vel


def _C(z):
    if z > 1e-9:
        return (1.0 - math.cos(math.sqrt(z))) / z
    if z < -1e-9:
        return (math.cosh(math.sqrt(-z)) - 1.0) / (-z)
    return 0.5


def _S(z):
    if z > 1e-9:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / sz**3
    if z < -1e-9:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / sz**3
    return 1.0 / 6.0


def _lambert(r1v, r2v, tof):
    """Prograde universal-variable Lambert (bisection). Returns (v1, v2) or None."""
    r1 = math.hypot(*r1v)
    r2 = math.hypot(*r2v)
    cross_z = r1v[0] * r2v[1] - r1v[1] * r2v[0]
    dot = r1v[0] * r2v[0] + r1v[1] * r2v[1]
    dth = math.acos(max(-1.0, min(1.0, dot / (r1 * r2))))
    if cross_z < 0:
        dth = 2 * math.pi - dth
    denom = 1.0 - math.cos(dth)
    if denom <= 1e-12:
        return None
    A = math.sin(dth) * math.sqrt(r1 * r2 / denom)
    if A == 0.0:
        return None

    def yv(z):
        return r1 + r2 + A * (z * _S(z) - 1.0) / math.sqrt(_C(z))

    def tof_of(z):
        y = yv(z)
        if y < 0:
            return None
        chi = math.sqrt(y / _C(z))
        return (chi**3 * _S(z) + A * math.sqrt(y)) / math.sqrt(MU)

    zlo, zhi = -4.0 * math.pi**2, 4.0 * math.pi**2 - 1e-3
    tlo = tof_of(zlo)
    tries = 0
    while tlo is None and tries < 80:
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
    y = yv(z)
    if y < 0:
        return None
    f = 1.0 - y / r1
    g = A * math.sqrt(y / MU)
    gdot = 1.0 - y / r2
    if g == 0.0:
        return None
    v1 = ((r2v[0] - f * r1v[0]) / g, (r2v[1] - f * r1v[1]) / g)
    v2 = ((gdot * r2v[0] - r1v[0]) / g, (gdot * r2v[1] - r1v[1]) / g)
    return v1, v2


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _mag(a):
    return math.hypot(a[0], a[1])


def _angle(a, b):
    na, nb = _mag(a), _mag(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return math.acos(max(-1.0, min(1.0, c)))


def _decode(u):
    """Map [0,1]^10 to (launch epoch, [5 ToF], [4 flyby planet names])."""
    dep = DEP_MIN + (DEP_MAX - DEP_MIN) * u[0]
    tof = [TOF_MIN + (TOF_MAX - TOF_MIN) * u[1 + k] for k in range(N_LEG)]
    k = len(FLYBY_CANDIDATES)
    flybys = [FLYBY_CANDIDATES[min(k - 1, int(u[6 + j] * k))] for j in range(N_FLYBY)]
    return dep, tof, flybys


def _total_dv(dep, tof, flybys):
    seq = [START] + flybys + [TARGET]
    epochs = [dep]
    for t in tof:
        epochs.append(epochs[-1] + t)
    # leg Lambert solutions
    legs = []
    for i in range(N_LEG):
        p1, _ = _state(seq[i], epochs[i])
        p2, _ = _state(seq[i + 1], epochs[i + 1])
        sol = _lambert(p1, p2, tof[i])
        if sol is None:
            return FAIL_DV
        legs.append(sol)

    # launch burn (relative to Earth)
    _, vE = _state(START, epochs[0])
    dv = _mag(_sub(legs[0][0], vE))
    # powered flybys at the intermediate bodies
    for j in range(N_FLYBY):
        _, vb = _state(flybys[j], epochs[j + 1])
        rel_in = _sub(legs[j][1], vb)
        rel_out = _sub(legs[j + 1][0], vb)
        dv += abs(_mag(rel_out) - _mag(rel_in))
        turn = _angle(rel_in, rel_out)
        if turn > TURN_MAX:
            dv += TURN_PENALTY * (turn - TURN_MAX)
    # arrival burn (relative to Saturn)
    _, vS = _state(TARGET, epochs[-1])
    dv += _mag(_sub(legs[-1][1], vS))
    return dv


def objective(u):
    """HumpDay-style objective: `u ∈ [0,1]^10` -> total mission Δv for the
    chosen launch epoch, leg times, and flyby planet sequence."""
    dep, tof, flybys = _decode(u)
    return _total_dv(dep, tof, flybys)


def decode(u):
    """Convenience: launch, leg times, flyby sequence, and Δv for a `[0,1]^10` point."""
    dep, tof, flybys = _decode(u)
    return {
        "launch": dep,
        "tof": tof,
        "sequence": [START] + flybys + [TARGET],
        "delta_v": _total_dv(dep, tof, flybys),
    }
