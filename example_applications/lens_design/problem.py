"""
Lens-design objective: focus a collimated beam to the tightest possible spot.

A pure-Python 2-D sequential ray trace. A bundle of parallel rays enters from
the left and refracts (Snell's law) at four spherical surfaces forming two glass
elements, then propagates to a fixed focal plane. The HumpDay objective takes a
4-D point in [0,1]^4, stretches it into the four surface curvatures, traces the
rays, and returns the **RMS spot size** on the focal plane (plus a small penalty
for rays lost off the aperture). Minimising it sharpens the focus.

This is the textbook example of an "easy to state, brutal to optimise" problem:
real Snell refraction produces spherical aberration, so the sharp-focus designs
sit in a narrow needle surrounded by blur. Random search basically can't find
it; interpolation / local methods (PRIMA_BOBYQA, Nelder-Mead) excel — the
mirror image of the high-dimensional examples where they fail.

Mirrors the browser demo docs/applications/lens.html.
"""

from __future__ import annotations

import math

N_DIM = 4
NG = 1.5  # glass refractive index
SURF_X = (3.0, 4.2, 6.0, 7.2)  # vertex x of each surface
GLASS = (True, False, True, False)  # entering glass after surface i?
FOCAL = 15.0  # focal plane x
APERTURE = 1.6  # half-aperture (max ray height)
N_RAYS = 21
CMAX = 0.9  # curvature magnitude bound (1/R)


def _scale(u):
    """[0,1]^4 -> signed curvatures in [-CMAX, CMAX]^4."""
    return [(2.0 * ui - 1.0) * CMAX for ui in u]


def _refract(dx, dy, nx, ny, ratio):
    """Vector Snell refraction; returns a unit direction or None on TIR."""
    cosi = -(dx * nx + dy * ny)
    if cosi < 0:
        nx, ny, cosi = -nx, -ny, -cosi
    sin2t = ratio * ratio * (1.0 - cosi * cosi)
    if sin2t > 1.0:
        return None
    ct = math.sqrt(1.0 - sin2t)
    rx = ratio * dx + (ratio * cosi - ct) * nx
    ry = ratio * dy + (ratio * cosi - ct) * ny
    m = math.hypot(rx, ry)
    return (rx / m, ry / m)


def _hit_surface(px, py, dx, dy, xv, c):
    """Intersect a ray with a spherical surface (vertex (xv,0), curvature c)."""
    if abs(c) < 1e-6:  # flat plane x = xv
        if abs(dx) < 1e-9:
            return None
        t = (xv - px) / dx
        if t <= 1e-7:
            return None
        return (xv, py + t * dy, -1.0, 0.0)
    r = 1.0 / c
    cx = xv + r
    ox, oy = px - cx, py
    b = 2.0 * (ox * dx + oy * dy)
    cc = ox * ox + oy * oy - r * r
    disc = b * b - 4.0 * cc
    if disc < 0:
        return None
    sd = math.sqrt(disc)
    t = math.inf
    for tc in ((-b - sd) / 2.0, (-b + sd) / 2.0):
        if tc > 1e-7:
            hx = px + tc * dx
            if abs(hx - xv) < abs(r) + APERTURE + 2 and tc < t:
                t = tc
    if not math.isfinite(t):
        return None
    hx, hy = px + t * dx, py + t * dy
    return (hx, hy, (hx - cx) / r, hy / r)


def _trace_ray(y0, curv):
    """Trace one ray of initial height y0; return focal-plane height or None."""
    px, py, dx, dy = 0.0, y0, 1.0, 0.0
    for i in range(4):
        h = _hit_surface(px, py, dx, dy, SURF_X[i], curv[i])
        if h is None or abs(h[1]) > APERTURE + 0.4:
            return None
        n1 = 1.0 if i == 0 else (NG if GLASS[i - 1] else 1.0)
        n2 = NG if GLASS[i] else 1.0
        r = _refract(dx, dy, h[2], h[3], n1 / n2)
        if r is None:
            return None
        px, py, dx, dy = h[0], h[1], r[0], r[1]
    if dx <= 1e-6:
        return None
    t = (FOCAL - px) / dx
    if t <= 0:
        return None
    return py + t * dy


def spot(u):
    """Return (rms_spot, n_lost) for a design in [0,1]^4."""
    curv = _scale(u)
    ys = []
    lost = 0
    for j in range(N_RAYS):
        y0 = -APERTURE + 2 * APERTURE * j / (N_RAYS - 1)
        yf = _trace_ray(y0, curv)
        if yf is None:
            lost += 1
        else:
            ys.append(yf)
    if len(ys) < 3:
        return 1e3, lost
    mean = sum(ys) / len(ys)
    rms = math.sqrt(sum((y - mean) ** 2 for y in ys) / len(ys))
    return rms, lost


def objective(u):
    """HumpDay objective: RMS spot size + lost-ray penalty (minimise)."""
    rms, lost = spot(u)
    return rms + 0.15 * lost
