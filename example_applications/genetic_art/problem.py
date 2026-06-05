"""
Genetic-art objective: approximate a picture with translucent triangles.

A pure-Python software rasteriser. Each triangle has 10 genes (3 vertices, RGB,
alpha), so N_TRIANGLES triangles is a 10*N_TRIANGLES-dimensional search. The
HumpDay objective takes a point in [0,1]^(10*N_TRIANGLES), paints the triangles
back-to-front onto a small RES x RES canvas, and returns the **pixel RMS error**
to a fixed target image. Minimising it makes the painting resemble the target.

This is the high-dimensional, multimodal end of the spectrum: with the default
6 triangles it is already 60-D, and countless arrangements score about the same.
Population methods (Differential Evolution, CMA, Particle Swarm) shine here;
local / interpolation methods that ace the low-D lens_design example fall to the
bottom. CMA-ES in particular pays an n-squared covariance cost that makes the
full 300-D browser version impractical — try raising N_TRIANGLES to feel it.

Mirrors the browser demo docs/applications/genetic-art.html.
"""

from __future__ import annotations

import math

RES = 40
N_TRIANGLES = 6
GENES = 10
N_DIM = N_TRIANGLES * GENES


def _blank():
    return [0.0] * (RES * RES * 3)


def _fill_tri(buf, x1, y1, x2, y2, x3, y3, r, g, b, a):
    min_x = max(0, math.floor(min(x1, x2, x3)))
    max_x = min(RES - 1, math.ceil(max(x1, x2, x3)))
    min_y = max(0, math.floor(min(y1, y2, y3)))
    max_y = min(RES - 1, math.ceil(max(y1, y2, y3)))
    if max_x < min_x or max_y < min_y:
        return
    d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(d) < 1e-9:
        return
    ia = 1 - a
    for py in range(min_y, max_y + 1):
        fy = py + 0.5
        for px in range(min_x, max_x + 1):
            fx = px + 0.5
            l1 = ((y2 - y3) * (fx - x3) + (x3 - x2) * (fy - y3)) / d
            l2 = ((y3 - y1) * (fx - x3) + (x1 - x3) * (fy - y3)) / d
            if l1 < 0 or l2 < 0 or (1 - l1 - l2) < 0:
                continue
            o = (py * RES + px) * 3
            buf[o] = buf[o] * ia + r * a
            buf[o + 1] = buf[o + 1] * ia + g * a
            buf[o + 2] = buf[o + 2] * ia + b * a


def render(u):
    buf = _blank()
    for k in range(N_TRIANGLES):
        o = k * GENES

        def X(v):
            return (v * 1.3 - 0.15) * RES

        _fill_tri(
            buf,
            X(u[o]), X(u[o + 1]), X(u[o + 2]), X(u[o + 3]), X(u[o + 4]), X(u[o + 5]),
            u[o + 6] * 255, u[o + 7] * 255, u[o + 8] * 255, 0.30 + u[o + 9] * 0.65,
        )
    return buf


def _make_target():
    """A full-bleed low-poly scene: sky gradient, a bright peak, a dark base."""
    buf = _blank()
    for y in range(RES):
        t = y / RES
        if t < 0.55:
            v = t / 0.55
            r, g, b = 70 + 120 * v, 110 + 120 * v, 200 - 30 * v
        else:
            r, g, b = 40, 46, 60
        for x in range(RES):
            o = (y * RES + x) * 3
            buf[o], buf[o + 1], buf[o + 2] = r, g, b
    _fill_tri(buf, 6, 30, 20, 10, 34, 30, 235, 235, 245, 1.0)
    for y in range(30, RES):
        for x in range(RES):
            o = (y * RES + x) * 3
            buf[o], buf[o + 1], buf[o + 2] = 30, 46, 40
    return buf


TARGET = _make_target()


def similarity(u):
    """Percentage similarity to the target (100 = identical)."""
    return 100.0 * (1.0 - objective(u) / 255.0)


def objective(u):
    """HumpDay objective: pixel RMS error to the target image (minimise)."""
    im = render(u)
    s = 0.0
    for i in range(RES * RES * 3):
        d = im[i] - TARGET[i]
        s += d * d
    return math.sqrt(s / (RES * RES * 3))
