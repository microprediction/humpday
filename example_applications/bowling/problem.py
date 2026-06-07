"""
Bowling objective: maximise the chain reaction through 105 pins.

A faithful pure-Python port of the browser demo's from-scratch (no physics
engine) rigid-body simulator. A heavy ball is launched into a dense 105-pin
triangle and plows through; impulse-based ball-pin and pin-pin collisions with
restitution and friction propagate the strike. The HumpDay objective takes a 4-D
point in [0,1]^4 — ball speed, launch angle, spin (Magnus curve) and release
position — and returns the negative number of pins knocked down (with a small
near-miss term so standing-pin proximity gives a gradient).

It is the chain-reaction / sensitivity regime: tiny changes in entry angle and
spin cascade into very different pin counts, so the landscape is rough and
discontinuous. There's no analytic objective — you have to run the collision sim.

Mirrors the browser demo docs/applications/bowling.html.
"""

from __future__ import annotations

import math

N_DIM = 4
W = 800.0
BALL_RADIUS = 18.0
PIN_COLLISION_R = 9.0
MASS_BALL, MASS_PIN = 10.0, 1.0
RESTITUTION_BALL_PIN = 0.5
RESTITUTION_PIN_PIN = 0.5
PIN_FRICTION = 0.96
BALL_FRICTION = 0.991
PIN_ROWS = 14
SPACING = 24.0
HEADPIN_Y = 380.0


def _pin_starts():
    starts = []
    for row in range(PIN_ROWS):
        for i in range(row + 1):
            starts.append(
                (400 + (i - row / 2) * SPACING, HEADPIN_Y - row * SPACING * 0.866)
            )
    return starts


PIN_STARTS = _pin_starts()
TOTAL_PINS = len(PIN_STARTS)


def decode(u):
    return [14 + 9 * u[0], -8 + 16 * u[1], -2.5 + 5 * u[2], 340 + 120 * u[3]]


def simulate_throw(params):
    """Return (pins_down, trajectory) for a throw [speed, angle_deg, spin, release_x]."""
    speed, angle_deg, spin, release_x = params
    angle = angle_deg * math.pi / 180
    bx, by = release_x, 460.0
    bvx, bvy = speed * math.sin(angle), -speed * math.cos(angle)
    # parallel pin arrays
    px = [s[0] for s in PIN_STARTS]
    py = [s[1] for s in PIN_STARTS]
    pvx = [0.0] * TOTAL_PINS
    pvy = [0.0] * TOTAL_PINS
    fallen = [False] * TOTAL_PINS
    minr = BALL_RADIUS + PIN_COLLISION_R
    pin_min = 2 * PIN_COLLISION_R
    traj = [(bx, by)]

    for _ in range(320):
        ball_v2 = bvx * bvx + bvy * bvy
        moving = [
            k for k in range(TOTAL_PINS) if pvx[k] * pvx[k] + pvy[k] * pvy[k] > 0.04
        ]
        if ball_v2 < 0.04 and not moving:
            break
        if by < 30 and ball_v2 < 4 and not moving:
            break

        bx += bvx
        by += bvy
        bvx += spin * 0.012
        spin *= 0.985
        bvx *= BALL_FRICTION
        bvy *= BALL_FRICTION
        if bx < 30:
            bx, bvx = 30, bvx * -0.55
        if bx > W - 30:
            bx, bvx = W - 30, bvx * -0.55

        for k in range(TOTAL_PINS):
            if pvx[k] == 0 and pvy[k] == 0:
                continue
            px[k] += pvx[k]
            py[k] += pvy[k]
            pvx[k] *= PIN_FRICTION
            pvy[k] *= PIN_FRICTION
            if px[k] < 30:
                px[k], pvx[k] = 30, pvx[k] * -0.4
            if px[k] > W - 30:
                px[k], pvx[k] = W - 30, pvx[k] * -0.4

        # ball vs pins
        for k in range(TOTAL_PINS):
            dx, dy = px[k] - bx, py[k] - by
            d2 = dx * dx + dy * dy
            if d2 >= minr * minr or d2 < 1e-6:
                continue
            d = math.sqrt(d2)
            nx, ny = dx / d, dy / d
            v_rel = (pvx[k] - bvx) * nx + (pvy[k] - bvy) * ny
            if v_rel > 0:
                continue
            j = -(1 + RESTITUTION_BALL_PIN) * v_rel / (1 / MASS_BALL + 1 / MASS_PIN)
            bvx -= j * nx / MASS_BALL
            bvy -= j * ny / MASS_BALL
            pvx[k] += j * nx / MASS_PIN
            pvy[k] += j * ny / MASS_PIN
            overlap = minr - d + 0.5
            bx -= nx * overlap * (MASS_PIN / (MASS_BALL + MASS_PIN))
            by -= ny * overlap * (MASS_PIN / (MASS_BALL + MASS_PIN))
            px[k] += nx * overlap * (MASS_BALL / (MASS_BALL + MASS_PIN))
            py[k] += ny * overlap * (MASS_BALL / (MASS_BALL + MASS_PIN))
            fallen[k] = True

        # pin vs pin — only pairs involving a moving pin
        moving = [
            k for k in range(TOTAL_PINS) if pvx[k] * pvx[k] + pvy[k] * pvy[k] > 1e-4
        ]
        mset = set(moving)
        for a in moving:
            for b in range(TOTAL_PINS):
                if b == a or (b < a and b in mset):
                    continue  # avoid double-processing moving-moving pairs
                dx, dy = px[b] - px[a], py[b] - py[a]
                d2 = dx * dx + dy * dy
                if d2 >= pin_min * pin_min or d2 < 1e-6:
                    continue
                d = math.sqrt(d2)
                nx, ny = dx / d, dy / d
                v_rel = (pvx[b] - pvx[a]) * nx + (pvy[b] - pvy[a]) * ny
                if v_rel > 0:
                    continue
                j = -(1 + RESTITUTION_PIN_PIN) * v_rel / (2 / MASS_PIN)
                pvx[a] -= j * nx / MASS_PIN
                pvy[a] -= j * ny / MASS_PIN
                pvx[b] += j * nx / MASS_PIN
                pvy[b] += j * ny / MASS_PIN
                overlap = pin_min - d + 0.5
                px[a] -= nx * overlap * 0.5
                py[a] -= ny * overlap * 0.5
                px[b] += nx * overlap * 0.5
                py[b] += ny * overlap * 0.5
                fallen[a] = True
                fallen[b] = True
        traj.append((bx, by))

    down = 0
    for k in range(TOTAL_PINS):
        moved = math.hypot(px[k] - PIN_STARTS[k][0], py[k] - PIN_STARTS[k][1])
        if moved > 6 or fallen[k]:
            down += 1
    return down, fallen, traj


def evaluate_throw(u):
    down, _, _ = simulate_throw(decode(u))
    return down


def objective(u):
    """HumpDay objective: negative pins knocked down + small near-miss term."""
    down, fallen, traj = simulate_throw(decode(u))
    near_miss = 0.0
    for k in range(TOTAL_PINS):
        if fallen[k]:
            continue
        x0, y0 = PIN_STARTS[k]
        min_d = min(math.hypot(tx - x0, ty - y0) for tx, ty in traj[::4])
        near_miss += 1 / (1 + min_d * 0.1)
    return -down - 0.05 * near_miss
