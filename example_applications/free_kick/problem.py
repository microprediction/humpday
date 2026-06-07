"""
Free-kick objective: bend the ball around the wall and past the keeper.

A pure-Python 3-D ball-flight simulator (no physics engine). The HumpDay
objective takes a 4-D point in [0,1]^4 — aim, loft, power and curve (sidespin) —
kicks the ball from a fixed spot, and integrates its flight at 120 Hz under
gravity, linear drag and a Magnus side-force from the spin. A defensive wall and
an actively diving goalkeeper stand in the way. It returns the negative score: a
clean goal scores ~100+, a wall-block or a save scores low.

This is the iconic "bend it" optimisation: the ball must rise over (or curl
around) the wall yet dip under the bar and reach the corner the keeper can't get
to. The score is shaped (near-misses, woodwork and keeper stretches give a
gradient), but the GOAL / saved / blocked outcomes carve a multimodal landscape
where the curve and loft must be co-ordinated.

Mirrors the browser demo docs/applications/free-kick.html.
"""

from __future__ import annotations

import math

N_DIM = 4
W, H = 800.0, 450.0
PX_PER_M = 13.5
DT = 1 / 120
MAX_T = 3.0
G = 9.81
DRAG = 0.04
MAGNUS_COEF = 0.0085

GOAL_Y = 60.0
GOAL_WIDTH_M, GOAL_HEIGHT_M = 7.32, 2.44
GOAL_LEFT = (W - GOAL_WIDTH_M * PX_PER_M) / 2
GOAL_RIGHT = (W + GOAL_WIDTH_M * PX_PER_M) / 2
GOAL_CX = W / 2
KICK_X, KICK_Y = W * 0.30, H - 50

_KG_DX = GOAL_CX - KICK_X
_KG_DY = GOAL_Y - KICK_Y
_KG_LEN = math.hypot(_KG_DX, _KG_DY)
_KG_UX, _KG_UY = _KG_DX / _KG_LEN, _KG_DY / _KG_LEN

N_WALL = 5
PLAYER_WIDTH_M = 0.5
WALL_PLAYER_DX = PLAYER_WIDTH_M * 1.2 * PX_PER_M
WALL_TOTAL_W_PX = (N_WALL - 1) * WALL_PLAYER_DX + PLAYER_WIDTH_M * PX_PER_M
WALL_HEIGHT_M = 1.85
WALL_DIST_PX = 9.15 * PX_PER_M
WALL_PERP_DX = -_KG_DY / _KG_LEN
WALL_PERP_DY = _KG_DX / _KG_LEN

KEEPER_START_X = W / 2 + 1.2 * PX_PER_M
KEEPER_REACTION_S = 0.35
KEEPER_MAX_SPEED_M = 3.8
KEEPER_REACH_M = 1.7


def decode(u):
    return [-25 + 50 * u[0], 5 + 35 * u[1], 14 + 18 * u[2], -3 + 6 * u[3]]


def run_kick(params):
    """Simulate one kick; return (score, result)."""
    aim_deg, loft_deg, power_mps, curve = params
    base_dir = math.atan2(GOAL_Y - KICK_Y, GOAL_CX - KICK_X)
    aim = base_dir + aim_deg * math.pi / 180
    loft = loft_deg * math.pi / 180
    v_horiz = power_mps * math.cos(loft)
    bx, by, bz = KICK_X, KICK_Y, 0.0
    vx = v_horiz * math.cos(aim) * PX_PER_M
    vy = v_horiz * math.sin(aim) * PX_PER_M
    vz = power_mps * math.sin(loft)
    keeper_x, keeper_vx = KEEPER_START_X, 0.0

    crossed_wall = wall_blocked = False
    wall_height_at_cross = None
    crossed_goal = False
    gx = gz = gkx = None
    landed = False
    land_y = None
    t = 0.0
    while t < MAX_T:
        v_horiz_px = math.hypot(vx, vy)
        if v_horiz_px > 1e-6:
            nx, ny = -vy / v_horiz_px, vx / v_horiz_px
            mag = curve * MAGNUS_COEF * v_horiz_px * PX_PER_M
            vx += nx * mag * DT
            vy += ny * mag * DT
        vx *= 1 - DRAG * DT
        vy *= 1 - DRAG * DT
        vz *= 1 - DRAG * DT
        vz -= G * DT
        bx += vx * DT
        by += vy * DT
        bz += vz * DT

        if t >= KEEPER_REACTION_S and vy < -1e-3:
            t_to_goal = (GOAL_Y - by) / vy
            if 0 < t_to_goal < 2.0:
                predicted_x = bx + vx * t_to_goal
                keeper_vx = (
                    (1 if predicted_x > keeper_x else -1)
                    * KEEPER_MAX_SPEED_M
                    * PX_PER_M
                )
        else:
            keeper_vx *= 0.5
        keeper_x += keeper_vx * DT
        keeper_x = max(GOAL_LEFT + 10, min(GOAL_RIGHT - 10, keeper_x))

        if not crossed_wall:
            dx, dy = bx - KICK_X, by - KICK_Y
            along = dx * _KG_UX + dy * _KG_UY
            if along > WALL_DIST_PX:
                crossed_wall = True
                wall_height_at_cross = bz
                perp = dx * WALL_PERP_DX + dy * WALL_PERP_DY
                half_wall = WALL_TOTAL_W_PX / 2 + PLAYER_WIDTH_M * PX_PER_M / 2
                if abs(perp) < half_wall and bz < WALL_HEIGHT_M:
                    wall_blocked = True
                    v_dot = vx * _KG_UX + vy * _KG_UY
                    e = 0.35
                    vx -= (1 + e) * v_dot * _KG_UX
                    vy -= (1 + e) * v_dot * _KG_UY
                    vz *= 0.55

        if not crossed_goal and by < GOAL_Y:
            crossed_goal = True
            gx, gz, gkx = bx, bz, keeper_x
            break

        if bz < 0:
            bz = 0.0
            vz = -vz * 0.42
            vx *= 0.72
            vy *= 0.72
            if abs(vz) < 0.4:
                landed = True
                land_y = by
                break
        t += DT

    # ---- scoring ----
    if wall_blocked:
        score = 0.0
        if wall_height_at_cross is not None:
            score = max(0.0, 5 - 10 * (WALL_HEIGHT_M - wall_height_at_cross))
        result = "blocked"
    elif not crossed_goal:
        if landed:
            dy_short = max(0.0, land_y - GOAL_Y)
            score = max(0.0, 30 * (1 - dy_short / (KICK_Y - GOAL_Y)))
            result = "short"
        else:
            score, result = 0.0, "in the air"
    else:
        in_posts = GOAL_LEFT < gx < GOAL_RIGHT
        under_bar = gz < GOAL_HEIGHT_M - 0.10
        if not in_posts:
            dist = min(abs(gx - GOAL_LEFT), abs(gx - GOAL_RIGHT))
            score, result = max(0.0, 10 - dist / 2), "wide"
        elif not under_bar:
            score, result = max(0.0, 10 - (gz - GOAL_HEIGHT_M) * 8), "over"
        else:
            dx_m = (gx - gkx) / PX_PER_M
            dz_m = gz - GOAL_HEIGHT_M * 0.45
            reach = math.hypot(dx_m, dz_m / 0.65)
            if reach < KEEPER_REACH_M:
                score, result = 20 + 30 * (1 - reach / KEEPER_REACH_M), "keeper saves"
            else:
                score = 100 + 10 * min(2, reach - KEEPER_REACH_M)
                result = "GOAL"
    score += min(5.0, max(0.0, (power_mps - 20) * 0.3))  # speed bonus
    if crossed_goal:  # woodwork bonus
        if (
            abs(gx - GOAL_LEFT) < 0.10 * PX_PER_M
            or abs(gx - GOAL_RIGHT) < 0.10 * PX_PER_M
            or abs(gz - GOAL_HEIGHT_M) < 0.10
        ):
            score += 5
    return score, result


def evaluate_kick(u):
    return run_kick(decode(u))


def objective(u):
    """HumpDay objective: negative free-kick score (minimise)."""
    return -run_kick(decode(u))[0]
