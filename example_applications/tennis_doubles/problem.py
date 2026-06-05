"""
Tennis-doubles strategy objective: beat a textbook team by tuning your tactics.

A pure-Python stylised doubles point simulator. Your team (A) plays a fixed
"textbook" team (B); the HumpDay objective takes a 6-D point in [0,1]^6 and
decodes it into your strategy — net position, baseline depth, poach aggression,
cross-court tendency, depth/risk, and lob rate — then plays a batch of seeded
points (you serving and receiving) and returns the negative **win percentage**.

The point model includes serve geometry (into the diagonal service box), rally
shot placement, a receiving net player who intercepts (poaches) balls passing
near them, and out / net errors that scale with how much risk your depth setting
takes. It is deterministic per point, so the in-sample win rate over a small
batch of training points flatters the strategy relative to a large held-out set —
the same in/out-of-sample overfitting story as the chess and tetris demos.

Mirrors the browser demo docs/applications/tennis-doubles.html.
"""

from __future__ import annotations

import math

N_DIM = 6
COURT_W, COURT_L, NET, CENTER = 36, 78, 39, 18
COVER, VOLLEY_REACH, SKILL_SD, MARGIN, MAX_SHOTS = 15.0, 9, 2.6, 0.5, 24
N_TRAIN = 6  # training points-per-server (×2 = 12 points)
TEST_SEEDS = tuple(range(5000, 5060))  # held-out cohort (×2 = 120 points)

TEXTBOOK_B = {
    "netX": 18,
    "baseDepth": 5,
    "poach": 0.30,
    "aimCross": 0.65,
    "depthRisk": 0.50,
    "lob": 0.15,
}


def _make_rng(seed):
    s = (seed & 0xFFFFFFFF) or 1

    def r():
        nonlocal s
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        return s / 0x7FFFFFFF

    return r


def _gauss(rng):
    return (rng() + rng() + rng() + rng() - 2) * 1.4142


def _positions(s, side):
    if side == "A":
        return {"net": (s["netX"], 33), "base": (CENTER, s["baseDepth"])}
    return {"net": (s["netX"], 45), "base": (CENTER, COURT_L - s["baseDepth"])}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def simulate_point(seed, A, B, server):
    """Play one point; return the winning side, 'A' or 'B'."""
    rng = _make_rng(seed * 131 + (7 if server == "A" else 99))
    pos_a, pos_b = _positions(A, "A"), _positions(B, "B")
    hitting = server
    hitter = (
        (CENTER, A["baseDepth"])
        if hitting == "A"
        else (CENTER, COURT_L - B["baseDepth"])
    )

    for shot in range(MAX_SHOTS):
        strat = A if hitting == "A" else B
        recv_side = "B" if hitting == "A" else "A"
        recv_strat = B if hitting == "A" else A
        recv_pos = pos_b if hitting == "A" else pos_a
        into_far = hitting == "A"
        base_y = COURT_L if into_far else 0
        mid_y = NET + (1 if into_far else -1) * 6

        if shot == 0:  # serve into the diagonal service box
            deuce = seed % 2 == 0
            near = into_far
            hitter = (25 if deuce else 11, 2 if near else COURT_L - 2)
            box_x = 11 if deuce else 25
            sx = (
                box_x
                + _gauss(rng) * 2.0
                + (strat["depthRisk"] - 0.5) * 5 * (-1 if deuce else 1)
            )
            sy = (NET + 5 + rng() * 13 if near else NET - 5 - rng() * 13) + _gauss(
                rng
            ) * 1.5
            box_x_ok = (4 < sx < 18) if deuce else (18 < sx < 32)
            box_y_ok = (NET + 1 < sy < NET + 21) if near else (NET - 1 > sy > NET - 21)
            if not box_x_ok or not box_y_ok or rng() < 0.03:
                return recv_side
            hitter = (sx, sy)
            hitting = recv_side
            continue

        is_lob = rng() < strat["lob"]
        if is_lob:
            tx = recv_pos["net"][0] + (1 if rng() < 0.5 else -1) * (4 + rng() * 6)
            ty = base_y + (-1 if into_far else 1) * (3 + rng() * 5)
        else:
            cross = rng() < strat["aimCross"]
            hitter_left = hitter[0] < CENTER
            aim_left = (not hitter_left) if cross else hitter_left
            tx = (4 + rng() * 12) if aim_left else (20 + rng() * 12)
            depth_frac = 0.45 + 0.5 * strat["depthRisk"]
            ty = mid_y + (base_y - mid_y) * depth_frac
        tx += _gauss(rng) * SKILL_SD
        ty += _gauss(rng) * SKILL_SD
        target = (tx, ty)

        out_geom = (
            tx < -MARGIN
            or tx > COURT_W + MARGIN
            or (
                ty > COURT_L + MARGIN or ty < NET
                if into_far
                else ty < -MARGIN or ty > NET
            )
        )
        out_risk = (not is_lob) and rng() < (0.012 + 0.16 * strat["depthRisk"] ** 2)
        if out_geom or out_risk:
            return recv_side
        net_err = rng() < (0.015 if is_lob else 0.02 + 0.035 * strat["depthRisk"])
        if net_err:
            return recv_side

        net_shot = abs(hitter[1] - NET) < 10
        time_factor = (1.35 if is_lob else (1.25 - 0.45 * strat["depthRisk"])) * (
            0.82 if net_shot else 1.0
        )

        # receiving net player intercepts balls passing near them
        np_ = recv_pos["net"]
        volleyed = False
        contact = None
        if not is_lob:
            denom = target[1] - hitter[1]
            if abs(denom) > 1e-6:
                frac = (np_[1] - hitter[1]) / denom
                if 0.05 < frac < 1:
                    bx = hitter[0] + (target[0] - hitter[0]) * frac
                    lat = abs(bx - np_[0])
                    if (
                        lat < VOLLEY_REACH
                        and rng()
                        < recv_strat["poach"] * (1 - lat / VOLLEY_REACH) + 0.05
                    ):
                        volleyed = True
                        contact = (bx, np_[1])
        if volleyed:
            hitter = contact
            hitting = recv_side
            continue

        base_reach = COVER * time_factor * (1.18 if is_lob else 1.0)
        reached = _dist(recv_pos["base"], target) <= base_reach
        if not reached:
            return hitting
        hitter = target
        hitting = recv_side

    return "A" if _make_rng(seed)() < 0.5 else "B"


def score_strategy(A, seeds, B=None):
    """Percentage of points won by team A across the seeds (each served both ways)."""
    if B is None:
        B = TEXTBOOK_B
    w = n = 0
    for seed in seeds:
        if simulate_point(seed, A, B, "A") == "A":
            w += 1
        n += 1
        if simulate_point(seed, A, B, "B") == "A":
            w += 1
        n += 1
    return 100.0 * w / n


def decode(u):
    return {
        "netX": 8 + 20 * u[0],
        "baseDepth": 1 + 10 * u[1],
        "poach": u[2],
        "aimCross": u[3],
        "depthRisk": u[4],
        "lob": 0.4 * u[5],
    }


def objective(u, seed_offset=0):
    """HumpDay objective: negative win % over the training points (minimise)."""
    seeds = range(seed_offset, seed_offset + N_TRAIN)
    return -score_strategy(decode(u), seeds)


def evaluate_strategy(u):
    """Held-out win % over the fixed test cohort."""
    return score_strategy(decode(u), TEST_SEEDS)
