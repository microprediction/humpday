"""
Tetris-brain objective: tune a greedy bot's four evaluation weights.

A pure-Python re-implementation of the heuristic Tetris player from the browser
demo (docs/applications/tetris.html). For the current piece the bot tries every
rotation and column, simulates the drop, and rates the resulting board by

    w1*aggregate_height + w2*lines_cleared + w3*holes + w4*bumpiness

playing the highest-rated landing. The HumpDay objective takes a 4-D point in
[0,1]^4, stretches it into the four weights in [-1,1]^4, plays a few games on a
stylised 7x16 board with random piece bags, and returns the **negative mean
lines cleared**. Minimising it maximises lines.

The piece order is random, so the score is NOISY — this is a noisy-objective
example, and it exposes the in-sample vs out-of-sample gap (train seeds vs a
held-out cohort). Empirically the optimiser does NOT rediscover the textbook
"reward lines, punish holes" weights: survivability matters more than greed, so
odd-looking vectors (sometimes even penalising line-clears) score well.
"""

from __future__ import annotations

import random

N_DIM = 4
W = 7  # board width (narrower than standard so weights matter)
H = 16  # board height
MAX_PIECES = 160
N_GAMES = 3  # train games per evaluation


def _build_pieces():
    base = {
        "I": [(0, 1), (1, 1), (2, 1), (3, 1)],
        "O": [(1, 0), (2, 0), (1, 1), (2, 1)],
        "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
        "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
        "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
        "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
        "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
    }

    def norm(cells):
        mnx = min(c[0] for c in cells)
        mny = min(c[1] for c in cells)
        return tuple(sorted((x - mnx, y - mny) for x, y in cells))

    def rot(cells):
        return norm([(-y, x) for x, y in cells])

    pieces = []
    for cells in base.values():
        states = []
        c = norm(cells)
        seen = set()
        for _ in range(4):
            if c not in seen:
                seen.add(c)
                states.append(c)
            c = rot(c)
        pieces.append(states)
    return pieces


PIECES = _build_pieces()


def _scale(u):
    return [2.0 * ui - 1.0 for ui in u]


def _features(grid):
    heights = [0] * W
    for c in range(W):
        for r in range(H):
            if grid[r * W + c]:
                heights[c] = H - r
                break
    agg = sum(heights)
    holes = 0
    bump = 0
    for c in range(W):
        filled_above = False
        for r in range(H):
            if grid[r * W + c]:
                filled_above = True
            elif filled_above:
                holes += 1
        if c < W - 1:
            bump += abs(heights[c] - heights[c + 1])
    return agg, holes, bump


def _place(grid, state, x):
    maxx = max(c[0] for c in state)
    if x + maxx >= W or x < 0:
        return None
    dy = -1
    for d in range(H):
        ok = True
        for cx, cy in state:
            gx, gy = x + cx, cy + d
            if gy >= H or (gy >= 0 and grid[gy * W + gx]):
                ok = False
                break
        if ok:
            dy = d
        else:
            break
    if dy < 0:
        return None
    ng = list(grid)
    topped = False
    for cx, cy in state:
        gy = cy + dy
        if gy < 0:
            topped = True
            continue
        ng[gy * W + (x + cx)] = 1
    if topped:
        return None
    lines = 0
    r = H - 1
    while r >= 0:
        if all(ng[r * W + c] for c in range(W)):
            lines += 1
            for rr in range(r, 0, -1):
                for c in range(W):
                    ng[rr * W + c] = ng[(rr - 1) * W + c]
            for c in range(W):
                ng[c] = 0
        else:
            r -= 1
    return ng, lines


def _best_move(grid, p, w):
    best = None
    best_score = -1e18
    for state in PIECES[p]:
        maxx = max(c[0] for c in state)
        for x in range(W - maxx):
            res = _place(grid, state, x)
            if res is None:
                continue
            ng, lines = res
            agg, holes, bump = _features(ng)
            score = w[0] * agg + w[1] * lines + w[2] * holes + w[3] * bump
            if score > best_score:
                best_score = score
                best = (ng, lines)
    return best


def play_game(w, seed):
    """Play one game; return lines cleared."""
    rng = random.Random(seed)
    grid = [0] * (W * H)
    lines = 0
    bag = []
    for _ in range(MAX_PIECES):
        if not bag:
            bag = [0, 1, 2, 3, 4, 5, 6]
            rng.shuffle(bag)
        p = bag.pop()
        mv = _best_move(grid, p, w)
        if mv is None:
            break
        grid, cleared = mv
        lines += cleared
    return lines


def objective(u, seed_offset=0):
    """HumpDay objective: negative mean lines over N_GAMES (minimise).

    `seed_offset` lets a test harness re-evaluate on a disjoint cohort; the
    optimiser shouldn't pass it."""
    w = _scale(u)
    total = sum(play_game(w, seed_offset + i) for i in range(N_GAMES))
    return -total / N_GAMES


def evaluate_weights(u, n_games=20, seed_offset=900_000):
    """Held-out evaluation on a disjoint seed cohort."""
    w = _scale(u)
    games = [play_game(w, seed_offset + i * 13) for i in range(n_games)]
    games.sort()
    n = len(games)
    return {
        "mean": sum(games) / n,
        "median": games[n // 2],
        "min": games[0],
        "max": games[-1],
    }
