"""
Chess piece-value objective: tune an evaluation to beat a textbook bot.

A self-contained, perft-faithful 0x88 chess engine with material + piece-square-
table evaluation and a depth-2 alpha-beta minimax search (a pure-Python port of
the browser demo's engine). Your bot and a "textbook" bot (standard 1/3/3/5/9
values, unit PST weights) play full games from random openings. The HumpDay
objective takes an 8-D point in [0,1]^8 — four piece values (N, B, R, Q) plus
four positional PST weights — and returns the negative **win percentage** of your
bot, playing both colours.

The lessons: (1) it does NOT rediscover the textbook values — it exploits the
shallow fixed opponent, so the "best" evaluation is an artefact of who it plays;
(2) random openings make the score NOISY, so a small training-seed batch
overfits relative to a held-out set (in/out-of-sample). This is the expensive
end of the spectrum — every evaluation plays several depth-2 games.

Mirrors the browser demo docs/applications/chess.html.
"""

from __future__ import annotations

N_DIM = 8
KNIGHT_D = (33, 31, 18, 14, -33, -31, -18, -14)
KING_D = (16, -16, 1, -1, 17, 15, -17, -15)
BISHOP_D = (17, 15, -17, -15)
ROOK_D = (16, -16, 1, -1)
QUEEN_D = (16, -16, 1, -1, 17, 15, -17, -15)
MATE = 1e7
DEPTH = 2
OPEN_PLIES = 6
HARD_CAP = 200
WIN_MARGIN = 200
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STD_VAL = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}
TEST_SEEDS = (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007)
N_TRAIN = 2  # training seeds (each played as both colours)


def _on_board(s):
    return 0 <= s < 128 and (s & 0x88) == 0


def _rank(s):
    return s >> 4


def _file(s):
    return s & 7


def _color(p):
    return None if p == "." else ("w" if p.isupper() else "b")


def parse_fen(fen):
    pl, side, ca, ep = fen.split()[:4]
    b = ["."] * 128
    for r, row in enumerate(pl.split("/")):
        rank = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                b[rank * 16 + f] = ch
                f += 1
    rights = {"K": "K" in ca, "Q": "Q" in ca, "k": "k" in ca, "q": "q" in ca}
    e = -1
    if ep and ep != "-":
        e = (int(ep[1]) - 1) * 16 + (ord(ep[0]) - 97)
    return {"board": b, "side": "w" if side == "w" else "b", "rights": rights, "ep": e}


def _is_attacked(b, sq, by):
    if by == "w":
        for d in (-15, -17):
            s = sq + d
            if _on_board(s) and b[s] == "P":
                return True
    else:
        for d in (15, 17):
            s = sq + d
            if _on_board(s) and b[s] == "p":
                return True
    kn = "N" if by == "w" else "n"
    for d in KNIGHT_D:
        s = sq + d
        if _on_board(s) and b[s] == kn:
            return True
    kg = "K" if by == "w" else "k"
    for d in KING_D:
        s = sq + d
        if _on_board(s) and b[s] == kg:
            return True
    bb = "B" if by == "w" else "b"
    rr = "R" if by == "w" else "r"
    qq = "Q" if by == "w" else "q"
    for d in BISHOP_D:
        s = sq + d
        while _on_board(s):
            p = b[s]
            if p != ".":
                if p == bb or p == qq:
                    return True
                break
            s += d
    for d in ROOK_D:
        s = sq + d
        while _on_board(s):
            p = b[s]
            if p != ".":
                if p == rr or p == qq:
                    return True
                break
            s += d
    return False


def _king_square(b, c):
    k = "K" if c == "w" else "k"
    for s in range(128):
        if _on_board(s) and b[s] == k:
            return s
    return -1


def _in_check(st, c):
    return _is_attacked(
        st["board"], _king_square(st["board"], c), "b" if c == "w" else "w"
    )


def _gen_pseudo(st):
    board, side, rights, ep = st["board"], st["side"], st["rights"], st["ep"]
    mv = []
    us = side
    them = "w" if side == "b" else "b"
    for sq in range(128):
        if not _on_board(sq):
            continue
        p = board[sq]
        if p == "." or _color(p) != us:
            continue
        up = p.upper()
        if up == "P":
            d = 16 if us == "w" else -16
            sr = 1 if us == "w" else 6
            pr = 7 if us == "w" else 0
            one = sq + d
            if _on_board(one) and board[one] == ".":
                if _rank(one) == pr:
                    for q in "QRBN":
                        mv.append(
                            {
                                "from": sq,
                                "to": one,
                                "piece": p,
                                "captured": ".",
                                "promo": q if us == "w" else q.lower(),
                            }
                        )
                else:
                    mv.append({"from": sq, "to": one, "piece": p, "captured": "."})
                two = sq + 2 * d
                if _rank(sq) == sr and board[two] == ".":
                    mv.append(
                        {
                            "from": sq,
                            "to": two,
                            "piece": p,
                            "captured": ".",
                            "flag": "double",
                        }
                    )
            for cd in (15, 17) if us == "w" else (-15, -17):
                t = sq + cd
                if not _on_board(t):
                    continue
                if board[t] != "." and _color(board[t]) == them:
                    if _rank(t) == pr:
                        for q in "QRBN":
                            mv.append(
                                {
                                    "from": sq,
                                    "to": t,
                                    "piece": p,
                                    "captured": board[t],
                                    "promo": q if us == "w" else q.lower(),
                                }
                            )
                    else:
                        mv.append(
                            {"from": sq, "to": t, "piece": p, "captured": board[t]}
                        )
                elif t == ep and ep >= 0:
                    cs = t - 16 if us == "w" else t + 16
                    mv.append(
                        {
                            "from": sq,
                            "to": t,
                            "piece": p,
                            "captured": board[cs],
                            "flag": "ep",
                        }
                    )
        elif up == "N":
            for d in KNIGHT_D:
                t = sq + d
                if _on_board(t) and (board[t] == "." or _color(board[t]) == them):
                    mv.append({"from": sq, "to": t, "piece": p, "captured": board[t]})
        elif up == "K":
            for d in KING_D:
                t = sq + d
                if _on_board(t) and (board[t] == "." or _color(board[t]) == them):
                    mv.append({"from": sq, "to": t, "piece": p, "captured": board[t]})
            hr = 0 if us == "w" else 7
            if _rank(sq) == hr and _file(sq) == 4 and not _is_attacked(board, sq, them):
                k_r = rights["K"] if us == "w" else rights["k"]
                q_r = rights["Q"] if us == "w" else rights["q"]
                rook = "R" if us == "w" else "r"
                if (
                    k_r
                    and board[sq + 1] == "."
                    and board[sq + 2] == "."
                    and board[sq + 3] == rook
                    and not _is_attacked(board, sq + 1, them)
                    and not _is_attacked(board, sq + 2, them)
                ):
                    mv.append(
                        {
                            "from": sq,
                            "to": sq + 2,
                            "piece": p,
                            "captured": ".",
                            "flag": "castleK",
                        }
                    )
                if (
                    q_r
                    and board[sq - 1] == "."
                    and board[sq - 2] == "."
                    and board[sq - 3] == "."
                    and board[sq - 4] == rook
                    and not _is_attacked(board, sq - 1, them)
                    and not _is_attacked(board, sq - 2, them)
                ):
                    mv.append(
                        {
                            "from": sq,
                            "to": sq - 2,
                            "piece": p,
                            "captured": ".",
                            "flag": "castleQ",
                        }
                    )
        else:
            dirs = BISHOP_D if up == "B" else ROOK_D if up == "R" else QUEEN_D
            for d in dirs:
                t = sq + d
                while _on_board(t):
                    if board[t] == ".":
                        mv.append({"from": sq, "to": t, "piece": p, "captured": "."})
                    else:
                        if _color(board[t]) == them:
                            mv.append(
                                {"from": sq, "to": t, "piece": p, "captured": board[t]}
                            )
                        break
                    t += d
    return mv


def _make_move(st, m):
    b = st["board"][:]
    us = st["side"]
    them = "b" if us == "w" else "w"
    rights = dict(st["rights"])
    ep = -1
    frm, to = m["from"], m["to"]
    b[to] = m.get("promo") or m["piece"]
    b[frm] = "."
    flag = m.get("flag")
    if flag == "ep":
        cs = to - 16 if us == "w" else to + 16
        b[cs] = "."
    elif flag == "castleK":
        b[frm + 1] = b[frm + 3]
        b[frm + 3] = "."
    elif flag == "castleQ":
        b[frm - 1] = b[frm - 4]
        b[frm - 4] = "."
    elif flag == "double":
        ep = to - 16 if us == "w" else to + 16
    if m["piece"] == "K":
        rights["K"] = rights["Q"] = False
    if m["piece"] == "k":
        rights["k"] = rights["q"] = False
    if frm == 0 or to == 0:
        rights["Q"] = False
    if frm == 7 or to == 7:
        rights["K"] = False
    if frm == 112 or to == 112:
        rights["q"] = False
    if frm == 119 or to == 119:
        rights["k"] = False
    return {"board": b, "side": them, "rights": rights, "ep": ep}


def _gen_legal(st):
    us = st["side"]
    opp = "b" if us == "w" else "w"
    out = []
    for m in _gen_pseudo(st):
        ns = _make_move(st, m)
        if not _is_attacked(ns["board"], _king_square(ns["board"], us), opp):
            out.append(m)
    return out


def board_string(st):
    return "".join(st["board"][r * 16 + f] for r in range(7, -1, -1) for f in range(8))


PST = {
    "P": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        10,
        10,
        -20,
        -20,
        10,
        10,
        5,
        5,
        -5,
        -10,
        0,
        0,
        -10,
        -5,
        5,
        0,
        0,
        0,
        20,
        20,
        0,
        0,
        0,
        5,
        5,
        10,
        25,
        25,
        10,
        5,
        5,
        10,
        10,
        20,
        30,
        30,
        20,
        10,
        10,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "N": [
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
        -40,
        -20,
        0,
        5,
        5,
        0,
        -20,
        -40,
        -30,
        5,
        10,
        15,
        15,
        10,
        5,
        -30,
        -30,
        0,
        15,
        20,
        20,
        15,
        0,
        -30,
        -30,
        5,
        15,
        20,
        20,
        15,
        5,
        -30,
        -30,
        0,
        10,
        15,
        15,
        10,
        0,
        -30,
        -40,
        -20,
        0,
        0,
        0,
        0,
        -20,
        -40,
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
    ],
    "B": [
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        5,
        0,
        0,
        0,
        0,
        5,
        -10,
        -10,
        10,
        10,
        10,
        10,
        10,
        10,
        -10,
        -10,
        0,
        10,
        10,
        10,
        10,
        0,
        -10,
        -10,
        5,
        5,
        10,
        10,
        5,
        5,
        -10,
        -10,
        0,
        5,
        10,
        10,
        5,
        0,
        -10,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
    ],
    "R": [
        0,
        0,
        0,
        5,
        5,
        0,
        0,
        0,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "Q": [
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
        -10,
        0,
        5,
        0,
        0,
        0,
        0,
        -10,
        -10,
        5,
        5,
        5,
        5,
        5,
        0,
        -10,
        -5,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        0,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        -10,
        0,
        5,
        5,
        5,
        5,
        5,
        -10,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
    ],
    "K": [
        20,
        30,
        10,
        0,
        0,
        10,
        30,
        20,
        20,
        20,
        0,
        0,
        0,
        0,
        20,
        20,
        -10,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -10,
        -20,
        -30,
        -30,
        -40,
        -40,
        -30,
        -30,
        -20,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
    ],
}


def _pst_val(p, sq):
    t = p.upper()
    rank, file = _rank(sq), _file(sq)
    idx = rank * 8 + file if _color(p) == "w" else (7 - rank) * 8 + file
    return PST[t][idx]


def std_params():
    return {
        "values": dict(STD_VAL),
        "scales": {"P": 1, "N": 1, "B": 1, "R": 1, "Q": 1, "K": 1},
    }


def _eval_cp(st, params):
    b = st["board"]
    side = st["side"]
    values, scales = params["values"], params["scales"]
    s = 0
    for sq in range(128):
        if not _on_board(sq):
            continue
        p = b[sq]
        if p == ".":
            continue
        t = p.upper()
        c = values[t] + scales[t] * _pst_val(p, sq)
        s += c if _color(p) == side else -c
    return s


def _order_moves(mv):
    mv.sort(
        key=lambda m: 0 if m["captured"] == "." else STD_VAL[m["captured"].upper()],
        reverse=True,
    )


def _search(st, depth, alpha, beta, params, k_sq):
    if depth == 0:
        return _eval_cp(st, params)
    us = st["side"]
    opp = "b" if us == "w" else "w"
    mv = _gen_pseudo(st)
    _order_moves(mv)
    best = -float("inf")
    any_legal = False
    for m in mv:
        ns = _make_move(st, m)
        our_k = m["to"] if m["piece"] in ("K", "k") else k_sq
        if _is_attacked(ns["board"], our_k, opp):
            continue
        any_legal = True
        sc = -_search(
            ns, depth - 1, -beta, -alpha, params, _king_square(ns["board"], opp)
        )
        if sc > best:
            best = sc
            if best > alpha:
                alpha = best
        if alpha >= beta:
            break
    if not any_legal:
        return -MATE - depth if _in_check(st, us) else 0
    return best


def _best_move(st, depth, params):
    us = st["side"]
    opp = "b" if us == "w" else "w"
    k_sq = _king_square(st["board"], us)
    mv = _gen_pseudo(st)
    _order_moves(mv)
    best = -float("inf")
    chosen = None
    for m in mv:
        ns = _make_move(st, m)
        our_k = m["to"] if m["piece"] in ("K", "k") else k_sq
        if _is_attacked(ns["board"], our_k, opp):
            continue
        sc = -_search(
            ns,
            depth - 1,
            -float("inf"),
            float("inf"),
            params,
            _king_square(ns["board"], opp),
        )
        if sc > best:
            best = sc
            chosen = m
    return chosen


def _true_balance(st):
    b = st["board"]
    s = 0
    for sq in range(128):
        if not _on_board(sq):
            continue
        p = b[sq]
        if p == ".":
            continue
        s += (1 if _color(p) == "w" else -1) * STD_VAL[p.upper()]
    return s


def _make_rng(seed):
    s = (seed & 0xFFFFFFFF) or 1

    def r():
        nonlocal s
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        return s / 0x7FFFFFFF

    return r


def _random_opening(seed, n_plies):
    st = parse_fen(START_FEN)
    rnd = _make_rng(seed + 1)
    for _ in range(n_plies):
        mv = _gen_legal(st)
        if not mv:
            break
        st = _make_move(st, mv[int(rnd() * len(mv))])
    return st


def _rep_key(st):
    r = st["rights"]
    return (
        board_string(st)
        + st["side"]
        + str(r["K"])
        + str(r["Q"])
        + str(r["k"])
        + str(r["q"])
        + str(st["ep"])
    )


def play_game(start_state, white_p, black_p):
    """Play a full game; return the result from White's view (1/0/-1)."""
    st = start_state
    half = 0
    seen = {_rep_key(st): 1}
    for _ in range(HARD_CAP):
        mv = _gen_legal(st)
        if not mv:
            mated = _in_check(st, st["side"])
            return 0 if not mated else (-1 if st["side"] == "w" else 1)
        params = white_p if st["side"] == "w" else black_p
        m = _best_move(st, DEPTH, params)
        reset = (
            (m["captured"] != ".")
            or (m["piece"] in ("P", "p"))
            or m.get("flag") == "ep"
        )
        half = 0 if reset else half + 1
        st = _make_move(st, m)
        k = _rep_key(st)
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= 3:
            return 0
        if half >= 100:
            return 0
    bal = _true_balance(st)
    return 1 if bal >= WIN_MARGIN else (-1 if bal <= -WIN_MARGIN else 0)


def decode(u):
    return [
        100 + 500 * u[0],
        100 + 500 * u[1],
        200 + 700 * u[2],
        300 + 1200 * u[3],
        2.5 * u[4],
        2.5 * u[5],
        2.5 * u[6],
        2.5 * u[7],
    ]


def params_from_array(a):
    return {
        "values": {"P": 100, "N": a[0], "B": a[1], "R": a[2], "Q": a[3], "K": 0},
        "scales": {"P": 1, "N": a[4], "B": a[5], "R": a[6], "Q": a[7], "K": 1},
    }


def score_candidate(a, seeds):
    """Win % of the candidate evaluation vs the textbook bot, playing both colours."""
    cand = params_from_array(a)
    std = std_params()
    total = 0.0
    n = 0
    for seed in seeds:
        op = _random_opening(seed, OPEN_PLIES)
        total += (play_game(op, cand, std) + 1) / 2  # candidate as White
        n += 1
        total += (-play_game(op, std, cand) + 1) / 2  # candidate as Black
        n += 1
    return 100.0 * total / n


def objective(u, seed_offset=0):
    """HumpDay objective: negative candidate win % over the training seeds."""
    seeds = range(seed_offset, seed_offset + N_TRAIN)
    return -score_candidate(decode(u), seeds)


def evaluate_candidate(u):
    """Held-out win % over the fixed test seed cohort."""
    return score_candidate(decode(u), TEST_SEEDS)
