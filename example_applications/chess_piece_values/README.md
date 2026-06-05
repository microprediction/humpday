# Chess Piece Values — Objective Design (and the Expensive End)

Tune a chess bot's evaluation to beat a textbook bot. A self-contained,
**perft-verified** 0x88 engine plays full games with depth-2 alpha-beta minimax;
the bots differ only in their evaluation. HumpDay's `[0,1]^8` cube sets four piece
values (N, B, R, Q) and four positional piece-square-table weights, and the
objective returns your bot's negative **win percentage**, playing both colours
from random openings.

## What this stresses

- **Objective design.** The optimiser does **not** rediscover the textbook
  1/3/3/5/9 values — it exploits the *shallow fixed opponent*, so the "best"
  evaluation is an artefact of who it plays. A cautionary tale about what your
  objective is really rewarding.
- **Noise + overfitting.** Random openings make each game different, so a small
  training-seed batch flatters the evaluation relative to a held-out cohort —
  the in/out-of-sample gap, on a real game.
- **Expensive evaluation.** Every objective call plays several depth-2 games;
  this is the costly regime where each sample must count. A run takes a few
  **minutes** — by design.

## Running

```bash
python -m example_applications.chess_piece_values.run
```

The engine passes perft (20 / 400 / 8902 from the start; 48 for Kiwipete). Mirrors
the browser demo [`docs/applications/chess.html`](../../docs/applications/chess.html).
