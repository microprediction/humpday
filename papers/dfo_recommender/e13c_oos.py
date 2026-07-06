"""E13c — Out-of-sample validation of the signed-composition winner.

Adds the canon:-1DE+2NM program (NM host, DE as anti-inspiration; selection
regret 0.179, the best single draw of any search on the E7 suite) to the
existing 13-way untouched-problem race from E10. Only the new entrant
executes; ranks recomputed over the merged 14-way field on the same 580
instances.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
from example_demos import DEMOS, disguise_demo  # noqa: E402
from e2_hardening import INF, atomic_dump, run_discovered  # noqa: E402
from simplex_blend import compile_optimizer  # noqa: E402

SRC = "runs/e10_oos.json"
OUT = "runs/e13c_oos.json"
NEW_NAME = "ablated_star"
NEW_CODE = "runs/e13_code/canon_-1DEp2NM_ablated.py"


def main():
    opt = compile_optimizer(open(NEW_CODE).read())
    src = json.load(open(SRC))
    assert src["done"]
    base_rows = src["rows"]
    by_name = {d.name: d for d in DEMOS}
    field = src["field"] + [NEW_NAME]

    rows = []
    done = {}
    if Path(OUT).exists():
        try:
            rows = json.load(open(OUT)).get("rows", [])
            done = {(r["budget"], r["demo"], r["seed"]): r for r in rows}
        except Exception:  # noqa: BLE001
            rows, done = [], {}

    total = len(base_rows)
    for c, br in enumerate(base_rows, 1):
        key = (br["budget"], br["demo"], br["seed"])
        if key in done:
            continue
        dm = by_name[br["demo"]]
        inst = disguise_demo(dm, br["seed"])
        vals = dict(br["vals"])
        v = run_discovered(opt, inst.objective, dm.n_dim, br["budget"], 9000 + br["seed"])
        vals[NEW_NAME] = None if v >= INF else v
        big = 1e18
        ranks = {
            o: 1 + sum(
                1 for x in field
                if (vals.get(x) if vals.get(x) is not None else big)
                < (vals.get(o) if vals.get(o) is not None else big) - 1e-12
            )
            for o in field
        }
        rows.append({**{k: br[k] for k in ("budget", "demo", "n", "seed")},
                     "vals": vals, "ranks": ranks})
        if c % 20 == 0 or c == total:
            print(f"[{c}/{total}]", flush=True)
            atomic_dump({"done": False, "field": field, "rows": rows}, OUT)

    summary = {}
    budgets = sorted({r["budget"] for r in rows})
    for b in budgets:
        brs = [r for r in rows if r["budget"] == b]
        summary[str(b)] = {
            o: {"mean_rank": round(sum(r["ranks"][o] for r in brs) / len(brs), 3),
                "wins": sum(1 for r in brs if r["ranks"][o] == 1)}
            for o in field
        }
    atomic_dump({"done": True, "field": field, "summary": summary, "rows": rows}, OUT)

    print("\n=== mean rank by budget (1=best of 14) ===")
    for b in budgets:
        sb_ = summary[str(b)]
        ordered = sorted(field, key=lambda o: sb_[o]["mean_rank"])
        print(f"  budget {b}: " + "  ".join(f"{o}={sb_[o]['mean_rank']}" for o in ordered[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
