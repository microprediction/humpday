"""E10 — Do the TUNED parametric winners generalize?

E7/E9 found that at matched 20-evaluation budgets on the selection suite,
cheap parametric search (14-gene template, hand-written portfolio family)
matches or beats per-point LLM generation. The open question is
out-of-sample: the template's in-suite scores flattered it before.

This adds the two tuned winners, reconstructed exactly from their
coordinates, to the existing 11-way untouched-demo race (E6b rows reused;
only the two new entrants execute):

  template_star  : algo_dev.make_candidate at E7's best dfo_template genome
  portfolio_star : portfolio_w.make_portfolio at E9's best weights

Ranks recomputed over the merged 13-way field on the same 580 instances.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
import algo_dev as ad  # noqa: E402
import simplex_blend as sb  # noqa: E402
from example_demos import DEMOS, disguise_demo  # noqa: E402
from e2_hardening import INF, atomic_dump, run_discovered  # noqa: E402
from portfolio_w import make_portfolio  # noqa: E402

from humpday.transforms.cubetosimplex import cube_to_simplex  # noqa: E402

TEMPLATE_U = [0.5002, 0.3013, 0.7923, 0.9816, 0.7018, 0.5489, 0.8527,
              0.4036, 0.8249, 0.3566, 0.9763, 0.2655, 0.9672, 0.7681]
PORTFOLIO_U = [0.7171, 0.0, 0.2693, 0.8481]

SRC = "runs/e6b_more_points.json"
OUT = "runs/e10_oos.json"


def main():
    w = list(cube_to_simplex(PORTFOLIO_U))
    progs = {
        "template_star": ad.make_candidate(TEMPLATE_U),
        "portfolio_star": make_portfolio(w, sb.weights_to_spec(w)),
    }
    src = json.load(open(SRC))
    assert src["done"]
    base_rows = src["rows"]
    by_name = {d.name: d for d in DEMOS}
    field = src["field"] + list(progs)

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
        for name, opt in progs.items():
            v = run_discovered(opt, inst.objective, dm.n_dim, br["budget"], 9000 + br["seed"])
            vals[name] = None if v >= INF else v
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

    print("\n=== mean rank by budget (1=best of 13) ===")
    for b in budgets:
        sb_ = summary[str(b)]
        ordered = sorted(field, key=lambda o: sb_[o]["mean_rank"])
        print(f"  budget {b}: " + "  ".join(f"{o}={sb_[o]['mean_rank']}" for o in ordered[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
