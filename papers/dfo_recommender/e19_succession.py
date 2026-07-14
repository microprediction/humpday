"""E19 — Succession match on fresh instances.

Alloy vs the two caricature-prompt twins vs the tuned portfolio, on the 29
untouched demos at disguise seeds 5-9, which no experiment has ever used.
Pre-registered: winner = best mean rank across budgets; pairwise sign tests
reported. No further contenders may be added after launch.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
import simplex_blend as sb  # noqa: E402
from e2_hardening import INF, atomic_dump, load_centroid, run_discovered  # noqa: E402
from e6_untouched import UNTOUCHED  # noqa: E402
from example_demos import DEMOS, disguise_demo  # noqa: E402
from portfolio_w import make_portfolio  # noqa: E402

from humpday.transforms.cubetosimplex import cube_to_simplex  # noqa: E402

SEEDS = (5, 6, 7, 8, 9)
BUDGETS = (60, 120, 240, 480)
OUT = "runs/e19_succession.json"

w = list(cube_to_simplex([0.7171, 0.0, 0.2693, 0.8481]))
FIELD = {
    "Alloy": load_centroid(),
    "twin_e15": sb.compile_optimizer(
        open("runs/e15_code/goodPSNM_caricature_d1.py")
        .read()
        .replace("ANTI_STRENGTH = 1.0", "ANTI_STRENGTH = 0.0")
    ),
    "twin_e17": sb.compile_optimizer(open("runs/e17_code/e17_star_ablated.py").read()),
    "portfolio": make_portfolio(w, sb.weights_to_spec(w)),
}

by = {d.name: d for d in DEMOS}
held = sorted((by[n] for n in UNTOUCHED), key=lambda d: d.n_dim)
rows = []
names = list(FIELD)
for budget in BUDGETS:
    for dm in held:
        for s in SEEDS:
            inst = disguise_demo(dm, s)
            vals = {}
            for name, opt in FIELD.items():
                v = run_discovered(opt, inst.objective, dm.n_dim, budget, 9500 + s)
                vals[name] = None if v >= INF else v
            big = 1e18
            ranks = {
                o: 1
                + sum(
                    1
                    for x in names
                    if (vals[x] if vals[x] is not None else big)
                    < (vals[o] if vals[o] is not None else big) - 1e-12
                )
                for o in names
            }
            rows.append(
                {
                    "budget": budget,
                    "demo": dm.name,
                    "seed": s,
                    "vals": vals,
                    "ranks": ranks,
                }
            )
            if len(rows) % 25 == 0:
                print(f"[{len(rows)}/580]", flush=True)
                atomic_dump({"done": False, "rows": rows}, OUT)

summary = {
    str(b): {
        o: round(
            sum(r["ranks"][o] for r in rows if r["budget"] == b)
            / sum(1 for r in rows if r["budget"] == b),
            3,
        )
        for o in names
    }
    for b in BUDGETS
}
atomic_dump({"done": True, "field": names, "summary": summary, "rows": rows}, OUT)
print(json.dumps(summary, indent=1))
