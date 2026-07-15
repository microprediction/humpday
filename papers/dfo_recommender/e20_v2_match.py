"""E20 — Round-2 winner (tuned regenerated CentroidBlend) vs Alloy.

Fresh-instance match on the untouched demos at disguise seeds 10-14 (never
used). Field: Alloy, the E12 basin winner at its tuned parameters, and the
same program at default parameters (its ablated twin in the tuning sense).
Pre-registered: pairwise sign tests decide; contenders locked at launch.
"""

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
from e2_hardening import INF, atomic_dump, load_centroid, run_discovered  # noqa: E402
from e6_untouched import UNTOUCHED  # noqa: E402
from example_demos import DEMOS, disguise_demo  # noqa: E402

SEEDS = (10, 11, 12, 13, 14)
BUDGETS = (60, 120, 240, 480)
OUT = "runs/e20_v2_match.json"


def load_e12(tuned):
    code = open("runs/e12_code/pure_CentroidBlend.py").read()
    m = types.ModuleType("e12w")
    exec(compile(code, "<e12>", "exec"), m.__dict__)  # noqa: S102
    if tuned:
        d = json.load(open("runs/e12_round2.json"))
        floor = next(
            f for f in d["descents"] if f["representative"] == "pure:CentroidBlend"
        )
        u = floor["best_params"]
        names = [
            n
            for n in m.PARAMS
            if n in m.PARAM_RANGES and isinstance(m.PARAMS[n], (int, float))
        ]
        for x, n in zip(u, names):
            lo, hi = m.PARAM_RANGES[n]
            val = lo + min(1.0, max(0.0, x)) * (hi - lo)
            if isinstance(m.PARAMS[n], int):
                val = int(round(val))
            m.PARAMS[n] = val
    return m.optimize


FIELD = {
    "Alloy": load_centroid(),
    "v2_tuned": load_e12(True),
    "v2_default": load_e12(False),
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
                v = run_discovered(opt, inst.objective, dm.n_dim, budget, 9700 + s)
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
