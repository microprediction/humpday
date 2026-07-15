"""E21b — Held-out validation for cache-eviction artifacts.

Scores named generated policies, the panel, and ARC on unseen instances:
the same six trace families at seeds 20-24 (fresh parameters and key
permutations; nothing has used these). Regret normalised per instance
against the panel.

    ../../.venv/bin/python e21b_holdout.py runs/e21_adaptive_code/centroid.py ...
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cache_sim as cs  # noqa: E402
from e21_cache_simplex import compile_policy  # noqa: E402

SEEDS = (20, 21, 22, 23, 24)
OUT = "runs/e21b_holdout.json"

instances = [cs.make_instance(f, s) for f in cs.FAMILIES for s in SEEDS]
print(f"{len(instances)} held-out instances; panel baselines...", flush=True)
cache = cs.build_panel_cache(instances)

field = {name: cls for name, cls in list(cs.PANEL.items()) + [("ARC", cs.ARC)]}
for path in sys.argv[1:]:
    p = Path(path)
    field[p.parent.name.replace("_code", "") + ":" + p.stem] = compile_policy(
        p.read_text()
    )

results = {}
for name, cls in field.items():
    results[name] = cs.score_policy(cls, instances, cache)
    print(f"  {name:32s} {results[name]:.4f}", flush=True)

Path(OUT).write_text(json.dumps({"seeds": list(SEEDS), "results": results}, indent=2))
print("\n=== held-out leaderboard ===")
for k, v in sorted(results.items(), key=lambda t: t[1]):
    print(f"  {v:.4f}  {k}")
