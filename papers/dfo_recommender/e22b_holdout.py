"""E22b — Held-out validation for generated forecasters (seeds 20-24)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import forecast_sim as fs  # noqa: E402
from e22_forecast_simplex import compile_forecaster  # noqa: E402

SEEDS = (20, 21, 22, 23, 24)
instances = [fs.make_instance(f, s) for f in fs.FAMILIES for s in SEEDS]
print(f"{len(instances)} held-out instances; panel baselines...", flush=True)
cache = fs.build_panel_cache(instances)

field = dict(list(fs.PANEL.items()) + [("laplace", fs.LaplaceRef)])
for path in sys.argv[1:]:
    p = Path(path)
    field[p.parent.name.replace("_code", "").replace("e22_", "") + ":" + p.stem] = (
        compile_forecaster(p.read_text())
    )

results = {}
for name, cls in field.items():
    results[name] = fs.score_forecaster(cls, instances, cache)
    print(f"  {name:34s} {results[name]:.4f}", flush=True)

Path("runs/e22b_holdout.json").write_text(
    json.dumps({"seeds": list(SEEDS), "results": results}, indent=2)
)
print("\n=== held-out leaderboard ===")
for k, v in sorted(results.items(), key=lambda t: t[1]):
    print(f"  {v:.4f}  {k}")
