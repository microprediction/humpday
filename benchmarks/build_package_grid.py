"""Derive the slim recommendation grid that ships inside the package.

`benchmarks/recommendation_grid.json` is the full analysis artifact: every run, every timing, the
Borda intermediates. `papers/dfo_recommender/analysis.py` needs all of it and the JavaScript asset
is generated from it, so it stays where it is.

The installed package needs four fields per entry and nothing else -- `eligibility.recommend`
reads `borda_score`, falls back to `median_best`, uses `mean_wall` in cost-aware mode, and
`borda_n` says how many runs back the score. Shipping the other four fields, and the full `runs`
list, costs 2.8MB in a wheel whose whole argument is that it installs anywhere.

    python benchmarks/build_package_grid.py

Re-run whenever benchmarks/recommendation_grid.json is rebuilt.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "benchmarks" / "recommendation_grid.json"
DST = REPO_ROOT / "humpday" / "data" / "recommendation_grid.json"

KEEP = ("borda_score", "median_best", "mean_wall", "borda_n")


def main() -> int:
    full = json.loads(SRC.read_text())
    slim = {
        "cells": {
            key: {
                name: {k: entry[k] for k in KEEP if k in entry}
                for name, entry in cell.items()
            }
            for key, cell in full.get("cells", {}).items()
        },
        "meta": dict(full.get("meta", {}), derived_from=SRC.name, fields=list(KEEP)),
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(slim, separators=(",", ":"), sort_keys=True))
    print(
        f"{SRC.stat().st_size / 1e6:.1f}MB -> {DST.stat().st_size / 1e6:.2f}MB "
        f"({len(slim['cells'])} cells)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
