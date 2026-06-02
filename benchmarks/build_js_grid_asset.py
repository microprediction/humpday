"""
Build the JS-loadable slim recommendation grid from the canonical
`benchmarks/recommendation_grid.json`.

The full grid (~2.8 MB) stores raw per-(objective, seed) results for
provenance — invaluable for re-aggregating, but pure noise from the
recommender's perspective. The recommender only consults the per-cell
aggregates (median_best, borda_score, borda_worst), so the JS port
ships a slimmed-down asset (~110 KB) with just those.

Regenerate this anytime `recommendation_grid.json` changes:

    python benchmarks/build_js_grid_asset.py

The output is `docs/js/modules/recommendation-grid.js` — a tiny ESM/CJS
module that exports the grid dict. It's checked into the repo so
browser-only users don't have to build anything.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "benchmarks" / "recommendation_grid.json"
DST = REPO_ROOT / "docs" / "js" / "modules" / "recommendation-grid.js"

# Per-algorithm aggregate fields the recommender actually reads.
KEEP_FIELDS = (
    "median_best",
    "borda_score",
    "borda_worst",
    "mean_wall",
    "n_runs",
    "n_failures",
    "skipped_too_slow",
)


def main() -> int:
    full = json.loads(SRC.read_text())
    slim = {"meta": full.get("meta", {}), "cells": {}}
    for cell_key, cell in full["cells"].items():
        slim["cells"][cell_key] = {}
        for algo, entry in cell.items():
            slim["cells"][cell_key][algo] = {
                k: entry[k] for k in KEEP_FIELDS if k in entry
            }

    # Write as JS so the asset is consumable in both browser (via
    # <script> tag — populates `globalThis.RecommendationGrid`) and Node
    # (via require()).
    js = f"""\
/**
 * Slim recommendation grid for humpday.eligibility (JS port).
 *
 * Auto-generated from `benchmarks/recommendation_grid.json` by
 * `benchmarks/build_js_grid_asset.py`. Don't hand-edit — re-run the
 * build script after a fresh sweep.
 *
 * The recommender uses borda_score as the primary key (with
 * borda_worst as a tie-break / risk-aversion signal), falling back to
 * median_best for cells that pre-date the Borda refactor.
 */
const RecommendationGrid = {json.dumps(slim, separators=(",", ":"))};

if (typeof module !== "undefined" && module.exports) {{
    module.exports = RecommendationGrid;
}} else if (typeof globalThis !== "undefined") {{
    globalThis.RecommendationGrid = RecommendationGrid;
}}
"""
    DST.write_text(js)
    size_kb = len(js) / 1024
    print(f"Wrote {DST.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")
    print(f"Cells: {len(slim['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
