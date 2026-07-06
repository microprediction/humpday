"""E6b — Do OTHER interior programs also generalize, or is the centroid special?

E6 validated only the centroid artifact out of sample. This extends the same
race (identical 29 untouched demos, budgets, seeds, deterministic disguise
instances) to the other strong selection-stage artifacts:

  rand14    : best of the 25 random interior points (selection regret 0.115)
  warm4     : best warm-neighbourhood point                        (0.152)
  warm7     : near-PatternSearch-corner point                      (0.155)
  pure_NM   : the LLM-generated pure Nelder-Mead vertex artifact   (0.160)

Existing field values (panel + centroid + surrogate + unstructured) are
REUSED from runs/e6_untouched.json — instances are deterministic
(disguise_demo(dm, seed), run seed 9000+seed) — so only the four new
programs are executed. Ranks are recomputed over the merged 11-way field.
Crash-safe; resume by re-running.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
from e2_hardening import INF, atomic_dump, run_discovered  # noqa: E402
from example_demos import DEMOS, disguise_demo  # noqa: E402

NEW_PROGRAMS = {
    "rand14": "runs/simplex_overnight_code/rand14.py",
    "warm4": "runs/simplex_warm_code/warm4.py",
    "warm7": "runs/simplex_warm_code/warm7.py",
    "pure_NM_gen": "runs/simplex_warm_code/pure_NelderMead.py",
}

SRC = "runs/e6_untouched.json"
OUT = "runs/e6b_more_points.json"


def load_program(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.optimize


def main():
    src = json.load(open(SRC))
    assert src["done"], "e6_untouched.json is not complete"
    base_rows = src["rows"]
    by_name = {d.name: d for d in DEMOS}
    progs = {k: load_program(v) for k, v in NEW_PROGRAMS.items()}
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
            v = run_discovered(
                opt, inst.objective, dm.n_dim, br["budget"], 9000 + br["seed"]
            )
            vals[name] = None if v >= INF else v
        big = 1e18
        ranks = {
            o: 1
            + sum(
                1
                for x in field
                if (vals.get(x) if vals.get(x) is not None else big)
                < (vals.get(o) if vals.get(o) is not None else big) - 1e-12
            )
            for o in field
        }
        rows.append(
            {
                **{k: br[k] for k in ("budget", "demo", "n", "seed")},
                "vals": vals,
                "ranks": ranks,
            }
        )
        if c % 10 == 0 or c == total:
            print(
                f"[{c}/{total}] {br['demo']} b={br['budget']} s={br['seed']}",
                flush=True,
            )
            atomic_dump({"done": False, "field": field, "rows": rows}, OUT)

    summary = {}
    budgets = sorted({r["budget"] for r in rows})
    for b in budgets:
        brs = [r for r in rows if r["budget"] == b]
        summary[str(b)] = {
            o: {
                "mean_rank": round(sum(r["ranks"][o] for r in brs) / len(brs), 3),
                "wins": sum(1 for r in brs if r["ranks"][o] == 1),
            }
            for o in field
        }
    atomic_dump({"done": True, "field": field, "summary": summary, "rows": rows}, OUT)

    print("\n=== mean rank by budget (1=best of 11) ===")
    for b in budgets:
        sb = summary[str(b)]
        ordered = sorted(field, key=lambda o: sb[o]["mean_rank"])
        print(
            f"  budget {b}: " + "  ".join(f"{o}={sb[o]['mean_rank']}" for o in ordered)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
