"""E9 — Is the simplex just hyperparameters? The decisive baseline.

portfolio_w.make_portfolio(w) is ONE hand-written program whose behaviour is
driven by the same slot shares the LLM prompts use (architecture copied from
the winning centroid program, so the baseline inherits the discovered glue).
No LLM anywhere; evaluations are cheap.

Arms (same suite, seeds, trials, and outer-search protocol as E7, so results
are directly comparable point-for-point):

  rand20_portfolio (x2 reps) : the SAME 20 random w's E7's rand_simplex arm
                               draws (same RNG seed), instantiated by the
                               portfolio instead of the LLM.
  dfo20_portfolio  (x2 reps) : centroid program as outer DFO over w, 20 evals
                               (mirrors E7's dfo_simplex).
  rand100_ceiling  (x1)      : 100 random w's to map the family's ceiling.

If the tuned portfolio matches the LLM blends, the coordinate is doing all
the work and 'concept space' collapses to hyperparameters. If the blends
win, the per-point semantic layer carries real information.

    ../../.venv/bin/python e9_portfolio.py --out runs/e9_portfolio.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simplex_blend as sb  # noqa: E402
from example_demos import DEMOS  # noqa: E402
from portfolio_w import make_portfolio  # noqa: E402

from humpday.transforms.cubetosimplex import cube_to_simplex  # noqa: E402

SUITE = [
    "espresso_dialin", "facility_location", "gear_ratios", "kalman_tuning",
    "pid_tuning", "tension_spring", "plinko_funnel", "cassini_minlp",
]
SEEDS = (0, 1)
TRIALS = 100


def load_outer():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "centroid_opt", str(HERE / "runs/simplex_warm_code/centroid.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.optimize


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/e9_portfolio.json")
    args = ap.parse_args()

    by_name = {d.name: d for d in DEMOS}
    base = [by_name[n] for n in SUITE]
    print("  precomputing panel baselines...", flush=True)
    panel_cache = sb.build_panel_cache(base, SEEDS, TRIALS)
    outer = load_outer()

    runs = []

    def save(final=False):
        tmp = Path(args.out + ".tmp")
        tmp.write_text(json.dumps({
            "done": final, "suite": SUITE, "seeds": list(SEEDS),
            "trials": TRIALS, "runs": runs,
        }, indent=2))
        tmp.replace(Path(args.out))

    def eval_w(u, history):
        w = list(cube_to_simplex([min(1.0, max(0.0, x)) for x in u]))
        spec = sb.weights_to_spec(w)
        opt = make_portfolio(w, spec)
        regret = sb.score_optimizer(opt, base, SEEDS, TRIALS, panel_cache=panel_cache)
        history.append({"u": [round(x, 4) for x in u],
                        "w": spec["inspiration"], "regret": regret})
        print(f"    eval {len(history):3d} regret={regret:.4f}", flush=True)
        return regret

    # matched arms
    for rep in (0, 1):
        for arm, n_evals in (("rand20_portfolio", 20), ("dfo20_portfolio", 20)):
            print(f"  === {arm} rep{rep} ===", flush=True)
            history = []
            if arm.startswith("rand"):
                rng = random.Random(4200 + rep)  # SAME points as E7 rand_simplex
                for _ in range(n_evals):
                    eval_w([rng.random() for _ in range(4)], history)
            else:
                random.seed(4300 + rep)  # mirrors E7 dfo_simplex
                try:
                    import numpy as np

                    np.random.seed(4300 + rep)
                except Exception:  # noqa: BLE001
                    pass
                outer(lambda u: eval_w(u, history), n_evals, 4)
            best = min(h["regret"] for h in history)
            runs.append({"arm": arm, "repeat": rep, "history": history, "best": best})
            print(f"  {arm} rep{rep}: best={best:.4f}", flush=True)
            save()

    # ceiling arm
    print("  === rand100_ceiling ===", flush=True)
    history = []
    rng = random.Random(777)
    for _ in range(100):
        eval_w([rng.random() for _ in range(4)], history)
    runs.append({"arm": "rand100_ceiling", "repeat": 0, "history": history,
                 "best": min(h["regret"] for h in history)})
    save(final=True)

    print("\n=== summary ===")
    for arm in ("rand20_portfolio", "dfo20_portfolio", "rand100_ceiling"):
        rs = [r for r in runs if r["arm"] == arm]
        if rs:
            print(f"  {arm:18s} best={min(r['best'] for r in rs):.4f} "
                  f"mean_best={mean(r['best'] for r in rs):.4f}")
    print("  (compare: LLM centroid drew 0.087 on the warm suite; E7 arms "
          "score on THIS suite — compare within E7/E9 only)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
