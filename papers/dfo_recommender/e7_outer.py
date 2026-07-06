"""E7 — Does the outer derivative-free loop have real bite?

Everything so far scored FIXED simplex points (vertices, centroid, random
sweeps). This experiment runs the actual outer search and asks two questions
with budget-matched arms (same evaluation count, same suite, same scoring):

  1. BITE:   does a DFO outer loop find better recipes than random sampling
             of the simplex? (dfo_simplex vs rand_simplex)
  2. SPACE:  is searching concept space (4-cube -> 5-simplex -> LLM-generated
             program) better per evaluation than searching the collapsed
             parametric alternative (14-gene template, no LLM)?
             (dfo_simplex vs dfo_template, rand_simplex vs rand_template)

The outer optimizer for the dfo arms is the centroid program itself
(runs/simplex_warm_code/centroid.py), i.e. the artifact the construction
discovered drives the next round of discovery. Each objective evaluation is
one draw of the noisy map: generate (LLM for simplex arms, decode for
template arms), compile, score on the selection suite. Compile failure
scores 1.0 rather than being retried, so arms pay for their failures.

Suite: 8 previously-used selection-era demos x seeds {0,1} x 100 trials
(deliberately drawn from already-burned demos so the untouched E6 pool stays
clean for final validation). Crash-safe per-evaluation checkpoints; resume
by re-running with the same --out.

    ../../.venv/bin/python e7_outer.py --dry-run --out runs/e7_smoke.json
    ../../.venv/bin/python e7_outer.py --out runs/e7_outer.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import algo_dev as ad  # noqa: E402
import simplex_blend as sb  # noqa: E402
from example_demos import DEMOS  # noqa: E402

from humpday.transforms.cubetosimplex import cube_to_simplex  # noqa: E402

# Selection-era demos (in the 2026-06-16 spread-16 set), NOT in the E6
# untouched pool. Verify by name at startup.
SUITE = [
    "espresso_dialin",
    "facility_location",
    "gear_ratios",
    "kalman_tuning",
    "pid_tuning",
    "tension_spring",
    "plinko_funnel",
    "cassini_minlp",
]
SEEDS = (0, 1)
TRIALS = 100
N_EVALS = 20
REPEATS = (0, 1)


def load_outer_optimizer():
    spec = importlib.util.spec_from_file_location(
        "centroid_opt", str(HERE / "runs/simplex_warm_code/centroid.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.optimize


def make_simplex_eval(base, panel_cache, args, history):
    """Objective on [0,1]^4: cube -> simplex weights -> LLM program -> regret."""

    def f(u):
        w = cube_to_simplex([min(1.0, max(0.0, x)) for x in u])
        spec = sb.weights_to_spec(list(w))
        prompt = sb.build_prompt(spec)
        try:
            code = (
                sb._DRY_TEMPLATE
                if args.dry_run
                else sb.generate_live(prompt, args.model)
            )
            opt = sb.compile_optimizer(code)
            regret = sb.score_optimizer(
                opt, base, SEEDS, TRIALS, panel_cache=panel_cache
            )
        except Exception as e:  # noqa: BLE001
            print(f"      eval failed ({e}); regret=1.0", flush=True)
            regret = 1.0
        history.append(
            {
                "u": [round(x, 4) for x in u],
                "w": dict(spec["inspiration"]),
                "regret": regret,
            }
        )
        print(f"    eval {len(history):2d}/{N_EVALS} regret={regret:.4f}", flush=True)
        return regret

    return f


def make_template_eval(base, panel_cache, history):
    """Objective on [0,1]^14: genome -> algo_dev candidate -> regret."""

    def f(u):
        g = [min(1.0, max(0.0, x)) for x in u]
        try:
            opt = ad.make_candidate(g)
            regret = sb.score_optimizer(
                opt, base, SEEDS, TRIALS, panel_cache=panel_cache
            )
        except Exception as e:  # noqa: BLE001
            print(f"      eval failed ({e}); regret=1.0", flush=True)
            regret = 1.0
        history.append({"u": [round(x, 4) for x in u], "regret": regret})
        print(f"    eval {len(history):2d}/{N_EVALS} regret={regret:.4f}", flush=True)
        return regret

    return f


def run_random(f, n_dim, n_evals, rng):
    for _ in range(n_evals):
        f([rng.random() for _ in range(n_dim)])


def run_dfo(outer, f, n_dim, n_evals, seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    outer(f, n_evals, n_dim)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/e7_outer.json")
    args = ap.parse_args()

    by_name = {d.name: d for d in DEMOS}
    missing = [n for n in SUITE if n not in by_name]
    if missing:
        raise SystemExit(f"suite demos missing: {missing}")
    base = [by_name[n] for n in SUITE]

    print("  precomputing panel baselines...", flush=True)
    panel_cache = sb.build_panel_cache(base, SEEDS, TRIALS)
    outer = load_outer_optimizer()

    out = Path(args.out)
    runs = []
    done = set()
    if out.exists():
        try:
            runs = json.load(open(out))["runs"]
            done = {
                (r["arm"], r["repeat"]) for r in runs if len(r["history"]) >= N_EVALS
            }
        except Exception:  # noqa: BLE001
            pass

    def save(final=False):
        tmp = Path(str(out) + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "done": final,
                    "suite": SUITE,
                    "seeds": list(SEEDS),
                    "trials": TRIALS,
                    "n_evals": N_EVALS,
                    "model": None if args.dry_run else args.model,
                    "outer": "centroid (runs/simplex_warm_code/centroid.py)",
                    "runs": runs,
                },
                indent=2,
            )
        )
        tmp.replace(out)

    arms = [
        ("rand_simplex", "random", 4, "simplex"),
        ("dfo_simplex", "dfo", 4, "simplex"),
        ("rand_template", "random", 14, "template"),
        ("dfo_template", "dfo", 14, "template"),
    ]
    for rep in REPEATS:
        for arm, kind, n_dim, space in arms:
            if (arm, rep) in done:
                print(f"  {arm} rep{rep}: already complete, skipping", flush=True)
                continue
            print(f"  === {arm} rep{rep} ===", flush=True)
            history = []
            if space == "simplex":
                f = make_simplex_eval(base, panel_cache, args, history)
            else:
                f = make_template_eval(base, panel_cache, history)
            if kind == "random":
                run_random(f, n_dim, N_EVALS, random.Random(4200 + rep))
            else:
                run_dfo(outer, f, n_dim, N_EVALS, 4300 + rep)
            best = min((h["regret"] for h in history), default=1.0)
            runs.append({"arm": arm, "repeat": rep, "history": history, "best": best})
            print(f"  {arm} rep{rep}: best={best:.4f}", flush=True)
            save()

    save(final=True)
    print("\n=== best regret found per arm (20 evals each) ===")
    from statistics import mean

    for arm, _, _, _ in arms:
        bests = [r["best"] for r in runs if r["arm"] == arm]
        if bests:
            print(
                f"  {arm:14s} mean_best={mean(bests):.4f}  ({', '.join(f'{b:.4f}' for b in bests)})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
