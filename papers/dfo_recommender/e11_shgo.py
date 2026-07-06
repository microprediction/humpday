"""E11 — SHGO-in-concept-space: the two-level search as an actual algorithm.

Phase 1 SAMPLE   : generate programs at the 5 vertices + centroid + 6 random
                   recipes (12 LLM calls). Each program must expose PARAMS
                   (tunable constants) and PARAM_RANGES (their bounds).
Phase 2 BASINS   : cluster the programs by behavior (per-instance score
                   vectors), k clusters via farthest-point seeding.
Phase 3 DESCEND  : one parametric descent per basin: tune the best member's
                   PARAMS with a 14-evaluation DFO (the centroid program as
                   inner optimizer). Basin floors, not landing points.
Phase 4 RETURN   : the best tuned floor.

Budget: 12 LLM generations (vs 20 for each E7 arm) plus cheap CPU scoring.
Compare the returned floor against E7/E9 arm bests on the same suite.
All generated code is SAVED this time (runs/e11_code/).

    ../../.venv/bin/python e11_shgo.py --dry-run --out runs/e11_smoke.json
    ../../.venv/bin/python e11_shgo.py --out runs/e11_shgo.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simplex_blend as sb  # noqa: E402
from example_demos import DEMOS, disguise_demo  # noqa: E402

from humpday.transforms.cubetosimplex import cube_to_simplex  # noqa: E402

SUITE = [
    "espresso_dialin", "facility_location", "gear_ratios", "kalman_tuning",
    "pid_tuning", "tension_spring", "plinko_funnel", "cassini_minlp",
]
SEEDS = (0, 1)
TRIALS = 100
N_RANDOM = 6
K_BASINS = 3
INNER_EVALS = 14

PARAMS_ADDENDUM = """

ADDITIONALLY, and importantly: expose the 4 to 8 most behaviour-critical
numeric constants of your design as module-level dicts

    PARAMS = {"name": default_value, ...}          # plain numbers
    PARAM_RANGES = {"name": (low, high), ...}      # sensible bounds

and READ every such constant inside `optimize` via PARAMS["name"], so that
overwriting a PARAMS entry changes the optimizer's behaviour. Defaults must
reproduce your intended design. Include things like population sizing
factors, step sizes, cooling rates, mutation scales, restart patience."""


def compile_with_params(code):
    """Exec code in a module; return (optimize, module) or raise."""
    import types

    m = types.ModuleType("gen_opt")
    exec(compile(code, "<generated>", "exec"), m.__dict__)  # noqa: S102
    if not callable(getattr(m, "optimize", None)):
        raise ValueError("no callable optimize")
    return m.optimize, m


def score_vec(opt, base, panel_cache):
    out = []
    INF = float("inf")
    for i, demo in enumerate(base):
        for s in SEEDS:
            inst = disguise_demo(demo, s)
            seed = 5000 + 31 * i + s
            random.seed(seed)
            try:
                import numpy as np

                np.random.seed(seed)
            except Exception:  # noqa: BLE001
                pass
            try:
                res = opt(inst.objective, TRIALS, inst.n_dim)
                cand = float(res[0]) if res else INF
            except Exception:  # noqa: BLE001
                cand = INF
            panel_vals = panel_cache[(i, s)]
            vals = [cand] + panel_vals
            finite = [v for v in vals if v < INF]
            if cand >= INF or not finite:
                out.append(1.0)
                continue
            mn, mx = min(finite), max(finite)
            out.append(0.0 if mx <= mn else (cand - mn) / (mx - mn))
    return out


def farthest_point_clusters(items, k):
    """items: list of dicts with 'vector'. Returns list of member-index lists."""
    if len(items) <= k:
        return [[i] for i in range(len(items))]

    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    centers = [min(range(len(items)), key=lambda i: items[i]["regret"])]
    while len(centers) < k:
        far = max(
            (i for i in range(len(items)) if i not in centers),
            key=lambda i: min(dist(items[i]["vector"], items[c]["vector"]) for c in centers),
        )
        centers.append(far)
    clusters = [[] for _ in centers]
    for i in range(len(items)):
        j = min(range(len(centers)), key=lambda c: dist(items[i]["vector"], items[centers[c]]["vector"]))
        clusters[j].append(i)
    return clusters


def load_inner_optimizer():
    spec = importlib.util.spec_from_file_location(
        "centroid_opt", str(HERE / "runs/simplex_warm_code/centroid.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.optimize


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/e11_shgo.json")
    ap.add_argument("--code-dir", default="runs/e11_code")
    args = ap.parse_args()

    by_name = {d.name: d for d in DEMOS}
    base = [by_name[n] for n in SUITE]
    code_dir = Path(args.code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    print("  precomputing panel baselines...", flush=True)
    panel_cache = sb.build_panel_cache(base, SEEDS, TRIALS)
    inner = load_inner_optimizer()

    state = {"phase1": [], "descents": [], "done": False}

    def save():
        tmp = Path(args.out + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.replace(Path(args.out))

    # ---- Phase 1: sample the complex -------------------------------------
    pts = [(f"pure:{v['name']}", i) for i, v in enumerate(sb.VERTICES)]
    recipes = []
    for label, i in pts:
        w = [0.001] * sb.N_VERTICES
        w[i] = 1.0 - 0.001 * (sb.N_VERTICES - 1)
        recipes.append((label, w))
    recipes.append(("centroid", list(cube_to_simplex([0.5] * (sb.N_VERTICES - 1)))))
    rng = random.Random(1100)
    for j in range(N_RANDOM):
        recipes.append((f"rand{j}", list(cube_to_simplex([rng.random() for _ in range(sb.N_VERTICES - 1)]))))

    items = []
    for label, w in recipes:
        prompt = sb.build_prompt(sb.weights_to_spec(w)) + PARAMS_ADDENDUM
        try:
            code = sb._DRY_TEMPLATE if args.dry_run else sb.generate_live(prompt, args.model)
            (code_dir / f"{label.replace(':', '_')}.py").write_text(code)
            opt, mod = compile_with_params(code)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:16s} FAILED: {e}", flush=True)
            state["phase1"].append({"label": label, "w": w, "regret": 1.0, "failed": True})
            save()
            continue
        vec = score_vec(opt, base, panel_cache)
        regret = mean(vec)
        n_params = len(getattr(mod, "PARAMS", {}) or {})
        items.append({"label": label, "w": w, "regret": regret, "vector": vec,
                      "code": code, "n_params": n_params})
        state["phase1"].append({"label": label, "w": w, "regret": regret,
                                "n_params": n_params})
        print(f"  {label:16s} regret={regret:.4f} params={n_params}", flush=True)
        save()

    # ---- Phase 2: basins ---------------------------------------------------
    clusters = farthest_point_clusters(items, K_BASINS)
    print("\n  basins:", [[items[i]["label"] for i in c] for c in clusters], flush=True)

    # ---- Phase 3: one descent per basin ------------------------------------
    for ci, members in enumerate(clusters):
        if not members:
            continue
        best_i = min(members, key=lambda i: items[i]["regret"])
        it = items[best_i]
        try:
            opt, mod = compile_with_params(it["code"])
        except Exception:  # noqa: BLE001
            continue
        params = dict(getattr(mod, "PARAMS", {}) or {})
        ranges = dict(getattr(mod, "PARAM_RANGES", {}) or {})
        names = [n for n in params if n in ranges and isinstance(params[n], (int, float))]
        floor = {"basin": ci, "members": [items[i]["label"] for i in members],
                 "representative": it["label"], "start": it["regret"],
                 "tuned": it["regret"], "best_params": None, "history": []}
        if names:
            def objective(u):
                for x, n in zip(u, names):
                    lo, hi = ranges[n]
                    val = lo + min(1.0, max(0.0, x)) * (hi - lo)
                    if isinstance(params[n], int):
                        val = int(round(val))
                    mod.PARAMS[n] = val
                vec = score_vec(opt, base, panel_cache)
                r = mean(vec)
                floor["history"].append({"u": [round(x, 3) for x in u], "regret": r})
                print(f"    basin{ci} inner {len(floor['history']):2d}/{INNER_EVALS} "
                      f"regret={r:.4f}", flush=True)
                return r

            random.seed(1100 + ci)
            inner(objective, INNER_EVALS, len(names))
            if floor["history"]:
                hbest = min(floor["history"], key=lambda h: h["regret"])
                floor["tuned"] = min(it["regret"], hbest["regret"])
                floor["best_params"] = hbest["u"]
        print(f"  basin{ci} [{it['label']}] start={floor['start']:.4f} "
              f"tuned floor={floor['tuned']:.4f}", flush=True)
        state["descents"].append(floor)
        save()

    # ---- Phase 4 ------------------------------------------------------------
    if state["descents"]:
        best = min(state["descents"], key=lambda f: f["tuned"])
        state["result"] = {"basin": best["basin"], "representative": best["representative"],
                           "floor": best["tuned"]}
        print(f"\n=== E11 result: floor={best['tuned']:.4f} from "
              f"{best['representative']} (E7 arm bests: 0.246-0.312) ===", flush=True)
    state["done"] = True
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
