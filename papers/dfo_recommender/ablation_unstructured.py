"""Ablation #1: does the Inspiration Simplex beat just asking Opus for the best
optimizer it can write, with NO simplex / no blend recipe?

If unstructured "write your strongest optimizer" generations match the best simplex
blend, the barycentric structure is decoration. If the blends win, the structure
adds value. Scores on the SAME suite/config as runs/simplex_warm.json (16 'spread'
demos x seeds 0,1,2 x 120 trials, panel-normalised regret) so numbers are directly
comparable to that leaderboard (centroid won at 0.0866).
"""
from __future__ import annotations
import argparse, json, os, tempfile
from simplex_blend import (
    generate_live, compile_optimizer, score_optimizer, build_panel_cache, select_demos,
)

PROMPT = """You are an expert in derivative-free optimization. Write the BEST \
general-purpose black-box optimizer you can, as one self-contained pure-Python \
function. Use whatever combination of techniques you judge strongest.

Exact contract:
- Define `def optimize(objective, n_trials, n_dim):`
- `objective` takes a list of `n_dim` floats in [0,1] and returns a float to MINIMISE.
- Make AT MOST `n_trials` calls to `objective`; respect the budget exactly.
- Return `(best_value, best_point)`.
- Standard library only (`math`, `random`); no numpy/scipy.
- Output only the code block.

This is independent design attempt #{k} of several; bring your strongest ideas \
(population methods, local search, models, restarts, annealing — your choice)."""


def atomic_dump(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument("--demos", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--save-code", default="runs/ablation_unstructured_code")
    ap.add_argument("--out", default="runs/ablation_unstructured.json")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    base = select_demos(args.demos, "spread")  # SAME 16 as the simplex run
    print(f"unstructured ablation: {args.attempts} Opus attempts vs simplex centroid (0.0866)\n"
          f"{len(base)} demos x {len(seeds)} seeds x {args.trials} trials\n", flush=True)
    print("  precomputing panel baselines (cached)...", flush=True)
    panel_cache = build_panel_cache(base, seeds, args.trials)

    save_dir = args.save_code
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    for k in range(args.attempts):
        label = f"unstructured{k}"
        try:
            code = generate_live(PROMPT.format(k=k + 1), args.model)
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: generation FAILED: {e}", flush=True)
            continue
        if save_dir:
            with open(os.path.join(save_dir, f"{label}.py"), "w") as fh:
                fh.write(code)
        try:
            opt = compile_optimizer(code)
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: compile FAILED: {e}", flush=True)
            continue
        regret = score_optimizer(opt, base, seeds, args.trials, panel_cache=panel_cache)
        results.append({"label": label, "regret": regret})
        print(f"  {label:16s} regret={regret:.4f}", flush=True)
        atomic_dump({"done": False, "results": results}, args.out)

    results.sort(key=lambda r: r["regret"])
    best = results[0]["regret"] if results else None
    atomic_dump({"done": True, "n_attempts": args.attempts, "demos": len(base),
                 "seeds": list(seeds), "trials": args.trials,
                 "best_unstructured": best, "simplex_centroid": 0.0866,
                 "results": results}, args.out)
    print("\n=== unstructured leaderboard (panel-normalised regret) ===")
    for r in results:
        print(f"  {r['regret']:.4f}  {r['label']}")
    if best is not None:
        verdict = ("SIMPLEX WINS (blend structure adds value)" if best > 0.0866 + 0.01
                   else "TIE/UNSTRUCTURED WINS (structure may be decoration)")
        print(f"\nbest unstructured = {best:.4f}  vs  simplex centroid = 0.0866  ->  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
