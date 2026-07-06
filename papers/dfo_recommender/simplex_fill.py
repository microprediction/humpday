"""
Fill the two evidential gaps in the simplex-of-inspirations warm run
(runs/simplex_warm.json):

1. pure:CMAEvolutionStrategy never scored — its generated code failed to
   compile and the point was dropped, so "centroid beats every vertex" has a
   hole. Regenerate (up to --retries attempts) and score it.

2. The winning centroid (regret 0.0866) is a single stochastic LLM draw.
   Generate --centroid-draws fresh centroids at the same equal weights and
   score each, so the paper can report the spread across draws rather than
   one possibly-lucky sample.

Config matches the warm run exactly: 16 spread-mode demos, seeds 0,1,2,
120 trials, claude-opus-4-8, same panel + normalisation via
simplex_blend.score_optimizer. Crash-safe per-point checkpoints.

    ../../.venv/bin/python simplex_fill.py --out runs/simplex_fill.json \
        --save-code runs/simplex_fill_code
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simplex_blend as sb  # noqa: E402


def one_hot(idx):
    w = [0.001] * sb.N_VERTICES
    w[idx] = 1.0 - 0.001 * (sb.N_VERTICES - 1)
    return w


def centroid_weights():
    from humpday.transforms.cubetosimplex import cube_to_simplex

    return cube_to_simplex([0.5] * (sb.N_VERTICES - 1))


def generate_scored(label, weights, args, panel_cache, base, seeds, save_dir):
    spec = sb.weights_to_spec(weights)
    prompt = sb.build_prompt(spec)
    for attempt in range(1, args.retries + 1):
        try:
            code = sb.generate_live(prompt, args.model)
        except Exception as e:  # noqa: BLE001
            print(f"  {label} attempt {attempt} generation FAILED: {e}", flush=True)
            continue
        if save_dir:
            fn = save_dir / f"{label.replace(':', '_')}_a{attempt}.py"
            fn.write_text(code)
        try:
            opt = sb.compile_optimizer(code)
        except Exception as e:  # noqa: BLE001
            print(f"  {label} attempt {attempt} compile FAILED: {e}", flush=True)
            continue
        regret = sb.score_optimizer(
            opt, base, seeds, args.trials, panel_cache=panel_cache
        )
        print(f"  {label} regret={regret:.4f} (attempt {attempt})", flush=True)
        return {
            "label": label,
            "regret": regret,
            "attempts": attempt,
            "inspiration": spec["inspiration"],
        }
    print(f"  {label} FAILED after {args.retries} attempts", flush=True)
    return {"label": label, "regret": None, "attempts": args.retries,
            "inspiration": spec["inspiration"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--centroid-draws", type=int, default=4)
    ap.add_argument("--out", default="runs/simplex_fill.json")
    ap.add_argument("--save-code", default="runs/simplex_fill_code")
    args = ap.parse_args()

    ckpt = Path(args.out)
    save_dir = Path(args.save_code)
    save_dir.mkdir(parents=True, exist_ok=True)

    base = sb.select_demos(args.demos, "spread")  # SAME 16 as the warm run
    seeds = tuple(int(s) for s in args.seeds.split(","))

    print("  precomputing panel baselines...", flush=True)
    panel_cache = sb.build_panel_cache(base, seeds, args.trials)

    results = []

    def write_ckpt(done):
        payload = {
            "done": done,
            "vertices": [v["name"] for v in sb.VERTICES],
            "n_demos": args.demos,
            "seeds": args.seeds,
            "trials": args.trials,
            "model": args.model,
            "results": results,
        }
        tmp = Path(str(ckpt) + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(ckpt)

    cma_idx = next(
        i for i, v in enumerate(sb.VERTICES) if v["name"] == "CMAEvolutionStrategy"
    )
    todo = [("pure:CMAEvolutionStrategy", one_hot(cma_idx))]
    cw = centroid_weights()
    for j in range(args.centroid_draws):
        todo.append((f"centroid_d{j + 1}", cw))

    for label, w in todo:
        results.append(
            generate_scored(label, w, args, panel_cache, base, seeds, save_dir)
        )
        write_ckpt(done=False)

    write_ckpt(done=True)
    scored = sorted(
        (r for r in results if r["regret"] is not None), key=lambda r: r["regret"]
    )
    print("\n=== Fill results (warm-run scale; centroid was 0.0866) ===")
    for r in scored:
        print(f"  {r['regret']:.4f}  {r['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
