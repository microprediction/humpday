"""E17 — Search the extended (signed) simplex with an actual outer DFO,
caricature semantics throughout.

Coordinates: u in [0,1]^5 maps to raw = -1 + 3u, shifted to sum to 1: the
signed affine slice with weights in roughly [-1, 2]. Positive weights blend
hosts (largest = architecture, others graft); each negative weight w_k < 0
names a caricature reference with exaggeration length lam_k = |w_k|.

Arms (one repeat each, same dimension-spread suite as E15/E16):
  dfo    : Alloy searches the space, 25 evaluations (sequential)
  random : 25 random signed points (shardable control)

Objective is the generated program's regret; the ablated twin (ANTI gate to
zero) is also scored at every point so the paired delta is logged and the
caricature's causal share is known point by point.

    ../../.venv/bin/python e17_search.py --arm dfo --out runs/e17_dfo.json
    ../../.venv/bin/python e17_search.py --arm random --out runs/e17_rand.json
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

import simplex_blend as sb  # noqa: E402
from e15_semantics import GATE, spread_suite  # noqa: E402

SEEDS = (0, 1)
TRIALS = 100
N_EVALS = 25
NAMES = [v["name"] for v in sb.VERTICES]


def u_to_signed(u):
    raw = [-1.0 + 3.0 * min(1.0, max(0.0, x)) for x in u]
    shift = (sum(raw) - 1.0) / len(raw)
    return [x - shift for x in raw]


def build_prompt(w):
    idea = {v["name"]: v["idea"] for v in sb.VERTICES}
    wmap = dict(zip(NAMES, w))
    pos = {k: v for k, v in wmap.items() if v > 0.05}
    neg = {k: -v for k, v in wmap.items() if v < -0.05}
    if not pos:
        pos = {max(wmap, key=wmap.get): 1.0}
    host = max(pos, key=pos.get)
    pos_tot = sum(pos.values())
    lines = [
        "Build a black-box numerical optimizer from the SIGNED recipe: "
        + ", ".join(f"{k} {v:+.0%}" for k, v in wmap.items() if abs(v) > 0.05),
        "",
        f"HOST ({int(100 * pos[host] / pos_tot)}% of positive mass): {host} — "
        f"{idea[host]}. Build the architecture from this.",
    ]
    grafts = sorted(((k, v) for k, v in pos.items() if k != host), key=lambda t: -t[1])
    if grafts:
        lines.append(
            "GRAFT into it: "
            + ", ".join(f"{int(100 * v / pos_tot)}% {k} ({idea[k]})" for k, v in grafts)
        )
    if neg:
        lines += [
            "",
            "Negative weights mean CARICATURE, extrapolation in design "
            "space: for each reference below, identify the axes on which "
            "the host most differs from it and EXAGGERATE the host along "
            "exactly those axes, with the size of each exaggeration "
            "proportional to the stated lam times ANTI_STRENGTH.",
        ]
        for k, mag in sorted(neg.items(), key=lambda t: -t[1]):
            lines.append(f"  CARICATURE AWAY FROM {k} (lam = {mag:.2f}): {idea[k]}")
    lines += [
        GATE.format(B=host),
        "",
        sb.CONTRACT,
        "",
        "Return ONLY a ```python code block containing the optimize function.",
    ]
    return "\n".join(lines)


def load_alloy():
    from humpday.optimizers.alloy import Alloy

    def optimize(objective, n_trials, n_dim):
        return Alloy(objective, n_trials, n_dim).optimize()

    return optimize


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("dfo", "random"), required=True)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-dir", default="runs/e17_code")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = spread_suite()
    code_dir = Path(args.code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    print("  precomputing panel baselines...", flush=True)
    panel_cache = sb.build_panel_cache(base, SEEDS, TRIALS)

    history = []

    def save(done):
        tmp = Path(args.out + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "done": done,
                    "arm": args.arm,
                    "suite": [d.name for d in base],
                    "n_evals": N_EVALS,
                    "model": None if args.dry_run else args.model,
                    "history": history,
                },
                indent=2,
            )
        )
        tmp.replace(Path(args.out))

    def score_code(code):
        opt = sb.compile_optimizer(code)
        return sb.score_optimizer(opt, base, SEEDS, TRIALS, panel_cache=panel_cache)

    def objective(u):
        w = u_to_signed(list(u))
        label = f"{args.arm}{args.seed}_e{len(history):02d}"
        row = {
            "label": label,
            "u": [round(float(x), 4) for x in u],
            "w": dict(zip(NAMES, [round(x, 3) for x in w])),
            "regret": 1.0,
            "ablated": None,
            "delta": None,
        }
        try:
            code = (
                sb._DRY_TEMPLATE
                if args.dry_run
                else sb.generate_live(build_prompt(w), args.model)
            )
            (code_dir / f"{label}.py").write_text(code)
            row["regret"] = score_code(code)
            if "ANTI_STRENGTH = 1.0" in code:
                row["ablated"] = score_code(
                    code.replace("ANTI_STRENGTH = 1.0", "ANTI_STRENGTH = 0.0")
                )
                row["delta"] = row["regret"] - row["ablated"]
        except Exception as e:  # noqa: BLE001
            print(f"  {label} FAILED: {e}", flush=True)
        history.append(row)
        d = row["delta"]
        print(
            f"  {label} regret={row['regret']:.4f}"
            + (f" twin={row['ablated']:.4f} delta={d:+.4f}" if d is not None else "")
            + f" best={min(h['regret'] for h in history):.4f}",
            flush=True,
        )
        save(False)
        return row["regret"]

    if args.arm == "random":
        rng = random.Random(1700 + args.seed)
        for _ in range(N_EVALS):
            objective([rng.random() for _ in range(5)])
    else:
        random.seed(1750 + args.seed)
        try:
            import numpy as np

            np.random.seed(1750 + args.seed)
        except Exception:  # noqa: BLE001
            pass
        load_alloy()(objective, N_EVALS, 5)

    save(True)
    best = min(history, key=lambda h: h["regret"])
    print(
        f"\n=== {args.arm}: best {best['regret']:.4f} at {best['w']} "
        f"(delta {best['delta']}) ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
