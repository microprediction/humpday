"""E13 — Signed compositions: thinking outside the simplex.

Extend recipes from barycentric (all weights >= 0, sum 1) to SIGNED affine
combinations (weights may be negative, still sum 1). A vertex with negative
weight is an ANTI-INSPIRATION: the generated program maintains a cheap
shadow of that algorithm's proposal logic (spending no objective calls on
it) and steers away from where it would go, with repulsion strength
proportional to the magnitude.

Arms (same 8-demo suite, seeds {0,1}, 100 trials, panel-normalised regret
as E7/E11, so numbers are directly comparable):

  signed_rand   : 12 random signed recipes, each with at least one genuinely
                  negative coordinate (the affine hull beyond the simplex).
  signed_canon  : 4 hand-picked recipes of the "-1 A, +2 B" form.
  unsigned_ctrl : 6 fresh random simplex points as an in-run control
                  (E7's rand arms got best-of-20 of 0.304/0.312; E11's
                  12-point sample got 0.298).

All code saved to runs/e13_code/. Compile failure scores 1.0.

    ../../.venv/bin/python e13_signed.py --dry-run --out runs/e13_smoke.json
    ../../.venv/bin/python e13_signed.py --out runs/e13_signed.json
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

from humpday.transforms.cubetosimplex import cube_to_simplex  # noqa: E402

SUITE = [
    "espresso_dialin", "facility_location", "gear_ratios", "kalman_tuning",
    "pid_tuning", "tension_spring", "plinko_funnel", "cassini_minlp",
]
SEEDS = (0, 1)
TRIALS = 100

CANONICAL = [
    ("canon:-1NM+2DE", {"NelderMead": -1.0, "DifferentialEvolution": 2.0,
                        "CMAEvolutionStrategy": 0.0, "PatternSearch": 0.0,
                        "SimulatedAnnealing": 0.0}),
    ("canon:-1DE+2NM", {"NelderMead": 2.0, "DifferentialEvolution": -1.0,
                        "CMAEvolutionStrategy": 0.0, "PatternSearch": 0.0,
                        "SimulatedAnnealing": 0.0}),
    ("canon:-0.5PS+1.5SA", {"NelderMead": 0.0, "DifferentialEvolution": 0.0,
                            "CMAEvolutionStrategy": 0.0, "PatternSearch": -0.5,
                            "SimulatedAnnealing": 1.5}),
    ("canon:-0.5SA+centroidish", {"NelderMead": 0.375, "DifferentialEvolution": 0.375,
                                  "CMAEvolutionStrategy": 0.375, "PatternSearch": 0.375,
                                  "SimulatedAnnealing": -0.5}),
]


def sample_signed(rng):
    """Random signed recipe: sums to 1, at least one coordinate <= -0.1,
    at least one >= 0.35 (a host must exist)."""
    while True:
        raw = [rng.uniform(-0.6, 1.4) for _ in range(sb.N_VERTICES)]
        shift = (sum(raw) - 1.0) / sb.N_VERTICES
        w = [x - shift for x in raw]
        if min(w) <= -0.1 and max(w) >= 0.35 and min(w) >= -1.0 and max(w) <= 2.0:
            return w


def build_signed_prompt(wmap):
    pos = {k: v for k, v in wmap.items() if v > 0.01}
    neg = {k: -v for k, v in wmap.items() if v < -0.01}
    host = max(pos, key=pos.get)
    pos_tot = sum(pos.values())
    idea = {v["name"]: v["idea"] for v in sb.VERTICES}

    lines = [
        "Build a black-box numerical optimizer from a SIGNED recipe of base "
        "algorithms. Positive weights are inspirations to blend, exactly as in "
        "asymmetric blending: the highest-weight method is the HOST architecture "
        "and the others donate grafted ideas in proportion to their weight.",
        "",
        "NEGATIVE weights are ANTI-INSPIRATIONS. For each negatively-weighted "
        "method, maintain a cheap SHADOW of its proposal logic (spend NO "
        "objective evaluations on it; just compute where it WOULD move next "
        "from the current state) and steer AWAY from that suggestion: penalise "
        "or resample candidates that fall within a repulsion radius of the "
        "shadow's suggestion. Repulsion strength and radius scale with the "
        "magnitude of the negative weight. The shadow must be real, working "
        "code, not a comment.",
        "",
        f"Signed recipe: " + ", ".join(
            f"{k} {v:+.0%}" for k, v in wmap.items() if abs(v) > 0.01
        ),
        "",
        f"HOST architecture ({int(100 * pos[host] / pos_tot)}% of positive mass): "
        f"{host} — {idea[host]}",
    ]
    grafts = sorted(((k, v) for k, v in pos.items() if k != host), key=lambda t: -t[1])
    if grafts:
        lines.append("GRAFT into it: " + ", ".join(
            f"{int(100 * v / pos_tot)}% {k} ({idea[k]})" for k, v in grafts))
    for k, mag in sorted(neg.items(), key=lambda t: -t[1]):
        lines.append(
            f"ANTI-INSPIRATION (strength {int(100 * mag)}%): {k} — shadow its "
            f"logic ({idea[k]}) and avoid where it would sample."
        )
    lines += ["", sb.CONTRACT, "",
              "Return ONLY a ```python code block containing the optimize function."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/e13_signed.json")
    ap.add_argument("--code-dir", default="runs/e13_code")
    args = ap.parse_args()

    by_name = {d.name: d for d in DEMOS}
    base = [by_name[n] for n in SUITE]
    code_dir = Path(args.code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    print("  precomputing panel baselines...", flush=True)
    panel_cache = sb.build_panel_cache(base, SEEDS, TRIALS)

    names = [v["name"] for v in sb.VERTICES]
    rng = random.Random(1300)
    points = []
    for j in range(12):
        w = sample_signed(rng)
        points.append((f"signed{j}", dict(zip(names, w)), True))
    for label, wmap in CANONICAL:
        points.append((label, wmap, True))
    for j in range(6):
        w = list(cube_to_simplex([rng.random() for _ in range(sb.N_VERTICES - 1)]))
        points.append((f"ctrl{j}", dict(zip(names, w)), False))

    results = []

    def save(done):
        tmp = Path(args.out + ".tmp")
        tmp.write_text(json.dumps({
            "done": done, "suite": SUITE, "seeds": list(SEEDS), "trials": TRIALS,
            "model": None if args.dry_run else args.model, "results": results,
        }, indent=2))
        tmp.replace(Path(args.out))

    for label, wmap, signed in points:
        if signed:
            prompt = build_signed_prompt(wmap)
        else:
            prompt = sb.build_prompt(sb.weights_to_spec([wmap[n] for n in names]))
        try:
            code = sb._DRY_TEMPLATE if args.dry_run else sb.generate_live(prompt, args.model)
            (code_dir / f"{label.replace(':', '_').replace('+', 'p')}.py").write_text(code)
            opt = sb.compile_optimizer(code)
            regret = sb.score_optimizer(opt, base, SEEDS, TRIALS, panel_cache=panel_cache)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:24s} FAILED: {e}", flush=True)
            regret = 1.0
        results.append({"label": label, "signed": signed, "w": wmap, "regret": regret})
        print(f"  {label:24s} regret={regret:.4f}", flush=True)
        save(False)

    save(True)
    print("\n=== leaderboard ===")
    for r in sorted(results, key=lambda r: r["regret"]):
        tag = "signed" if r["signed"] else "ctrl"
        print(f"  {r['regret']:.4f}  [{tag:6s}] {r['label']}")
    sg = [r["regret"] for r in results if r["signed"]]
    ct = [r["regret"] for r in results if not r["signed"]]
    if sg and ct:
        print(f"\n  best signed {min(sg):.4f} (mean {mean(sg):.4f}) vs "
              f"best unsigned-ctrl {min(ct):.4f} (mean {mean(ct):.4f}); "
              f"E7 rand best-of-20: 0.304/0.312, E11 12-pt best: 0.298")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
