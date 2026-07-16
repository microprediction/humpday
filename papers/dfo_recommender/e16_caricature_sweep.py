"""E16 — Complete caricature sweep of the signed pair space.

E15 found one semantics that survives the paired-ablation protocol:
caricature, where "(1+lam)*B - lam*A" means build B and exaggerate B along
the axes where it most differs from A, further as lam grows. This sweep
covers the space:

  - all 20 ordered (host B, reference A) pairs at lam = 1, two draws each;
  - a magnitude ladder lam in {0.5, 2, 3} on the E15 winner pair
    (PatternSearch vs NelderMead), two draws each.

Every program gates its exaggerations through ANTI_STRENGTH so the ablated
twin (= plain B) scores on identical instances; only the paired delta
counts. Deliverable: the 5x5 caricature matrix of mean paired deltas, to be
set against the ancestor rank-correlation matrix.

Shardable via --hosts / --ladder for parallel runs.

    ../../.venv/bin/python e16_caricature_sweep.py --hosts NelderMead --out runs/e16_nm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simplex_blend as sb  # noqa: E402
from e15_semantics import GATE, spread_suite  # noqa: E402

SEEDS = (0, 1)
TRIALS = 100
DRAWS = 2
NAMES = [v["name"] for v in sb.VERTICES]
LADDER_PAIR = ("PatternSearch", "NelderMead")
LADDER = (0.5, 2.0, 3.0)

CARICATURE = """The recipe means CARICATURE, an extrapolation in design space with
extrapolation length lam = {lam}: start from {A}, pass through {B}, and keep
going lam times as far again. Identify the two or three axes on which {B}
most differs from {A} (step geometry, memory, acceptance, adaptivity) and
EXAGGERATE {B} along exactly those axes, with the SIZE of every exaggeration
proportional to lam * ANTI_STRENGTH. lam = {lam} here: {intensity}."""

INTENSITY = {
    0.5: "a restrained caricature, exaggerations mild",
    1.0: "a firm caricature, exaggerations clearly beyond plain behaviour",
    2.0: "a strong caricature, exaggerations large",
    3.0: "an extreme caricature, exaggerations pushed as far as still sane",
}


def build_prompt(b, a, lam):
    idea = {v["name"]: v["idea"] for v in sb.VERTICES}
    w_b, w_a = 1.0 + lam, lam
    parts = [
        f"Build a black-box numerical optimizer from the SIGNED recipe: "
        f"{b} +{int(w_b * 100)}%, {a} -{int(w_a * 100)}%.",
        "",
        f"HOST (+{int(w_b * 100)}%): {b} — {idea[b]}. Build the architecture from this.",
        "",
        CARICATURE.format(A=a, B=b, lam=lam, intensity=INTENSITY[lam]),
        GATE.format(B=b),
        "",
        sb.CONTRACT,
        "",
        "Return ONLY a ```python code block containing the optimize function.",
    ]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/e16_sweep.json")
    ap.add_argument("--code-dir", default="runs/e16_code")
    ap.add_argument("--hosts", default="", help="comma list of hosts to run")
    ap.add_argument(
        "--ladder", action="store_true", help="run the magnitude ladder only"
    )
    args = ap.parse_args()
    only_hosts = set(args.hosts.split(",")) if args.hosts else None

    base = spread_suite()
    code_dir = Path(args.code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    print("  precomputing panel baselines...", flush=True)
    panel_cache = sb.build_panel_cache(base, SEEDS, TRIALS)

    results = []

    def save(done):
        tmp = Path(args.out + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "done": done,
                    "suite": [d.name for d in base],
                    "seeds": list(SEEDS),
                    "trials": TRIALS,
                    "model": None if args.dry_run else args.model,
                    "results": results,
                },
                indent=2,
            )
        )
        tmp.replace(Path(args.out))

    def score_code(code):
        opt = sb.compile_optimizer(code)
        return sb.score_optimizer(opt, base, SEEDS, TRIALS, panel_cache=panel_cache)

    def run_arm(b, a, lam, k):
        label = f"{b[:4]}_not_{a[:4]}_lam{lam}_d{k}"
        row = {
            "host": b,
            "anti": a,
            "lam": lam,
            "draw": k,
            "label": label,
            "regret": None,
            "ablated": None,
            "delta": None,
        }
        try:
            code = (
                sb._DRY_TEMPLATE
                if args.dry_run
                else sb.generate_live(build_prompt(b, a, lam), args.model)
            )
            (code_dir / f"{label}.py").write_text(code)
            row["regret"] = score_code(code)
            if "ANTI_STRENGTH = 1.0" in code:
                row["ablated"] = score_code(
                    code.replace("ANTI_STRENGTH = 1.0", "ANTI_STRENGTH = 0.0")
                )
                row["delta"] = row["regret"] - row["ablated"]
            else:
                print(f"    ({label}: no gate)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:32s} FAILED: {e}", flush=True)
            row["regret"] = 1.0
        results.append(row)
        d = row["delta"]
        print(
            f"  {label:32s} regret={row['regret']:.4f}"
            + (f" twin={row['ablated']:.4f} delta={d:+.4f}" if d is not None else ""),
            flush=True,
        )
        save(False)

    if args.ladder:
        b, a = LADDER_PAIR
        for lam in LADDER:
            for k in range(DRAWS):
                run_arm(b, a, lam, k)
    else:
        for b in NAMES:
            if only_hosts is not None and b not in only_hosts:
                continue
            for a in NAMES:
                if a == b:
                    continue
                for k in range(DRAWS):
                    run_arm(b, a, 1.0, k)

    save(True)
    gated = [r for r in results if r["delta"] is not None]
    print("\n=== mean paired delta by pair (negative = caricature helps) ===")
    pairs = sorted({(r["host"], r["anti"]) for r in gated})
    for b, a in pairs:
        ds = [r["delta"] for r in gated if r["host"] == b and r["anti"] == a]
        print(f"  {b[:12]:12s} not {a[:12]:12s} lam-mix: {mean(ds):+.4f} ({len(ds)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
