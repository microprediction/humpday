"""E15 — Five semantics for negative coefficients, tested under the paired
ablation protocol with theory-guided recipe selection.

Ancestor rank correlations on the real suite (runs/rankcorr.json) predict
where a negative weight can pay: subtraction harvests correlation, so the
signed term should help on positively correlated pairs and not on
anticorrelated ones. Recipes tested (B hosts at +200%, A shorted at -100%):

  good pairs (predicted to benefit):
    +2 NelderMead  -1 SimulatedAnnealing   (corr +0.18)
    +2 CMA-ES      -1 DifferentialEvolution (corr +0.14)
    +2 PatternSearch -1 NelderMead          (corr +0.16)
  bad-pair control (predicted NOT to benefit):
    +2 NelderMead  -1 DifferentialEvolution (corr -0.20; the E13 pair)

Semantics (each renders the same signed recipe differently in the prompt):
  density    : signed mixture of proposal densities; rejection-thin B's
               proposals by w_A q_A / w_B q_B BEFORE evaluation (exact,
               costs no objective calls)
  caricature : extrapolation in design space; exaggerate what distinguishes
               B from A along their contrast axes
  pitfalls   : design out A's characteristic failure modes from B
  baseline   : A's shadow sets the bar; falling behind its improvement rate
               triggers strategy escalation
  repulsion  : E13's avoidance (known-bad reference)

Every program must gate ALL its negative-weight machinery through a module
constant ANTI_STRENGTH (default 1.0), so the ablated twin is the same
program with ANTI_STRENGTH = 0.0, scored on identical instances. The paired
delta regret(signed) - regret(ablated) is the only statistic that counts.
Additionally a small pool of plain +100%-host generations calibrates the
host band.

Selection suite: 8 burned demos stride-picked across the dimension range
(fixing E13's low-dimensional selection bias). All code saved.

    ../../.venv/bin/python e15_semantics.py --dry-run --out runs/e15_smoke.json
    ../../.venv/bin/python e15_semantics.py --out runs/e15_semantics.json
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
from e6_untouched import UNTOUCHED  # noqa: E402
from example_demos import DEMOS  # noqa: E402

SEEDS = (0, 1)
TRIALS = 100
DRAWS = 2

RECIPES = [
    ("goodNMSA", "NelderMead", "SimulatedAnnealing", "good"),
    ("goodCMADE", "CMAEvolutionStrategy", "DifferentialEvolution", "good"),
    ("goodPSNM", "PatternSearch", "NelderMead", "good"),
    ("badNMDE", "NelderMead", "DifferentialEvolution", "bad"),
]

GATE = """

MANDATORY GATE: define a module-level constant

    ANTI_STRENGTH = 1.0

and scale EVERY behaviour introduced by the negative weight through it, so
that ANTI_STRENGTH = 0.0 recovers plain {B} exactly. Read it at call time
(inside optimize), not import time."""

SEMANTICS = {
    "density": """The negative weight acts on your PROPOSAL DENSITY, not on space.
Represent {A}'s proposal at the current state as an explicit density q_A you
can evaluate cheaply (for instance a Gaussian centred on {A}'s would-be next
move), and know your own proposal density q_B. BEFORE spending an objective
evaluation on a candidate x, thin it: with probability
min(0.9, ANTI_STRENGTH * q_A(x) / (2 * q_B(x))), discard x and repropose
(reproposing costs no objective calls). This reshapes your sampling toward
the signed mixture max(2 q_B - q_A, 0): {B} sharpened away from its overlap
with {A}.""",
    "caricature": """The negative weight means CARICATURE: the recipe 2*{B} - 1*{A} is an
extrapolation in design space. Identify the two or three axes on which {B}
most differs from {A} (step geometry, memory, acceptance, adaptivity) and
EXAGGERATE {B} along exactly those axes, going further from {A} than plain
{B} would. Implement each exaggeration as an adjustment away from plain-{B}
behaviour scaled by ANTI_STRENGTH.""",
    "pitfalls": """The negative weight means: design OUT the characteristic failure modes
of {A}. List (in comments) the three best-known pathologies of {A} and build
explicit countermeasures into {B} so those pathologies cannot occur, each
countermeasure scaled by ANTI_STRENGTH. Do not avoid {A}'s search regions;
avoid {A}'s mistakes.""",
    "baseline": """The negative weight makes {A} a BASELINE to outperform. Maintain a cheap
shadow of {A} from the same start state (its would-be moves cost no
objective calls; estimate its progress from your own evaluations when your
trajectory passes nearby). Track your improvement rate against the shadow's
predicted rate, and when you fall behind, escalate: enlarge steps, restart,
or switch mechanism. Escalation aggressiveness scales with ANTI_STRENGTH.""",
    "repulsion": """The negative weight is an ANTI-INSPIRATION. Maintain a cheap SHADOW of
{A}'s proposal logic (spend NO objective evaluations on it) and steer AWAY
from its suggestion: penalise or resample candidates within a repulsion
radius of the shadow's suggestion, radius and strength scaled by
ANTI_STRENGTH.""",
    "residual": """The negative weight means ORTHOGONALIZATION: take {A}, remove everything
it has in common with {B}, and keep only what remains: the mechanisms {A}
has that {B} does not. EXAGGERATE exactly those unique {A} mechanisms and
graft them into the {B} architecture, each scaled by ANTI_STRENGTH, so the
shared machinery appears only once and {A} contributes only what {B}
cannot express.""",
}


def spread_suite(n=8):
    untouched = set(UNTOUCHED)
    pool = sorted((d for d in DEMOS if d.name not in untouched), key=lambda d: d.n_dim)
    idx = sorted({round(k * (len(pool) - 1) / (n - 1)) for k in range(n)})
    return [pool[i] for i in idx]


def build_prompt(b, a, semantics_text):
    idea = {v["name"]: v["idea"] for v in sb.VERTICES}
    parts = [
        f"Build a black-box numerical optimizer from the SIGNED recipe: "
        f"{b} +200%, {a} -100%.",
        "",
        f"HOST (+200%): {b} — {idea[b]}. Build the architecture from this.",
        "",
        semantics_text.format(A=a, B=b),
        GATE.format(B=b),
        "",
        sb.CONTRACT,
        "",
        "Return ONLY a ```python code block containing the optimize function.",
    ]
    return "\n".join(parts)


def build_pure_prompt(b):
    idea = {v["name"]: v["idea"] for v in sb.VERTICES}
    return "\n".join(
        [
            f"Build a black-box numerical optimizer: a strong, budget-aware "
            f"implementation of {b} — {idea[b]}.",
            "",
            sb.CONTRACT,
            "",
            "Return ONLY a ```python code block containing the optimize function.",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/e15_semantics.json")
    ap.add_argument("--code-dir", default="runs/e15_code")
    ap.add_argument("--recipes", default="", help="comma list to run (default all)")
    ap.add_argument("--semantics", default="", help="comma list to run (default all)")
    ap.add_argument("--skip-pure", action="store_true")
    args = ap.parse_args()
    only_r = set(args.recipes.split(",")) if args.recipes else None
    only_s = set(args.semantics.split(",")) if args.semantics else None

    base = spread_suite()
    print("suite:", [(d.name, d.n_dim) for d in base], flush=True)
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
                    "dims": [d.n_dim for d in base],
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

    # pure-host calibration pool
    for host in () if args.skip_pure else sorted({b for _, b, _, _ in RECIPES}):
        for k in range(DRAWS):
            label = f"pure_{host}_d{k}"
            try:
                code = (
                    sb._DRY_TEMPLATE
                    if args.dry_run
                    else sb.generate_live(build_pure_prompt(host), args.model)
                )
                (code_dir / f"{label}.py").write_text(code)
                regret = score_code(code)
            except Exception as e:  # noqa: BLE001
                print(f"  {label:34s} FAILED: {e}", flush=True)
                regret = 1.0
            results.append(
                {"kind": "pure", "host": host, "label": label, "regret": regret}
            )
            print(f"  {label:34s} regret={regret:.4f}", flush=True)
            save(False)

    # signed arms
    for rname, b, a, klass in RECIPES:
        if only_r is not None and rname not in only_r:
            continue
        for sem, text in SEMANTICS.items():
            if only_s is not None and sem not in only_s:
                continue
            for k in range(DRAWS):
                label = f"{rname}_{sem}_d{k}"
                row = {
                    "kind": "signed",
                    "recipe": rname,
                    "host": b,
                    "anti": a,
                    "class": klass,
                    "semantics": sem,
                    "label": label,
                    "regret": None,
                    "ablated": None,
                    "delta": None,
                }
                try:
                    code = (
                        sb._DRY_TEMPLATE
                        if args.dry_run
                        else sb.generate_live(build_prompt(b, a, text), args.model)
                    )
                    (code_dir / f"{label}.py").write_text(code)
                    row["regret"] = score_code(code)
                    if "ANTI_STRENGTH = 1.0" in code:
                        ab = code.replace("ANTI_STRENGTH = 1.0", "ANTI_STRENGTH = 0.0")
                        row["ablated"] = score_code(ab)
                        row["delta"] = row["regret"] - row["ablated"]
                    else:
                        print(f"    ({label}: no ANTI_STRENGTH gate!)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  {label:34s} FAILED: {e}", flush=True)
                    row["regret"] = 1.0
                results.append(row)
                d = row["delta"]
                print(
                    f"  {label:34s} regret={row['regret']:.4f}"
                    + (
                        f" ablated={row['ablated']:.4f} delta={d:+.4f}"
                        if d is not None
                        else ""
                    ),
                    flush=True,
                )
                save(False)

    save(True)

    print("\n=== paired deltas by semantics (negative = signed term helps) ===")
    for sem in SEMANTICS:
        for klass in ("good", "bad"):
            ds = [
                r["delta"]
                for r in results
                if r.get("semantics") == sem
                and r.get("class") == klass
                and r.get("delta") is not None
            ]
            if ds:
                print(
                    f"  {sem:10s} [{klass:4s} pairs] mean delta {mean(ds):+.4f} "
                    f"({', '.join(f'{x:+.3f}' for x in ds)})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
