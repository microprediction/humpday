"""E8 — Does the simplex keep an evolutionary concept-space search stretched out?

Hypothesis (user's): existing LLM-driven evolutionary searches over programs
(FunSearch-style improve-the-parent loops) tend to collapse toward the
generator's defaults or one basin; anchoring the population to simplex
coordinates preserves exploration (the population stays "stretched") while
still narrowing, because variation happens in a bounded, meaningful
coordinate space.

Three budget-matched arms (24 evaluations each, steady-state population of 4,
replace-worst selection, identical suite and scoring):

  freeform_evolve : classic LLM evolution. Init 4 free-form programs; each
                    step picks a tournament parent and asks the model to
                    IMPROVE it given its measured regret. No coordinates.
  simplex_evolve  : same loop, but candidates are simplex points. Variation
                    mutates the parent's coordinates (Gaussian on the
                    simplex) and regenerates from the slot spec at the new
                    point. The program itself is never shown to the model.
  simplex_stretch : as simplex_evolve, but each step proposes three
                    coordinate mutations and keeps the one farthest from the
                    current population (explicit stretch), before generating.

Metrics per evaluation: best-so-far regret AND population behavioral
diversity (mean pairwise L2 distance between per-instance score vectors of
the current population) — the latter is comparable across all arms, unlike
coordinate dispersion, which is also logged for the simplex arms.

Suite: the same 8 burned selection-era demos as E7 (untouched E6 pool stays
clean), seeds {0,1}, 100 trials. Compile failure scores 1.0 and consumes the
evaluation. Crash-safe per-evaluation checkpoints; resume by re-running.

    ../../.venv/bin/python e8_stretch.py --dry-run --out runs/e8_smoke.json
    ../../.venv/bin/python e8_stretch.py --out runs/e8_stretch.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simplex_blend as sb  # noqa: E402
from example_demos import DEMOS  # noqa: E402

SUITE = [
    "espresso_dialin", "facility_location", "gear_ratios", "kalman_tuning",
    "pid_tuning", "tension_spring", "plinko_funnel", "cassini_minlp",
]
SEEDS = (0, 1)
TRIALS = 100
N_EVALS = 24
POP = 4
REPEATS = (0, 1)

FREEFORM_INIT = """You are an expert in derivative-free optimization. Write the BEST \
general-purpose black-box optimizer you can, as one self-contained pure-Python \
function. Use whatever combination of techniques you judge strongest. This is \
independent design attempt #{k} of several; bring your strongest ideas.

{contract}

Return ONLY a ```python code block containing the optimize function."""

FREEFORM_IMPROVE = """You are an expert in derivative-free optimization, improving a \
program by iteration. Below is an optimizer and its measured normalised regret on a \
benchmark of disguised real-world problems (0 = matches the best of a strong panel \
everywhere, 1 = matches the worst). Improve it: change whatever you judge is holding \
it back, keeping what works.

Measured regret: {regret:.4f}

```python
{code}
```

{contract}

Return ONLY a ```python code block containing the improved optimize function."""


def score_vector(opt, base, panel_cache):
    """Per-instance normalised regrets (the same numbers score_optimizer means)."""
    import algo_dev as ad

    INF = float("inf")
    out = []
    for i, demo in enumerate(base):
        for s in SEEDS:
            from example_demos import disguise_demo

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


def behavioral_diversity(pop):
    vecs = [c["vector"] for c in pop if c.get("vector")]
    if len(vecs) < 2:
        return 0.0
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(vecs[i], vecs[j]))))
    return mean(dists)


def w_dispersion(pop):
    ws = [c["w"] for c in pop if c.get("w")]
    if len(ws) < 2:
        return 0.0
    dists = []
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            dists.append(sum(abs(a - b) for a, b in zip(ws[i], ws[j])))
    return mean(dists)


def mutate_w(w, rng, spread=0.15):
    out = [max(1e-3, x + rng.gauss(0, spread)) for x in w]
    t = sum(out)
    return [x / t for x in out]


def random_w(rng):
    from humpday.transforms.cubetosimplex import cube_to_simplex

    return list(cube_to_simplex([rng.random() for _ in range(sb.N_VERTICES - 1)]))


def generate(prompt, args):
    return sb._DRY_TEMPLATE if args.dry_run else sb.generate_live(prompt, args.model)


def evaluate(code, base, panel_cache):
    try:
        opt = sb.compile_optimizer(code)
    except Exception as e:  # noqa: BLE001
        print(f"      compile failed ({e}); regret=1.0", flush=True)
        return 1.0, None
    vec = score_vector(opt, base, panel_cache)
    return mean(vec), vec


def run_arm(arm, rep, base, panel_cache, args, trace):
    rng = random.Random(8600 + 97 * rep + hash(arm) % 1000)
    pop = []
    n_eval = 0

    def record(regret):
        trace.append({
            "eval": n_eval,
            "regret": regret,
            "best": min(t["regret"] for t in trace) if trace else regret,
            "diversity": behavioral_diversity(pop),
            "w_dispersion": w_dispersion(pop),
        })
        trace[-1]["best"] = min(trace[-1]["best"], regret)
        print(f"    eval {n_eval:2d}/{N_EVALS} regret={regret:.4f} "
              f"best={trace[-1]['best']:.4f} div={trace[-1]['diversity']:.3f}",
              flush=True)

    def spawn(code, w):
        nonlocal n_eval
        n_eval += 1
        regret, vec = evaluate(code, base, panel_cache)
        cand = {"regret": regret, "vector": vec, "code": code, "w": w}
        if len(pop) < POP:
            pop.append(cand)
        else:
            worst = max(range(POP), key=lambda i: pop[i]["regret"])
            if regret < pop[worst]["regret"]:
                pop[worst] = cand
        record(regret)

    # ---- init: POP evaluations ----
    for k in range(POP):
        if arm == "freeform_evolve":
            prompt = FREEFORM_INIT.format(k=k + 1, contract=sb.CONTRACT)
            w = None
        else:
            w = random_w(rng)
            prompt = sb.build_prompt(sb.weights_to_spec(w))
        try:
            code = generate(prompt, args)
        except Exception as e:  # noqa: BLE001
            print(f"      generation failed ({e})", flush=True)
            n_eval += 1
            record(1.0)
            continue
        spawn(code, w)

    # ---- evolution ----
    while n_eval < N_EVALS:
        a, b = rng.sample(range(len(pop)), 2) if len(pop) >= 2 else (0, 0)
        parent = pop[a] if pop[a]["regret"] <= pop[b]["regret"] else pop[b]
        if arm == "freeform_evolve":
            prompt = FREEFORM_IMPROVE.format(
                regret=parent["regret"], code=parent["code"], contract=sb.CONTRACT
            )
            w = None
        else:
            if arm == "simplex_stretch":
                proposals = [mutate_w(parent["w"], rng) for _ in range(3)]
                others = [c["w"] for c in pop if c is not parent and c.get("w")]

                def spread_of(wp):
                    return min(
                        (sum(abs(x - y) for x, y in zip(wp, o)) for o in others),
                        default=0.0,
                    )

                w = max(proposals, key=spread_of)
            else:
                w = mutate_w(parent["w"], rng)
            prompt = sb.build_prompt(sb.weights_to_spec(w))
        try:
            code = generate(prompt, args)
        except Exception as e:  # noqa: BLE001
            print(f"      generation failed ({e})", flush=True)
            n_eval += 1
            record(1.0)
            continue
        spawn(code, w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/e8_stretch.json")
    ap.add_argument(
        "--arms", default="",
        help="only run these arm:rep pairs, e.g. 'freeform_evolve:1,simplex_stretch:1'",
    )
    args = ap.parse_args()
    only = None
    if args.arms:
        only = {(a.split(":")[0], int(a.split(":")[1])) for a in args.arms.split(",")}

    by_name = {d.name: d for d in DEMOS}
    base = [by_name[n] for n in SUITE]
    print("  precomputing panel baselines...", flush=True)
    panel_cache = sb.build_panel_cache(base, SEEDS, TRIALS)

    out = Path(args.out)
    runs = []
    done = set()
    if out.exists():
        try:
            runs = json.load(open(out))["runs"]
            done = {(r["arm"], r["repeat"]) for r in runs if len(r["trace"]) >= N_EVALS}
        except Exception:  # noqa: BLE001
            pass

    def save(final=False):
        tmp = Path(str(out) + ".tmp")
        tmp.write_text(json.dumps({
            "done": final, "suite": SUITE, "seeds": list(SEEDS), "trials": TRIALS,
            "n_evals": N_EVALS, "pop": POP,
            "model": None if args.dry_run else args.model, "runs": runs,
        }, indent=2))
        tmp.replace(out)

    for rep in REPEATS:
        for arm in ("freeform_evolve", "simplex_evolve", "simplex_stretch"):
            if only is not None and (arm, rep) not in only:
                continue
            if (arm, rep) in done:
                print(f"  {arm} rep{rep}: already complete, skipping", flush=True)
                continue
            print(f"  === {arm} rep{rep} ===", flush=True)
            trace = []
            run_arm(arm, rep, base, panel_cache, args, trace)
            best = min((t["regret"] for t in trace), default=1.0)
            late_div = mean(t["diversity"] for t in trace[len(trace) // 2:])
            runs.append({"arm": arm, "repeat": rep, "trace": trace,
                         "best": best, "late_diversity": round(late_div, 4)})
            print(f"  {arm} rep{rep}: best={best:.4f} late_diversity={late_div:.3f}",
                  flush=True)
            save()

    save(final=True)
    print("\n=== summary (24 evals each) ===")
    for arm in ("freeform_evolve", "simplex_evolve", "simplex_stretch"):
        rs = [r for r in runs if r["arm"] == arm]
        if rs:
            print(f"  {arm:16s} mean_best={mean(r['best'] for r in rs):.4f}  "
                  f"mean_late_diversity={mean(r['late_diversity'] for r in rs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
