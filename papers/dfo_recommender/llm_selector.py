"""Can LLM 'common-sense' reasoning predict which optimizer to use per problem?

For each real-world problem we give an LLM only the problem DESCRIPTION + its
dimension + the eval budget (NO benchmark results) and textbook one-line descriptions
of candidate optimizers, then ask it to REASON about the landscape and RANK them.
Repeat R times (shuffled order / optional subgroups) and aggregate with Plackett-Luce
(Bradley-Terry for rankings, Hunter MM) -> a latent strength per optimizer per problem.
We only use the induced ORDER, so cross-call scale arbitrariness never enters.

Then (if runs/rankcorr.json exists) we evaluate the LLM predictor against the
EMPIRICAL real-world ranking, and compare it to the synthetic-benchmark predictor:
  - tau(LLM per-problem ranking, empirical)  vs  tau(synthetic-global, empirical)
  - selection test: does LLM-top-1 per problem beat always-best-single / synthetic-fav?

    ../../.venv/bin/python llm_selector.py --dry-run          # no API; smoke
    ../../.venv/bin/python llm_selector.py --problems 15 --rounds 3 --out runs/llm_selector.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path("../../").resolve()))
sys.path.insert(0, ".")
from example_demos import DEMOS  # noqa: E402

# Candidate optimizers (all present in E1's panel) + textbook characterisations.
CANDIDATES = {
    "NelderMead": "downhill simplex (reflect/expand/contract/shrink); derivative-free local search, no curvature model",
    "Powell": "conjugate-direction line searches; derivative-free, good on smooth, low-coupling functions",
    "PRIMA_NEWUOA": "Powell's NEWUOA: builds a quadratic trust-region model from samples; very strong on smooth functions, assumes low noise",
    "PRIMA_BOBYQA": "bound-constrained quadratic-model trust-region (Powell BOBYQA); smooth-function specialist",
    "CMAEvolutionStrategy": "covariance-matrix-adaptation evolution strategy; robust global search that learns scaling, handles ill-conditioning/ruggedness",
    "DifferentialEvolution": "population with difference-vector mutation + crossover; robust global, good on multimodal/rugged landscapes",
    "ParticleSwarm": "swarm with velocity updates; global, broad exploration, handles multimodality",
    "SimulatedAnnealing": "stochastic hill-climb with cooling acceptance; escapes local minima, sample-hungry",
    "PatternSearch": "generalized pattern / coordinate direct search with shrinking step; robust, model-free",
    "CoordinateDescent": "optimizes one coordinate at a time; strong when variables are near-separable",
}


def demo_description(name):
    base = re.sub(r"_\d+d$", "", name)  # scaled demos: wind_farm_60d -> wind_farm
    try:
        m = importlib.import_module(f"example_applications.{base}.problem")
        doc = (m.__doc__ or "").strip()
        para = doc.split("\n\n")[0].replace("\n", " ").strip()
        return para or name
    except Exception:  # noqa: BLE001
        return name


# --------------------------- LLM elicitation -------------------------------
def ask_llm(prompt, model):
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=model, max_tokens=1800, thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in r.content if b.type == "text")


def build_prompt(desc, n_dim, budget, cand_subset):
    lines = [f"  - {c}: {CANDIDATES[c]}" for c in cand_subset]
    return (
        "You are selecting a derivative-free optimizer for a real-world problem using "
        "ONLY reasoning about the problem's character. You will NOT see any benchmark "
        "results.\n\n"
        f"PROBLEM ({n_dim} continuous parameters, all in [0,1]; budget = {budget} "
        f"objective evaluations; minimisation):\n{desc}\n\n"
        "CANDIDATE OPTIMIZERS:\n" + "\n".join(lines) + "\n\n"
        "Think briefly about the likely landscape (smooth vs rugged, separable vs "
        "coupled, multimodal, noisy, and how the tight budget interacts), then rank ALL "
        "the candidates from MOST to LEAST likely to find the best solution within the "
        "budget. End with exactly one line:\n"
        "RANKING: <name> > <name> > ... (every candidate, best first)"
    )


def parse_ranking(text, cand_subset):
    line = ""
    for ln in text.splitlines():
        if ln.strip().upper().startswith("RANKING:"):
            line = ln.split(":", 1)[1]
    order = []
    for tok in line.split(">"):
        t = tok.strip()
        for c in cand_subset:
            if c.lower() in t.lower() and c not in order:
                order.append(c)
                break
    # append any missing candidates (keeps it a full ranking)
    for c in cand_subset:
        if c not in order:
            order.append(c)
    return order


def dry_ranking(cand_subset, seed):
    r = random.Random(seed)
    o = list(cand_subset)
    r.shuffle(o)
    return o


# --------------------------- Plackett-Luce (Hunter MM) ----------------------
def plackett_luce(rankings, items, iters=400):
    g = dict.fromkeys(items, 1.0)
    for _ in range(iters):
        num = dict.fromkeys(items, 0.0)
        den = dict.fromkeys(items, 0.0)
        for r in rankings:
            for p in range(len(r) - 1):
                remaining = r[p:]
                s = sum(g[j] for j in remaining)
                if s <= 0:
                    continue
                num[r[p]] += 1.0
                for j in remaining:
                    den[j] += 1.0 / s
        newg = {i: (num[i] / den[i] if den[i] > 0 else g[i]) for i in items}
        tot = sum(newg.values()) or 1.0
        g = {i: newg[i] * len(items) / tot for i in items}
    return g  # higher = stronger


def kendall_tau(order_a, order_b):
    common = [x for x in order_a if x in order_b]
    ra = {x: i for i, x in enumerate(order_a)}
    rb = {x: i for i, x in enumerate(order_b)}
    c = d = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            s = (ra[x] - ra[y]) * (rb[x] - rb[y])
            c += s > 0
            d += s < 0
    return (c - d) / (c + d) if (c + d) else 0.0


def atomic_dump(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


# --------------------------- empirical ranking from E1 ----------------------
def empirical_orders(budget, cands):
    """Per-demo empirical candidate order from runs/rankcorr.json (real suite),
    plus the synthetic GLOBAL candidate order. Returns (per_demo, synth_global)."""
    if not os.path.exists("runs/rankcorr.json"):
        return {}, None
    res = json.load(open("runs/rankcorr.json"))["results"]
    per = {}
    by_demo = {}
    for r in res:
        if r["budget"] != budget or r["suite"] != "real":
            continue
        by_demo.setdefault(r["label"], []).append(r["ranks"])
    for demo, rlists in by_demo.items():
        mr = {c: mean(rl[c] for rl in rlists) for c in cands if c in rlists[0]}
        per[demo] = sorted(mr, key=mr.get)
    syn = [r["ranks"] for r in res if r["budget"] == budget and r["suite"] == "synthetic"]
    synth = None
    if syn:
        mr = {c: mean(rl[c] for rl in syn) for c in cands if c in syn[0]}
        synth = sorted(mr, key=mr.get)
    return per, synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--subset-size", type=int, default=0, help="0 = full candidate set")
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/llm_selector.json")
    a = ap.parse_args()

    cands = list(CANDIDATES)
    m = a.subset_size or len(cands)

    # problems: dim-spread real demos. Prefer demos already present in E1 at this
    # budget so every elicited problem gets an empirical comparison.
    pool = DEMOS
    if os.path.exists("runs/rankcorr.json"):
        labs = {r["label"] for r in json.load(open("runs/rankcorr.json"))["results"]
                if r["suite"] == "real" and r["budget"] == a.budget}
        if labs:
            pool = [d for d in DEMOS if d.name in labs]
    ds = sorted(pool, key=lambda d: d.n_dim)
    n = min(a.problems, len(ds))
    idx = sorted({round(k * (len(ds) - 1) / max(n - 1, 1)) for k in range(n)})
    probs = [ds[i] for i in idx]

    print(f"LLM-selector: {len(probs)} problems x {a.rounds} rounds "
          f"({'DRY' if a.dry_run else a.model}), budget={a.budget}\n", flush=True)

    llm_orders = {}
    raw = []
    for pi, dm in enumerate(probs):
        desc = demo_description(dm.name)
        rankings = []
        for rd in range(a.rounds):
            sub = list(cands)
            random.Random(1000 * pi + rd).shuffle(sub)
            sub = sub[:m]
            if a.dry_run:
                order = dry_ranking(sub, 1000 * pi + rd)
                txt = "DRY"
            else:
                txt = ask_llm(build_prompt(desc, dm.n_dim, a.budget, sub), a.model)
                order = parse_ranking(txt, sub)
            rankings.append(order)
            raw.append({"demo": dm.name, "round": rd, "order": order})
        strengths = plackett_luce(rankings, cands)
        llm_orders[dm.name] = sorted(cands, key=lambda c: -strengths[c])
        print(f"[{pi+1}/{len(probs)}] {dm.name:24s} n={dm.n_dim:3d} "
              f"LLM top3={llm_orders[dm.name][:3]}", flush=True)
        atomic_dump({"done": False, "budget": a.budget, "candidates": cands,
                     "llm_orders": llm_orders, "raw": raw}, a.out)

    # ---- evaluation against empirical (E1) ----
    per_emp, synth = empirical_orders(a.budget, cands)
    evaluation = None
    if per_emp:
        common = [d for d in llm_orders if d in per_emp]
        tau_llm = mean(kendall_tau(llm_orders[d], per_emp[d]) for d in common)
        tau_syn = mean(kendall_tau(synth, per_emp[d]) for d in common) if synth else None
        tau_rand = mean(kendall_tau(random.Random(d.__hash__() & 255).sample(cands, len(cands)),
                                    per_emp[d]) for d in common)
        # selection test: empirical rank of each predictor's top-1 (lower=better)
        def emp_rank_of(demo, opt):
            return per_emp[demo].index(opt) + 1 if opt in per_emp[demo] else len(cands)
        sel_llm = mean(emp_rank_of(d, llm_orders[d][0]) for d in common)
        sel_syn = mean(emp_rank_of(d, synth[0]) for d in common) if synth else None
        # always-best-single = candidate with best average empirical rank across demos
        avg = {c: mean(emp_rank_of(d, c) for d in common) for c in cands}
        best_single = min(avg, key=avg.get)
        sel_fixed = avg[best_single]
        sel_oracle = mean(1 for _ in common)  # top-1 is always rank 1 by definition
        evaluation = {
            "n_problems_scored": len(common),
            "tau_llm_vs_empirical": round(tau_llm, 3),
            "tau_synthetic_vs_empirical": (round(tau_syn, 3) if tau_syn is not None else None),
            "tau_random_vs_empirical": round(tau_rand, 3),
            "selection_mean_emp_rank": {
                "LLM_top1": round(sel_llm, 3),
                "synthetic_top1": (round(sel_syn, 3) if sel_syn is not None else None),
                f"always_best_single({best_single})": round(sel_fixed, 3),
                "oracle": round(sel_oracle, 3),
            },
        }
    atomic_dump({"done": True, "budget": a.budget, "candidates": cands,
                 "llm_orders": llm_orders, "evaluation": evaluation, "raw": raw}, a.out)

    print("\n=== evaluation ===")
    if evaluation is None:
        print("  (runs/rankcorr.json not available yet — elicitation saved; "
              "re-run evaluation once E1 has the matching budget.)")
    else:
        e = evaluation
        print(f"  predictiveness (Kendall-tau vs empirical real ranking, {e['n_problems_scored']} problems):")
        print(f"    LLM common-sense : {e['tau_llm_vs_empirical']}")
        print(f"    synthetic bench. : {e['tau_synthetic_vs_empirical']}")
        print(f"    random           : {e['tau_random_vs_empirical']}")
        print("  selection (mean empirical rank of chosen optimizer, lower=better):")
        for k, v in e["selection_mean_emp_rank"].items():
            print(f"    {k:32s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
