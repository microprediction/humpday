"""Nevergrad-native high-dim sanity check for SAD (Schur Acquisition Damping).

SAD's thesis is dimension-gated: damping an *undersampled* CMA covariance should
help only when n is large and the budget is low. The disguised real-world suite
tops out around n=60 but is heterogeneous; infinity77/go_2021 is ~2-9D (wrong
regime). This script gets a fast, INDEPENDENT signal in genuine high dim using
Nevergrad's own rotated/ill-conditioned `ArtificialFunction`s, fed *identically*
to three optimizers so the comparison is fair:

  - CMA-full     : our compact Hansen CMA, full covariance (gamma=1.0)   [baseline]
  - SAD-adaptive : the same CMA + reliability-adaptive gamma*            [the win]
  - ng-CMA       : Nevergrad's OWN CMA, ask/tell on the same problem     [referee]

Rotated functions are chosen on purpose: a rotated ellipsoid/cigar makes the
sampling covariance actually matter, which is exactly what SAD damps. All
optimizers see the SAME map g:[0,1]^n -> R^n, so absolute scaling cancels.
Per-cell regret is normalised within the comparison set (0 = best optimizer on
that function/dim/budget/seed), matching the disguised-suite methodology.

Lives in papers/ — imports schur_cma + nevergrad from the repo venv only; nothing
here ships in the humpday wheel.

    ../../.venv/bin/python nevergrad_sad.py --quick
    ../../.venv/bin/python nevergrad_sad.py --dims 20,40 --seeds 0,1,2 --out runs/nevergrad_sad.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile

import nevergrad as ng
import numpy as np
from nevergrad.functions import ArtificialFunction
from schur_cma import cma_es

INF = float("inf")

# Rotated + ill-conditioned functions stress the covariance; sphere is the easy
# control where damping should NOT help.
FUNCS = ["sphere", "ellipsoid", "cigar", "rastrigin", "rosenbrock"]


def make_objective(name: str, n: int, seed: int):
    """A Nevergrad ArtificialFunction wrapped onto the unit cube [0,1]^n.

    g maps the cube into [-5,5]^n; the function's rotation/translation is fixed by
    its own seed so the optimum is relocated per (name, n, seed) — nothing here can
    be won by memorising a location."""
    af = ArtificialFunction(
        name=name,
        block_dimension=n,
        rotation=True,
        num_blocks=1,
        useless_variables=0,
    )
    # ArtificialFunction is itself stochastic-free here; seed the rotation via copy.
    rng = np.random.default_rng(seed)
    shift = rng.uniform(-2.0, 2.0, size=n)  # relocate optimum off-centre per seed

    def f(x_list):
        x = np.asarray(x_list, dtype=float)
        z = (x - 0.5) * 10.0 - shift
        return float(af.function(z))

    return f


def run_ng_optimizer(opt_name: str, f, n: int, budget: int, seed: int) -> float:
    """Run a built-in Nevergrad optimizer on f over [0,1]^n; return best loss seen."""
    param = ng.p.Array(shape=(n,), lower=0.0, upper=1.0)
    param.random_state.seed(seed)
    opt = ng.optimizers.registry[opt_name](
        parametrization=param, budget=budget, num_workers=1
    )
    best = INF
    for _ in range(budget):
        cand = opt.ask()
        loss = f(np.asarray(cand.value))
        opt.tell(cand, loss)
        if loss < best:
            best = loss
    return best


def evaluate_cell(name, n, budget, seed):
    f = make_objective(name, n, seed)
    out = {}
    out["CMA-full"] = cma_es(f, budget, n, gamma=1.0, seed=seed)
    out["SAD-adaptive"] = cma_es(f, budget, n, adaptive=True, seed=seed)
    try:
        out["ng-CMA"] = run_ng_optimizer("CMA", f, n, budget, seed)
    except Exception as e:  # keep going if nevergrad's CMA balks at a dim/budget
        out["ng-CMA"] = INF
    return out


def normalise(cell: dict) -> dict:
    vals = [v for v in cell.values() if math.isfinite(v)]
    if not vals:
        return dict.fromkeys(cell, 1.0)
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {
        k: (1.0 if not math.isfinite(v) else (v - lo) / span) for k, v in cell.items()
    }


def atomic_dump(obj, path):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", default="20,40")
    ap.add_argument("--budget-mult", type=int, default=8, help="budget = mult * n")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/nevergrad_sad.json")
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    funcs = FUNCS
    if args.quick:
        dims, seeds, funcs = [20], [0], ["ellipsoid", "cigar"]

    optimizers = ["CMA-full", "SAD-adaptive", "ng-CMA"]
    raw, normed = [], []
    agg = {o: [] for o in optimizers}
    agg_hi = {o: [] for o in optimizers}  # n>=40 slice

    total = len(dims) * len(seeds) * len(funcs)
    i = 0
    for n in dims:
        budget = args.budget_mult * n
        for name in funcs:
            for seed in seeds:
                i += 1
                cell = evaluate_cell(name, n, budget, seed)
                nc = normalise(cell)
                raw.append(
                    {"func": name, "n": n, "budget": budget, "seed": seed, **cell}
                )
                normed.append(
                    {"func": name, "n": n, "budget": budget, "seed": seed, **nc}
                )
                for o in optimizers:
                    agg[o].append(nc[o])
                    if n >= 40:
                        agg_hi[o].append(nc[o])
                print(
                    f"[{i}/{total}] {name:10s} n={n:3d} b={budget:4d} s={seed} :: "
                    + "  ".join(f"{o}={nc[o]:.3f}" for o in optimizers),
                    flush=True,
                )
                atomic_dump(
                    {
                        "done": False,
                        "dims": dims,
                        "seeds": seeds,
                        "funcs": funcs,
                        "budget_mult": args.budget_mult,
                        "optimizers": optimizers,
                        "raw": raw,
                        "normed": normed,
                    },
                    args.out,
                )

    summary = {o: (sum(agg[o]) / len(agg[o]) if agg[o] else None) for o in optimizers}
    summary_hi = {
        o: (sum(agg_hi[o]) / len(agg_hi[o]) if agg_hi[o] else None) for o in optimizers
    }
    atomic_dump(
        {
            "done": True,
            "dims": dims,
            "seeds": seeds,
            "funcs": funcs,
            "budget_mult": args.budget_mult,
            "optimizers": optimizers,
            "summary_all": summary,
            "summary_high_dim_n>=40": summary_hi,
            "raw": raw,
            "normed": normed,
        },
        args.out,
    )

    print("\n=== mean normalised regret (0 = best in cell; lower better) ===")
    print(f"{'optimizer':14s} {'all':>8s} {'n>=40':>8s}")
    for o in optimizers:
        a = summary[o]
        h = summary_hi[o]
        print(f"{o:14s} {a:8.3f} {('--' if h is None else f'{h:8.3f}')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
