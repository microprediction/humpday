"""E22 — Third worked example: the inspiration simplex over distributional
time-series forecasters.

Vertices: exponential smoothing, autoregression, GARCH, Student tails, and
a regime watcher. Parts: conditional mean, conditional scale, distribution
shape, adaptation, regime handling. The skaters package's `laplace` is the
reference champion (the ARC analog), reported but not in the panel.

PRE-REGISTERED PREDICTION (from the paper's two-example lesson, stated
before this run): forecasting keeps its value in cross-step invariants
(volatility memory, regime tracking), so static shares should fail as they
did for caches, and adaptive shares should be where the action is.

Rounds: --adaptive off (static shares) and on (shares as starting
allocation, adapted online). Selection seeds {0,1}; seeds 20-24 reserved
for holdout.

    ../../.venv/bin/python e22_forecast_simplex.py --out runs/e22_static.json
    ../../.venv/bin/python e22_forecast_simplex.py --adaptive --out runs/e22_adaptive.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import forecast_sim as fs  # noqa: E402
import simplex_blend as sb  # noqa: E402

from humpday.transforms.cubetosimplex import cube_to_simplex  # noqa: E402

SLOTS = [
    "conditional_mean",
    "conditional_scale",
    "distribution_shape",
    "adaptation",
    "regime_handling",
]

VERTICES = [
    {
        "name": "ExponentialSmoothing",
        "idea": "track a slowly moving level by exponential smoothing and forecast it; "
        "estimate scale from smoothed residuals",
        "slots": ["conditional_mean", "adaptation"],
    },
    {
        "name": "AutoRegression",
        "idea": "predict the next value from a linear combination of recent values or "
        "differences, coefficients fit online",
        "slots": ["conditional_mean"],
    },
    {
        "name": "GARCH",
        "idea": "conditional variance: tomorrow's variance blends a long-run level, "
        "today's squared surprise, and today's variance",
        "slots": ["conditional_scale", "adaptation"],
    },
    {
        "name": "StudentTails",
        "idea": "heavy-tailed (Student-t) predictive densities and outlier-robust "
        "scale estimation",
        "slots": ["distribution_shape"],
    },
    {
        "name": "RegimeWatcher",
        "idea": "monitor forecast errors for structural breaks; on a suspected break, "
        "inflate predictive variance and refresh the estimates",
        "slots": ["regime_handling", "conditional_scale"],
    },
]
N = len(VERTICES)

CONTRACT = '''Write a single Python class with EXACTLY this interface:

    class Forecaster:
        def __init__(self):
            """No arguments. All state internal."""
        def logpdf(self, y):
            """Log density of the NEXT observation under your current
            one-step-ahead predictive distribution, evaluated at y,
            BEFORE updating on y. Must be a proper density in y
            (integrates to 1). Return a finite float."""
        def update(self, y):
            """Then observe y and update all internal state."""

Hard rules:
  - Pure Python. Stdlib only: math, random.
  - The caller always invokes logpdf(y) first, then update(y), tick by tick.
  - logpdf must never return NaN or +inf; guard all variances away from zero.
  - Series arrive on arbitrary scales and offsets; self-normalise online.

Return ONLY a ```python code block containing the Forecaster class.'''


def weights_to_spec(w):
    spec = {"inspiration": {VERTICES[i]["name"]: round(w[i], 3) for i in range(N)}}
    for slot in SLOTS:
        contenders = [(i, w[i]) for i in range(N) if slot in VERTICES[i]["slots"]]
        tot = sum(x for _, x in contenders) or 1.0
        spec[slot] = [
            (VERTICES[i]["name"], round(x / tot, 3))
            for i, x in contenders
            if x / tot > 0.01
        ]
    return spec


def build_prompt(spec, adaptive=False):
    insp = spec["inspiration"]
    host = max(insp, key=insp.get)
    idea = {v["name"]: v["idea"] for v in VERTICES}
    grafts = sorted(
        ((k, v) for k, v in insp.items() if k != host and v > 0.01), key=lambda t: -t[1]
    )
    lines = [
        "Build an online, one-step-ahead DISTRIBUTIONAL FORECASTER for a "
        "univariate series by BLENDING established ideas - asymmetrically, "
        "not as a 50/50 average. The highest-weight idea is the HOST "
        "architecture; the others donate GRAFTED mechanisms in proportion to "
        "their weight. '70% A, 30% B' means 'A, but borrow an idea from B', "
        "not a chimera.",
        "",
        f"HOST architecture ({int(insp[host] * 100)}%): {host} — {idea[host]}",
        "GRAFT into it: "
        + (
            ", ".join(f"{int(v * 100)}% {k}" for k, v in grafts)
            if grafts
            else "(nothing — pure host)"
        ),
        "",
        "Per-part guidance (shares are how strongly each mechanism must "
        "shape that part):",
    ]
    for slot in SLOTS:
        owners = spec[slot]
        if owners:
            parts = "; ".join(f"{k} ({int(p * 100)}%): {idea[k]}" for k, p in owners)
            lines.append(f"  - {slot}: {parts}")
    lines += [
        "",
        "Write connective glue so the parts cooperate (the scale model feeds "
        "the density; the regime monitor can reset the mean model). Favour a "
        "real, working blend over a faithful copy of any single method.",
        *(
            [
                "",
                "IMPORTANT - ADAPTIVE SHARES: the percentages above are the "
                "STARTING allocation, not a fixed one. Track each mechanism's "
                "recent predictive performance (for example running log-scores of "
                "component predictions) and REALLOCATE influence online toward "
                "whichever ancestors the current stretch of the series rewards, "
                "drifting back when the series changes character.",
            ]
            if adaptive
            else []
        ),
        "",
        CONTRACT,
    ]
    return "\n".join(lines)


def compile_forecaster(code):
    import types

    m = types.ModuleType("gen_forecaster")
    exec(compile(code, "<generated>", "exec"), m.__dict__)  # noqa: S102
    cls = getattr(m, "Forecaster", None)
    if cls is None:
        raise ValueError("no Forecaster class")
    f = cls()
    import math as _m

    for y in (0.0, 1.0, -2.0, 100.0, 100.5):
        v = f.logpdf(y)
        if not isinstance(v, (int, float)) or _m.isnan(v) or v == float("inf"):
            raise ValueError("logpdf must return finite float")
        f.update(y)
    return cls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--out", default="runs/e22_static.json")
    ap.add_argument("--code-dir", default="")
    args = ap.parse_args()
    code_dir = Path(args.code_dir or args.out.replace(".json", "_code"))
    code_dir.mkdir(parents=True, exist_ok=True)

    instances = [fs.make_instance(f, s) for f in fs.FAMILIES for s in (0, 1)]
    print("  panel baselines...", flush=True)
    cache = fs.build_panel_cache(instances)
    refs = {
        n: fs.score_forecaster(c, instances, cache)
        for n, c in list(fs.PANEL.items()) + [("laplace", fs.LaplaceRef)]
    }
    print("  references:", {k: round(v, 3) for k, v in refs.items()}, flush=True)

    pts = []
    for i, v in enumerate(VERTICES):
        w = [0.001] * N
        w[i] = 1.0 - 0.001 * (N - 1)
        pts.append((f"pure:{v['name']}", w))
    pts.append(("centroid", list(cube_to_simplex([0.5] * (N - 1)))))
    rng = random.Random(2200)
    for j in range(6):
        pts.append(
            (f"rand{j}", list(cube_to_simplex([rng.random() for _ in range(N - 1)])))
        )

    DRY = (
        "import math\nclass Forecaster:\n"
        "    def __init__(self): self.last=0.0; self.var=1.0; self.n=0\n"
        "    def logpdf(self, y):\n"
        "        if not self.n: return -30.0\n"
        "        v=max(self.var,1e-12)\n"
        "        return -0.5*math.log(2*math.pi*v)-0.5*(y-self.last)**2/v\n"
        "    def update(self, y):\n"
        "        if self.n: d=y-self.last; self.var=0.95*self.var+0.05*d*d\n"
        "        self.last=y; self.n+=1\n"
    )

    results = []

    def save(done):
        tmp = Path(args.out + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "done": done,
                    "adaptive": args.adaptive,
                    "references": refs,
                    "families": list(fs.FAMILIES),
                    "seeds": [0, 1],
                    "model": None if args.dry_run else args.model,
                    "results": results,
                },
                indent=2,
            )
        )
        tmp.replace(Path(args.out))

    for label, w in pts:
        spec = weights_to_spec(w)
        prompt = build_prompt(spec, adaptive=args.adaptive)
        try:
            code = (
                DRY
                if args.dry_run
                else sb.generate_live(prompt, args.model, max_tokens=16000)
            )
            (code_dir / f"{label.replace(':', '_')}.py").write_text(code)
            cls = compile_forecaster(code)
            regret = fs.score_forecaster(cls, instances, cache)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:26s} FAILED: {e}", flush=True)
            regret = 1.0
        results.append({"label": label, "w": spec["inspiration"], "regret": regret})
        print(f"  {label:26s} regret={regret:.4f}", flush=True)
        save(False)

    save(True)
    print(
        "\n=== leaderboard (references: "
        + ", ".join(f"{k} {v:.3f}" for k, v in sorted(refs.items(), key=lambda t: t[1]))
        + ") ==="
    )
    for r in sorted(results, key=lambda r: r["regret"]):
        print(f"  {r['regret']:.4f}  {r['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
