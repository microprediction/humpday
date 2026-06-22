"""
Hybrid numerical/semantic optimizer discovery — the "simplex of inspirations".

Idea (see IDEAS.md): place N base optimizers at the VERTICES of a probability
simplex. A point on the simplex is a convex weighting over them — "30% inspired
by A, 20% by B, ...". An LLM (Claude) is the genotype->phenotype map: given the
weights it GENERATES the source of a new optimizer whose mechanisms are blended
according to the weights. The generated optimizer is scored on the *disguised*
benchmark (memorisation-proof), and that regret is the fitness a numerical search
over the simplex minimises. Hybrid: numerical outer search + semantic
genotype->phenotype + numerical evaluation.

ENFORCED STRUCTURAL BLEND (the chosen design): rather than vague prompt-steering,
the weights map to concrete *slot ownership*. An optimizer is decomposed into
slots — initialization, move-generation, acceptance, adaptation, restart — and
each vertex algorithm contributes its characteristic mechanism to the slots it's
known for. The weights become per-slot probabilities the generated code must
implement, so the simplex coordinate provably bites (30% vs 35% changes the
spec), while the LLM still writes novel connective glue.

The simplex is searched through the repo's own cube<->simplex bijection
(`humpday.transforms.cubetosimplex`): the outer optimizer moves in [0,1]^(N-1),
lifted to barycentric weights. This is the same "optimize on the simplex" trick
the library is built around, one level up — the points are allocations of
algorithmic *inspiration*.

Live generation uses Claude (model `claude-opus-4-8`). Set ANTHROPIC_API_KEY.
Without a key, `--dry-run` swaps in a deterministic template generator so the
whole pipeline (simplex -> spec -> prompt -> sandbox -> score) is verifiable.

    python papers/dfo_recommender/simplex_blend.py --dry-run            # no API, plumbing test
    python papers/dfo_recommender/simplex_blend.py --points vertices    # live, 5 pure vertices + centroid
    python papers/dfo_recommender/simplex_blend.py --points 8 --demos 5  # live, 8 random simplex points
"""

from __future__ import annotations

import argparse
import math
import random
import signal
import sys
import textwrap
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import algo_dev as ad  # noqa: E402
from example_demos import DEMOS, disguise_demo  # noqa: E402

from humpday.transforms.cubetosimplex import cube_to_simplex  # noqa: E402

INF = float("inf")

# --------------------------------------------------------------------------
# The simplex vertices: base optimizers and the slots each is known for. The
# `slots` list is what makes the structural blend enforceable — a vertex only
# competes for ownership of the slots it actually has a mechanism for.
# --------------------------------------------------------------------------
SLOTS = ["initialization", "move_generation", "acceptance", "adaptation", "restart"]

VERTICES = [
    {
        "name": "NelderMead",
        "idea": "downhill simplex: reflect/expand/contract/shrink a set of n+1 points",
        "slots": ["move_generation", "adaptation"],
    },
    {
        "name": "DifferentialEvolution",
        "idea": "population + difference-vector mutation (rand/1, current-to-best/1) and crossover",
        "slots": ["initialization", "move_generation"],
    },
    {
        "name": "CMAEvolutionStrategy",
        "idea": "sample from a Gaussian whose covariance adapts to successful steps",
        "slots": ["move_generation", "adaptation"],
    },
    {
        "name": "PatternSearch",
        "idea": "Hooke-Jeeves coordinate moves with a step size that shrinks on failure",
        "slots": ["move_generation", "adaptation"],
    },
    {
        "name": "SimulatedAnnealing",
        "idea": "accept uphill moves with probability exp(-dF/T); cool T over time",
        "slots": ["acceptance", "restart"],
    },
]
N_VERTICES = len(VERTICES)


def weights_to_spec(weights):
    """Map simplex weights (one per vertex, summing to 1) to a per-slot blend.

    For each slot, only vertices that own a mechanism for it compete; their
    global weights are renormalised within the slot. Returns
    {slot: [(vertex_name, pct), ...]} plus the raw inspiration weights."""
    spec = {"inspiration": {VERTICES[i]["name"]: round(weights[i], 3) for i in range(N_VERTICES)}}
    for slot in SLOTS:
        contenders = [(i, weights[i]) for i in range(N_VERTICES) if slot in VERTICES[i]["slots"]]
        tot = sum(w for _, w in contenders) or 1.0
        spec[slot] = [
            (VERTICES[i]["name"], round(w / tot, 3)) for i, w in contenders if w / tot > 0.01
        ]
    return spec


CONTRACT = textwrap.dedent(
    '''\
    Write a single Python function with EXACTLY this signature and contract:

        def optimize(objective, n_trials, n_dim):
            """Minimise `objective` (a callable taking a list of n_dim floats in
            [0,1] and returning a float). Use AT MOST n_trials calls to
            objective. Return (best_value, best_point)."""

    Hard rules:
      - Pure Python. You may `import math` and `import random` only. No numpy, no I/O.
      - Every candidate must be clipped into [0,1] before evaluation.
      - Count objective calls; never exceed n_trials. Stop and return the best
        seen as soon as the budget is spent.
      - Return (best_value, best_point) where best_point is a list of n_dim floats.
      - No global state, no prints. Just the function.
    '''
)


def build_prompt(spec):
    """Turn a blend spec into the structural-blend instruction for the LLM."""
    insp = spec["inspiration"]
    host = max(insp, key=insp.get)
    grafts = sorted(((k, v) for k, v in insp.items() if k != host and v > 0.01),
                    key=lambda t: -t[1])
    lines = [
        "Build a black-box numerical optimizer by BLENDING base algorithms — but "
        "blending is ASYMMETRIC, not a 50/50 average. The highest-weight method "
        "is the HOST architecture; the others donate GRAFTED ideas in proportion "
        "to their weight (a 30% method contributes a prominent borrowed mechanism; "
        "a 5% method only a light inflection). '70% A, 30% B' means 'A, but borrow "
        "an idea from B', not a chimera.",
        "",
        f"HOST architecture ({int(insp[host] * 100)}%): {host} — build the skeleton from this.",
        "GRAFT into it: "
        + (", ".join(f"{int(v * 100)}% {k}" for k, v in grafts) if grafts else "(nothing — pure host)"),
        "",
        "Inspiration weights: "
        + ", ".join(f"{k} {int(v * 100)}%" for k, v in insp.items() if v > 0.01),
        "",
        "Per-slot guidance (host owns a slot unless a graft's weight dominates it):",
    ]
    idea = {v["name"]: v["idea"] for v in VERTICES}
    for slot in SLOTS:
        owners = spec[slot]
        if not owners:
            continue
        parts = "; ".join(f"{name} ({int(pct * 100)}%): {idea[name]}" for name, pct in owners)
        lines.append(f"  - {slot}: {parts}")
    lines += [
        "",
        "Write connective glue as needed so the slots cooperate (e.g. the "
        "acceptance rule gates the move-generation output; adaptation updates "
        "the step/covariance used by move-generation). Favour a real, working "
        "blend over a faithful copy of any single algorithm.",
        "",
        CONTRACT,
        "",
        "Return ONLY a ```python code block containing the optimize function.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Generation: Claude (live) or a deterministic template (dry-run).
# --------------------------------------------------------------------------
def _extract_code(text):
    if "```python" in text:
        text = text.split("```python", 1)[1]
    elif "```" in text:
        text = text.split("```", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip()


def generate_live(prompt, model="claude-opus-4-8", max_tokens=8000):
    import anthropic

    client = anthropic.Anthropic()
    system = (
        "You are an expert in derivative-free optimization. You write correct, "
        "self-contained pure-Python optimizers that respect their evaluation "
        "budget exactly. You output only the requested code block."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return _extract_code(text)


_DRY_TEMPLATE = textwrap.dedent(
    """\
    import math, random
    def optimize(objective, n_trials, n_dim):
        def clip(x): return [min(1.0, max(0.0, xi)) for xi in x]
        evals = [0]; best_f = [float('inf')]; best_x = [None]
        def ev(x):
            x = clip(x); f = objective(x); evals[0]+= 1
            if f < best_f[0]: best_f[0], best_x[0] = f, x[:]
            return f
        pop = max(4, min(10, n_trials//3))
        P = [[random.random() for _ in range(n_dim)] for _ in range(pop)]
        F = [ev(p) for p in P]; sigma = 0.2
        while evals[0] < n_trials:
            i = min(range(pop), key=lambda k: F[k])
            cand = [P[i][d] + random.gauss(0, sigma) for d in range(n_dim)]
            fc = ev(cand)
            if fc < F[i]: P[i], F[i], sigma = clip(cand), fc, sigma*1.1
            else: sigma = max(1e-3, sigma*0.95)
        return best_f[0], best_x[0]
    """
)


def compile_optimizer(code, eval_timeout_s=20):
    """Exec generated code in a restricted namespace; return a wrapped optimize
    that enforces the eval budget and a wall-clock timeout. Raises on bad code."""
    allowed = {"math": math, "random": random}

    def _import(name, *a, **k):
        if name in allowed:
            return allowed[name]
        raise ImportError(f"import '{name}' not allowed in sandbox")

    safe_builtins = {
        n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
        for n in (
            "range len min max abs sum sorted float int list tuple enumerate "
            "zip map filter round pow divmod reversed bool any all next iter "
            "set dict frozenset str isinstance type slice hash repr format "
            "ValueError ZeroDivisionError OverflowError ArithmeticError Exception"
        ).split()
    }
    safe_builtins["__import__"] = _import
    g = {"__builtins__": safe_builtins, "math": math, "random": random}
    exec(compile(code, "<generated>", "exec"), g)  # noqa: S102 (sandboxed)
    fn = g.get("optimize")
    if not callable(fn):
        raise ValueError("generated code defines no callable `optimize`")

    def wrapped(objective, n_trials, n_dim):
        state = {"n": 0}

        class _Budget(Exception):
            pass

        def counting(x):
            if state["n"] >= n_trials:
                raise _Budget()
            state["n"] += 1
            return objective(x)

        def _alarm(*_):
            raise TimeoutError("optimizer exceeded wall-clock budget")

        had_alarm = hasattr(signal, "SIGALRM")
        if had_alarm:
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(eval_timeout_s)
        try:
            return fn(counting, n_trials, n_dim)
        except _Budget:
            return None  # ran out of budget without returning; scorer handles
        finally:
            if had_alarm:
                signal.alarm(0)

    return wrapped


# --------------------------------------------------------------------------
# Scoring: regret vs the panel on disguised demos (mirrors algo_dev fitness,
# but the candidate is an arbitrary optimize(objective, n_trials, n_dim)).
# --------------------------------------------------------------------------
def build_panel_cache(base_demos, seeds, n_trials, panel=ad.PANEL):
    """Precompute panel baselines once per (demo, seed). The panel does not depend
    on the candidate, so caching it makes evaluating many simplex points ~Kx
    cheaper (K = number of points) instead of rerunning NM/DE/CMA every time."""
    cache = {}
    for i, demo in enumerate(base_demos):
        for s in seeds:
            inst = disguise_demo(demo, s)
            seed = 5000 + 31 * i + s
            cache[(i, s)] = [ad._panel_best(a, inst, n_trials, seed) for a in panel]
    return cache


def score_optimizer(opt, base_demos, seeds, n_trials, panel=ad.PANEL, panel_cache=None):
    scores = []
    for i, demo in enumerate(base_demos):
        for s in seeds:
            inst = disguise_demo(demo, s)
            seed = 5000 + 31 * i + s
            random.seed(seed)
            try:
                import numpy as np

                np.random.seed(seed)
            except Exception:  # noqa: BLE001
                pass
            try:
                res = opt(inst.objective, n_trials, inst.n_dim)
                cand = float(res[0]) if res else INF
            except Exception:  # noqa: BLE001
                cand = INF
            if panel_cache is not None:
                panel_vals = panel_cache[(i, s)]
            else:
                panel_vals = [ad._panel_best(a, inst, n_trials, seed) for a in panel]
            vals = [cand] + panel_vals
            finite = [v for v in vals if v < INF]
            if cand >= INF or not finite:
                scores.append(1.0)
                continue
            mn, mx = min(finite), max(finite)
            scores.append(0.0 if mx <= mn else (cand - mn) / (mx - mn))
    return mean(scores) if scores else 1.0


def select_demos(n, mode="head"):
    """Pick `n` demos. 'head' = first n (legacy). 'spread' = an even stride over
    the dimension-sorted suite, so the subset spans low- to high-dim (incl. the
    scaled n>=16 problems) rather than only the low-dim head."""
    if mode == "spread" and n < len(DEMOS):
        ds = sorted(DEMOS, key=lambda d: d.n_dim)
        idx = sorted({round(k * (len(ds) - 1) / max(n - 1, 1)) for k in range(n)})
        return [ds[i] for i in idx]
    return DEMOS[:n]


# Previously-best recipe (runs/simplex_overnight.json: rand14) — PatternSearch-
# dominant with CMA seasoning. The warm start seeds the search here rather than at
# random, then perturbs to explore the neighbourhood.
WARM_CENTER = {
    "NelderMead": 0.047,
    "DifferentialEvolution": 0.055,
    "CMAEvolutionStrategy": 0.211,
    "PatternSearch": 0.638,
    "SimulatedAnnealing": 0.049,
}


def _perturb_weights(center, spread, seed):
    random.seed(seed)
    w = [max(1e-3, c + random.gauss(0, spread)) for c in center]
    tot = sum(w)
    return [x / tot for x in w]


def _vertex_baselines():
    """Pure vertices + centroid — the baselines an interior blend must beat."""
    pts = []
    for i, v in enumerate(VERTICES):
        w = [0.001] * N_VERTICES
        w[i] = 1.0 - 0.001 * (N_VERTICES - 1)
        pts.append((f"pure:{v['name']}", w))
    pts.append(("centroid", cube_to_simplex([0.5] * (N_VERTICES - 1))))
    return pts


def simplex_points(mode, n_warm=8):
    """Return a list of (label, weights) to evaluate.

    'vertices' — pure vertices + centroid.
    'warm'     — baselines (pure vertices + centroid) AND a warm-started search:
                 the previously-best recipe (WARM_CENTER) plus a perturbed
                 neighbourhood, so the interior-vs-vertex question is answered in
                 one comprehensive run.
    integer    — that many random interior points (legacy)."""
    if mode == "vertices":
        return _vertex_baselines()
    if mode == "warm":
        pts = _vertex_baselines()
        center = [WARM_CENTER[v["name"]] for v in VERTICES]
        pts.append(("warm:center", center))
        for j in range(n_warm):
            pts.append((f"warm{j}", _perturb_weights(center, 0.12, 7000 + j)))
        return pts
    n = int(mode)
    pts = []
    for j in range(n):
        random.seed(1234 + j)
        u = [random.random() for _ in range(N_VERTICES - 1)]
        pts.append((f"rand{j}", cube_to_simplex(u)))
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", default="vertices", help="'vertices', 'warm', or an integer count")
    ap.add_argument("--n-warm", type=int, default=8, help="warm-cluster size (mode 'warm')")
    ap.add_argument("--demos", type=int, default=5)
    ap.add_argument("--demo-mode", choices=("head", "spread"), default="head",
                    help="'spread' samples across the dimension-sorted suite (incl. high-dim)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true", help="no API; use template generator")
    ap.add_argument("--save-code", default="", help="dir to write each generated optimizer")
    ap.add_argument("--out", default="", help="JSON path for crash-safe per-point checkpoints")
    args = ap.parse_args()

    ckpt = Path(args.out) if args.out else None

    def write_ckpt(results, done):
        if ckpt is None:
            return
        import json

        payload = {
            "done": done,
            "vertices": [v["name"] for v in VERTICES],
            "n_demos": args.demos,
            "seeds": args.seeds,
            "trials": args.trials,
            "model": args.model,
            "results": [{"label": l, "regret": r, "inspiration": ins} for l, r, ins in results],
        }
        tmp = Path(str(ckpt) + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(ckpt)

    base = select_demos(args.demos, args.demo_mode)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    pts = simplex_points(args.points, args.n_warm)
    save_dir = Path(args.save_code) if args.save_code else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    print("  precomputing panel baselines (cached across all points)...", flush=True)
    panel_cache = build_panel_cache(base, seeds, args.trials)

    print(
        f"Simplex-of-inspirations over {N_VERTICES} vertices "
        f"({', '.join(v['name'] for v in VERTICES)})\n"
        f"{len(pts)} simplex points x {len(base)} demos x {len(seeds)} seeds, "
        f"{args.trials} trials. mode={'DRY-RUN (template)' if args.dry_run else 'LIVE ' + args.model}\n"
    )

    results = []
    for label, w in pts:
        spec = weights_to_spec(w)
        prompt = build_prompt(spec)
        try:
            code = _DRY_TEMPLATE if args.dry_run else generate_live(prompt, args.model)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:24s} generation FAILED: {e}", flush=True)
            continue
        if save_dir:
            (save_dir / f"{label.replace(':', '_')}.py").write_text(code)
        try:
            opt = compile_optimizer(code)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:24s} compile FAILED: {e}", flush=True)
            continue
        regret = score_optimizer(opt, base, seeds, args.trials, panel_cache=panel_cache)
        results.append((label, regret, spec["inspiration"]))
        print(f"  {label:24s} regret={regret:.4f}", flush=True)
        write_ckpt(results, done=False)

    write_ckpt(results, done=True)
    if results:
        results.sort(key=lambda t: t[1])
        print("\n=== Leaderboard (normalised regret vs panel; lower = better) ===")
        for label, regret, insp in results:
            flavour = ", ".join(f"{k[:4]}{int(v * 100)}" for k, v in insp.items() if v > 0.05)
            print(f"  {regret:.4f}  {label:24s}  [{flavour}]")
        print(
            "\nThe numerical search over the simplex would now move toward the "
            "low-regret region; each point's fitness came from LLM-generated code."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
