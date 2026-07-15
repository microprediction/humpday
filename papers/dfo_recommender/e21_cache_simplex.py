"""E21 — Second worked example: the inspiration simplex over cache-eviction
policies.

Vertices: LRU, LFU, FIFO, CLOCK, RANDOM. The parts of an eviction policy:
on-hit bookkeeping, victim selection, on-insert bookkeeping, aging, and
adaptation. A recipe divides each part among the vertices that own a
mechanism for it, exactly as in the optimizer example.

ARC (Megiddo & Modha 2003) is the known hand-designed interior point: on
the selection suite the panel scores 0.47-0.57 and ARC scores 0.20, so this
interior provably contains much better policies. The question is whether
the construction finds one.

Points: 5 pure vertices + centroid + 6 random interior, one draw each.
Selection: 6 trace families x 2 seeds, parameter-randomized and
key-permuted per seed. Held-out validation later uses unseen seeds.

    ../../.venv/bin/python e21_cache_simplex.py --out runs/e21_cache.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cache_sim as cs  # noqa: E402
import simplex_blend as sb  # noqa: E402

from humpday.transforms.cubetosimplex import cube_to_simplex  # noqa: E402

SLOTS = ["on_hit", "victim", "on_insert", "aging", "adaptation"]

VERTICES = [
    {
        "name": "LRU",
        "idea": "recency: on every hit move the key to the front; evict the least recently used",
        "slots": ["on_hit", "victim"],
    },
    {
        "name": "LFU",
        "idea": "frequency: count accesses per key; evict the least frequently used; decay counts so stale popularity fades",
        "slots": ["on_hit", "victim", "aging"],
    },
    {
        "name": "FIFO",
        "idea": "arrival order: remember insertion order only; evict the oldest inhabitant",
        "slots": ["on_insert", "victim"],
    },
    {
        "name": "CLOCK",
        "idea": "second chance: a reference bit per key set on hit; a rotating hand clears bits and evicts the first unset key",
        "slots": ["on_hit", "victim"],
    },
    {
        "name": "RANDOM",
        "idea": "evict a uniformly random inhabitant; no bookkeeping at all",
        "slots": ["victim"],
    },
]
N = len(VERTICES)

CONTRACT = '''Write a single Python class with EXACTLY this interface:

    class Policy:
        def __init__(self, capacity):
            """capacity = maximum number of keys the cache may hold."""
        def access(self, key):
            """Record an access. Return True if key was cached (hit),
            False otherwise (miss). On a miss the key MUST be inserted,
            evicting one cached key of your choice if at capacity."""

Hard rules:
  - Pure Python. Stdlib only: math, random, collections.
  - Never hold more than `capacity` keys.
  - Deterministic apart from `random`; no prints, no I/O, no globals.

Return ONLY a ```python code block containing the Policy class.'''


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
        "Build a cache EVICTION POLICY by BLENDING established policies - "
        "asymmetrically, not as a 50/50 average. The highest-weight policy is "
        "the HOST architecture; the others donate GRAFTED ideas in proportion "
        "to their weight. '70% A, 30% B' means 'A, but borrow an idea from B', "
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
        "Per-part guidance (shares are how often or how strongly each "
        "mechanism must operate):",
    ]
    for slot in SLOTS:
        owners = spec[slot]
        if owners:
            parts = "; ".join(f"{k} ({int(p * 100)}%): {idea[k]}" for k, p in owners)
            lines.append(f"  - {slot}: {parts}")
    lines += [
        "",
        "Write connective glue so the parts cooperate (e.g. the victim rule "
        "consults whatever bookkeeping the hit rule maintains). Favour a real, "
        "working blend over a faithful copy of any single policy.",
        *(
            [
                "",
                "IMPORTANT - ADAPTIVE SHARES: the percentages above are the "
                "STARTING allocation, not a fixed one. Maintain cheap feedback "
                "structures (for example ghost lists of recently evicted keys, "
                "or per-mechanism hit counters) and REALLOCATE the shares online: "
                "when evictions attributable to one mechanism keep getting "
                "re-requested, shift weight away from that mechanism toward the "
                "others. The blend should drift toward whichever of its ancestors "
                "the current workload rewards, and drift back when the workload "
                "changes.",
            ]
            if adaptive
            else []
        ),
        "",
        CONTRACT,
    ]
    return "\n".join(lines)


def compile_policy(code):
    import types

    m = types.ModuleType("gen_policy")
    exec(compile(code, "<generated>", "exec"), m.__dict__)  # noqa: S102
    cls = getattr(m, "Policy", None)
    if cls is None:
        raise ValueError("no Policy class")
    p = cls(4)
    for k in (1, 2, 3, 4, 5, 1):
        r = p.access(k)
        if not isinstance(r, bool):
            raise ValueError("access must return bool")
    return cls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/e21_cache.json")
    ap.add_argument("--code-dir", default="runs/e21_code")
    ap.add_argument(
        "--adaptive",
        action="store_true",
        help="shares are a starting point the policy adapts online",
    )
    args = ap.parse_args()

    code_dir = Path(args.code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)

    instances = [cs.make_instance(f, s) for f in cs.FAMILIES for s in (0, 1)]
    print("  precomputing panel baselines...", flush=True)
    cache = cs.build_panel_cache(instances)

    refs = {
        name: cs.score_policy(cls, instances, cache)
        for name, cls in list(cs.PANEL.items()) + [("ARC", cs.ARC)]
    }
    print("  references:", {k: round(v, 3) for k, v in refs.items()}, flush=True)

    pts = []
    for i, v in enumerate(VERTICES):
        w = [0.001] * N
        w[i] = 1.0 - 0.001 * (N - 1)
        pts.append((f"pure:{v['name']}", w))
    pts.append(("centroid", list(cube_to_simplex([0.5] * (N - 1)))))
    rng = random.Random(2100)
    for j in range(6):
        pts.append(
            (f"rand{j}", list(cube_to_simplex([rng.random() for _ in range(N - 1)])))
        )

    results = []

    def save(done):
        tmp = Path(args.out + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "done": done,
                    "references": refs,
                    "families": list(cs.FAMILIES),
                    "seeds": [0, 1],
                    "model": None if args.dry_run else args.model,
                    "results": results,
                },
                indent=2,
            )
        )
        tmp.replace(Path(args.out))

    DRY = (
        "class Policy:\n"
        "    def __init__(self, capacity):\n"
        "        from collections import OrderedDict\n"
        "        self.cap = capacity; self.d = OrderedDict()\n"
        "    def access(self, key):\n"
        "        if key in self.d:\n"
        "            self.d.move_to_end(key); return True\n"
        "        if len(self.d) >= self.cap: self.d.popitem(last=False)\n"
        "        self.d[key] = True; return False\n"
    )

    for label, w in pts:
        spec = weights_to_spec(w)
        prompt = build_prompt(spec, adaptive=args.adaptive)
        try:
            code = DRY if args.dry_run else sb.generate_live(prompt, args.model)
            (code_dir / f"{label.replace(':', '_')}.py").write_text(code)
            cls = compile_policy(code)
            regret = cs.score_policy(cls, instances, cache)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:18s} FAILED: {e}", flush=True)
            regret = 1.0
        results.append({"label": label, "w": spec["inspiration"], "regret": regret})
        print(f"  {label:18s} regret={regret:.4f}", flush=True)
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
