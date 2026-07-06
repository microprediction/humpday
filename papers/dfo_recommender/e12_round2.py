"""E12 — Recursive round 2: discovered artifacts become vertices.

The simplex is re-anchored: two validated artifacts from round 1 (the
centroid blend and warm4, both top-ranked out of sample) join three
classical wildcards as vertices. Artifact descriptors carry behavioral
signatures ("nuanced inspiration"), not just mechanism lists.

Lessons from E8/E11 folded in: params capped at 4-6 with 20 inner
evaluations (E11's 8-9 params x 14 evals descents went nowhere); the
selection suite is ROTATED to 8 different burned demos (compounding
selection bias guard); all code saved.

Pipeline as E11: sample 12 recipes -> cluster into 3 behavioral basins ->
one parametric descent per basin -> best tuned floor.

    ../../.venv/bin/python e12_round2.py --dry-run --out runs/e12_smoke.json
    ../../.venv/bin/python e12_round2.py --out runs/e12_round2.json
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simplex_blend as sb  # noqa: E402

# ---- Round-2 vertex set: 2 validated artifacts + 3 classical wildcards ----
ROUND2_VERTICES = [
    {
        "name": "CentroidBlend",
        "idea": ("the validated round-1 champion: DE-population init feeding a "
                 "Nelder-Mead simplex; each step picks one of four move types "
                 "(NM reflect/expand, DE mutation+crossover, adaptive Gaussian, "
                 "coordinate probe); SA acceptance gate with reheat restarts. "
                 "Held-out signature: best mean rank at every budget, most "
                 "consistent across budgets"),
        "slots": ["initialization", "move_generation", "acceptance", "restart"],
    },
    {
        "name": "Warm4Blend",
        "idea": ("a validated pattern-search-dominant blend with CMA seasoning: "
                 "aggressive coordinate-probe moves with shrinking steps, "
                 "Gaussian jumps scaled by an adapted diagonal covariance. "
                 "Held-out signature: the strongest optimizer at budgets "
                 "under 240, fades at 480 (exploits fast, refines less)"),
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
        "name": "SimulatedAnnealing",
        "idea": "accept uphill moves with probability exp(-dF/T); cool T over time",
        "slots": ["acceptance", "restart"],
    },
]

# Rotated selection suite: burned demos, disjoint from the E7/E8/E9/E11 suite
# and from the E6 untouched pool.
ROTATED_SUITE = [
    "cantilever_beam", "darts_aim", "ebola_response", "genetic_art",
    "lennard_jones_cluster", "pressure_vessel", "tennis_doubles",
    "tuned_mass_damper",
]

PARAMS_ADDENDUM = """

ADDITIONALLY, and importantly: expose the 4 to 6 most behaviour-critical
numeric constants of your design as module-level dicts

    PARAMS = {"name": default_value, ...}          # plain numbers
    PARAM_RANGES = {"name": (low, high), ...}      # sensible bounds

and READ every such constant inside `optimize` via PARAMS["name"], so that
overwriting a PARAMS entry changes the optimizer's behaviour. Defaults must
reproduce your intended design."""


def main() -> int:
    # Re-anchor the simplex, then reuse the E11 pipeline wholesale.
    sb.VERTICES = ROUND2_VERTICES
    sb.N_VERTICES = len(ROUND2_VERTICES)

    import e11_shgo as e11

    e11.SUITE = ROTATED_SUITE
    e11.PARAMS_ADDENDUM = PARAMS_ADDENDUM
    e11.INNER_EVALS = 20
    sys.argv = [a if a != "e12_round2.py" else "e11_shgo.py" for a in sys.argv]
    if "--out" not in sys.argv:
        sys.argv += ["--out", "runs/e12_round2.json"]
    if "--code-dir" not in sys.argv:
        sys.argv += ["--code-dir", "runs/e12_code"]
    return e11.main()


if __name__ == "__main__":
    raise SystemExit(main())
