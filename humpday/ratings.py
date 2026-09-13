"""Recorded optimizer ratings: one table, one way to read it.

There used to be three rating artifacts, built by different scripts at different times over
different objectives, feeding different selection paths. A Borda grid over nine analytic surfaces
drove ``minimize``; two Elo tables over sphere and Rosenbrock in two dimensions drove little; a
per-dimension two-suite Elo table drove ``suggest``. The broadest of them measured the suite that
the benchmark-validity study finds has no rank correlation with real-world performance, so the
best-organised evidence was also the least informative.

One table now, indexed by the three things that change the answer.

``dimension``
    Which optimizer wins changes with it, and not smoothly: a method can work at five variables and
    fail at ten. **Never interpolated.** An unmeasured dimension reports nothing.

``budget``
    Evaluations allowed, in absolute terms, because that is what a caller knows. NEWUOA and BOBYQA
    need roughly ``2n+1`` points before proposing anything and UOBYQA needs ``(n+1)(n+2)/2``, which
    is 5,151 at a hundred variables, so a budget that is generous at ten is nothing at a hundred.
    Interpolated **downward only**: a cell recorded at a smaller budget says what an optimizer does
    with less room, which is the safe direction to be wrong in.

``suite``
    ``surfaces`` are classic analytic functions, randomly shifted, rotated, rescaled and noised per
    run. ``engineering`` are worked physics and engineering problems. They disagree systematically,
    which is why both are recorded rather than averaged into one number.
"""

from __future__ import annotations

import json
from pathlib import Path

SUITES = ("surfaces", "engineering")

# Weight of the prior when shrinking a cell's rank toward an optimizer's cross-suite mean. A cell
# backed by this many problems is trusted halfway.
PRIOR_PROBLEMS = 15.0

_TABLE: dict | None = None


def _load() -> dict:
    global _TABLE
    if _TABLE is None:
        try:
            raw = json.loads(
                (Path(__file__).parent / "data" / "ratings.json").read_text()
            )
            _TABLE = raw.get("cells", {})
        except Exception:  # pragma: no cover - the table is optional
            _TABLE = {}
    return _TABLE


def cells() -> dict:
    """Raw table: ``{"<dim>/<budget>/<suite>": {"ratings", "problems", "timed_out"}}``."""
    return _load()


def _ranked(cell: dict) -> list:
    """Optimizers in this cell, best first, with the ones that could not return placed last.

    An optimizer is listed under ``timed_out`` when it twice failed to come back inside a wall clock
    allowance set at a multiple of what the objective's own evaluations cost, so the allowance
    tracks the problem and the finding is about the method. Being unable to return is worse than
    losing, and it is kept out of the Elo because it is not a result; here it simply ranks last.

    A disqualification usually happens partway through a cell, so the optimizer carries a rating
    from the problems it did finish as well as a place in ``timed_out``. The timeout wins: a method
    that stopped returning is not recommendable at that size on the strength of the rounds it
    completed before it stopped.
    """
    out = sorted(cell.get("timed_out", []))
    ratings = {n: r for n, r in cell.get("ratings", {}).items() if n not in set(out)}
    return sorted(ratings, key=lambda n: -ratings[n]) + out


def recorded_dimensions() -> list:
    return sorted({int(k.split("/")[0]) for k in _load()})


def _budgets(n_dim: int, suite: str) -> list:
    return sorted(
        int(k.split("/")[1])
        for k in _load()
        if k.startswith(f"{n_dim}/") and k.endswith(f"/{suite}")
    )


def _pick(budgets: list, n_trials: int):
    """The largest recorded budget not exceeding ``n_trials``, else the smallest recorded.

    Interpolating downward is the safe direction to be wrong in: a cell recorded at a smaller
    budget says what an optimizer does with less room. When every recorded budget is larger, the
    smallest is used, on the grounds that evidence measured under a more generous budget is worth
    having and worth knowing about, rather than silence.
    """
    if not budgets:
        return None
    usable = [b for b in budgets if b <= n_trials]
    return max(usable) if usable else min(budgets)


def cell_for(n_dim: int, suite: str, n_trials: int):
    """The one cell answering a single-suite question, or ``None``."""
    budget = _pick(_budgets(n_dim, suite), n_trials)
    return None if budget is None else _load().get(f"{n_dim}/{budget}/{suite}")


def cells_at(n_dim: int, n_trials: int) -> dict:
    """One cell per suite at this dimension, all at the **same** budget.

    A common budget rather than the best available per suite, which is the whole reason this
    function exists rather than two calls to :func:`cell_for`. Ranks from different budgets are not
    comparable: at fifty variables a thousand evaluations is enough for NEWUOA to build its
    interpolation set and fifty is not, so pairing one suite measured at a thousand against another
    measured at fifty would read a budget effect as a suite effect. If the suites were recorded at
    different budgets -- one cell abandoned on wall clock, say -- the largest budget they share is
    used, and if they share none this returns whatever single suite exists, which
    :func:`robust_order` then declines to rank.
    """
    per_suite = {suite: _budgets(n_dim, suite) for suite in SUITES}
    present = {suite: b for suite, b in per_suite.items() if b}
    if not present:
        return {}

    shared = set.intersection(*(set(b) for b in present.values()))
    budget = _pick(sorted(shared), n_trials)
    if budget is None:  # nothing in common: answer for whichever suite exists, alone
        return {
            suite: _load()[f"{n_dim}/{_pick(b, n_trials)}/{suite}"]
            for suite, b in present.items()
        }
    return {suite: _load()[f"{n_dim}/{budget}/{suite}"] for suite in present}


def robust_order(n_dim: int, n_trials: int = 100) -> list:
    """Optimizers from most to least dependable across the suites. Empty where unmeasured.

    Ratings are compared *within* a cell only. Elo scales differ between suites, the analytic
    tournaments running several hundred points wider, so a rating means something against its own
    cell and nothing against another. Everything here works on within-cell ranks.

    The score is a pessimistic rank rather than a mean. A mean rewards a specialist that wins one
    suite and sits near the bottom of the other, which is the recommendation two suites exist to
    avoid. A raw worst rank, though, hangs the ordering on a single cell, so each rank is first
    shrunk toward the optimizer's cross-suite average by ``w = n / (n + PRIOR_PROBLEMS)``, where
    ``n`` is how many problems back that cell. A rank from a fifty-problem tournament moves the
    score more than one from four. With equally sized cells this reduces to a penalised mean; as a
    cell grows it approaches the true worst case.

    That average is itself weighted by ``w``. Shrinking toward an unweighted mean leaves the thin
    cell contaminating the target it is being shrunk toward, so a four-problem tournament that
    happened to run out of wall clock still swings the ordering through the back door. A cell
    should be discounted once, in both places it is used.
    """
    at = cells_at(n_dim, n_trials)
    if len(at) < 2:
        return []

    ranks: dict = {}
    for suite, cell in at.items():
        for position, name in enumerate(_ranked(cell), start=1):
            ranks.setdefault(name, {})[suite] = position

    weight = {
        suite: float(cell.get("problems", 0))
        / (float(cell.get("problems", 0)) + PRIOR_PROBLEMS)
        for suite, cell in at.items()
    }
    total = sum(weight.values()) or 1.0

    scored = []
    for name, per_suite in ranks.items():
        if len(per_suite) != len(at):
            continue  # only rank what every suite scored
        target = sum(weight[s] * r for s, r in per_suite.items()) / total
        worst = max(
            weight[s] * r + (1.0 - weight[s]) * target for s, r in per_suite.items()
        )
        scored.append((worst, target, name))
    scored.sort()
    return [name for _, _, name in scored]


def suite_order(n_dim: int, suite: str, n_trials: int = 100) -> list:
    """Optimizers ordered by measured rating on one suite. Empty if that cell is unrecorded."""
    cell = cell_for(n_dim, suite, n_trials)
    return [] if not cell else _ranked(cell)


def rating(n_dim: int, suite: str, name: str, n_trials: int = 100):
    """Measured rating, or ``None`` where nothing was recorded or the optimizer timed out."""
    cell = cell_for(n_dim, suite, n_trials)
    if not cell or name in cell.get("timed_out", []):
        return None
    return cell.get("ratings", {}).get(name)


def timed_out(n_dim: int, suite: str, n_trials: int = 100) -> list:
    """Optimizers that could not return inside the allowance in this cell. Ranked last, not rated."""
    cell = cell_for(n_dim, suite, n_trials)
    return sorted(cell.get("timed_out", [])) if cell else []
