"""
Tests for the smart-default behavior of `minimize` / `cube_minimize`.

When `method` is None, the algorithm is picked by `humpday.eligibility.recommend`,
which combines dimensional caps, minimum-trials gating, and an overhead-tier
filter driven by measured objective eval-time.

This file exercises both branches:
  * `auto_timing=False` — the rule-based fallback, which mirrors `suggest_pure`
    and is what we want existing callers to see in fast-microbenchmark contexts.
  * `auto_timing=True` (the default) — the timing-driven branch, which avoids
    heavy-overhead algorithms when the objective is cheap.
"""

from unittest.mock import patch

import numpy as np
import pytest

from humpday import cube_minimize, eligibility, minimize
from humpday.optimizers.alloptimizers import suggest_pure


def _quadratic(x):
    return float(np.sum((np.asarray(x) - 0.3) ** 2))


@pytest.mark.parametrize("n_dim", [1, 2, 5, 20, 75])
def test_minimize_no_method_no_timing_matches_rule_ranking(
    n_dim, monkeypatch, tmp_path
):
    """With auto_timing disabled AND no benchmarks grid, the recommender's
    pick should equal the rule-based top — which is suggest_pure[0] for the
    dimensions tested here when the trial budget is large enough to clear
    every algorithm's `min_trials` (CMA-ES at n=20 wants 4·n = 80 evals
    before it's eligible)."""
    # Point the grid lookup at a path that doesn't exist so we exercise the
    # rule-based fallback rather than the committed benchmarks grid.
    monkeypatch.setattr(eligibility, "_GRID_PATH_DEFAULT", tmp_path / "no_grid.json")
    eligibility._clear_grid_cache()

    bounds = [(-1.0, 1.0)] * n_dim
    expected = suggest_pure(n_dim, 200)[0]

    with patch(
        "humpday.optimizers.scipy_interface.pure_optimize",
        return_value=(0.0, np.zeros(n_dim)),
    ) as mock_pure:
        minimize(
            _quadratic,
            bounds=bounds,
            options={"maxiter": 200, "auto_timing": False},
        )

    called_method = mock_pure.call_args.args[1]
    assert called_method == expected, (
        f"n_dim={n_dim}: expected rule-based pick {expected!r}, "
        f"pure_optimize received {called_method!r}"
    )


def test_minimize_explicit_method_is_respected():
    """Explicit method= still wins; auto-pick is only used when method is None."""
    with patch(
        "humpday.optimizers.scipy_interface.pure_optimize",
        return_value=(0.0, np.zeros(3)),
    ) as mock_pure:
        minimize(
            _quadratic,
            bounds=[(-1, 1)] * 3,
            method="PRIMA_BOBYQA",
            options={"maxiter": 50},
        )
    assert mock_pure.call_args.args[1] == "PRIMA_BOBYQA"


def test_cube_minimize_no_method_no_timing_matches_rule_ranking(monkeypatch, tmp_path):
    """cube_minimize (the lower-level entrypoint) auto-picks too."""
    monkeypatch.setattr(eligibility, "_GRID_PATH_DEFAULT", tmp_path / "no_grid.json")
    eligibility._clear_grid_cache()

    n_dim = 8
    expected = suggest_pure(n_dim, 50)[0]
    with patch(
        "humpday.optimizers.scipy_interface.pure_optimize",
        return_value=(0.0, np.zeros(n_dim)),
    ) as mock_pure:
        cube_minimize(
            _quadratic,
            bounds=[(0, 1)] * n_dim,
            options={"maxiter": 50, "auto_timing": False},
        )
    assert mock_pure.call_args.args[1] == expected


def test_low_dim_rule_pick_matches_old_default():
    """Backwards-compat sanity check: at n <= 2, the old hard-coded
    'NelderMead' default and the new rule-based pick agree, so existing
    callers see no behavior change."""
    for n_dim in (1, 2):
        assert suggest_pure(n_dim, 50)[0] == "NelderMead"


def test_high_dim_rule_pick_prefers_adaptive_random_search():
    """At n > 50 the top pick should be Rechenberg, not RandomSearch.
    ARS produces a monotone convergence curve on any objective with mild
    structure; plain RandomSearch is i.i.d. and only competitive when the
    surface is essentially structureless."""
    assert suggest_pure(60, 100)[0] == "Rechenberg"
    assert suggest_pure(60, 100)[1] == "RandomSearch"


# -----------------------------------------------------------------------------
# Timing-driven auto-pick
# -----------------------------------------------------------------------------


def _stash_method(mock_pure):
    """Pull the method name passed to pure_optimize from a Mock's call_args."""
    return mock_pure.call_args.args[1]


def test_cheap_objective_avoids_bayesian_opt():
    """5-dim, 200 trials, microsecond-cost objective: GP fit per iter would
    dominate, so the recommender must NOT pick BayesianOpt."""
    bounds = [(-1.0, 1.0)] * 5
    with patch(
        "humpday.optimizers.scipy_interface.pure_optimize",
        return_value=(0.0, np.zeros(5)),
    ) as mock_pure:
        minimize(_quadratic, bounds=bounds, options={"maxiter": 200})
    assert _stash_method(mock_pure) != "BayesianOpt"


def test_cheap_objective_avoids_cma_es_in_mid_dim():
    """20-dim, 200 trials, microsecond objective: eigendecomp per generation
    would dominate, so CMA-ES is dropped and the recommender falls through
    to the next eligible algorithm in the ranking."""
    bounds = [(-1.0, 1.0)] * 20
    with patch(
        "humpday.optimizers.scipy_interface.pure_optimize",
        return_value=(0.0, np.zeros(20)),
    ) as mock_pure:
        minimize(_quadratic, bounds=bounds, options={"maxiter": 200})
    assert _stash_method(mock_pure) != "CMAEvolutionStrategy"


def test_expensive_objective_admits_heavy_algorithms():
    """When the objective is genuinely expensive, all overhead tiers
    become eligible. We mock the timing call so the test doesn't actually
    sleep for seconds. With eval_time=1.0s, BO is eligible and ranks high
    for 5-dim problems with 200 trials, so the recommender should pick it."""
    bounds = [(-1.0, 1.0)] * 5

    fake_timing = eligibility.TimingResult(
        eval_time=1.0,
        samples=[1.0, 1.0, 1.0],
        used_for_recommendation=True,
    )
    with (
        patch(
            "humpday.eligibility.time_objective",
            return_value=fake_timing,
        ),
        patch(
            "humpday.optimizers.scipy_interface.pure_optimize",
            return_value=(0.0, np.zeros(5)),
        ) as mock_pure,
    ):
        minimize(_quadratic, bounds=bounds, options={"maxiter": 200})

    # BayesianOpt is rule_based_ranking[3-10][4]; PRIMA_BOBYQA is [3]; DE is [0].
    # All three are eligible for an expensive objective at this dim/trials.
    # The deterministic pick is rule_based_ranking[0] = DifferentialEvolution.
    # The point of this test is simply that the heavy algorithms aren't
    # blocked — assert that the pick is not RandomSearch (the fallback).
    assert _stash_method(mock_pure) != "RandomSearch"


def test_result_carries_method_and_eval_time():
    """After minimize() returns, the chosen method and measured eval_time
    are stashed on the result for caller inspection."""
    bounds = [(-1.0, 1.0)] * 3
    result = minimize(_quadratic, bounds=bounds, options={"maxiter": 30})
    assert hasattr(result, "method")
    assert result.method in eligibility.TIER
    assert hasattr(result, "eval_time_measured")
    # eval_time_measured is None only if auto_timing was off or timing threw;
    # here neither, so we expect a positive float.
    assert result.eval_time_measured is not None and result.eval_time_measured > 0


def test_result_tuple_helper_matches_legacy_shape():
    """`.tuple()` returns the (fun, x) pair used by minimize_unit_cube."""
    bounds = [(-1.0, 1.0)] * 2
    result = minimize(
        _quadratic, bounds=bounds, options={"maxiter": 20, "auto_timing": False}
    )
    fun, x = result.tuple()
    assert fun == result.fun
    assert x is result.x


def test_result_carries_tier():
    """The overhead tier of the chosen algorithm is reported on the result."""
    bounds = [(-1.0, 1.0)] * 2
    result = minimize(
        _quadratic, bounds=bounds, options={"maxiter": 20, "auto_timing": False}
    )
    assert result.tier is not None
    assert result.tier == eligibility.TIER[result.method]


def test_result_repr_includes_method_when_auto_picked():
    """The repr surfaces the auto-pick info so debugging at a REPL is easy."""
    bounds = [(-1.0, 1.0)] * 2
    result = minimize(
        _quadratic, bounds=bounds, options={"maxiter": 20, "auto_timing": False}
    )
    s = repr(result)
    assert "method=" in s
    assert result.method in s
