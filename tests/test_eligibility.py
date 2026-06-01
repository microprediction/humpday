"""Tests for humpday.eligibility — the dimensional + overhead-tier
recommender that powers humpday.minimize()'s auto-selection.

These tests deliberately do NOT touch any optimizer classes — the module
under test takes algorithm names as strings and returns names as strings.
"""

import time

import numpy as np
import pytest

from humpday import eligibility as E

# -----------------------------------------------------------------------------
# Catalogue invariants
# -----------------------------------------------------------------------------

def test_every_known_algorithm_has_a_tier():
    """If a new algorithm gets added to PURE_OPTIMIZERS, this test should fail
    until someone classifies its overhead tier in eligibility.TIER."""
    from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

    missing = [name for name in PURE_OPTIMIZERS if name not in E.TIER]
    assert not missing, (
        f"Algorithms missing an eligibility tier — add to humpday.eligibility.TIER: {missing}"
    )


def test_tiers_are_in_valid_range():
    for name, tier in E.TIER.items():
        assert tier in E.MIN_EVAL_TIME_FOR_TIER, (
            f"{name} has tier {tier} but no MIN_EVAL_TIME_FOR_TIER entry"
        )


def test_thresholds_are_monotone():
    """Higher tier → higher minimum eval-time required."""
    prev = -1.0
    for tier in sorted(E.MIN_EVAL_TIME_FOR_TIER):
        threshold = E.MIN_EVAL_TIME_FOR_TIER[tier]
        assert threshold >= prev, f"tier {tier} threshold {threshold} not monotone"
        prev = threshold


# -----------------------------------------------------------------------------
# Dimensional filter
# -----------------------------------------------------------------------------

def test_grid_search_blocked_in_high_dim():
    assert E.passes_dim("GridSearch", 4)
    assert not E.passes_dim("GridSearch", 5)
    assert not E.passes_dim("GridSearch", 100)


def test_bayesian_opt_blocked_in_high_dim():
    assert E.passes_dim("BayesianOpt", 10)
    assert not E.passes_dim("BayesianOpt", 11)


def test_uobyqa_blocked_above_12():
    assert E.passes_dim("PRIMA_UOBYQA", 12)
    assert not E.passes_dim("PRIMA_UOBYQA", 13)


def test_uncapped_algorithms_pass_at_any_dim():
    assert E.passes_dim("RandomSearch", 1)
    assert E.passes_dim("RandomSearch", 10_000)
    assert E.passes_dim("DifferentialEvolution", 500)


def test_filter_by_dim_strips_caps():
    names = ["GridSearch", "BayesianOpt", "RandomSearch", "DifferentialEvolution"]
    survivors = E.filter_by_dim(names, n_dim=50)
    assert "GridSearch" not in survivors
    assert "BayesianOpt" not in survivors
    assert "RandomSearch" in survivors
    assert "DifferentialEvolution" in survivors


# -----------------------------------------------------------------------------
# Trials filter
# -----------------------------------------------------------------------------

def test_uobyqa_needs_quadratic_trials():
    # n_dim=10 → (11*12)/2 + 5 = 71 (well below the n_dim=12 hard cap)
    assert E.min_trials("PRIMA_UOBYQA", 10) >= 66
    assert not E.passes_trials("PRIMA_UOBYQA", n_dim=10, n_trials=50)
    assert E.passes_trials("PRIMA_UOBYQA", n_dim=10, n_trials=200)


def test_grid_search_needs_3_to_the_n_trials():
    # n_dim=3 → 27 grid points
    assert not E.passes_trials("GridSearch", n_dim=3, n_trials=20)
    assert E.passes_trials("GridSearch", n_dim=3, n_trials=27)


def test_default_minimum_is_ten():
    assert E.min_trials("RandomSearch", 5) == 10
    assert E.passes_trials("RandomSearch", n_dim=5, n_trials=10)
    assert not E.passes_trials("RandomSearch", n_dim=5, n_trials=5)


# -----------------------------------------------------------------------------
# Eval-time filter
# -----------------------------------------------------------------------------

def test_trivial_tier_always_passes_eval_time():
    assert E.passes_eval_time("RandomSearch", 0.0)
    assert E.passes_eval_time("RandomSearch", 1e-9)


def test_bayesian_opt_blocked_for_cheap_objective():
    # 1 microsecond per eval — GP fit will dominate
    assert not E.passes_eval_time("BayesianOpt", 1e-6)
    # 1 second per eval — GP fit is negligible
    assert E.passes_eval_time("BayesianOpt", 1.0)


def test_cma_es_threshold_at_one_ms():
    assert not E.passes_eval_time("CMAEvolutionStrategy", 1e-5)
    assert E.passes_eval_time("CMAEvolutionStrategy", 1e-2)


# -----------------------------------------------------------------------------
# Combined eligibility
# -----------------------------------------------------------------------------

def test_eligible_combines_all_three_filters():
    # 50-dim, 1000 trials, 10us per eval:
    #   - dim filter blocks GridSearch, BayesianOpt, UOBYQA, NelderMead, ...
    #   - eval-time 10us blocks CMA-ES (heavy) and BayesianOpt (very heavy)
    #   - 1000 trials is enough for everything else
    survivors = E.eligible(
        E.TIER.keys(),
        n_dim=50,
        n_trials=1000,
        eval_time=1e-5,
    )
    assert "GridSearch" not in survivors
    assert "BayesianOpt" not in survivors
    assert "PRIMA_UOBYQA" not in survivors
    assert "NelderMead" not in survivors
    assert "CMAEvolutionStrategy" not in survivors
    assert "RandomSearch" in survivors
    assert "DifferentialEvolution" in survivors


def test_eligible_eval_time_none_skips_overhead_filter():
    # eval_time=None should let CMA-ES and BO through (overhead filter off)
    survivors = E.eligible(
        ["CMAEvolutionStrategy", "BayesianOpt", "RandomSearch"],
        n_dim=5,
        n_trials=200,
        eval_time=None,
    )
    assert "CMAEvolutionStrategy" in survivors
    assert "BayesianOpt" in survivors


# -----------------------------------------------------------------------------
# Recommendation
# -----------------------------------------------------------------------------

def test_recommend_low_dim_expensive_picks_quadratic_model():
    # 2D, 200 trials, 1 second per eval — UOBYQA dominates
    pick = E.recommend(n_dim=2, n_trials=200, eval_time=1.0)
    assert pick == "NelderMead" or pick.startswith("PRIMA")


def test_recommend_low_dim_cheap_avoids_bayesian():
    # 2D but each eval is 100 ns — GP overhead would dominate
    pick = E.recommend(n_dim=2, n_trials=200, eval_time=1e-7)
    assert pick != "BayesianOpt"


def test_recommend_high_dim_avoids_grid_and_bo():
    pick = E.recommend(n_dim=100, n_trials=500, eval_time=1.0)
    assert pick not in ("GridSearch", "BayesianOpt", "PRIMA_UOBYQA", "NelderMead")


def test_recommend_falls_back_to_random_search_when_nothing_passes():
    # n_trials=1 — below every algorithm's min_trials. Should still return
    # something legitimate rather than raise.
    pick = E.recommend(n_dim=10, n_trials=1, eval_time=None)
    assert pick == "RandomSearch"


def test_recommend_respects_available_whitelist():
    pick = E.recommend(
        n_dim=5,
        n_trials=200,
        eval_time=1.0,
        available=["RandomSearch", "GridSearch"],
    )
    assert pick == "RandomSearch"  # GridSearch fails dim cap, RS survives


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------

def test_time_objective_returns_positive_seconds():
    def f(x):
        return float(np.sum(x ** 2))

    result = E.time_objective(f, np.zeros(5), n_warmup=1, n_measure=3)
    assert result.eval_time > 0.0
    assert len(result.samples) == 3
    assert all(s > 0.0 for s in result.samples)


def test_time_objective_detects_slow_function():
    def f(x):
        time.sleep(0.005)  # 5 ms per call
        return float(np.sum(x ** 2))

    result = E.time_objective(f, np.zeros(3), n_warmup=1, n_measure=3)
    # Should comfortably exceed the LIGHT-tier threshold (10us) and clear
    # MEDIUM (100us); around the HEAVY threshold (1ms) is up for grabs depending
    # on machine noise, but we set 5ms specifically so MEDIUM holds robustly.
    assert result.eval_time > E.MIN_EVAL_TIME_FOR_TIER[E.TIER_MEDIUM]


def test_time_objective_uses_median():
    """The median should ignore a single slow outlier."""
    times = iter([0.0001, 0.0001, 0.5])  # one big outlier
    call_count = {"n": 0}

    def f(x):
        # First call is warmup; subsequent calls do whatever the iterator says.
        if call_count["n"] == 0:
            call_count["n"] += 1
            return 0.0
        sleep_for = next(times)
        time.sleep(sleep_for)
        return 0.0

    result = E.time_objective(f, np.zeros(2), n_warmup=1, n_measure=3)
    # median of [0.0001, 0.0001, ~0.5] is 0.0001 — well under 0.1s
    assert result.eval_time < 0.1


# -----------------------------------------------------------------------------
# Grid-driven recommendation
# -----------------------------------------------------------------------------

def _write_grid(tmp_path, cells: dict) -> "Path":
    import json
    p = tmp_path / "grid.json"
    p.write_text(json.dumps({"meta": {}, "cells": cells}))
    E._clear_grid_cache()
    return p


def test_recommend_uses_grid_when_present(tmp_path):
    """When a grid is on disk, the recommender picks the eligible algorithm
    with the smallest median_best on the matching cell — not the rule top."""
    # Rule top for n_dim=5 is DifferentialEvolution. Make CMAEvolutionStrategy
    # the grid winner and confirm the grid overrides the rule.
    grid_path = _write_grid(
        tmp_path,
        {
            "5/200": {
                "DifferentialEvolution": {"median_best": 1e-3},
                "CMAEvolutionStrategy": {"median_best": 1e-9},  # much better
                "ParticleSwarm": {"median_best": 1e-2},
            }
        },
    )
    pick = E.recommend(
        n_dim=5,
        n_trials=200,
        eval_time=1.0,  # expensive enough to allow CMA-ES (tier 3)
        grid_path=grid_path,
    )
    assert pick == "CMAEvolutionStrategy"


def test_recommend_falls_back_to_rule_when_grid_missing(tmp_path):
    """A grid path that doesn't exist should not blow up — fall back to rule."""
    pick = E.recommend(
        n_dim=2,
        n_trials=200,
        eval_time=None,
        grid_path=tmp_path / "no_such_file.json",
    )
    assert pick == "NelderMead"  # rule_based_ranking[n<=2][0]


def test_recommend_grid_skips_blocked_algorithms(tmp_path):
    """Grid-listed algorithms that fail the dim/trials/eval-time filters
    should be ignored even if they'd score best."""
    # n_dim=11 makes BayesianOpt fail the dim cap (≤ 10). Grid still says BO
    # is best — recommender must skip it.
    grid_path = _write_grid(
        tmp_path,
        {
            "11/200": {
                "BayesianOpt": {"median_best": 1e-15},
                "DifferentialEvolution": {"median_best": 1e-5},
            }
        },
    )
    pick = E.recommend(
        n_dim=11,
        n_trials=200,
        eval_time=1.0,
        grid_path=grid_path,
    )
    assert pick != "BayesianOpt"
    assert pick == "DifferentialEvolution"


def test_recommend_grid_snaps_to_nearest_lower_cell(tmp_path):
    """When the caller's exact (n_dim, n_trials) isn't in the grid, we snap
    to the nearest cell with smaller-or-equal coords."""
    grid_path = _write_grid(
        tmp_path,
        {
            "5/200": {
                "CMAEvolutionStrategy": {"median_best": 1e-9},
                "DifferentialEvolution": {"median_best": 1e-3},
            }
        },
    )
    # Caller asks for n_dim=8 / n_trials=500 — should snap onto 5/200.
    pick = E.recommend(
        n_dim=8, n_trials=500, eval_time=1.0, grid_path=grid_path
    )
    assert pick == "CMAEvolutionStrategy"


def test_recommend_reads_aggregated_median_best_only(tmp_path):
    """The recommender's read path only consults `median_best` — it should
    not need the new raw `runs` dict to function. This guarantees that
    grids written by the incremental script and grids written by hand both
    work."""
    grid_path = _write_grid(
        tmp_path,
        {
            "5/200": {
                # No `runs` key, just the aggregate the recommender needs.
                "DifferentialEvolution": {"median_best": 1e-3},
                "CMAEvolutionStrategy": {"median_best": 1e-9},
            }
        },
    )
    pick = E.recommend(
        n_dim=5, n_trials=200, eval_time=1.0, grid_path=grid_path
    )
    assert pick == "CMAEvolutionStrategy"


def test_recommend_skips_inf_median_best(tmp_path):
    """A grid entry whose median_best is +inf (every run failed or
    skipped_too_slow) must not be picked."""
    grid_path = _write_grid(
        tmp_path,
        {
            "5/200": {
                "DifferentialEvolution": {"median_best": 1e-3},
                "CMAEvolutionStrategy": {
                    "median_best": float("inf"),
                    "skipped_too_slow": True,
                },
            }
        },
    )
    pick = E.recommend(
        n_dim=5, n_trials=200, eval_time=1.0, grid_path=grid_path
    )
    assert pick == "DifferentialEvolution"


def test_recommend_grid_with_no_eligible_entries_falls_back_to_rule(tmp_path):
    """If the grid cell exists but lists only ineligible algorithms, fall back
    to the rule-based ranking instead of returning RandomSearch."""
    grid_path = _write_grid(
        tmp_path,
        {
            "2/100": {
                "BayesianOpt": {"median_best": 1e-15},  # cheap eval_time blocks BO
            }
        },
    )
    pick = E.recommend(
        n_dim=2, n_trials=100, eval_time=1e-9, grid_path=grid_path
    )
    # Eval_time 1ns blocks every non-trivial tier — only tier-0 algorithms
    # remain eligible. HillClimbing is the first tier-0 entry in the
    # n_dim<=2 rule-based ranking, so it wins the fallback.
    assert pick == "HillClimbing"
