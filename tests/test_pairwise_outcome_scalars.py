"""Finite NumPy scalars score the same as Python floats in the tournament (#392)"""

import math

import pytest

from humpday.optimizers.adaptive_optimizer import (
    pairwise_outcome,
    run_algorithm_tournament,
)

np = pytest.importorskip("numpy")

SCALAR_TYPES = [float, int, np.float32, np.float64, np.int64, np.int32, np.array]


@pytest.mark.parametrize("to_a", SCALAR_TYPES)
@pytest.mark.parametrize("to_b", SCALAR_TYPES)
def test_finite_scalars_compare_by_value(to_a, to_b):
    assert pairwise_outcome(to_a(1), to_b(2)) == 1.0
    assert pairwise_outcome(to_a(2), to_b(1)) == 0.0
    assert pairwise_outcome(to_a(2), to_b(2)) == 0.5


@pytest.mark.parametrize(
    "bad", [math.nan, math.inf, -math.inf, np.float32("nan"), np.float32("inf")]
)
def test_nonfinite_still_loses_and_two_failures_are_not_compared(bad):
    assert pairwise_outcome(np.float32(1), bad) == 1.0
    assert pairwise_outcome(bad, np.int64(1)) == 0.0
    assert pairwise_outcome(bad, math.inf) is None


@pytest.mark.parametrize("not_a_score", ["1", [1.0], None, 1 + 0j])
def test_nonscalars_are_not_scores(not_a_score):
    assert pairwise_outcome(not_a_score, 2.0) == 0.0
    assert pairwise_outcome(not_a_score, not_a_score) is None


@pytest.mark.parametrize("cast", [float, np.float32, np.float64])
def test_tournament_records_matches_whatever_the_scalar_type(cast):
    elo = run_algorithm_tournament(
        iter([lambda x: cast(sum(v * v for v in x))]),
        trials_per_problem=20,
        n_problems=1,
        n_dim=2,
        algorithms_to_test=["RandomSearch", "NelderMead"],
    )
    assert len(elo.match_history) == 1
