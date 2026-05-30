"""
Tests for the smart-default behavior of `minimize` / `cube_minimize`.

When `method` is None, the algorithm should be picked by `suggest_pure`
using the parsed problem dimension, instead of falling back to the old
hard-coded "NelderMead" default.
"""

from unittest.mock import patch

import numpy as np
import pytest

from humpday import cube_minimize, minimize
from humpday.optimizers.alloptimizers import suggest_pure


def _quadratic(x):
    return float(np.sum((np.asarray(x) - 0.3) ** 2))


@pytest.mark.parametrize("n_dim", [1, 2, 5, 20, 75])
def test_minimize_no_method_picks_suggest_pure_top(n_dim):
    """Omitting `method` must dispatch to suggest_pure's #1 pick for that n_dim."""
    bounds = [(-1.0, 1.0)] * n_dim
    expected = suggest_pure(n_dim, 50)[0]

    with patch(
        "humpday.optimizers.scipy_interface.pure_optimize",
        return_value=(0.0, np.zeros(n_dim)),
    ) as mock_pure:
        minimize(_quadratic, bounds=bounds, options={"maxiter": 50})

    called_method = mock_pure.call_args.args[1]
    assert called_method == expected, (
        f"n_dim={n_dim}: expected auto-pick {expected!r}, "
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


def test_cube_minimize_no_method_picks_auto():
    """cube_minimize (the lower-level entrypoint) auto-picks too."""
    n_dim = 8
    expected = suggest_pure(n_dim, 50)[0]
    with patch(
        "humpday.optimizers.scipy_interface.pure_optimize",
        return_value=(0.0, np.zeros(n_dim)),
    ) as mock_pure:
        cube_minimize(_quadratic, bounds=[(0, 1)] * n_dim, options={"maxiter": 50})
    assert mock_pure.call_args.args[1] == expected


def test_low_dim_auto_pick_matches_old_default():
    """Backwards-compat sanity check: at n <= 2, the old hard-coded
    'NelderMead' default and the new auto-pick agree, so existing callers
    see no behavior change."""
    for n_dim in (1, 2):
        assert suggest_pure(n_dim, 50)[0] == "NelderMead"


def test_high_dim_auto_pick_prefers_adaptive_random_search():
    """At n > 50 the top pick should be Rechenberg, not RandomSearch.
    ARS produces a monotone convergence curve on any objective with mild
    structure; plain RandomSearch is i.i.d. and only competitive when the
    surface is essentially structureless."""
    assert suggest_pure(60, 100)[0] == "Rechenberg"
    assert suggest_pure(60, 100)[1] == "RandomSearch"
