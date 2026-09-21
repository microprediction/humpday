"""CMA-ES's path gate, step-size update and covariance update follow the standard
recurrence (#348), checked against the equations rather than against the
JavaScript twin, which could share the same mistake.

The port compared ||p_sigma|| against 1.4 + 2/(n+1) with no chi_n = E||N(0,I_n)||
factor, divided by sqrt(n) instead of chi_n in the step-size update, and omitted
the c1 (1 - hsig) cc (2 - cc) C compensation when the path update was gated off.
The first two grow like sqrt(n), so in moderate dimension every normal path was
classified as excessive and the rank-one covariance update was suppressed.
"""

import math

import pytest

from humpday import _array as _A
from humpday.optimizers.evolutionary_algorithms import CMAEvolutionStrategy


def _chi_n(n):
    return math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))


def _sphere(x):
    return sum((float(v) - 0.4) ** 2 for v in x)


def _generations(n, budget, seed=42):
    """Drive one generation at a time and yield the per-generation trace."""
    _A.seed(seed)
    opt = CMAEvolutionStrategy(_sphere, budget, n)
    while True:
        xs = opt.suggest_batch()
        if xs is None:
            return
        opt.tell_batch([_sphere(x) for x in xs])
        trace = getattr(opt, "_cma_trace", None)
        if trace is not None:
            yield trace


@pytest.mark.parametrize("n", [2, 10, 20, 50])
def test_every_generation_follows_the_standard_equations(n):
    traces = list(_generations(n, budget=40 * n))
    assert len(traces) >= 3
    seen_generations = set()
    for t in traces:
        if t["generation"] in seen_generations:
            continue
        seen_generations.add(t["generation"])
        assert t["chi_n"] == pytest.approx(_chi_n(n), rel=1e-15)

        # hsig gate with the chi_n factor.
        cs, g = t["cs"], t["generation"]
        normalised = t["ps_norm"] / math.sqrt(1 - (1 - cs) ** (2 * g))
        expected_hsig = 1 if normalised < (1.4 + 2 / (n + 1)) * t["chi_n"] else 0
        assert t["hsig"] == expected_hsig

        # Step size: sigma <- sigma exp(cs/damps (||ps||/chi_n - 1)).
        expected_sigma = t["sigma_before"] * math.exp(
            (t["cs"] / t["damps"]) * (t["ps_norm"] / t["chi_n"] - 1)
        )
        assert t["sigma_after"] == pytest.approx(expected_sigma, rel=1e-12)

        # Covariance: (1 - c1 - cmu + c1 (1 - hsig) cc (2 - cc)) C + ...
        if t["cov_base"] is not None:
            cc, c1, cmu = t["cc"], t["c1"], t["cmu"]
            expected_base = 1 - c1 - cmu + c1 * (1 - t["hsig"]) * cc * (2 - cc)
            assert t["cov_base"] == pytest.approx(expected_base, rel=1e-15)


@pytest.mark.parametrize("n", [10, 20, 50])
def test_a_normal_path_is_not_gated_off_in_moderate_dimension(n):
    # A fresh evolution path has E||p_sigma|| ~ chi_n; the gate sits at
    # roughly 1.4 chi_n. The old gate compared against ~1.5 with no chi_n, so at
    # n = 20 (chi_n ~ 4.4) it fired every generation.
    traces = list(_generations(n, budget=40 * n))
    first = [t for t in traces if t["generation"] <= 5]
    assert first and all(t["hsig"] == 1 for t in first), [t["hsig"] for t in first]


def test_the_audit_case_n20_generation_one():
    # #348: at n=20, generation 1, normalised path norm ~3.12; the old
    # threshold was ~1.495 (hsig=0), the standard one ~6.60 (hsig=1).
    t = next(iter(_generations(20, budget=500)))
    assert t["generation"] == 1
    threshold = (1.4 + 2 / 21) * _chi_n(20)
    assert threshold == pytest.approx(6.60, abs=0.02)
    assert t["hsig"] == 1
