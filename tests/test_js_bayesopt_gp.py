"""The JavaScript BayesianOpt conditions the same Gaussian process as Python's (#408).

The JS port used to score candidates with a nearest-neighbour heuristic whose "uncertainty",
exp(-2 * distance to the nearest observation), is largest at the points already sampled. With
one observation at (0.5, 0.5) it preferred to resample that point. These tests fix the
observations and the query points, so there is no RNG and no optimizer outcome in the way, and
compare the surrogate itself: posterior mean, posterior standard deviation and expected
improvement, against `BayesianOpt._gp_predict` and `_expected_improvement`.

The tolerance is rounding, not behaviour. The two sides share the kernel, the Cholesky, the solve
and the normal CDF; what differs is the last bit of libm's exp and erf against JavaScript's (and,
under the numpy backend, LAPACK's factorisation in place of the pure loop). The first JavaScript
GP missed this tolerance by 3e-8 on EI, through an Abramowitz-Stegun erf.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from humpday.optimizers.evolutionary_algorithms import BayesianOpt

NODE = shutil.which("node")
RUNNER = Path(__file__).parent / "js_bayesopt_gp_runner.js"

pytestmark = pytest.mark.skipif(not NODE, reason="node not on PATH")

TOL = 1e-10


def _js(n_dim, X, y, queries):
    spec = {"n_dim": n_dim, "X": X, "y": y, "queries": queries}
    result = subprocess.run(
        [NODE, str(RUNNER)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _py(n_dim, X, y, queries):
    b = BayesianOpt(lambda x: 0.0, 50, n_dim)
    b.X_observed = [list(r) for r in X]
    b.y_observed = list(y)
    mu, sigma, ei = [], [], []
    for q in queries:
        m, s = b._gp_predict(list(q))
        mu.append(float(m))
        sigma.append(float(s))
        ei.append(float(b._expected_improvement(list(q))))
    return {"mu": mu, "sigma": sigma, "ei": ei}


def _assert_close(js, py):
    for key in ("mu", "sigma", "ei"):
        for i, (a, b) in enumerate(zip(js[key], py[key])):
            assert abs(a - b) <= TOL * max(1.0, abs(b)), (key, i, a, b)


def _halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _design(n, n_dim, offset=1):
    bases = [2, 3, 5, 7][:n_dim]
    return [[_halton(i + offset, b) for b in bases] for i in range(n)]


def test_one_observation_std_grows_away_from_it():
    """The issue's example: one observation at the centre, queries walking to a corner."""
    X, y = [[0.5, 0.5]], [0.0]
    ts = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    queries = [[0.5 - t, 0.5 - t] for t in ts] + [[1.0, 1.0]]
    js = _js(2, X, y, queries)
    py = _py(2, X, y, queries)
    _assert_close(js, py)

    sigma = js["sigma"]
    # At the observation the posterior has nothing left but the noise: sqrt(1e-6).
    assert sigma[0] == pytest.approx(1e-3, rel=1e-6)
    # Uncertainty grows with distance from the observation, to the prior's 1 at the corners.
    assert all(a < b for a, b in zip(sigma[: len(ts)], sigma[1 : len(ts)]))
    assert sigma[len(ts) - 1] == pytest.approx(1.0, abs=1e-5)
    assert sigma[-1] == sigma[len(ts) - 1]
    # And so does the acquisition: the observed point is the worst place to look next.
    ei = js["ei"]
    assert ei[0] < 1e-20
    assert all(a < b for a, b in zip(ei[: len(ts)], ei[1 : len(ts)]))
    # The figures the issue quotes for Python. Its EI was taken under the old approximate CDF;
    # with math.erf the corner scores 0.39396148403.
    assert ei[-1] == pytest.approx(0.3939614840, abs=1e-10)
    assert sigma[-1] == pytest.approx(0.9999981367, abs=1e-9)


@pytest.mark.parametrize("n_dim, n_obs", [(2, 7), (3, 12)])
def test_posterior_and_acquisition_agree_on_a_fixed_design(n_dim, n_obs):
    X = _design(n_obs, n_dim)
    y = [math.fsum((v - 0.3) ** 2 for v in x) + 0.1 * math.sin(7.0 * x[0]) for x in X]
    queries = _design(40, n_dim, offset=101) + X[:3]
    js = _js(n_dim, X, y, queries)
    py = _py(n_dim, X, y, queries)
    _assert_close(js, py)
    # Same ranking of the candidates, so both would choose the same next point.
    assert max(range(len(queries)), key=js["ei"].__getitem__) == max(
        range(len(queries)), key=py["ei"].__getitem__
    )
    order_js = sorted(range(len(queries)), key=lambda i: -js["ei"][i])
    order_py = sorted(range(len(queries)), key=lambda i: -py["ei"][i])
    assert order_js[:10] == order_py[:10]


def test_conditioning_set_is_the_same_past_the_cap():
    """Past 128 observations both sides keep the best 64 and the newest 64."""
    n_dim = 2
    X = _design(150, n_dim)
    y = [math.fsum((v - 0.6) ** 2 for v in x) for x in X]
    queries = _design(8, n_dim, offset=501)
    _assert_close(_js(n_dim, X, y, queries), _py(n_dim, X, y, queries))
