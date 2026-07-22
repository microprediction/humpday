"""Replay the golden transition vectors against the live roster.

Bit-exact: every point the optimizer asks for and every value it is told
must match the recorded IEEE-754 bits. A failure means either optimizer
behavior drifted, or (when it fails only on one platform) a libm
transcendental is steering a trajectory branch and needs replacing with
a portable version. Ports replay this same file in their own language.
"""

import json
import struct
from pathlib import Path

import pytest

from humpday import _array as _A
from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

VECTORS = Path(__file__).parent.parent / "parity" / "transition_vectors.json"
_DATA = json.loads(VECTORS.read_text())


def sphere03(x):
    s = 0.0
    for v in x:
        d = float(v) - 0.3
        s = s + d * d
    return s


def rosen01(x):
    s = 0.0
    i = 0
    n = len(x)
    while i < n - 1:
        a = float(x[i + 1]) - float(x[i]) * float(x[i])
        b = 1.0 - float(x[i])
        s = s + 100.0 * (a * a) + b * b
        i += 1
    return s


OBJECTIVES = {"sphere03": sphere03, "rosen01": rosen01}


def _bits(v):
    return struct.pack(">d", float(v)).hex()


@pytest.fixture(autouse=True)
def _restore_legacy():
    yield
    _A.use_legacy_rng()


pytestmark = pytest.mark.skipif(
    _A.BACKEND != "pure",
    reason="portable contract is defined on the pure backend (no BLAS/LAPACK)",
)


@pytest.mark.parametrize(
    "case",
    _DATA["cases"],
    ids=lambda c: f"{c['optimizer']}-{c['objective']}-d{c['n_dim']}",
)
def test_replay_transition_vector(case):
    cls = PURE_OPTIMIZERS[case["optimizer"]]
    objective = OBJECTIVES[case["objective"]]
    _A.use_portable_rng(case["seed"], case["seq"])
    opt = cls(objective, case["n_trials"], case["n_dim"])
    i = 0
    while True:
        x = opt.suggest_next()
        if x is None:
            break
        assert i < len(case["x"]), f"extra transition {i}"
        got = [_bits(c) for c in x]
        assert got == case["x"][i], f"point {i} diverged"
        v = objective(x)
        assert _bits(v) == case["f"][i], f"value {i} diverged"
        opt.receive_update(v)
        i += 1
    assert i == len(case["x"]), f"run ended early: {i} < {len(case['x'])}"
