"""Bit-exact parity between humpday/_prng.py and docs/js/modules/prng.js.

The Python module is the reference; the JS twin must reproduce every
stream exactly. Doubles are compared as IEEE-754 bit patterns, so a
single-ulp libm difference would fail loudly (that's the point — the
generator avoids libm transcendentals entirely).

Also pins Python-side golden values so the reference itself can't drift:
these constants ARE the cross-language spec, and every future port
(Rust, Julia, R) must reproduce them.
"""

import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from humpday._prng import PCG32, portable_exp, portable_log

NODE = shutil.which("node")
RUNNER = Path(__file__).parent / "js_prng_runner.js"

SEEDS = [(42, 54), (0, 0), (123456789, 987654321), (2**63, 2**31)]
N = 500


def _bits(x: float) -> str:
    return struct.pack(">d", x).hex()


def _python_streams(seed, seq, n):
    out = {}
    g = PCG32(seed, seq)
    out["u32"] = [g.next_u32() for _ in range(n)]
    g = PCG32(seed, seq)
    out["random"] = [_bits(g.random()) for _ in range(n)]
    g = PCG32(seed, seq)
    out["gauss"] = [_bits(g.gauss()) for _ in range(n)]
    g = PCG32(seed, seq)
    out["uniform"] = [_bits(g.uniform(-3.5, 11.25)) for _ in range(n)]
    g = PCG32(seed, seq)
    mods = [1, 2, 3, 7, 10, 100, 1000, 4294967295]
    out["randbelow"] = [g.randbelow(mods[i % 8]) for i in range(n)]
    g = PCG32(seed, seq)
    arr = list(range(30))
    g.shuffle(arr)
    out["shuffle"] = arr
    g = PCG32(seed, seq)
    out["sample"] = g.sample(list(range(30)), 12)
    return out


# ---- Python-side golden values (the spec) --------------------------------


def test_pcg32_reference_vector():
    # O'Neill's own demo seeding (42, 54): first outputs of the official
    # pcg32 C implementation.
    g = PCG32(42, 54)
    first = [g.next_u32() for _ in range(6)]
    assert first == [
        0xA15C02B7,
        0x7B47F409,
        0xBA1D3330,
        0x83D2F293,
        0xBFA4784B,
        0xCBED606E,
    ]


def test_double_and_gauss_golden():
    # Golden values pinned at spec time; any change means the portable
    # algorithms changed and every port must be re-verified.
    g = PCG32(42, 54)
    assert _bits(g.random()) == "3fe42b8055ed1fd0"
    g = PCG32(42, 54)
    assert _bits(g.gauss()) == "3fe9a1fbd078a681"


def test_portable_exp_close_to_libm_and_golden():
    for x in [-740.0, -10.0, -0.5, 0.0, 1e-9, 0.5, 1.0, 10.0, 700.0, 709.0]:
        assert math.isclose(portable_exp(x), math.exp(x), rel_tol=5e-15)
    # Golden bits — part of the cross-language spec.
    assert _bits(portable_exp(1.0)) == "4005bf0a8b145768"
    assert _bits(portable_exp(-0.5)) == "3fe368b2fc6f960c"
    assert portable_exp(-800.0) == 0.0
    assert portable_exp(800.0) == float("inf")


def test_portable_log_close_to_libm():
    for x in [1e-300, 1e-9, 0.1, 0.5, 1.0, 1.5, 2.0, math.pi, 1e9, 1e300]:
        assert math.isclose(portable_log(x), math.log(x), rel_tol=1e-14, abs_tol=1e-14)
    with pytest.raises(ValueError):
        portable_log(0.0)


def test_distribution_sanity():
    g = PCG32(7, 1)
    xs = [g.gauss() for _ in range(20000)]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    assert abs(mean) < 0.03
    assert abs(var - 1.0) < 0.05
    g = PCG32(7, 1)
    us = [g.random() for _ in range(20000)]
    assert 0.49 < sum(us) / len(us) < 0.51
    assert all(0.0 <= u < 1.0 for u in us)


def test_randbelow_unbiased_range():
    g = PCG32(11, 3)
    draws = [g.randbelow(7) for _ in range(7000)]
    assert set(draws) == set(range(7))
    counts = [draws.count(i) for i in range(7)]
    assert max(counts) - min(counts) < 300


def test_sample_and_shuffle_are_permutations():
    g = PCG32(5, 9)
    arr = list(range(50))
    g.shuffle(arr)
    assert sorted(arr) == list(range(50))
    s = g.sample(list(range(50)), 20)
    assert len(set(s)) == 20 and set(s) <= set(range(50))


# ---- JS bit-exactness -----------------------------------------------------


@pytest.mark.skipif(not NODE, reason="node not on PATH")
@pytest.mark.parametrize("seed,seq", SEEDS)
def test_js_streams_bit_identical(seed, seq):
    result = subprocess.run(
        [NODE, str(RUNNER), str(seed), str(seq), str(N)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    js = json.loads(result.stdout)
    py = _python_streams(seed, seq, N)
    for key in ["u32", "random", "gauss", "uniform", "randbelow", "shuffle", "sample"]:
        assert js[key] == py[key], f"stream {key!r} diverged for seed={seed}"
