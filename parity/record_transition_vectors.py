"""Record golden transition vectors for every optimizer.

For each (optimizer, seed, dim, budget, objective) case, drive the
optimizer via threadless ask/tell under the portable PRNG and record the
exact sequence of (point out, value in) transitions, doubles encoded as
big-endian IEEE-754 hex. These vectors are the cross-language contract:
tests/test_transition_vectors.py replays them in Python, and each port
(JS, Rust, Julia, R) must replay them bit-for-bit.

Objectives are restricted to polynomial arithmetic with a fixed
evaluation order — no libm — so the objective itself cannot introduce
platform variation.

The vectors are recorded and replayed on the PURE backend only: the
numpy backend routes dot/eigh/cholesky through BLAS/LAPACK, whose FMA
and reduction order vary by platform (the CI probe showed PRIMA, CMA-ES
and LBFGSB diverging between macOS and Linux under numpy, while all 23
matched under the pure backend's fixed-order arithmetic). Ports
implement the pure-backend semantics.

KNOWN CAVEAT (to be lifted by the summation-portability pass): CPython
3.12+ computes builtin sum() on floats with Neumaier compensation, so
these vectors — recorded on 3.13 — replay exactly only on 3.12+ until
every trajectory-relevant float sum() becomes an explicit left fold
(which is also what the ports will implement). CI pins the replay job
to 3.13 meanwhile.

Regenerate with:  HUMPDAY_FORCE_PURE_ARRAY=1 python parity/record_transition_vectors.py
"""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from humpday import _array as _A  # noqa: E402
from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS  # noqa: E402

if _A.BACKEND != "pure":
    sys.exit(
        "The portable contract is defined on the pure backend (fixed-order "
        "arithmetic, no BLAS/LAPACK). Rerun with HUMPDAY_FORCE_PURE_ARRAY=1."
    )

OUT = Path(__file__).parent / "transition_vectors.json"

SEEDS = [(11, 3), (42, 54)]
CASES = [(2, 40), (3, 30)]  # (n_dim, n_trials)


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


def record_case(cls, seed, seq, n_dim, n_trials, obj_name):
    objective = OBJECTIVES[obj_name]
    _A.use_portable_rng(seed, seq)
    opt = cls(objective, n_trials, n_dim)
    xs, fs = [], []
    while True:
        x = opt.suggest_next()
        if x is None:
            break
        v = objective(x)
        xs.append([_bits(c) for c in x])
        fs.append(_bits(v))
        opt.receive_update(v)
    _A.use_legacy_rng()
    return {
        "optimizer": cls.__name__,
        "seed": seed,
        "seq": seq,
        "n_dim": n_dim,
        "n_trials": n_trials,
        "objective": obj_name,
        "x": xs,
        "f": fs,
    }


def main():
    cases = []
    for name in sorted(PURE_OPTIMIZERS):
        cls = PURE_OPTIMIZERS[name]
        for (seed, seq), (n_dim, n_trials), obj_name in zip(
            SEEDS * 2, CASES * 2, ["sphere03", "sphere03", "rosen01", "rosen01"]
        ):
            cases.append(record_case(cls, seed, seq, n_dim, n_trials, obj_name))
    OUT.write_text(
        json.dumps(
            {
                "spec": "humpday transition vectors v1: portable PCG32 "
                "(humpday/_prng.py), doubles as big-endian IEEE-754 hex",
                "cases": cases,
            },
            indent=None,
            separators=(",", ":"),
        )
    )
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
