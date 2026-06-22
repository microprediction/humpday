"""The ask/tell interface (suggest_next / receive_update) must be a behavior-
preserving view onto each optimizer's own optimize(): driving an optimizer
incrementally must reproduce optimize()'s best value and evaluation count exactly.

See notes/asktell-optimizer-interface.md.
"""

import random

try:
    import pytest
except ImportError:  # allow running the __main__ self-check without pytest
    pytest = None

from humpday.optimizers.alloptimizers import PURE_OPTIMIZERS

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None


def _seed(s):
    random.seed(s)
    if np is not None:
        np.random.seed(s)


def _sphere(x):
    return float(sum((xi - 0.3) ** 2 for xi in x))


def _run_monolithic(cls, seed, n_trials, n_dim):
    _seed(seed)
    opt = cls(_sphere, n_trials, n_dim)
    bv, _ = opt.optimize()
    return float(bv), opt.evaluations


def _run_asktell(cls, seed, n_trials, n_dim):
    _seed(seed)
    opt = cls(_sphere, n_trials, n_dim)
    while True:
        x = opt.suggest_next()
        if x is None:
            break
        opt.receive_update(_sphere(x))
    bv, _ = opt.best()
    return float(bv), opt.evaluations


def _run_asktell_batch(cls, seed, n_trials, n_dim):
    _seed(seed)
    opt = cls(_sphere, n_trials, n_dim)
    max_group = 1
    while True:
        xs = opt.suggest_batch()
        if xs is None:
            break
        max_group = max(max_group, len(xs))
        opt.tell_batch([_sphere(x) for x in xs])
    bv, _ = opt.best()
    return float(bv), opt.evaluations, max_group


if pytest is not None:

    @pytest.mark.parametrize("name", sorted(PURE_OPTIMIZERS))
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_asktell_equivalent_to_optimize(name, seed):
        cls = PURE_OPTIMIZERS[name]
        n_dim, n_trials = 5, 60
        mono_v, mono_e = _run_monolithic(cls, seed, n_trials, n_dim)
        at_v, at_e = _run_asktell(cls, seed, n_trials, n_dim)
        assert at_v == mono_v, (
            f"{name}: best value differs (mono={mono_v}, ask/tell={at_v})"
        )
        assert at_e == mono_e, (
            f"{name}: eval count differs (mono={mono_e}, ask/tell={at_e})"
        )

    @pytest.mark.parametrize("name", sorted(PURE_OPTIMIZERS))
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_batch_view_equivalent_to_optimize(name, seed):
        cls = PURE_OPTIMIZERS[name]
        n_dim, n_trials = 5, 60
        mono_v, mono_e = _run_monolithic(cls, seed, n_trials, n_dim)
        at_v, at_e, _ = _run_asktell_batch(cls, seed, n_trials, n_dim)
        assert at_v == mono_v, (
            f"{name}: batch best differs (mono={mono_v}, batch={at_v})"
        )
        assert at_e == mono_e, f"{name}: batch eval count differs ({mono_e} vs {at_e})"

    def test_cma_emits_real_batches_de_does_not():
        # CMA-ES is synchronous -> generation surfaces as a multi-point group.
        _, _, cma_group = _run_asktell_batch(
            PURE_OPTIMIZERS["CMAEvolutionStrategy"], 0, 120, 8
        )
        assert cma_group > 1, f"expected CMA to emit batches >1, got {cma_group}"
        # DE is immediate-selection -> serial, groups of 1.
        _, _, de_group = _run_asktell_batch(
            PURE_OPTIMIZERS["DifferentialEvolution"], 0, 120, 8
        )
        assert de_group == 1, f"expected DE to be serial (group=1), got {de_group}"

    def test_mixed_mode_guarded():
        cls = PURE_OPTIMIZERS["NelderMead"]
        opt = cls(_sphere, 20, 4)
        opt.optimize()
        with pytest.raises(RuntimeError):
            opt.suggest_next()  # already evaluated -> must refuse ask/tell

    def test_protocol_guards():
        cls = PURE_OPTIMIZERS["NelderMead"]
        opt = cls(_sphere, 20, 4)
        with pytest.raises(RuntimeError):
            opt.receive_update(0.0)  # no matching suggest_next
        x = opt.suggest_next()
        assert x is not None
        with pytest.raises(RuntimeError):
            opt.suggest_next()  # two suggests without a receive_update
        opt.close()
        assert opt.is_done()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    ok = True
    for nm in sorted(PURE_OPTIMIZERS):
        cls = PURE_OPTIMIZERS[nm]
        mism = []
        for s in (0, 1, 2):
            mv, me = _run_monolithic(cls, s, 60, 5)
            av, ae = _run_asktell(cls, s, 60, 5)
            if not (mv == av and me == ae):
                mism.append((s, mv, av, me, ae))
        flag = "OK " if not mism else "MISMATCH"
        ok = ok and not mism
        print(f"  [{flag}] {nm}")
        for s, mv, av, me, ae in mism:
            print(f"          seed={s} mono=({mv:.8g},{me}) asktell=({av:.8g},{ae})")
    print("\nALL EQUIVALENT" if ok else "\nMISMATCHES FOUND")
    raise SystemExit(0 if ok else 1)
