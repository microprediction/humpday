"""De-risking prototype for the ask/tell optimizer interface (see
notes/asktell-optimizer-interface.md). Lives OUTSIDE the humpday package and does
NOT modify it: it wraps the real optimizer classes and inverts control at the
single objective chokepoint, then proves the inverted run reproduces optimize()
exactly.

Inversion trick (approach C, thread-based, stdlib only): run the optimizer's own
optimize() in a worker thread, and swap `instance.objective` for a handshake that
hands each clipped point to the driver and blocks for its value. Because
BaseOptimizer.evaluate() is otherwise untouched, the algorithm's RNG draws and
control flow are byte-identical to a monolithic run (lock-step handshake => no
concurrency in the algorithm's logic). suggest_next/receive_update therefore
*cannot* change the trajectory — which the equivalence gate below verifies.
"""
from __future__ import annotations
import queue
import random
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))

from humpday.optimizers.scipy_algorithms import NelderMead
from humpday.optimizers.evolutionary_algorithms import (
    DifferentialEvolution,
    CMAEvolutionStrategy,
)

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None


def _seed_all(s):
    random.seed(s)
    if np is not None:
        np.random.seed(s)


class AskTell:
    """Wrap a constructed optimizer instance with a suggest_next/receive_update
    interface, without modifying the optimizer or the package."""

    _DONE = object()

    def __init__(self, instance):
        self.inst = instance
        self.real_objective = instance.objective  # the actual objective
        self._req = queue.Queue(maxsize=1)
        self._resp = queue.Queue(maxsize=1)
        self._thread = None
        self._done = False
        self._error = None
        self._result = None

        def handshake(x_clipped):
            # evaluate() already clipped x; hand it to the driver, block for value
            self._req.put(("X", x_clipped))
            return self._resp.get()

        self.inst.objective = handshake

    def _run(self):
        try:
            self._result = self.inst.optimize()
        except BaseException as e:  # noqa: BLE001 — propagate to driver
            self._error = e
        finally:
            self._req.put(("DONE", None))

    def suggest_next(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        tag, x = self._req.get()
        if tag == "DONE":
            self._done = True
            self._thread.join(timeout=5)
            if self._error is not None:
                raise self._error
            return None
        return x

    def receive_update(self, y):
        self._resp.put(y)

    def is_done(self):
        return self._done

    def best(self):
        return self.inst.best_value, self.inst.best_x


def drive(instance):
    """Run an optimizer purely through the ask/tell interface."""
    at = AskTell(instance)
    n = 0
    while True:
        x = at.suggest_next()
        if x is None:
            break
        y = at.real_objective(x)
        at.receive_update(y)
        n += 1
    bv, bx = at.best()
    return bv, bx, instance.evaluations, n


def sphere(x):
    return float(sum((xi - 0.3) ** 2 for xi in x))


def run_monolithic(Cls, seed, n_trials, n_dim):
    _seed_all(seed)
    inst = Cls(sphere, n_trials, n_dim)
    bv, bx = inst.optimize()
    return float(bv), inst.evaluations


def run_asktell(Cls, seed, n_trials, n_dim):
    _seed_all(seed)
    inst = Cls(sphere, n_trials, n_dim)
    bv, bx, evals, n_steps = drive(inst)
    return float(bv), evals, n_steps


def main():
    n_dim, n_trials = 5, 60
    cases = [("NelderMead", NelderMead),
             ("DifferentialEvolution", DifferentialEvolution),
             ("CMAEvolutionStrategy", CMAEvolutionStrategy)]
    print(f"equivalence gate: ask/tell vs monolithic  (n_dim={n_dim}, n_trials={n_trials})\n")
    all_ok = True
    for name, Cls in cases:
        for seed in (0, 1, 2):
            mv, mevals = run_monolithic(Cls, seed, n_trials, n_dim)
            av, aevals, asteps = run_asktell(Cls, seed, n_trials, n_dim)
            ok = (mv == av) and (mevals == aevals)
            all_ok = all_ok and ok
            flag = "OK " if ok else "MISMATCH"
            print(f"  [{flag}] {name:24s} seed={seed}  "
                  f"best: mono={mv:.10g} asktell={av:.10g}  "
                  f"evals: mono={mevals} asktell={aevals} (steps={asteps})")
    print("\nEQUIVALENCE GATE:", "PASS — inversion is behavior-preserving" if all_ok
          else "FAIL — trajectories differ; do NOT fold into base.py")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
