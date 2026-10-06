"""
Base class for all pure optimization algorithms.

Provides common functionality for objective evaluation, best value tracking,
and path visualization support.

Numpy-optional
--------------
This module reads array primitives from `humpday._array` rather than from
numpy directly. The shim transparently selects a numpy backend when numpy
is installed (no performance penalty — the operations re-export numpy
directly) and a pure-Python backend otherwise.

External API is unchanged: callers still get `.best_x`, `.best_value`,
`.evaluations`, `.path`, and `.track_path` with the same semantics.
The runtime type of `.best_x` is `numpy.ndarray` under the numpy backend
and `humpday._array_pure._Vec` (a `list` subclass) under the pure
backend; both are iterable, indexable, and support arithmetic.
"""

import math
import queue
import threading
from typing import Callable

from humpday import _array as _A

# --- ask/tell support (additive; see notes/asktell-optimizer-interface.md) ----
# The old API (optimize() owning the loop) is unchanged. The ask/tell interface
# below lets a CALLER own the loop by driving this optimizer's own optimize() in a
# worker thread, swapping self.objective for a lock-step queue handshake. Because
# evaluate() is otherwise untouched, the driven trajectory is identical to a
# monolithic run (proven in tests/test_asktell.py).
_DONE = object()
_ABORT = object()


class _AbortAskTell(BaseException):
    """Raised inside the worker to unwind a partially-driven optimize() on close()."""


class Batch(list):
    """Marker for a generator batch yield. A _run() generator may
    `values = yield Batch([x1, ..., xm])` to surface a whole synchronous
    generation (CMA-ES pattern); the driver evaluates the points in order —
    each with evaluate()'s exact increment/clip/bookkeep sequence — and sends
    back the list of values. Under ask/tell the group is served intact by
    suggest_batch() (for parallel evaluation) or point-by-point by
    suggest_next(). A plain list would be ambiguous (the pure backend's _Vec
    is a list subclass), hence the explicit marker type."""


class _AskTellState:
    """Lock-step handshake for driving optimize() incrementally. The worker emits a
    GROUP of points — size 1 from evaluate(), size m from evaluate_batch() — and the
    driver returns a list of that many values."""

    def __init__(self):
        self.req = queue.Queue()  # worker -> driver: ("X", [points]) or (_DONE, None)
        self.resp = queue.Queue()  # driver -> worker: [values], or _ABORT
        self.thread = None
        self.done = False
        self.error = None
        self.result = None
        self.mode = None  # 'scalar' | 'batch' (no mixing on one instance)
        self.cur_points = None  # current group being served
        self.cur_idx = 0  # scalar view: index within the group
        self.cur_vals = None  # scalar view: values collected so far
        self.awaiting = False  # scalar view: a point is out, value pending
        self.awaiting_batch = False

    def single_handshake(self, x_clipped):
        # Installed as self.objective inside the worker (used by evaluate()): hand a
        # one-point group to the driver and block for its value.
        self.req.put(("X", [x_clipped]))
        vals = self.resp.get()
        if vals is _ABORT:
            raise _AbortAskTell()
        return vals[0]

    def batch_handshake(self, points):
        # Used by evaluate_batch(): hand the whole group to the driver at once.
        self.req.put(("X", list(points)))
        vals = self.resp.get()
        if vals is _ABORT:
            raise _AbortAskTell()
        return list(vals)


class BaseOptimizer:
    """Base class for all pure optimization algorithms."""

    def __init__(self, objective: Callable, n_trials: int, n_dim: int):
        self.objective = objective
        if n_trials != int(n_trials) or int(n_trials) < 0:
            raise ValueError(
                f"n_trials must be a non-negative integer, got {n_trials!r}"
            )
        self.n_trials = int(n_trials)
        self.n_dim = n_dim
        self.evaluations = 0
        self.best_value = float("inf")
        self.best_x = _A.random_uniform(n_dim)
        self.track_path = False
        self.path = []
        self._at = None  # ask/tell driver state (lazy; see suggest_next)

    def _bookkeep(self, x_clipped, value):
        """Shared post-evaluation bookkeeping (path sampling, best tracking).
        The counter increment happens BEFORE the objective call — see
        evaluate() and the generator drivers, which replicate its order."""

        # Track path for visualization. Sample at ~20 evenly-spaced points
        # across the run by default; always record the first evaluation.
        if self.track_path and (
            self.evaluations % max(1, self.n_trials // 20) == 0 or self.evaluations == 1
        ):
            # `.copy()` exists on both numpy.ndarray and `_Vec` (which is a
            # list subclass). Either yields an independent snapshot.
            self.path.append(x_clipped.copy())

        if value < self.best_value:
            self.best_value = value
            self.best_x = x_clipped.copy()

        return value

    def evaluate(self, x) -> float:
        """Evaluate objective with tracking. `x` may be a numpy array, a
        `_Vec`, or any sequence of floats — `_A.clip` normalises it."""
        self.evaluations += 1
        x_clipped = _A.clip(x, 0, 1)
        value = self.objective(x_clipped)
        return self._bookkeep(x_clipped, value)

    # ------------------------------------------------------------------ #
    # Online (generator) protocol. A converted optimizer defines          #
    #   def _run(self):  ...  value = yield point  ...                    #
    # yielding candidate points and receiving their objective values.     #
    # The driver owns clipping, counting, path and best tracking, in      #
    # exactly evaluate()'s order, so trajectories match the loop-owning   #
    # form statement for statement. optimize() below drives _run with     #
    # self.objective; the ask/tell methods drive it with caller-supplied  #
    # values and no worker thread.                                        #
    # ------------------------------------------------------------------ #
    def optimize(self):
        run = getattr(type(self), "_run", None)
        if run is None:
            raise NotImplementedError(
                f"{type(self).__name__} defines neither optimize() nor _run()"
            )
        # n_trials is a hard cap on objective calls, enforced here rather than
        # trusted to every algorithm: a population method that wants more
        # points than the budget allows gets the budget, and the best of the
        # points it did evaluate is already banked by _bookkeep.
        if self.n_trials <= 0:
            return self.best_value, self.best_x
        gen = self._run()
        try:
            yielded = next(gen)
            while True:
                if isinstance(yielded, Batch):
                    # Synchronous generation: evaluate each point with the
                    # same increment/clip/bookkeep order as evaluate(), so
                    # the trajectory matches the evaluate_batch() form.
                    vals = []
                    for p in yielded:
                        if self.evaluations >= self.n_trials:
                            break
                        self.evaluations += 1
                        p_clipped = _A.clip(p, 0, 1)
                        v = self.objective(p_clipped)
                        self._bookkeep(p_clipped, v)
                        vals.append(v)
                    if len(vals) < len(yielded):
                        break  # the budget ran out inside a generation
                    yielded = gen.send(vals)
                else:
                    if self.evaluations >= self.n_trials:
                        break
                    self.evaluations += 1
                    x_clipped = _A.clip(yielded, 0, 1)
                    value = self.objective(x_clipped)
                    self._bookkeep(x_clipped, value)
                    yielded = gen.send(value)
        except StopIteration:
            pass
        finally:
            gen.close()
        return self.best_value, self.best_x

    def evaluate_batch(self, points):
        """Evaluate several points as one group. In a direct optimize() run this is
        exactly `[self.evaluate(p) for p in points]`. Under ask/tell it surfaces the
        whole group through suggest_batch() so the caller can evaluate it in parallel.

        Use this only in *synchronous* population methods — where the generation is
        built from a fixed distribution/state and no point depends on another point's
        value within the generation (e.g. CMA-ES). Do NOT use it in
        immediate-selection methods (e.g. DE), which are serial by construction."""
        if self._at is None:
            return [self.evaluate(p) for p in points]
        pts = [_A.clip(p, 0, 1) for p in points]
        values = [float(v) for v in self._at.batch_handshake(pts)]
        for p, v in zip(pts, values):
            self.evaluations += 1
            if v < self.best_value:
                self.best_value = v
                self.best_x = p.copy()
        return values

    # ------------------------------------------------------------------ #
    # Ask/tell interface (additive). A CALLER owns the loop. Two views:  #
    #   scalar:  while (x := opt.suggest_next()) is not None:            #
    #                opt.receive_update(objective(x))                    #
    #   batch:   while (xs := opt.suggest_batch()) is not None:          #
    #                opt.tell_batch([objective(x) for x in xs])          #
    # Both drive the optimizer's own (unchanged) optimize() in a worker  #
    # thread, so the trajectory matches a monolithic run. Sequential     #
    # algorithms emit groups of 1; synchronous population methods that   #
    # call evaluate_batch() emit their whole generation. Use a FRESH     #
    # instance per ask/tell run; don't mix views or a direct optimize(). #
    # ------------------------------------------------------------------ #
    class _GenDrive:
        """Threadless ask/tell state for optimizers that define _run().
        `group` is the current pending group of clipped points — size 1 for a
        scalar yield, size m for a Batch yield — mirroring the thread shim's
        group semantics."""

        __slots__ = (
            "gen",
            "group",
            "was_batch",
            "idx",
            "vals",
            "awaiting",
            "done",
            "mode",
        )

        def __init__(self, gen):
            self.gen = gen
            self.group = None
            self.was_batch = False
            self.idx = 0
            self.vals = []
            self.awaiting = False
            self.done = False
            self.mode = None

    def _gen_start(self):
        if self.evaluations > 0:
            raise RuntimeError(
                "ask/tell on an instance that has already evaluated; "
                "construct a fresh optimizer."
            )
        gd = self._gd = BaseOptimizer._GenDrive(self._run())
        try:
            self._gen_stage(gd, next(gd.gen))
        except StopIteration:
            gd.done = True
        return gd

    def _gen_stage(self, gd, yielded):
        """Install the generator's latest yield as the pending group."""
        while True:
            gd.was_batch = isinstance(yielded, Batch)
            points = yielded if gd.was_batch else [yielded]
            gd.group = [_A.clip(p, 0, 1) for p in points]
            gd.idx = 0
            gd.vals = []
            if gd.group:
                return
            # Empty Batch: nothing to evaluate — answer with [] and move on.
            yielded = gd.gen.send([])

    def _gen_feed(self, gd, value):
        """Bank one value for the group's next point; advance the generator
        once the group is fully answered."""
        self.evaluations += 1
        self._bookkeep(gd.group[gd.idx], float(value))
        gd.vals.append(float(value))
        gd.idx += 1
        if self.evaluations >= self.n_trials:
            # Budget spent: the run is over whether or not the generator's
            # current group was fully answered (the driver, not the
            # algorithm, owns the cap).
            gd.done = True
            gd.group = None
            gd.gen.close()
            return
        if gd.idx < len(gd.group):
            return
        try:
            sent = gd.vals if gd.was_batch else gd.vals[0]
            self._gen_stage(gd, gd.gen.send(sent))
        except StopIteration:
            gd.done = True
            gd.group = None

    def _asktell_start(self):
        if self.evaluations > 0:
            raise RuntimeError(
                "ask/tell on an instance that has already evaluated; "
                "construct a fresh optimizer."
            )
        at = self._at = _AskTellState()
        self.objective = at.single_handshake  # route evaluate() through the handshake
        at.thread = threading.Thread(target=self._asktell_run, daemon=True)
        at.thread.start()
        return at

    def _next_group(self, at):
        """Block for the worker's next group; returns the list of points or None."""
        tag, group = at.req.get()
        if tag is _DONE:
            at.done = True
            at.thread.join(timeout=5)
            if at.error is not None:
                raise at.error
            return None
        return list(group)

    def _asktell_budget_spent(self):
        """True when no evaluation is left to ask for. An instance that spent its
        budget in optimize() rather than through ask/tell still refuses to switch
        views, the way a fresh call to either driver would."""
        if self.evaluations < self.n_trials:
            return False
        if self.evaluations and getattr(self, "_gd", None) is None and self._at is None:
            raise RuntimeError(
                "ask/tell on an instance that has already evaluated; "
                "construct a fresh optimizer."
            )
        return True

    def suggest_next(self):
        """Scalar view: next point to evaluate (clipped [0,1]^n) or None when done.
        Works over synchronous population methods too — their generation is served
        one point at a time, and the values are handed back once the group fills."""
        if self._asktell_budget_spent():
            return None
        if getattr(type(self), "_run", None) is not None:
            gd = getattr(self, "_gd", None) or self._gen_start()
            if gd.mode == "batch":
                raise RuntimeError("instance already driven via suggest_batch()")
            gd.mode = "scalar"
            if gd.done:
                return None
            if gd.awaiting:
                raise RuntimeError(
                    "suggest_next() called again before receive_update()"
                )
            gd.awaiting = True
            return gd.group[gd.idx]
        at = self._at or self._asktell_start()
        if at.mode == "batch":
            raise RuntimeError("instance already driven via suggest_batch()")
        at.mode = "scalar"
        if at.done:
            return None
        if at.awaiting:
            raise RuntimeError("suggest_next() called again before receive_update()")
        if not at.cur_points:
            group = self._next_group(at)
            if group is None:
                return None
            at.cur_points = group
            at.cur_idx = 0
            at.cur_vals = [None] * len(group)
        at.awaiting = True
        return at.cur_points[at.cur_idx]

    def receive_update(self, value):
        """Scalar view: report the value for the most recent suggest_next() point."""
        gd = getattr(self, "_gd", None)
        if gd is not None:
            if not gd.awaiting:
                raise RuntimeError("receive_update() without a matching suggest_next()")
            gd.awaiting = False
            self._gen_feed(gd, value)
            return
        at = self._at
        if at is None or not at.awaiting:
            raise RuntimeError("receive_update() without a matching suggest_next()")
        at.awaiting = False
        at.cur_vals[at.cur_idx] = float(value)
        at.cur_idx += 1
        if at.cur_idx >= len(at.cur_points):
            at.resp.put(at.cur_vals)  # whole group's values -> unblock the worker
            at.cur_points = None

    def suggest_batch(self):
        """Batch view: the next group of points to evaluate together (size 1 for
        sequential algorithms, the generation size for synchronous population
        methods), or None when done. Pair with tell_batch()."""
        if self._asktell_budget_spent():
            return None
        if getattr(type(self), "_run", None) is not None:
            gd = getattr(self, "_gd", None) or self._gen_start()
            if gd.mode == "scalar":
                raise RuntimeError("instance already driven via suggest_next()")
            gd.mode = "batch"
            if gd.done:
                return None
            if gd.awaiting:
                raise RuntimeError("suggest_batch() called again before tell_batch()")
            # A generation larger than what is left of the budget is served
            # only as far as the budget goes; tell_batch() then ends the run.
            remaining = self.n_trials - self.evaluations
            if len(gd.group) > remaining:
                gd.group = gd.group[:remaining]
            gd.awaiting = True
            return list(gd.group)
        at = self._at or self._asktell_start()
        if at.mode == "scalar":
            raise RuntimeError("instance already driven via suggest_next()")
        at.mode = "batch"
        if at.done:
            return None
        if at.awaiting_batch:
            raise RuntimeError("suggest_batch() called again before tell_batch()")
        group = self._next_group(at)
        if group is None:
            return None
        at.cur_points = group
        at.awaiting_batch = True
        return list(group)

    def tell_batch(self, values):
        """Batch view: report values for the most recent suggest_batch() group."""
        gd = getattr(self, "_gd", None)
        if gd is not None:
            if not gd.awaiting:
                raise RuntimeError("tell_batch() without a matching suggest_batch()")
            values = [float(v) for v in values]
            if len(values) != len(gd.group):
                raise ValueError(
                    f"tell_batch expected {len(gd.group)} values, got {len(values)}"
                )
            gd.awaiting = False
            for v in values:
                self._gen_feed(gd, v)
            return
        at = self._at
        if at is None or not at.awaiting_batch:
            raise RuntimeError("tell_batch() without a matching suggest_batch()")
        values = [float(v) for v in values]
        if len(values) != len(at.cur_points):
            raise ValueError(
                f"tell_batch expected {len(at.cur_points)} values, got {len(values)}"
            )
        at.awaiting_batch = False
        at.resp.put(values)
        at.cur_points = None

    def is_done(self):
        """True once the driven run has completed (or been closed)."""
        if self.evaluations >= self.n_trials:
            return True
        gd = getattr(self, "_gd", None)
        if gd is not None:
            return gd.done
        return self._at is not None and self._at.done

    def best(self):
        """Current best (value, point)."""
        return self.best_value, self.best_x

    def close(self):
        """Abandon a partially-driven ask/tell run, unwinding the worker cleanly."""
        gd = getattr(self, "_gd", None)
        if gd is not None:
            if not gd.done:
                gd.gen.close()
                gd.done = True
                gd.awaiting = False
            return
        at = self._at
        if at is not None and not at.done:
            at.resp.put(_ABORT)  # unblock the worker so optimize() can unwind
            at.thread.join(timeout=5)
            at.done = True
            at.awaiting = False
            at.awaiting_batch = False

    def _asktell_run(self):
        at = self._at
        try:
            at.result = self.optimize()
        except _AbortAskTell:
            pass
        except BaseException as e:  # noqa: BLE001 — re-raised via suggest_next()
            at.error = e
        finally:
            at.req.put((_DONE, None))

    # ------------------------------------------------------------------ #
    # Shared L-BFGS-B (Byrd-Lu-Nocedal-Zhu 1995, simple-bounds form).    #
    # ------------------------------------------------------------------ #
    # Used as a polish stage by DifferentialEvolution (matches
    # scipy.differential_evolution `polish=True`), SimulatedAnnealing
    # (matches scipy.dual_annealing's local-search stage), and BayesianOpt.
    # Also used directly by the LBFGSB optimizer.
    #
    # This is a faithful pure-Python port of scipy's `_minimize_lbfgsb`
    # with four elements the previous "two-loop + Armijo" did not have:
    #
    #   1. **Bound-aware direction projection** — when x[i] sits on a
    #      bound and the unconstrained direction would push past it,
    #      that component is zeroed (the "Cauchy-point" idea reduced
    #      to its essence for simple bounds).
    #   2. **Projected-gradient convergence test** — terminate when the
    #      sup-norm of the bound-projected gradient falls below `pgtol`,
    #      not the unconstrained gradient norm.
    #   3. **f-tolerance termination** — terminate when the relative
    #      decrease in f falls below `factr * eps_mach`, scipy's
    #      headline convergence criterion.
    #   4. **Feasibility-limited step** — cap the initial step length so
    #      x + step*direction stays in [0,1]^n before backtracking,
    #      avoiding wasted line-search iterations that get clipped.
    #
    # Following, but not reproducing: Byrd, Lu, Nocedal, Zhu (1995), "A Limited Memory
    # Algorithm for Bound Constrained Optimization", SIAM J. Sci. Comput. 16(5), and
    # scipy/optimize/lbfgsb_src/ + _minimize_lbfgsb. See the note on _lbfgs_polish_gen for
    # which parts are here and which are not (#407).

    # scipy's default `factr` is 1e7 (moderate accuracy). 1e2 = "high
    # accuracy" per scipy's docstring; we use the moderate default.
    _LBFGS_FACTR = 1e7
    _LBFGS_PGTOL = 1e-5
    _LBFGS_EPS_MACH = 2.220446049250313e-16
    _LBFGS_MEMORY = 10  # scipy `maxcor` default

    def _drive_gen(self, gen):
        """Drive a point-yielding generator with self.evaluate (legacy path)."""
        try:
            x = next(gen)
            while True:
                x = gen.send(self.evaluate(x))
        except StopIteration as st:
            return st.value

    def _lbfgs_polish(self):
        return self._drive_gen(self._lbfgs_polish_gen())

    # What this is, and what it is not.
    #
    # Projected L-BFGS on a box, not the Byrd-Lu-Nocedal-Zhu bound-constrained algorithm. It
    # takes the unconstrained two-loop direction, zeroes components pointing out of an active
    # bound, and backtracks along it. What it does not have is the generalised Cauchy point
    # along the piecewise projected-gradient path, or the free-variable subspace minimisation
    # that follows it -- the two pieces that make L-BFGS-B what it is. Calling direction
    # clipping "the simple-bounds reduction of the Cauchy point" overstated the case, and the
    # docstrings said "faithful port of scipy" where they should have said this (#407).
    #
    # It also uses Armijo backtracking where scipy uses a strong-Wolfe line search.
    #
    # The pieces that were simply wrong are fixed: the projected-gradient norm is scipy's
    # `projgr`, and the two-loop recursion scales H0 by s.y/y.y as scipy's does. What remains is
    # a different algorithm from scipy's, performing comparably on these problems, and it should
    # be described that way rather than as a port.
    def _lbfgs_polish_gen(self, start=None, start_value=None):
        """Polish from `start`, or from the best point seen when it is None.

        A polish that always begins at the running best cannot restart: once it has converged,
        every further pass re-derives the same point and the budget buys nothing. `start` lets
        a caller run an independent descent from somewhere else, which is what a multi-start
        method needs -- the global best is tracked by _bookkeep either way, so a pass that ends
        worse than the incumbent costs evaluations but cannot cost the answer.
        """
        n = self.n_dim
        memory = min(self._LBFGS_MEMORY, max(1, n))

        lo = [0.0] * n
        hi = [1.0] * n

        x = self.best_x.copy() if start is None else _A.asarray(list(start))
        f = self.best_value if start is None else float(start_value)
        grad = yield from self._fd_gradient_polish_gen(x)

        s_list: list = []
        y_list: list = []

        # Budget reservation: each iteration costs at least one gradient
        # (2·n evals) plus one or more candidate evaluations. Stop early
        # enough to leave room for the final gradient computation.
        while self.evaluations < self.n_trials - 2 * n:
            # (1) Convergence on the projected gradient, scipy's `projgr`.
            if self._proj_grad_sup_norm(x, grad) < self._LBFGS_PGTOL:
                break

            # (2) The limited-memory model, in compact form.
            theta, W, Minv = self._lbfgsb_compact(s_list, y_list)

            # (3) The generalised Cauchy point chooses the active set by minimising that model
            # along the piecewise projected-gradient path.
            xcp, free, c = self._lbfgsb_cauchy(x, grad, theta, W, Minv, lo, hi)

            # (4) Subspace minimisation over whatever is still free, truncated to the box.
            xbar = self._lbfgsb_subspace(x, xcp, grad, theta, W, Minv, c, free, lo, hi)

            direction = [xbar[k] - float(x[k]) for k in range(n)]
            if all(abs(v) <= 1e-300 for v in direction):
                break

            gd = _A.fold_sum(float(grad[k]) * direction[k] for k in range(n))
            if gd >= 0.0:
                # The model's step is not a descent direction -- stale curvature, or a
                # subspace solve that failed. Fall back to the projected gradient, which
                # always is one unless the point is stationary.
                direction = [-float(grad[k]) for k in range(n)]
                for k in range(n):
                    xk = float(x[k])
                    if (xk <= lo[k] and direction[k] < 0.0) or (
                        xk >= hi[k] and direction[k] > 0.0
                    ):
                        direction[k] = 0.0
                gd = _A.fold_sum(float(grad[k]) * direction[k] for k in range(n))
                if gd >= 0.0:
                    break
                s_list.clear()
                y_list.clear()

            # (5) Backtracking line search. Every trial point is feasible by construction, so
            # the step needs no separate projection.
            step = 1.0
            new_x = x
            new_f = f
            accepted = False
            while step > 1e-14:
                if self.evaluations >= self.n_trials:
                    break
                candidate = _A.asarray(
                    [
                        min(hi[k], max(lo[k], float(x[k]) + step * direction[k]))
                        for k in range(n)
                    ]
                )
                cand_f = yield candidate
                if cand_f <= f + 1e-4 * step * gd:
                    new_x = candidate
                    new_f = cand_f
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                break

            # (6) scipy's factr * eps_mach test on the relative decrease.
            f_scale = max(abs(f), abs(new_f), 1.0)
            if (f - new_f) < self._LBFGS_FACTR * self._LBFGS_EPS_MACH * f_scale:
                x, f = new_x, new_f
                break

            new_grad = yield from self._fd_gradient_polish_gen(new_x)

            # (7) Curvature pair, kept only when it is one.
            s = _A.asarray([float(new_x[k]) - float(x[k]) for k in range(n)])
            y = _A.asarray([float(new_grad[k]) - float(grad[k]) for k in range(n)])
            sy = self._lbfgsb_dot(s, y)
            ss = self._lbfgsb_dot(s, s)
            yy = self._lbfgsb_dot(y, y)
            if sy > 1e-12 * math.sqrt(ss * yy + 1e-300):
                s_list.append([float(v) for v in s])
                y_list.append([float(v) for v in y])
                if len(s_list) > memory:
                    s_list.pop(0)
                    y_list.pop(0)
            x, f, grad = new_x, new_f, new_grad

    # ---- Byrd-Lu-Nocedal-Zhu machinery -------------------------------------------------
    #
    # The three pieces that make L-BFGS-B what it is: the compact limited-memory
    # representation, the generalised Cauchy point that chooses the active set by minimising
    # the model along the piecewise projected-gradient path, and subspace minimisation over
    # whatever is still free. The polish previously had none of them -- it clipped an
    # unconstrained direction at active bounds and backtracked, which is projected L-BFGS and a
    # different algorithm (#407).
    #
    # Byrd, Lu, Nocedal & Zhu (1995), "A Limited Memory Algorithm for Bound Constrained
    # Optimization", SIAM J. Sci. Comput. 16(5), sections 4-5; scipy's lbfgsb_src (`cauchy`,
    # `subsm`, `projgr`). Validated against scipy on bound-active quadratics in
    # tests/test_lbfgsb_algorithm.py.

    @staticmethod
    def _lbfgsb_dot(a, b):
        return _A.fold_sum(float(a[i]) * float(b[i]) for i in range(len(a)))

    @classmethod
    def _lbfgsb_solve(cls, A, b):
        """Gaussian elimination with partial pivoting; None if singular."""
        k = len(b)
        M = [list(A[i]) + [float(b[i])] for i in range(k)]
        for col in range(k):
            pivot = max(range(col, k), key=lambda r: abs(M[r][col]))
            if abs(M[pivot][col]) < 1e-300:
                return None
            M[col], M[pivot] = M[pivot], M[col]
            inv = 1.0 / M[col][col]
            for r in range(col + 1, k):
                factor = M[r][col] * inv
                if factor:
                    for c in range(col, k + 1):
                        M[r][c] -= factor * M[col][c]
        out = [0.0] * k
        for r in range(k - 1, -1, -1):
            total = M[r][k]
            for c in range(r + 1, k):
                total -= M[r][c] * out[c]
            out[r] = total / M[r][r]
        return out

    @classmethod
    def _lbfgsb_compact(cls, s_list, y_list):
        """theta, W (as columns) and M^-1 for B = theta I - W M W^T."""
        m = len(s_list)
        if m == 0:
            return 1.0, [], []
        sy_last = cls._lbfgsb_dot(s_list[-1], y_list[-1])
        yy_last = cls._lbfgsb_dot(y_list[-1], y_list[-1])
        theta = yy_last / sy_last if sy_last > 1e-300 else 1.0

        W = [[float(v) for v in y] for y in y_list]
        W += [[theta * float(v) for v in s] for s in s_list]

        D = [cls._lbfgsb_dot(s_list[i], y_list[i]) for i in range(m)]
        L = [
            [cls._lbfgsb_dot(s_list[i], y_list[j]) if i > j else 0.0 for j in range(m)]
            for i in range(m)
        ]
        SS = [
            [theta * cls._lbfgsb_dot(s_list[i], s_list[j]) for j in range(m)]
            for i in range(m)
        ]

        size = 2 * m
        Minv = [[0.0] * size for _ in range(size)]
        for i in range(m):
            Minv[i][i] = -D[i]
            for j in range(m):
                Minv[i][m + j] = L[j][i]
                Minv[m + i][j] = L[i][j]
                Minv[m + i][m + j] = SS[i][j]
        return theta, W, Minv

    @classmethod
    def _lbfgsb_cauchy(cls, x, g, theta, W, Minv, lo, hi):
        """Generalised Cauchy point: the first minimiser of the model along the piecewise
        projected steepest-descent path. Returns (xcp, free, c)."""
        n = len(x)
        m2 = len(W)

        t = [0.0] * n
        d = [0.0] * n
        for i in range(n):
            gi = float(g[i])
            if gi < 0.0:
                t[i] = (float(x[i]) - hi[i]) / gi
            elif gi > 0.0:
                t[i] = (float(x[i]) - lo[i]) / gi
            else:
                t[i] = float("inf")
            d[i] = 0.0 if t[i] == 0.0 else -gi

        xcp = [float(v) for v in x]
        free = [i for i in range(n) if t[i] > 0.0]
        if not free:
            return xcp, [], [0.0] * m2

        p = [cls._lbfgsb_dot(col, d) for col in W] if m2 else []
        c = [0.0] * m2
        fp = -cls._lbfgsb_dot(d, d)
        if m2:
            Mp = cls._lbfgsb_solve(Minv, p)
            fpp = -theta * fp - (cls._lbfgsb_dot(p, Mp) if Mp is not None else 0.0)
        else:
            fpp = -theta * fp
        dt_min = -fp / fpp if fpp > 1e-300 else float("inf")

        # Only variables that can move are breakpoints. One whose breakpoint is zero sits on the
        # bound the gradient pushes it against: it joins the active set at the start, and
        # walking it here adds its gradient to fp as though the path had travelled along it.
        order = sorted((i for i in free if t[i] < float("inf")), key=lambda i: t[i])
        t_old = 0.0
        for b in order:
            dt = t[b] - t_old
            if dt_min < dt:
                break
            for i in range(n):
                if d[i] != 0.0:
                    xcp[i] += dt * d[i]
            xcp[b] = hi[b] if d[b] > 0.0 else lo[b]
            zb = xcp[b] - float(x[b])
            gb = float(g[b])
            if m2:
                wb = [W[j][b] for j in range(m2)]
                for j in range(m2):
                    c[j] += dt * p[j]
                Mc = cls._lbfgsb_solve(Minv, c)
                Mw = cls._lbfgsb_solve(Minv, wb)
                Mp = cls._lbfgsb_solve(Minv, p)
                fp += dt * fpp + gb * gb + theta * gb * zb
                if Mc is not None:
                    fp -= gb * cls._lbfgsb_dot(wb, Mc)
                fpp -= theta * gb * gb
                if Mp is not None:
                    fpp -= 2.0 * gb * cls._lbfgsb_dot(wb, Mp)
                if Mw is not None:
                    fpp -= gb * gb * cls._lbfgsb_dot(wb, Mw)
                for j in range(m2):
                    p[j] += gb * wb[j]
            else:
                fp += dt * fpp + gb * gb + theta * gb * zb
                fpp -= theta * gb * gb
            d[b] = 0.0
            t_old = t[b]
            dt_min = -fp / fpp if fpp > 1e-300 else float("inf")
            if fp >= 0.0:
                dt_min = 0.0
                break

        dt_min = max(dt_min, 0.0)
        for i in range(n):
            if d[i] != 0.0:
                xcp[i] += dt_min * d[i]
        for i in range(n):
            xcp[i] = min(hi[i], max(lo[i], xcp[i]))
        if m2:
            for j in range(m2):
                c[j] += dt_min * p[j]

        free = [i for i in range(n) if lo[i] < xcp[i] < hi[i]]
        return xcp, free, c

    @classmethod
    def _lbfgsb_subspace(cls, x, xcp, g, theta, W, Minv, c, free, lo, hi):
        """Minimise the model over the variables still free at the Cauchy point, truncated to
        the box. Variables the Cauchy point fixed stay fixed: that active set is what it is for."""
        if not free:
            return list(xcp)
        m2 = len(W)

        Mc = cls._lbfgsb_solve(Minv, c) if m2 else None
        r = []
        for i in free:
            ri = float(g[i]) + theta * (xcp[i] - float(x[i]))
            if Mc is not None:
                ri -= _A.fold_sum(W[j][i] * Mc[j] for j in range(m2))
            r.append(ri)

        k = len(free)
        B = [[0.0] * k for _ in range(k)]
        for a in range(k):
            B[a][a] = theta
        if m2:
            MW = []
            for i in free:
                wi = [W[j][i] for j in range(m2)]
                MW.append(cls._lbfgsb_solve(Minv, wi) or [0.0] * m2)
            for a in range(k):
                wa = [W[j][free[a]] for j in range(m2)]
                for b in range(k):
                    B[a][b] -= cls._lbfgsb_dot(wa, MW[b])

        step = cls._lbfgsb_solve(B, [-ri for ri in r])
        if step is None:
            return list(xcp)

        alpha = 1.0
        for a, i in enumerate(free):
            if step[a] > 1e-300:
                alpha = min(alpha, (hi[i] - xcp[i]) / step[a])
            elif step[a] < -1e-300:
                alpha = min(alpha, (lo[i] - xcp[i]) / step[a])
        alpha = max(0.0, min(1.0, alpha))

        out = list(xcp)
        for a, i in enumerate(free):
            out[i] = min(hi[i], max(lo[i], xcp[i] + alpha * step[a]))
        return out

    def _lbfgs_two_loop(self, grad, s_list, y_list):
        """Two-loop recursion for the L-BFGS search direction (Nocedal 1980).

        The middle step scales by gamma = s.y / y.y from the newest curvature pair, which is
        what scipy's recurrence does and what sets the step length before any line search. The
        previous version used the identity there, which is a legitimate variant but a slower one
        and not the recurrence it claimed to be (#407).
        """
        n = len(grad)
        direction = [-float(g) for g in grad]
        alpha = [0.0] * len(s_list)
        for i in range(len(s_list) - 1, -1, -1):
            sy = _A.fold_sum(
                float(s_list[i][k]) * float(y_list[i][k]) for k in range(n)
            )
            if abs(sy) < 1e-30:
                continue
            rho = 1.0 / sy
            alpha[i] = rho * _A.fold_sum(
                float(s_list[i][k]) * direction[k] for k in range(n)
            )
            direction = [
                direction[k] - alpha[i] * float(y_list[i][k]) for k in range(n)
            ]

        # H0 = gamma * I with gamma = s.y / y.y from the newest pair, as in scipy. This is the
        # scale of the step before any line search, and using the identity here -- as this did --
        # leaves the first trial step wrong by whatever the curvature is (#407).
        if s_list:
            newest_s = s_list[-1]
            newest_y = y_list[-1]
            sy = _A.fold_sum(float(newest_s[k]) * float(newest_y[k]) for k in range(n))
            yy = _A.fold_sum(float(newest_y[k]) * float(newest_y[k]) for k in range(n))
            if yy > 1e-30 and sy > 0.0:
                gamma = sy / yy
                direction = [gamma * d for d in direction]

        for i in range(len(s_list)):
            sy = _A.fold_sum(
                float(s_list[i][k]) * float(y_list[i][k]) for k in range(n)
            )
            if abs(sy) < 1e-30:
                continue
            rho = 1.0 / sy
            beta = rho * _A.fold_sum(
                float(y_list[i][k]) * direction[k] for k in range(n)
            )
            direction = [
                direction[k] + (alpha[i] - beta) * float(s_list[i][k]) for k in range(n)
            ]
        return direction

    def _proj_grad_sup_norm(self, x, grad):
        """Sup-norm of the bound-projected gradient: ||P(x - g) - x||_inf, on [0, 1]^n.

        This is scipy's `projgr`: a positive gradient component is bounded by how far the
        variable can travel down to its lower bound, and a negative one by how far up to its
        upper bound. The distance matters at every point, not only on the boundary -- a step of
        100 from x = 0.01 moves 0.01, whatever the gradient says.

        The previous version clipped only when a variable sat exactly on a bound, so it returned
        the raw gradient everywhere inside the cube: at x = [0.01, 0.5] with g = [100, 0] it
        reported 100 where the projected step is 0.01 (#407). That is the quantity the
        convergence test compares against pgtol, so the test was reading a number four orders
        too large and the polish carried on past its own stopping criterion.
        """
        n = len(grad)
        m = 0.0
        for k in range(n):
            gk = float(grad[k])
            xk = float(x[k])
            if gk < 0.0:
                gk = max(gk, xk - 1.0)  # bounded by the distance to the upper bound
            else:
                gk = min(gk, xk)  # bounded by the distance to the lower bound
            if abs(gk) > m:
                m = abs(gk)
        return m

    def _fd_gradient_for_polish(self, x):
        """Central-difference gradient with budget guards (legacy driver of
        the generator twin below; mirrors LBFGSB._fd_gradient)."""
        return self._drive_gen(self._fd_gradient_polish_gen(x))

    def _fd_gradient_polish_gen(self, x):
        n = self.n_dim
        h = 1e-6
        grad = [0.0] * n
        for i in range(n):
            if self.evaluations >= self.n_trials:
                break
            x_plus = x.copy()
            x_plus[i] = min(1.0, float(x[i]) + h)
            f_plus = yield x_plus
            if self.evaluations >= self.n_trials:
                break
            x_minus = x.copy()
            x_minus[i] = max(0.0, float(x[i]) - h)
            f_minus = yield x_minus
            denom = float(x_plus[i]) - float(x_minus[i])
            if denom > 0:
                grad[i] = (f_plus - f_minus) / denom
        return _A.asarray(grad)
