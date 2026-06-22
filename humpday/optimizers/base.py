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


class _AskTellState:
    """Lock-step handshake for driving optimize() incrementally. The worker emits a
    GROUP of points — size 1 from evaluate(), size m from evaluate_batch() — and the
    driver returns a list of that many values."""

    def __init__(self):
        self.req = queue.Queue()   # worker -> driver: ("X", [points]) or (_DONE, None)
        self.resp = queue.Queue()  # driver -> worker: [values], or _ABORT
        self.thread = None
        self.done = False
        self.error = None
        self.result = None
        self.mode = None           # 'scalar' | 'batch' (no mixing on one instance)
        self.cur_points = None     # current group being served
        self.cur_idx = 0           # scalar view: index within the group
        self.cur_vals = None       # scalar view: values collected so far
        self.awaiting = False      # scalar view: a point is out, value pending
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
        self.n_trials = n_trials
        self.n_dim = n_dim
        self.evaluations = 0
        self.best_value = float("inf")
        self.best_x = _A.random_uniform(n_dim)
        self.track_path = False
        self.path = []
        self._at = None  # ask/tell driver state (lazy; see suggest_next)

    def evaluate(self, x) -> float:
        """Evaluate objective with tracking. `x` may be a numpy array, a
        `_Vec`, or any sequence of floats — `_A.clip` normalises it."""
        self.evaluations += 1
        x_clipped = _A.clip(x, 0, 1)
        value = self.objective(x_clipped)

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

    def suggest_next(self):
        """Scalar view: next point to evaluate (clipped [0,1]^n) or None when done.
        Works over synchronous population methods too — their generation is served
        one point at a time, and the values are handed back once the group fills."""
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
        at = self._at
        if at is None or not at.awaiting:
            raise RuntimeError("receive_update() without a matching suggest_next()")
        at.awaiting = False
        at.cur_vals[at.cur_idx] = float(value)
        at.cur_idx += 1
        if at.cur_idx >= len(at.cur_points):
            at.resp.put(at.cur_vals)   # whole group's values -> unblock the worker
            at.cur_points = None

    def suggest_batch(self):
        """Batch view: the next group of points to evaluate together (size 1 for
        sequential algorithms, the generation size for synchronous population
        methods), or None when done. Pair with tell_batch()."""
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
        return self._at is not None and self._at.done

    def best(self):
        """Current best (value, point)."""
        return self.best_value, self.best_x

    def close(self):
        """Abandon a partially-driven ask/tell run, unwinding the worker cleanly."""
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
    # See: Byrd, Lu, Nocedal, Zhu (1995), "A Limited Memory Algorithm
    # for Bound Constrained Optimization", SIAM J. Sci. Comput. 16(5).
    # scipy reference: scipy/optimize/lbfgsb_src/ + _minimize_lbfgsb.

    # scipy's default `factr` is 1e7 (moderate accuracy). 1e2 = "high
    # accuracy" per scipy's docstring; we use the moderate default.
    _LBFGS_FACTR = 1e7
    _LBFGS_PGTOL = 1e-5
    _LBFGS_EPS_MACH = 2.220446049250313e-16
    _LBFGS_MEMORY = 10  # scipy `maxcor` default

    def _lbfgs_polish(self):
        n = self.n_dim
        memory = min(self._LBFGS_MEMORY, max(1, n))

        x = self.best_x.copy()
        f = self.best_value
        grad = self._fd_gradient_for_polish(x)

        s_list: list = []
        y_list: list = []

        # Budget reservation: each iteration costs at least one gradient
        # (2·n evals) plus one or more candidate evaluations. Stop early
        # enough to leave room for the final gradient computation.
        while self.evaluations < self.n_trials - 2 * n:
            # (1) Projected-gradient convergence test.
            pg_inf = self._proj_grad_sup_norm(x, grad)
            if pg_inf < self._LBFGS_PGTOL:
                break

            # (2) Two-loop recursion for the unconstrained L-BFGS direction.
            direction = self._lbfgs_two_loop(grad, s_list, y_list)

            # (3) Project direction at active bounds: zero out components
            # that would push x past the bound it already sits on. This
            # is the simple-bounds reduction of the Cauchy-point step.
            for k in range(n):
                xk = float(x[k])
                if xk <= 0.0 and direction[k] < 0.0:
                    direction[k] = 0.0
                elif xk >= 1.0 and direction[k] > 0.0:
                    direction[k] = 0.0

            gd = sum(float(grad[k]) * direction[k] for k in range(n))
            if gd > -1e-30:
                # Direction isn't a descent (zero curvature, memory drift,
                # or every component clipped). Reset memory and fall back
                # to the projected-gradient steepest-descent direction.
                s_list.clear()
                y_list.clear()
                direction = [-float(g) for g in grad]
                for k in range(n):
                    xk = float(x[k])
                    if xk <= 0.0 and direction[k] < 0.0:
                        direction[k] = 0.0
                    elif xk >= 1.0 and direction[k] > 0.0:
                        direction[k] = 0.0
                gd = sum(float(grad[k]) * direction[k] for k in range(n))
                if gd > -1e-30:
                    break  # truly stuck — projected gradient is zero

            # (4) Cap step length so x + step*direction stays in [0,1]^n.
            step_max = float("inf")
            for k in range(n):
                dk = direction[k]
                if dk > 0.0:
                    step_max = min(step_max, (1.0 - float(x[k])) / dk)
                elif dk < 0.0:
                    step_max = min(step_max, (0.0 - float(x[k])) / dk)
            step = min(1.0, step_max) if step_max > 0.0 else 1.0

            # (5) Armijo backtracking with feasibility-clipped candidates.
            c1 = 1e-4
            new_x = x.copy()
            new_f = f
            accepted = False
            while step > 1e-12:
                if self.evaluations >= self.n_trials:
                    break
                candidate = _A.clip(
                    _A.asarray([float(x[k]) + step * direction[k] for k in range(n)]),
                    0,
                    1,
                )
                cand_f = self.evaluate(candidate)
                if cand_f <= f + c1 * step * gd:
                    new_x = candidate
                    new_f = cand_f
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                break  # line search failed — no further descent

            # (6) f-tolerance termination: stop when the relative decrease
            # falls below scipy's `factr * eps_mach` criterion.
            f_scale = max(abs(f), abs(new_f), 1.0)
            if (f - new_f) < self._LBFGS_FACTR * self._LBFGS_EPS_MACH * f_scale:
                x, f = new_x, new_f
                break

            new_grad = self._fd_gradient_for_polish(new_x)

            # (7) L-BFGS memory update.
            s = _A.asarray([float(new_x[k]) - float(x[k]) for k in range(n)])
            y = _A.asarray([float(new_grad[k]) - float(grad[k]) for k in range(n)])
            if float(_A.dot(s, y)) > 1e-12:
                s_list.append(s)
                y_list.append(y)
                if len(s_list) > memory:
                    s_list.pop(0)
                    y_list.pop(0)
            x, f, grad = new_x, new_f, new_grad

    def _lbfgs_two_loop(self, grad, s_list, y_list):
        """Two-loop recursion for the L-BFGS search direction
        (Nocedal 1980)."""
        n = len(grad)
        direction = [-float(g) for g in grad]
        alpha = [0.0] * len(s_list)
        for i in range(len(s_list) - 1, -1, -1):
            sy = sum(float(s_list[i][k]) * float(y_list[i][k]) for k in range(n))
            if abs(sy) < 1e-30:
                continue
            rho = 1.0 / sy
            alpha[i] = rho * sum(float(s_list[i][k]) * direction[k] for k in range(n))
            direction = [
                direction[k] - alpha[i] * float(y_list[i][k]) for k in range(n)
            ]
        for i in range(len(s_list)):
            sy = sum(float(s_list[i][k]) * float(y_list[i][k]) for k in range(n))
            if abs(sy) < 1e-30:
                continue
            rho = 1.0 / sy
            beta = rho * sum(float(y_list[i][k]) * direction[k] for k in range(n))
            direction = [
                direction[k] + (alpha[i] - beta) * float(s_list[i][k]) for k in range(n)
            ]
        return direction

    def _proj_grad_sup_norm(self, x, grad):
        """Sup-norm of the bound-projected gradient: ||P(x - g) - x||_inf.

        For simple bounds [0, 1] this reduces to clipping g componentwise
        wherever a bound is active in the steepest-descent direction —
        the standard convergence criterion for box-constrained L-BFGS.
        """
        n = len(grad)
        m = 0.0
        for k in range(n):
            gk = float(grad[k])
            xk = float(x[k])
            if xk <= 0.0 and gk > 0.0:
                continue  # bound active, gradient points outward
            if xk >= 1.0 and gk < 0.0:
                continue  # bound active, gradient points outward
            if abs(gk) > m:
                m = abs(gk)
        return m

    def _fd_gradient_for_polish(self, x):
        """Central-difference gradient with budget guards (mirrors
        LBFGSB._fd_gradient)."""
        n = self.n_dim
        h = 1e-6
        grad = [0.0] * n
        for i in range(n):
            if self.evaluations >= self.n_trials:
                break
            x_plus = x.copy()
            x_plus[i] = min(1.0, float(x[i]) + h)
            f_plus = self.evaluate(x_plus)
            if self.evaluations >= self.n_trials:
                break
            x_minus = x.copy()
            x_minus[i] = max(0.0, float(x[i]) - h)
            f_minus = self.evaluate(x_minus)
            denom = float(x_plus[i]) - float(x_minus[i])
            if denom > 0:
                grad[i] = (f_plus - f_minus) / denom
        return _A.asarray(grad)
