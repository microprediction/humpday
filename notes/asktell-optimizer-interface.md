# Design note: a uniform ask/tell (`suggest_next` / `receive_update`) optimizer interface

**Status:** design + de-risking prototype. Nothing in the shipped `humpday/`
package changes until the equivalence prototype (below) passes.

## 1. Motivation

Today every optimizer is *monolithic*: it owns its loop and calls the objective
itself. That makes the things we actually want hard:

- **Switching / hot-swapping** one optimizer for another mid-run.
- **Crossover / blending** two optimizers (the runtime version of what the
  "Inspiration Simplex" did by generating fused code).
- **Shared budget accounting, pausing, checkpointing, and instrumentation.**

A uniform incremental interface —

```python
opt.suggest_next(batch=1) -> x | [x]   # propose point(s) to evaluate
opt.receive_update(x, y)               # report observed value(s)
opt.is_done() -> bool
opt.best() -> (best_value, best_x)
```

— makes all of these first-class: the *caller* owns the loop and can interleave,
swap, or fuse optimizers freely.

## 2. What the existing implementation actually looks like (checked)

(Map from a full sweep of `humpday/optimizers/`.)

- **21 optimizers**, all faithful pure-Python ports that **own their loop** and
  call `self.evaluate(x)` internally. None wrap an external ask/tell library
  (cmaes / nevergrad / skopt are reimplemented, not wrapped), so there is nothing
  to delegate ask/tell to — but also no external loop to fight.
- **One chokepoint:** every objective call in every optimizer funnels through
  `BaseOptimizer.evaluate(x)` (`base.py:39–59`): it clips to `[0,1]^n`, calls
  `self.objective`, samples the path, updates `best_value`/`best_x`/`evaluations`,
  returns the value. **This is the single inversion point.**
- **Public contract everything depends on:** `optimize() -> (best_value, best_x)`;
  functional wrapper `pure_optimize(objective, name, n_trials, n_dim)`; the
  `cube_*` transform layer in `scipy_interface.py`; `adaptive_optimizer.py`'s
  tournament; and `tests/test_optimizers.py` (smoke test: runs each optimizer,
  enforces budget ≤ 3× n_trials, unpacks `(best_value, best_x[, count])`).
- **Population vs sequential:** DE / CMA-ES / PSO / GA naturally evaluate a
  *generation* (batch) per step; NelderMead / Powell / PRIMA / pattern-search /
  coordinate-descent are inherently sequential (next point depends on the last
  value). 5 algorithms add an L-BFGS finite-difference polish (coupled evals).

## 3. The key consequence: invert once, not 21 times

Because all objective calls pass through `evaluate()`, ask/tell can be retrofitted
**in one place** by inverting control *at that method*, leaving every algorithm's
body untouched. The usual nightmare — rewriting each trust-region/simplex method
as an explicit state machine to externalize its control flow — is avoided.

Three ways to invert, with the trade-offs:

| approach | algorithm code changes | new dependency | determinism | notes |
|---|---|---|---|---|
| **A. Rewrite each as a state machine** | all 21, heavy | none | full | the "horrible" path; ~500–800 LoC, high risk |
| **B. Generator (`yield` at evaluate)** | every `evaluate` call-site (`y = yield ...`) | none | full | not transparent — touches each algorithm |
| **C. Coroutine inversion via worker thread** | **none** | stdlib `threading`/`queue` | full (lock-step) | transparent; recommended |
| C′. Same via `greenlet` | none | **greenlet** (new dep) | full | faster than threads, but violates the zero-runtime-dep rule |

**Recommendation: C (thread-based, stdlib-only).** Run the algorithm's existing
`optimize()` in a worker thread; intercept `evaluate(x)` so that instead of
calling the objective it **hands `x` to the driver and blocks for the value**.
The handshake is strictly lock-step (one outstanding point at a time), so:

- the algorithm's RNG draws and branch decisions happen in the exact same order
  as a monolithic run → **identical trajectory for the same seed** (this is
  testable and is the gate, §5);
- no algorithm code changes, no new runtime dependency;
- the polish stage and any multi-eval inner steps work transparently (the worker
  just pauses at whatever `evaluate()` comes next).

Cost is ~100–200 LoC in/around `base.py`, additive, with `optimize()` rewritten
as a thin loop over `suggest_next`/`receive_update` so there is a **single source
of truth** and the existing contract is preserved byte-for-byte in behavior.

## 4. The one real subtlety: batching for population methods

Lock-step single-point ask/tell is *exactly equivalent* for replaying a run
(every `evaluate()` is mirrored 1:1), so it is correct for the equivalence gate.
But for the **crossover use case** it under-serves population methods: DE/CMA want
to emit a whole generation, get all values, then update. So the production
interface is `suggest_next(batch=k)` / `receive_update(points, values)`:

- sequential methods ignore `batch` (always emit 1);
- population methods buffer internally and emit their generation when asked, then
  consume the matching values.

Phase 1 proves equivalence with `batch=1`; batching is Phase 4 and is where the
"blend a population method with a sequential one" case must be validated.

## 5. Equivalence gate (must pass before any package change)

For a representative set — **NelderMead** (sequential), **DifferentialEvolution**
(population), **CMAEvolutionStrategy** (population + polish) — assert that, for the
same seed and objective, the ask/tell-driven run reproduces `optimize()`'s
`best_value` **exactly** and the same evaluation count. If this holds, the
inversion is behavior-preserving and we can fold it into `BaseOptimizer`.

The prototype lives **outside the package** (`papers/dfo_recommender/
asktell_prototype.py`) and wraps the real optimizer classes without modifying
them, so de-risking costs the shipped code nothing.

## 6. Staged plan

1. **Interface** on `BaseOptimizer`: `suggest_next(batch=1)`, `receive_update`,
   `is_done()`, `best()`; keep auto-tracking of best/eval-count/path.
2. **Invert once** (approach C) in the base class; `optimize()` becomes a thin
   driver loop → single source of truth. Existing tests unchanged.
3. **Equivalence gate** (§5) across all 21, not just the three.
4. **Batch support** for population methods (§4) + tests for blended batches.
5. **Crossover/blend harness** (the actual goal): interleave / hot-swap / seed one
   optimizer's history into another at runtime.

## 7. Risks / things to watch

- **Thread teardown & exceptions:** the worker must propagate exceptions to the
  driver and never deadlock if the caller abandons the loop (timeout/`is_done`).
- **Determinism:** seed `random` *and* numpy before constructing the optimizer
  (the constructor draws `random_uniform(n_dim)` for `best_x`).
- **`with_count` / budget semantics:** preserve the ≤3× n_trials allowance the
  smoke test enforces.
- **Don't break the public contract:** `optimize()`, `pure_optimize`, the `cube_*`
  layer, and `tests/test_optimizers.py` must pass untouched.
- **Bloat:** stay stdlib-only (no greenlet) so the package keeps zero runtime deps.
