# Online optimizers: the generator inversion

Every optimizer becomes natively online — accepting one observation at a
time — by inverting who owns the loop. This note records the design, the
migration protocol, and why it precedes the Julia/Rust/R ports.

## The protocol

A converted optimizer defines `_run()`, a generator:

    def _run(self):
        ...
        value = yield point          # was: value = self.evaluate(point)
        ...

The driver (BaseOptimizer) owns clipping, the evaluation counter, path
sampling and best tracking, in exactly `evaluate()`'s historical order, so
a converted algorithm's seeded trajectory is *identical* to its
loop-owning form — statement for statement, RNG draw for RNG draw. Nested
helpers convert with `yield from` (see `_lbfgs_polish_gen`, whose legacy
`_lbfgs_polish()` is now a thin driver, upgrading every optimizer that
polishes).

Three drive modes share the protocol:

- `optimize()` — the classic API, now a loop over `_run` in BaseOptimizer.
- `suggest_next()/receive_update()` and `suggest_batch()/tell_batch()` —
  the existing public ask/tell surface, served without a worker thread
  for converted classes. The thread-handshake shim remains only for
  unconverted classes and is retired class by class.
- Parity vectors (planned) — the ports check every `(value in) -> (point
  out)` transition, far tighter than endpoint comparisons.

## Two tiers

**Tier 1 — native.** Anything we own converts mechanically:
`self.evaluate(x)` becomes `(yield x)`; list comprehensions over evaluate
expand to loops (yield is illegal in comprehensions); helpers become
sub-generators. Pilot: DifferentialEvolution (easy, with polish) and
NelderMead (nested restart/shrink control flow). Both proved
trajectory-identical against frozen pre-conversion copies
(`tests/reference_impls_pre_online.py`, `tests/test_online_pilot.py`).

**Tier 2 — buffered facade.** For cores we can't yield-ify (foreign
libraries, future ports wrapping batch routines): `ask()` serves from a
queue of planned points, `tell()` banks results, and the inner batch code
advances only when its plan is exhausted. Same interface, documented
"plans in chunks" semantics. No current roster member needs this, but the
tier is part of the contract so ports may use it where a language lacks
coroutines.

## Migration gates

Each conversion ships with its proof: freeze the pre-conversion class
verbatim in `tests/reference_impls_pre_online.py`, and require exact
trajectory equality across seeds x dimensions x objectives x budgets,
plus native-ask/tell == optimize equality. A conversion that changes any
trajectory is wrong by definition, whatever the leaderboards say.

## Why now

The Rust/Julia/R ports multiply the cost of the loop-owning architecture:
the thread shim does not port (WASM has no threads; R neither), and
inverting five codebases later is five times this work. Ports written
against the online form — and its transition-level parity vectors — never
know the loop-owning era existed.

## Roster status

Complete. All 23 optimizers are natively online (PRs #296-#300): every
class defines _run(), ask/tell runs threadless across the roster, and no
loop-owning optimizer code remains. PRIMA converted as Tier 1 after all —
the three trust-region methods are in-repo pure Python, so their
interpolation-set init helpers became sub-generators and the buffered
facade was never needed (it remains part of the contract for ports
wrapping foreign batch code).

Batch yields, added in the third wave: a generator may
`values = yield Batch([x1, ..., xm])` (the marker class in base.py) to
surface a whole synchronous generation. The drivers evaluate the group in
evaluate()'s exact per-point order; suggest_batch() serves it intact for
parallel evaluation and suggest_next() serves it point by point. CMA-ES
uses this for its generations.

Every conversion carries its frozen pre-conversion twin in
tests/reference_impls_pre_online.py and an exact trajectory-equivalence
proof in tests/test_online_pilot.py (299 tests at completion).

## What this unlocks

Runtime composition. With every optimizer speaking ask/tell without
threads, a meta-optimizer can interleave several optimizers on one
evaluation stream — round-robin, bandit allocation over suggestions,
mid-run switching between hosts — with no glue beyond the interface.
This is a new, purely mechanical composition family alongside the
inspiration simplex's semantic blends, and a natural baseline for that
paper: does an LLM-blended artifact beat a scheduler juggling the same
vertex algorithms at runtime?

Ports. The generator protocol is language-neutral: Rust generators via
explicit state machines or coroutines, Julia via Channels or closures, R
via closure-based iterators, JS via native generators. Transition-level
parity vectors (every value-in/point-out pair, not endpoint comparisons)
become the cross-language contract once the portable RNG lands.
