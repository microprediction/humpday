# Reference ↔ Python ↔ JavaScript equivalence work

Working notes for issue [#78](https://github.com/microprediction/humpday/issues/78).

**Goal: all three of (a) the canonical reference algorithm, (b) the HumpDay
Python port, and (c) the HumpDay JavaScript port are equivalent within
reason.** Not just "Python == JS" — the **published reference** is the
target. A bug in the Python port mirrored faithfully in the JS port is
*not* done. Both ports must track the published algorithm.

For each algorithm the work is:

1. **Read the reference.** Powell's BOBYQA / NEWUOA / UOBYQA papers and the
   PRIMA Fortran source; SciPy's Nelder-Mead implementation; scikit-optimize's
   `gp_minimize`; etc. Pin down the parameters, initialisation, control
   flow, fallback rules. The reference defines the algorithm.
2. **Read the HumpDay Python port** with the reference open in another
   pane. Note every deviation. Decide whether the deviation is intentional
   (and worth keeping) or a bug to fix.
3. **Read the HumpDay JavaScript port** the same way. Note deviations from
   either the reference or the Python port.
4. **Bring all three into alignment** — usually that means patching Python
   to track the reference more closely, then patching JS to match the
   (improved) Python.
5. **Verify** with `tests/test_port_behavior.py` and trace inspection.

Python is not the authoritative source. The published algorithm is.

---

## Status (2026-05-27)

Baseline characterisation in `benchmarks/port_characterisation.json` (regen
with `pytest tests/test_port_behavior.py -m slow -s`). Ratios = js_median /
py_median on 2-D Rosenbrock at 300 evals; the most discriminating cell.

| Algorithm | Before BOBYQA fix | Current | State | Next action |
|---|---|---|---|---|
| **PRIMA_BOBYQA** | 88× worse | **0.98×** ✅ | Fixed in PR #83 | — |
| PRIMA_NEWUOA | 20× worse | 20× worse | Pending | Apply BOBYQA recipe |
| PRIMA_UOBYQA | 2× worse | 2× worse | Pending | Apply BOBYQA recipe |
| LBFGSB | 171× worse | 171× worse | Pending — also misnamed (see #80) | Re-port; or rename |
| Powell | 8× worse | 8× worse | Pending | Apply recipe |
| BayesianOpt | 8× worse | 8× worse | Pending | Apply recipe (kernel + GP path) |
| AntColonyOpt | JS 6× better | JS 6× better | Python-side port needs work | Reverse direction |
| TabuSearch | JS 5× better | JS 5× better | Python-side port needs work | Reverse direction |
| HillClimbing | JS 14× better | JS 14× better | Python-side port needs work | Reverse direction |

Remaining algorithms (13) are already equivalent within ~2× both ways.

---

## The recipe (validated on PRIMA_BOBYQA, PR #83)

1. **Read the reference algorithm first.** Pull up the paper / canonical
   implementation (PRIMA Fortran, scipy source, skopt source) and write
   down: initialisation, model form, trust-region rule, point-replacement
   rule, fallback paths, parameter values. The reference is what we're
   chasing — neither the Python nor JS port is authoritative.

2. **Read the HumpDay Python port** with the reference next to it. Note
   every deviation. If the deviation looks like a simplification (e.g.
   diagonal Hessian instead of full symmetric), decide whether to keep it
   for tractability or upgrade Python to match the reference. If the
   deviation looks like a bug (e.g. arithmetic that double-counts a term),
   patch Python first.

3. **Trace a single Python run** with `evaluate()` monkey-patched to
   capture `(x, value)` pairs. Look at the first 20 calls. This catches
   non-obvious behaviour — base-point shifts, fallback-step kicks,
   model-fit failures — that won't be visible from reading the code alone.

4. **Read the HumpDay JavaScript port** with both the reference and the
   (now-aligned) Python port open. Note deviations from either.

5. **Port method-for-method**. Use the same function names with a
   `_camelCase` prefix in JS (e.g. `_initBOBYQAPoints`, `_buildBOBYQAModel`,
   `_solveBoundConstrainedTR`). One-to-one mapping makes review easier and
   catches missed methods.

6. **Match the arithmetic verbatim** with whichever port has been brought
   into alignment with the reference. If Python has a peculiarity that
   matches the reference (e.g. BOBYQA's `new_pos = XPT[kopt] + (xnew - xbase)`
   if that's what Powell's Fortran does), JS must do the same. If the
   peculiarity is a Python bug that drifts from the reference, fix
   Python *and* don't propagate it to JS.

7. **Mirror the exception fallbacks**. The Python ports use
   `except Exception:` to fall back from a failing surrogate fit to a
   finite-difference gradient + identity Hessian. JS's Gaussian elimination
   returns `null` instead of throwing — the JS port has to check for that
   and route to the same fallback. **This was the single largest fix in
   PR #83** — without it the JS BOBYQA terminated after 8 evals while
   Python continued to 66.

8. **Avoid normal-equations conditioning loss**. When the regression matrix
   is exactly square (`npt == n_terms == 2n + 1` in BOBYQA, common in 2-D),
   solve `A · x = b` directly via Gauss elim. The normal-equations path
   `(AᵀA) x = Aᵀb` squares the condition number and silently produces
   wrong coefficients once the interpolation set starts clustering.

9. **Verify with the characterisation harness**. Run
   `pytest tests/test_port_behavior.py -m slow -s` and compare the new row
   for the algorithm against the baseline in
   `benchmarks/port_characterisation.json`. The ratio is a sanity check,
   not the criterion — see "How to know we're done" below.

---

## Common JS gotchas the recipe will keep catching

- **Square-matrix Gauss elim returns null on singular**, not an exception.
  Wrap calls accordingly. Python raises.
- **No `_A.linalg.qr` / `_A.linalg.eigh` in JS.** Diagonal-Hessian models
  (BOBYQA, UOBYQA) don't need them — the eigenvalues are just the diagonal
  entries. Full-Hessian algorithms (NEWUOA) will need a small SPD-check
  helper or a 2-D-only specialisation.
- **`Math.random()` ≠ `numpy.random.standard_normal()`.** Algorithms that
  use Gaussian noise (CMA-ES, ES) need an explicit Box-Muller transform in
  JS, not `Math.random()`.
- **Class declaration scoping**. The four per-family modules each used to
  declare `const Optimizer = ...` at top level — fine in Node, breaks in the
  browser because regular `<script>` tags share global scope. Pattern is
  fixed (PR #77) — assign to `globalThis` inside the `if module.exports`
  branch instead. Don't reintroduce.

---

## Next target: PRIMA_NEWUOA

Picked next because:

- Shares infrastructure with PRIMA_BOBYQA (the same `_solveLeastSquaresStatic`,
  same fallback pattern, same model structure modulo bounds).
- Current 20× gap is the largest remaining among algorithms with healthy
  Python ports.
- NEWUOA's interpolation set is also 2n+1 points but **without** bound
  constraints, so the model fit is simpler than BOBYQA's — should be a
  more localised change.

Estimated effort: ~half of BOBYQA's, since most of the helpers
(`_solveLeastSquaresStatic`, `_finiteDifferenceGradient*`) already exist
on the BOBYQA class and can be lifted to a module-scope helper or copied.

After NEWUOA: **PRIMA_UOBYQA** (small gap, but trivially same recipe);
**Powell** and **BayesianOpt** are larger rewrites because their JS ports
diverge structurally, not just numerically.

The Python-dominant-JS trio (AntColonyOpt, TabuSearch, HillClimbing) is
deferred to the end — fixing those means improving the **Python** port to
match the **JS**, which is a different exercise and may reveal real bugs
in the Python side.

---

## How to know we're done

**Definition of done: algorithmic equivalence within reason.** The JS port
runs the same algorithm as the Python port — same control flow, same
arithmetic, same fallbacks — and produces the same result up to
floating-point noise from arithmetic-order or library differences.

The performance ratio in `benchmarks/port_characterisation.json` is a
symptom, not the criterion. A port could pass an arbitrary ratio
threshold by luck without being faithful; conversely a truly faithful
port might land at ratio 1.02× because of floating-point order, and that
should still pass.

Concrete checks per algorithm, *all* required:

1. **Same algorithm structure.** Public + private methods in JS map
   1-to-1 to those in Python (same names, same responsibilities). No
   helper exists in only one port unless it's a language-specific
   plumbing concern (e.g. `_solveLeastSquaresStatic` replacing
   `_A.linalg.qr` + `_A.linalg.solve`).

2. **Same control flow on the same starting point.** Trace Python and JS
   on the same objective with the same start. The number of evaluations
   should match, and the first ~20 evaluated points should match in both
   coordinates to ~3 significant figures.

3. **Same fallback behaviour.** Each `except Exception:` in Python has a
   parallel `if (!result) ...` or `try { ... } catch ...` in JS that
   routes to the same fallback.

4. **Final value within floating-point reason.** When step 2 holds, the
   final reported `best_value` differs only by accumulated floating-point
   noise (typically ≤ 1–2% relative). If the final value diverges by an
   order of magnitude despite (1)–(3) above, there's a missing case.

5. **Documented equivalence.** Either removed from
   `KNOWN_DIVERGENT_PORTS` in `tests/test_js_parity.py`, or the comment
   updated to "deterministic equivalence — both ports take the same
   path, tied in pairwise comparisons" (as for PRIMA_BOBYQA).

When all 9 algorithms satisfy (1)–(5), issue #78 closes.
