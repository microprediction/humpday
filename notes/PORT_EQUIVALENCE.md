# Reference ↔ Python ↔ JavaScript equivalence work

Working notes for issue [#78](https://github.com/microprediction/humpday/issues/78).

**Goal: all three of (a) a reliable, trusted, third-party reference
implementation, (b) the HumpDay Python port, and (c) the HumpDay JavaScript
port are equivalent within reason.** Not just "Python == JS" — the
**reference implementation** is the target. A bug in the Python port
mirrored faithfully in the JS port is *not* done. Both ports must track
the reference.

## Rule of thumb (read this carefully)

**Do not implement from a paper.** Find a trusted reference implementation
and port that. A published paper hand-waves over corner cases that the
reference code has had to deal with; a port from the paper will reproduce
the paper's claims but lose the years of bug-fix history baked into the
reference.

For each algorithm, the trusted reference is one of:

| Family | Reference implementation |
|---|---|
| Nelder-Mead, Powell (conj. dir.), L-BFGS-B, Differential Evolution, dual annealing | [SciPy](https://github.com/scipy/scipy/tree/main/scipy/optimize) (read the actual Python/Fortran source under `scipy/optimize/`) |
| Bayesian optimisation (GP-based) | [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) — `gp_minimize` |
| BOBYQA | [Py-BOBYQA](https://github.com/numericalalgorithmsgroup/pybobyqa) — pure-Python, MIT, by Cartis & Roberts. NAG-sponsored. |
| NEWUOA, UOBYQA, BOBYQA (alternative) | [libprima/prima](https://github.com/libprima/prima) — Zaikun Zhang's modern Fortran rewrite of Powell's code. Use as a Fortran reference to port the *structure* into Python. |
| CMA-ES | [`cmaes`](https://github.com/CyberAgentAILab/cmaes) — pure-Python, MIT, or Hansen's [`pycma`](https://github.com/CMA-ES/pycma) |
| Particle Swarm | [`pyswarms`](https://github.com/ljvmiranda921/pyswarms) — pure-Python, MIT |
| Genetic Algorithm | [DEAP](https://github.com/DEAP/deap), [pymoo](https://github.com/anyoptimization/pymoo) — both pure-Python |
| Simulated Annealing | [`scipy.optimize.dual_annealing`](https://github.com/scipy/scipy/blob/main/scipy/optimize/_dual_annealing.py) |
| Firefly, Tabu, Harmony, Ant Colony | various pure-Python packages, quality varies; pick the most-cited / best-maintained |

## For each algorithm

1. **Identify the reference implementation** from the table above. If
   none of the listed packages fits, find one (Google Scholar for cites,
   PyPI for downloads, GitHub stars for community vetting).
2. **Read the reference implementation's source code**, not the paper.
   Pin down its parameters, initialisation, control flow, and fallback
   rules from the code.
3. **Read the HumpDay Python port** with the reference source open in
   another pane. Note every deviation. Patch Python to match.
4. **Read the HumpDay JavaScript port** the same way. Patch JS to match
   the (now-aligned) Python.
5. **Verify** with `tests/test_port_behavior.py` and trace inspection.

The published paper is useful **only** as background. Implementation
detail comes from the reference codebase.

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

1. **Find the reference implementation** from the table at the top of this
   file (or, if not listed, locate a trusted, well-cited, well-maintained
   one). Clone or `pip install`. **Do not work from a paper** — paper
   pseudocode hand-waves over corner cases that the reference code has
   solved over years. The code is the reference, not the prose.

2. **Read the reference source code** — actual `.py` or `.f90` — and
   write down: parameter defaults, initialisation, control flow,
   fallback rules, point-update rule.

3. **Read the HumpDay Python port** with the reference next to it. Note
   every deviation. If the deviation looks like a simplification (e.g.
   diagonal Hessian instead of full symmetric), decide whether to keep
   it for tractability or upgrade Python to match the reference. If the
   deviation looks like a bug, patch Python to match the reference.

4. **Trace a single Python run** with `evaluate()` monkey-patched to
   capture `(x, value)` pairs. Look at the first 20 calls. This catches
   non-obvious behaviour — base-point shifts, fallback-step kicks,
   model-fit failures — that won't be visible from reading the code
   alone.

5. **Read the HumpDay JavaScript port** with both the reference and the
   (now-aligned) Python port open. Note deviations from either.

6. **Port method-for-method**. Use the same function names with a
   `_camelCase` prefix in JS (e.g. `_initBOBYQAPoints`, `_buildBOBYQAModel`,
   `_solveBoundConstrainedTR`). One-to-one mapping makes review easier and
   catches missed methods.

7. **Match the arithmetic verbatim** with whichever port has been brought
   into alignment with the reference. If Python has a peculiarity that
   matches the reference, JS must do the same. If the peculiarity is a
   Python bug that drifts from the reference, fix Python *and* don't
   propagate it to JS.

8. **Mirror the exception fallbacks**. The Python ports use
   `except Exception:` to fall back from a failing surrogate fit to a
   finite-difference gradient + identity Hessian. JS's Gaussian
   elimination returns `null` instead of throwing — the JS port has to
   check for that and route to the same fallback. **This was the single
   largest fix in PR #83** — without it the JS BOBYQA terminated after
   8 evals while Python continued to 66.

9. **Avoid normal-equations conditioning loss**. When the regression
   matrix is exactly square (`npt == n_terms == 2n + 1` in BOBYQA, common
   in 2-D), solve `A · x = b` directly via Gauss elim. The normal-equations
   path `(AᵀA) x = Aᵀb` squares the condition number and silently
   produces wrong coefficients once the interpolation set starts
   clustering.

10. **Verify with the characterisation harness**. Run
    `pytest tests/test_port_behavior.py -m slow -s` and compare the new
    row for the algorithm against the baseline in
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

## Next targets — and where the reference lives

The list below pairs each algorithm with the **trusted reference** to port
from. No paper implementations. If the only reference is Fortran, that's
the source — we port the Fortran structure, not the paper's pseudocode.

| Algorithm | Reference to port from |
|---|---|
| **PRIMA_BOBYQA** ← done (PR #83) | (was: ad-hoc, before the rule existed). Re-do later from [Py-BOBYQA](https://github.com/numericalalgorithmsgroup/pybobyqa) — pure-Python, MIT — or [libprima](https://github.com/libprima/prima/blob/main/fortran/bobyqa) Fortran. |
| **PRIMA_NEWUOA** ← next | [libprima Fortran NEWUOA](https://github.com/libprima/prima/tree/main/fortran/newuoa). No pure-Python reference exists; libprima's Fortran is the canonical modern implementation. |
| PRIMA_UOBYQA | [libprima Fortran UOBYQA](https://github.com/libprima/prima/tree/main/fortran/uobyqa). |
| Powell (conj. dir.) | [scipy `_minimize_powell`](https://github.com/scipy/scipy/blob/main/scipy/optimize/_optimize.py) (search for `_minimize_powell`). |
| NelderMead | [scipy `_minimize_neldermead`](https://github.com/scipy/scipy/blob/main/scipy/optimize/_optimize.py). |
| LBFGSB | Decide: rename our class (it's not L-BFGS-B) per #80, or port [scipy's `_lbfgsb_py.py`](https://github.com/scipy/scipy/tree/main/scipy/optimize) which itself wraps Zhu/Byrd/Lu/Nocedal Fortran. |
| BayesianOpt | [`scikit-optimize gp_minimize`](https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/optimizer/gp.py). |
| SimulatedAnnealing | [`scipy.optimize._dual_annealing`](https://github.com/scipy/scipy/blob/main/scipy/optimize/_dual_annealing.py). |
| DifferentialEvolution | [`scipy.optimize._differentialevolution`](https://github.com/scipy/scipy/blob/main/scipy/optimize/_differentialevolution.py). |
| CMAEvolutionStrategy | [`cmaes`](https://github.com/CyberAgentAILab/cmaes) — pure-Python, MIT. |
| ParticleSwarm | [`pyswarms`](https://github.com/ljvmiranda921/pyswarms). |
| GeneticAlgorithm | [DEAP](https://github.com/DEAP/deap) — pure-Python, LGPL. |
| HillClimbing / TabuSearch / FireflyAlgorithm / AntColonyOpt / HarmonySearch / AdaptiveRandomSearch / CoordinateDescent / PatternSearch / EvolutionStrategy / RandomSearch | Pick the most-cited / best-maintained pure-Python implementation in each case. |

Order of attack (driven by which divergent ports are blocking issue #78):

1. **PRIMA_NEWUOA** — Read libprima's Fortran `newuoa.f90` and friends.
   Port the Fortran logic into pure Python. Mirror to JS.
2. **PRIMA_UOBYQA** — Same procedure with libprima's UOBYQA.
3. **PRIMA_BOBYQA** revisit — PR #83 aligned Python and JS but predates
   this rule. Re-align both against Py-BOBYQA (pure-Python reference).
4. **LBFGSB** — pending decision on rename vs. reimplementation.
5. **Powell, BayesianOpt** — port from scipy / scikit-optimize sources.
6. **AntColonyOpt, TabuSearch, HillClimbing** — pick pure-Python references,
   port both Python and JS to them.

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
