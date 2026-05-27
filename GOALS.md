# HumpDay Goals & Status

Single source of truth for what's done and what's left, across all 22 algorithms.
**Supersedes** `RESUME.md` (stale, from a prior session) and `NUMPY_REMOVAL_RESUME.md` (focused on one work-stream; merge into this file when its goal column is fully ✅).

Legend: ✅ done · 🟡 partial / in-progress · ❌ not done · ❓ unknown / unverified

## Status matrix

| Algorithm | 3rd party | JS | no-numpy | demo | academic | contests | int links | ext links | logic | perf |
|---|---|---|---|---|---|---|---|---|---|---|
| PRIMA_UOBYQA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| PRIMA_NEWUOA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| PRIMA_BOBYQA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| NelderMead | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| Powell | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| LBFGSB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ❓ |
| DifferentialEvolution | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| ParticleSwarm | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| SimulatedAnnealing | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| GeneticAlgorithm | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| RandomSearch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| BayesianOpt | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ❓ |
| CMAEvolutionStrategy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| TabuSearch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| FireflyAlgorithm | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| AntColonyOpt | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| EvolutionStrategy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| HillClimbing | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| HarmonySearch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| AdaptiveRandomSearch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| CoordinateDescent | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| PatternSearch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |

## Column meanings

| Column | What "✅" means | Where to look |
|---|---|---|
| **3rd party** | Algorithm has been validated against a reference implementation (SciPy / PRIMA / etc.). | `tests/test_implementation_validation.py`, `tests/test_complete_validation.py`, PR #43 fidelity-audit notes |
| **JS** | Python output cross-validated against the JavaScript port at the same seed / inputs. | `tests/test_python_js_validation.py`, `docs/js/modules/*.js` |
| **no-numpy** | Algorithm works under `HUMPDAY_FORCE_PURE_ARRAY=1` (the shim's pure-Python backend). | `tests/test_ported_algorithms.py::PORTED` |
| **demo** | The page's "Interactive 3D Visualization" panel actually loads — Three.js + visualizer scripts wire up, `onReady` fires, the loading overlay clears. Pages that don't include a visualization panel get 🟡. | `docs/algorithms/<algorithm>.html`, `docs/js/algorithm-visualizer.js` |
| **academic** | Page uses the canonical academic-styled template (Georgia/Times serif, container card, implementation table, performance-notes box, expandable resources). Visual sanity: the `<style>` block parses (no rot), brand colors are intentional, no broken/half-stripped text. | `docs/algorithms/<algorithm>.html` |
| **contests** | Algorithm is registered in the contest framework and runnable from the live contest page. | `docs/contest.html`, `humpday/optimizers/adaptive_optimizer.py` |
| **int links** | Every internal anchor / file link on the algorithm's docs page resolves. | `docs/algorithms/<algorithm>.html`, `docs/index.html` |
| **ext links** | Every external paper / repo link still returns 200. | algorithm docs pages, `humpday/optimizers/*.py` docstrings |
| **logic** | The algorithm's implementation has been read end-to-end and matches its reference; no known numerical bugs. | code review + `tests/test_optimizers.py::test_compendium` |
| **perf** | Algorithm has a current Elo / relative-performance benchmark recorded. | `humpday/optimizers/adaptive_optimizer.py`, benchmark scripts |

## Cell-status notes (where current marks deviate from a blanket ✅)

- **All algorithms · JS** — Python↔JS parity test landed in `tests/test_js_parity.py`. For each of the 22 algorithms it runs the Python port directly and the JS port via a Node subprocess (`tests/js_parity_runner.js`) on the same 2-D sphere objective and asserts both reach the optimum. Sweep runs in under 4 seconds. The old `test_python_js_validation.py` (which skipped because of `eval()`-loading issues) was deleted.
- **EvolutionStrategy · demo** — ✅ as of PR #66: the embedded 3D demo panel was added (the page added in PR #61 was originally a minimal template). All 22 algorithms now have a working in-browser demo.
- **All algorithms · int links** — ✅ verified by audit script. Site-wide top nav (Home · Contest · Algorithms · GitHub) added to all 36 docs pages in PR #66, plus a new `docs/algorithms.html` listing page grouped by family. No remaining `../js/optimizers.js` stale references, no `inde.html` typos.
- **All algorithms · ext links** — ✅ verified by Crossref / direct-fetch audit. Six paper links that previously pointed to the WRONG paper were corrected: uobyqa (was a TSP paper → Powell 2002), bobyqa (was Hager–Zhang CG_DESCENT → Powell 2009 NA report), tabu-search (was Feo–Resende GRASP → Glover 1986), ant-colony (was the harmony-search paper → Dorigo 1996), harmony-search (replaced unresolvable Kluwer DOI → Geem 2001 canonical DOI), adaptive-random-search (dead Kluwer DOI → Solis & Wets 1981). Source-code buttons all anchor to specific class lines.
- **LBFGSB · logic** — 🟡 because the class is named L-BFGS-B but is structurally finite-difference gradient + momentum, not a faithful L-BFGS-B (which would require analytical gradients HumpDay does not have). The page now says this honestly ("Finite-Difference Quasi-Newton — HumpDay's `LBFGSB` class, a derivative-free baseline"). Either rename the class or replace its body with a real quasi-Newton implementation.
- **BayesianOpt · logic** — 🟡 because the GP kernel had broadcasting tricks (numpy path) and a separate pure-Python path with explicit loops, gated on `_A.BACKEND`. Both paths verified against the sphere, but no rigorous validation against `scikit-optimize` or `BoTorch`.
- **All algorithms · perf** — ❓ until a current Elo benchmark sweep is recorded.

## Project-level work items (separate from per-algorithm goals)

### Documentation bugs

1. **`docs/RESUME.md`** — older session notes; delete or move to a historical folder once this `GOALS.md` is fully established as the canonical doc.

### Test-infrastructure debts

1. **`test_compendium`** and **`test_portfolio`** rely on seeded `random.choice` to avoid pre-existing algorithm flakes. Keep the seeds + the `BudgetExceeded` guard in `test_compendium` — both protect against future regressions.
2. **Pure-backend subprocess test** runs all 22 ported algorithms in one forked process with `HUMPDAY_FORCE_PURE_ARRAY=1`. Currently uses `n_trials=50` to fit a 180s timeout on CI hardware; numpy-backend tests still use 200.
3. **Python↔JS parity** — `tests/test_python_js_validation.py` has 10 skipped cases. Either resurrect them or replace with a clean reference suite.

### CI sanity

1. The publish workflow (`.github/workflows/publish.yml`) uses OIDC trusted publishing. PyPI side must be configured with: owner `microprediction`, repo `humpday`, workflow filename `publish.yml`, environment `pypi`.
2. After the recent suspension incident, avoid bursts of many PRs in a short window with AI-attributed commits — that pattern looks like spam to GitHub's automation.

## How to update this file

When you finish work that flips a cell:

1. Change the cell glyph in the status matrix.
2. If the change clears a row entirely, the algorithm is "done" for that goal; consider whether the goal-column itself can be marked complete project-wide.
3. If the change adds a new project-level concern, append to the appropriate section.
4. Don't grow this file with per-PR narratives. Those belong in commit messages and PR descriptions. This is a *status* doc, not a history.

## Quick verification commands

```bash
# Numpy-optional progress: count of algorithms in PORTED
grep -c '^    (' tests/test_ported_algorithms.py | head

# Lint + format clean across the repo
uvx ruff check .
uvx ruff format --check .

# Full test suite (with pure-backend subprocess)
python -m pytest tests/ --ignore=tests/validation -q

# Force-pure smoke test for any individual algorithm
HUMPDAY_FORCE_PURE_ARRAY=1 python -c "
from humpday.optimizers.evolutionary_algorithms import CMAEvolutionStrategy
def f(x): return float(sum((xi - 0.5)**2 for xi in x))
opt = CMAEvolutionStrategy(f, n_trials=100, n_dim=5)
v, x = opt.optimize()
print(v, list(x))
"
```
