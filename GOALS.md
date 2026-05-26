# HumpDay Goals & Status

Single source of truth for what's done and what's left, across all 22 algorithms.
**Supersedes** `RESUME.md` (stale, from a prior session) and `NUMPY_REMOVAL_RESUME.md` (focused on one work-stream; merge into this file when its goal column is fully ✅).

Legend: ✅ done · 🟡 partial / in-progress · ❌ not done · ❓ unknown / unverified

## Status matrix

| Algorithm | 3rd party | JS | no-numpy | demo | academic | contests | int links | ext links | logic | perf |
|---|---|---|---|---|---|---|---|---|---|---|
| PRIMA_UOBYQA | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| PRIMA_NEWUOA | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| PRIMA_BOBYQA | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| NelderMead | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| Powell | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| LBFGSB | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ❓ |
| DifferentialEvolution | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| ParticleSwarm | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| SimulatedAnnealing | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| GeneticAlgorithm | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| RandomSearch | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| BayesianOpt | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ❓ |
| CMAEvolutionStrategy | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| TabuSearch | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| FireflyAlgorithm | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| AntColonyOpt | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| EvolutionStrategy | ✅ | ❓ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| HillClimbing | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| HarmonySearch | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| AdaptiveRandomSearch | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| CoordinateDescent | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |
| PatternSearch | ✅ | ❓ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ |

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

- **PRIMA trio · no-numpy** — three algorithms remain numpy-dependent; need `svd` added to the shim first. See "PRIMA trio port" below.
- **All algorithms · JS** — the JavaScript port exists at `docs/js/modules/*.js`, but I am unaware of an automated Python↔JS parity sweep that runs in CI. `tests/test_python_js_validation.py` exists; most of its cases are skipped (10 skips in the test inventory). Mark every cell as ❓ until that gap is verified.
- **All algorithms · demo** — `docs/index.html` Algorithm Categories section was fixed in PR #49 to list all 22 algorithms, and the 9 wrong `<em>Not in Python core</em>` rows were replaced with proper Python source links in PR #57. The visualizer panel on each `docs/algorithms/<algo>.html` was broken in PR #60 and earlier — the page's script depends on `THREE.Scene`/`OrbitControls` but never loaded three.min.js. PR #61 injects the missing `<script src="https://cdnjs.cloudflare.com/.../three.min.js">` + OrbitControls into every page that uses the visualizer; only `harmony-search.html` had this previously. **EvolutionStrategy** is 🟡 because the page (added in PR #61) currently has no embedded 3D demo panel — the algorithm class exists in the visualizer's selector but the page template was minimal.
- **All algorithms · academic** — every algorithm page restored in PR #61 uses the canonical academic template established by `b9b91ce` and `c05fd86`. Pages built on this template were briefly destroyed by `fc6a046`'s regex rot (CSS `{}` braces stripped, `padding: 20px` → `padding: 2p`, etc.) and restored in PR #61 commit `475342a`.
- **LBFGSB · logic** — 🟡 because the class is named L-BFGS-B but is structurally finite-difference gradient + momentum (no actual L-BFGS quasi-Newton). Naming is misleading; behaviour is reasonable for a derivative-free baseline. Either rename the class or genuinely implement L-BFGS quasi-Newton updates.
- **BayesianOpt · logic** — 🟡 because the GP kernel had broadcasting tricks (numpy path) and a separate pure-Python path with explicit loops, gated on `_A.BACKEND`. Both paths verified against the sphere, but no rigorous validation against `scikit-optimize` or `BoTorch`.
- **All algorithms · int links / ext links / perf** — ❓ across the board pending a sweep.

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
