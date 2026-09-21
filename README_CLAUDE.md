# Humpday Repository Organization

This document describes the organization and structure of the Humpday optimization library repository to help contributors, users, and AI assistants understand how everything fits together.

## High-Level Structure

```
humpday/
├── humpday/                    # Main package
│   ├── __init__.py            # Main API exports
│   ├── __main__.py            # The `humpday` console command
│   ├── _array.py              # Backend shim: numpy when present, pure Python otherwise
│   ├── _prng.py               # PCG32 and the portable transcendentals shared with the JS port
│   ├── eligibility.py         # Which optimizers can use a given dimension and budget
│   ├── ratings.py             # Reads the recorded tournament in humpday/data/ratings.json
│   ├── optimizers/            # The algorithms, one module per family
│   ├── objectives/            # Test functions and benchmarks
│   └── transforms/            # Cube, simplex and bounds mappings
├── benchmarks/                 # record_ratings.py: writes the tournament the package ships
├── docs/                       # The published site, including the JavaScript port
├── examples/                   # Usage examples
├── example_applications/       # ~80 worked problems, each with a browser demo
├── experiments/                # Research experiments and validation
├── papers/                     # Academic paper materials
├── parity/                     # Transition vectors shared by the Python and JS twins
├── tests/                      # Test suite
└── pyproject.toml              # Project configuration
```

## Core Package (`humpday/`)

### Main Interface (`__init__.py`)
- **Primary functions**: `suggest()`, `minimize()`, `pure_optimize()`
- **New adaptive system**: `adaptive_optimize()`, `EloRatingSystem`
- **Algorithm registry**: `PURE_OPTIMIZERS`, `ALGORITHM_NAMES`

The main interface provides simple functions for optimization without requiring deep knowledge of the underlying algorithms.

### Optimizers (`humpday/optimizers/`)

**Core Files:**
- `base.py` - `Optimizer`, the ask/tell loop, the budget cap and the best-point bookkeeping
- `prima_algorithms.py` - UOBYQA, NEWUOA, BOBYQA
- `scipy_algorithms.py` - NelderMead, Powell, LBFGSB (implemented here, not called out to SciPy)
- `evolutionary_algorithms.py` - DE, PSO, CMA-ES, ES, GA, SA, Harmony, Firefly, ACOR, Bayesian
- `search_algorithms.py` - Rechenberg, CoordinateDescent, PatternSearch, HillClimbing, GridSearch,
  RandomSearch
- `alloy.py` - Alloy, the portfolio-of-methods optimizer
- `alloptimizers.py` - the `PURE_OPTIMIZERS` registry and `pure_optimize`
- `adaptive_optimizer.py` - Elo rating system for algorithm selection
- `scipy_interface.py` - `minimize`, `OptimizeResult` and the bounds transforms

There is no `optimizers.py`, and there are no wrapper modules: `expanded_scipy.py`,
`comprehensive_derivative_free.py`, `primacube.py`, `pysotcube.py` and `nloptcube.py` were all
removed along with the third-party packages behind them.

**Current Philosophy:** every algorithm is implemented here, and `pip install humpday` has no
dependencies at all. numpy is the `fast` extra: `humpday/_array.py` dispatches to it when it is
installed and to a pure-Python backend when it is not, and the algorithms call the shim either
way. `PURE_OPTIMIZERS` holds 23 algorithms, all of which have a JavaScript counterpart in
`docs/js/modules/`.

### Objectives (`humpday/objectives/`)

**DO NOT redefine test functions!** Use existing implementations.

**Core Files:**
- `deapobjectives.py` - Standard benchmark functions (sphere, rosenbrock, ackley, rastrigin, etc.)
- `classic.py` - Cube-normalized versions of standard functions
- `bbob_inspired_suite.py` - BBOB-style benchmark suite
- `stochastic_surfaces.py` - Noisy/stochastic test problems

**Usage Pattern:**
```python
from humpday.objectives.deapobjectives import sphere, rosenbrock, ackley
# Functions return tuples: (value,)
result = sphere([0.1, 0.2])[0]  # Extract the value
```

### Ratings and eligibility (`humpday/ratings.py`, `humpday/eligibility.py`)
- `ratings.py` reads the recorded tournament: per (dimension, budget, suite) cell, an Elo table
  over the optimizers that raced there. Nothing is interpolated across dimensions.
- `eligibility.py` says which optimizers can use a given dimension and budget at all, and
  `recommend` picks one, optionally trading quality against wall-clock cost.
- `benchmarks/record_ratings.py` is what writes the table; it is not imported by the package.

## Documentation (`docs/`)

**Structure:**
- `algorithm-visualization-demo.html` - Interactive 3D visualization demo
- `contest.html` - Algorithm comparison interface
- `js/modules/` - the JavaScript port of the algorithms, loaded as plain script tags
- `algorithms/`, `applications/` - one page per algorithm, one per worked problem
- `adaptive-optimization.md` - Guide to the Elo rating system

**Key Pages:**
- Main demo with embeddable 3D visualization
- Speed-controlled algorithm demonstrations
- CSP-secured interfaces

## Examples (`examples/`)
- `adaptive_optimization_example.py` - Complete example of Elo system usage
- Working code demonstrating proper API usage

## Experiments (`experiments/`)
- Research code for algorithm validation
- Benchmark comparisons
- Performance studies

## Tests (`tests/`)

**Structure:**
- `integration/` - End-to-end algorithm tests
- `performance/` - Speed and accuracy benchmarks  
- `validation/` - JavaScript vs Python validation

## Key Design Principles

### 1. Lightweight and Self-Contained
- **No dependencies at all** in the core package; numpy is the `fast` extra
- Every algorithm implemented here rather than wrapped
- No complex build requirements

### 2. JavaScript Compatibility
- All 23 algorithms have a JavaScript counterpart, and 13 of them are bit-exact twins that
  replay `parity/transition_vectors.json` point for point on the shared PCG32 stream
- The other 10 agree on behaviour, not on every last bit; #78 tracks the differences and #325
  tracks the JavaScript recommender lagging the recorded table
- Claims beyond that are not tested and should not be made

### 3. User-Friendly API
- Simple `suggest()` and `minimize()` functions for basic use
- Progressive complexity: basic → advanced → research level APIs
- Sensible defaults for all parameters

### 4. Adaptive Intelligence
- Elo rating system learns which algorithms work best
- Objective generators create diverse test problems
- Data-driven algorithm recommendations

## Algorithm Organization

### The 23 Algorithms

Every one subclasses `Optimizer` in `humpday/optimizers/base.py`, and
`humpday.ALGORITHM_NAMES` is the registry itself rather than a copy of it:

**Derivative-Free Methods:**
1. `PRIMA_UOBYQA` - Trust region, unconstrained
2. `PRIMA_NEWUOA` - Interpolation-based  
3. `PRIMA_BOBYQA` - Bound-constrained
4. `NelderMead` - Simplex method
5. `Powell` - Conjugate directions

**Gradient-Based:**
6. `LBFGSB` - Limited memory BFGS

**Evolutionary:**
7. `DifferentialEvolution` - DE algorithm
8. `ParticleSwarm` - PSO algorithm  
9. `CMAEvolutionStrategy` - CMA-ES
10. `EvolutionStrategy` - Basic ES
11. `GeneticAlgorithm` - Simple GA

**Metaheuristic:**
12. `BayesianOpt` - Simplified Bayesian optimization
13. `RandomSearch` - Pure random sampling
14. `GridSearch` - Regular Cartesian grid (baseline)
15. `Rechenberg` - (1+1)-ES with 1/5-success-rule (formerly `AdaptiveRandomSearch`)
16. `HillClimbing` - Local search with restarts
17. `CoordinateDescent` - Coordinate-wise optimization
18. `PatternSearch` - Hooke-Jeeves direct search
19. `SimulatedAnnealing` - Simulated annealing
20. `HarmonySearch` - Harmony search
21. `FireflyAlgorithm` - Firefly algorithm
22. `AntColonyOpt` - Socha-Dorigo continuous ACOR
23. `Alloy` - a portfolio of the above, spending its budget across several

### Algorithm Selection Philosophy

1. **Auto-selection**: Use `adaptive_optimize()` to learn best algorithms
2. **Manual selection**: Use `suggest_pure()` for heuristic recommendations  
3. **Direct use**: Call specific algorithms via `pure_optimize()`

## File Naming Conventions

- `snake_case` for Python files
- `kebab-case` for HTML/JS files
- `CamelCase` for class names
- `UPPER_CASE` for algorithm registries and constants

## Common Patterns

### Adding New Algorithms
1. Inherit from `Optimizer` in `humpday/optimizers/base.py`, in the module for its family
2. Implement the `_run()` generator, so the algorithm works ask/tell as well as batch
3. Add to the `PURE_OPTIMIZERS` registry in `alloptimizers.py`
4. Write the JavaScript twin in `docs/js/modules/`, and record transition vectors if it is exact
5. Give it a tier and caps in `eligibility.py`, or it cannot be recommended

### Adding New Objectives  
1. Add to appropriate file in `objectives/`
2. Follow DEAP pattern: return tuple `(value,)`
3. Ensure input is numpy array compatible
4. Domain should work on `[0,1]^n` or provide domain mapping

### API Design
- Simple interfaces for common cases
- Optional parameters with sensible defaults
- Consistent return formats: `(best_value, best_point)`
- Docstrings with clear examples

## Current Development Focus

### ✅ Completed
- Implementations of 23 algorithms, with no mandatory dependency
- Elo rating system for adaptive selection
- Interactive 3D visualization demos
- Comprehensive test functions library
- Ultra-lightweight dependency approach

### 🔄 In Progress  
- Algorithm performance analysis
- Domain-specific objective generators
- Extended validation against more benchmarks

### 📋 Future Plans
- Multi-objective optimization support
- Parallel algorithm execution
- Advanced visualization features
- Integration with more benchmark suites

## Working with the Repository

### For Contributors
1. Check `humpday/optimizers/` for the current implementations, one module per family
2. Use existing test functions from `objectives/`
3. Follow the lightweight, dependency-free philosophy: call `humpday._array`, never numpy directly
4. Test against the JavaScript implementations for consistency

### For Users
- Start with `suggest()` and `minimize()` functions
- Use `adaptive_optimize()` for automatic algorithm selection
- Refer to `examples/` for usage patterns
- Check `docs/` for interactive demonstrations

### For AI Assistants  
- **DON'T** redefine existing test functions
- **DO** use functions from `humpday.objectives.deapobjectives`
- **FOCUS** on `humpday/optimizers/`, whose modules replaced the single `optimizers.py`
- **REMEMBER** that every Python change to a trajectory needs its JavaScript twin updated in
  lockstep, and `parity/record_transition_vectors.py` re-run
- **DON'T** import numpy in package code; use the `humpday._array` shim, or the install with no
  extras breaks
- **LEVERAGE** the existing adaptive optimization system instead of building new selection logic

This organization reflects the evolution from a complex multi-dependency system to a streamlined, self-contained optimization library focused on reliability and ease of use.