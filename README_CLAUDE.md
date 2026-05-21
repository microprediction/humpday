# Humpday Repository Organization

This document describes the organization and structure of the Humpday optimization library repository to help contributors, users, and AI assistants understand how everything fits together.

## High-Level Structure

```
humpday/
├── humpday/                    # Main package
│   ├── __init__.py            # Main API exports
│   ├── optimizers/            # Optimization algorithms
│   ├── objectives/            # Test functions and benchmarks  
│   └── analysis/              # Performance analysis tools
├── docs/                      # Documentation and demos
├── examples/                  # Usage examples
├── experiments/               # Research experiments and validation
├── tests/                     # Test suite
├── paper/                     # Academic paper materials
└── pyproject.toml            # Project configuration
```

## Core Package (`humpday/`)

### Main Interface (`__init__.py`)
- **Primary functions**: `suggest()`, `minimize()`, `pure_optimize()`
- **New adaptive system**: `adaptive_optimize()`, `EloRatingSystem`
- **Algorithm registry**: `PURE_OPTIMIZERS`, `ALGORITHM_NAMES`

The main interface provides simple functions for optimization without requiring deep knowledge of the underlying algorithms.

### Optimizers (`humpday/optimizers/`)

**Core Files:**
- `optimizers.py` - **THE MAIN FILE**: Contains all 22 validated pure Python algorithms
- `alloptimizers.py` - Wrapper functions for backward compatibility
- `adaptive_optimizer.py` - Elo rating system for algorithm selection

**Legacy Files (mostly deprecated):**
- `expanded_scipy.py` - SciPy-based optimizers (being phased out)
- `comprehensive_derivative_free.py` - External package wrappers (deprecated)
- `primacube.py` - PRIMA algorithm wrappers (deprecated)

**Current Philosophy:** Pure Python implementations only, no external dependencies beyond numpy. All algorithms are in `optimizers.py` with the registry `PURE_OPTIMIZERS` containing exactly 22 algorithms that match the JavaScript implementations.

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

### Analysis (`humpday/analysis/`)
- Performance analysis tools
- Algorithm categorization
- Results processing

## Documentation (`docs/`)

**Structure:**
- `algorithm-visualization-demo.html` - Interactive 3D visualization demo
- `contest.html` - Algorithm comparison interface
- `js/` - JavaScript implementations of algorithms (for web demos)
- `adaptive-optimization.md` - Guide to new Elo rating system

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
- **Only numpy dependency** in core package
- Pure Python implementations preferred over external packages
- No complex build requirements

### 2. JavaScript Compatibility
- 22 algorithms exactly match JavaScript implementations
- Validation tests ensure consistency
- Web demos use identical algorithm logic

### 3. User-Friendly API
- Simple `suggest()` and `minimize()` functions for basic use
- Progressive complexity: basic → advanced → research level APIs
- Sensible defaults for all parameters

### 4. Adaptive Intelligence
- Elo rating system learns which algorithms work best
- Objective generators create diverse test problems
- Data-driven algorithm recommendations

## Algorithm Organization

### The 22 Validated Algorithms

All algorithms are in `optimizers.py` with base class `BaseOptimizer`:

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
14. `AdaptiveRandomSearch` - Adaptive step sizes
15. `HillClimbing` - Local search with restarts
16. `CoordinateDescent` - Coordinate-wise optimization
17. `PatternSearch` - Direct search
18. `SimulatedAnnealing` - Simulated annealing
19. `TabuSearch` - Tabu search
20. `HarmonySearch` - Harmony search
21. `FireflyAlgorithm` - Firefly algorithm
22. `AntColonyOpt` - Ant colony optimization

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
1. Inherit from `BaseOptimizer` in `optimizers.py`
2. Implement `optimize()` method
3. Add to `PURE_OPTIMIZERS` registry
4. Test against JavaScript equivalent

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
- Pure Python implementations of 22 algorithms
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
1. Check `optimizers.py` for current algorithm implementations
2. Use existing test functions from `objectives/`
3. Follow the lightweight, dependency-free philosophy
4. Test against JavaScript implementations for consistency

### For Users
- Start with `suggest()` and `minimize()` functions
- Use `adaptive_optimize()` for automatic algorithm selection
- Refer to `examples/` for usage patterns
- Check `docs/` for interactive demonstrations

### For AI Assistants  
- **DON'T** redefine existing test functions
- **DO** use functions from `humpday.objectives.deapobjectives`
- **FOCUS** on `optimizers.py` for algorithm implementations
- **REMEMBER** the 22-algorithm limit and JavaScript compatibility requirements
- **LEVERAGE** the existing adaptive optimization system instead of building new selection logic

This organization reflects the evolution from a complex multi-dependency system to a streamlined, self-contained optimization library focused on reliability and ease of use.