# Cube-Based Interface with Rectangular Bounds

Humpday provides a cube-based interface that allows you to optimize functions on arbitrary rectangular domains using automatic transformation to the unit hypercube [0,1]^n.

## Quick Start

```python
from humpday import cube_minimize

# Define your objective function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Optimize on rectangular domain
bounds = [(-2, 2), (-1, 3)]  # x ∈ [-2,2], y ∈ [-1,3]
result = cube_minimize(rosenbrock, bounds=bounds, method='NelderMead')

print(f"Solution: {result.x}")        # [1.0, 1.0] (approximately)
print(f"Function value: {result.fun}")  # ~0 (minimum value)
```

## Main Interface

### `cube_minimize()`

The primary interface function with cube-based optimization:

```python
cube_minimize(
    fun,                    # Objective function to minimize
    x0=None,               # Initial guess (currently ignored)
    args=(),               # Extra arguments (not yet supported)  
    method='NelderMead',   # Optimization algorithm
    bounds=None,           # Rectangular bounds
    options=None           # Solver options
)
```

**Parameters:**

- **`fun`**: Objective function that takes numpy array and returns scalar
- **`method`**: Any of the 22 Humpday algorithms:
  - `'NelderMead'`, `'DifferentialEvolution'`, `'ParticleSwarm'`
  - `'PRIMA_UOBYQA'`, `'PRIMA_NEWUOA'`, `'PRIMA_BOBYQA'`
  - `'CMAEvolutionStrategy'`, `'EvolutionStrategy'`, `'GeneticAlgorithm'`
  - `'BayesianOpt'`, `'RandomSearch'`, `'Rechenberg'`
  - `'HillClimbing'`, `'CoordinateDescent'`, `'PatternSearch'`
  - `'SimulatedAnnealing'`, `'HarmonySearch'`
  - `'FireflyAlgorithm'`, `'AntColonyOpt'`, `'Powell'`, `'LBFGSB'`
- **`bounds`**: Bounds specification (see below)
- **`options`**: Dictionary with `'maxiter'` for function evaluation limit

**Returns:** `OptimizeResult` object with attributes:
- `x`: Solution array  
- `fun`: Function value at solution
- `nfev`: Number of function evaluations
- `success`: Whether optimization succeeded
- `message`: Termination description

## Bounds Specification

Three formats are supported:

### 1. List of (min, max) tuples
```python
bounds = [(0, 1), (-5, 5), (10, 20)]  # Different bounds per dimension
```

### 2. Single (min, max) tuple
```python
bounds = (-2, 2)  # Same bounds for all dimensions
```

### 3. None (default unit hypercube)
```python
bounds = None  # Uses [0,1]^n
```

## Convenience Functions

### Algorithm-Specific Functions
```python
from humpday import (
    minimize_nelder_mead,
    minimize_differential_evolution,
    minimize_particle_swarm,
    minimize_cma_es,
    minimize_prima_uobyqa
)

# Each function has the same interface:
result = minimize_nelder_mead(objective, bounds=bounds, options=options)
```

### Scalar (1D) Optimization
```python
from humpday import minimize_scalar

def f(x):
    return x**2 - 4*x + 3

result = minimize_scalar(f, bounds=(-10, 10), method='DifferentialEvolution')
print(f"Minimum at x = {result.x}")  # Should be ~2.0
```

## Domain Transformations

The interface automatically transforms between your rectangular domain and the unit hypercube [0,1]^n that Humpday algorithms use internally.

### Manual Transformations
```python
from humpday import transform_to_unit_cube, transform_from_unit_cube

# Your domain: x ∈ [-10, 10], y ∈ [0, 100]
bounds = [(-10, 10), (0, 100)]
point_real = [5, 25]  # Point in your domain

# Transform to unit cube
point_unit = transform_to_unit_cube(point_real, bounds)
# [0.75, 0.25] in [0,1]^2

# Transform back
recovered = transform_from_unit_cube(point_unit, bounds)
# [5, 25] - same as original
```

## Examples

### Multi-Modal Function
```python
def ackley(x):
    """Ackley function - challenging multimodal landscape."""
    a, b, c = 20, 0.2, 2*np.pi
    n = len(x)
    
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c*xi) for xi in x)
    
    return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.e

# Search on [-5, 5]^3
bounds = [(-5, 5)] * 3
result = scipy_minimize(ackley, bounds=bounds, method='DifferentialEvolution')

print(f"Global minimum: {result.fun:.6f}")  # Should be ~0
print(f"At point: {result.x}")              # Should be ~[0, 0, 0]
```

### Portfolio Optimization
```python
def portfolio_risk(weights):
    """Simple portfolio risk minimization."""
    # Correlation matrix (example)
    corr = np.array([[1.0, 0.3, 0.1],
                    [0.3, 1.0, 0.2], 
                    [0.1, 0.2, 1.0]])
    
    risk = np.dot(weights, np.dot(corr, weights))
    
    # Constraint: weights must sum to 1
    penalty = 1000 * (np.sum(weights) - 1)**2
    
    return risk + penalty

# Each weight between 0 and 1
bounds = [(0, 1), (0, 1), (0, 1)]
result = scipy_minimize(portfolio_risk, bounds=bounds, method='DifferentialEvolution')

print(f"Optimal weights: {result.x}")
print(f"Sum: {np.sum(result.x):.4f}")  # Should be ~1.0
```

### High-Dimensional Problem
```python
# 100-dimensional sphere function
def high_dim_sphere(x):
    return np.sum((x - 0.3)**2)  # Minimum at x = [0.3, 0.3, ..., 0.3]

bounds = [(0, 1)] * 100
result = scipy_minimize(high_dim_sphere, bounds=bounds, 
                       method='ParticleSwarm',
                       options={'maxiter': 2000})

print(f"Solution error: {np.linalg.norm(result.x - 0.3):.4f}")
```

## Algorithm Recommendations

Different algorithms work better for different problem types:

### Smooth, Low-Dimensional (n ≤ 10)
- **Nelder-Mead**: Fast, reliable for smooth functions
- **PRIMA algorithms**: Robust trust-region methods

### Multimodal, Medium-Dimensional (10 < n ≤ 50)
- **Differential Evolution**: Excellent global search
- **Particle Swarm**: Good balance of exploration/exploitation
- **CMA-ES**: Adaptive covariance, handles ill-conditioning

### High-Dimensional (n > 50)
- **CMA-ES**: State-of-the-art for continuous optimization
- **Particle Swarm**: Scalable population-based search
- **Random Search**: Simple baseline, surprisingly effective

### Noisy or Discontinuous
- **Differential Evolution**: Robust to noise
- **Simulated Annealing**: Accepts worse solutions probabilistically

## Comparison with SciPy

| Feature | Humpday | SciPy |
|---------|---------|--------|
| Pure Python | ✅ | ❌ (C/Fortran dependencies) |
| Easy Installation | ✅ | ❌ (complex build requirements) |
| Algorithm Count | 22 validated | ~15 mainstream |
| Global Optimizers | ✅ Many | ❌ Few |
| Unit Hypercube Specialization | ✅ | ❌ |
| Adaptive Selection | ✅ Elo system | ❌ |

## Under the Hood

The SciPy interface works by:

1. **Parsing bounds** into lower/upper arrays
2. **Creating wrapper** that transforms [0,1]^n to your domain
3. **Calling Humpday optimizer** on transformed problem  
4. **Transforming solution** back to your domain
5. **Returning SciPy-compatible** result object

The transformation is:
```
x_real = lower + x_unit * (upper - lower)
```

This ensures all the mathematical guarantees and optimizations of Humpday's unit hypercube algorithms apply to your arbitrary rectangular domain.

## Performance Tips

1. **Algorithm selection**: Use `adaptive_optimize()` to learn which works best for your problem type
2. **Bounds scaling**: Avoid extremely large ranges that might cause numerical issues
3. **Function evaluation budget**: Set `options={'maxiter': N}` based on problem complexity
4. **Multiple runs**: For stochastic algorithms, run multiple times and take the best result

The SciPy interface makes Humpday's powerful optimization algorithms accessible with familiar syntax, while maintaining the lightweight, dependency-free philosophy that makes Humpday easy to deploy anywhere.