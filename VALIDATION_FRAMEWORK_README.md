# HumpDay Cross-Validation Framework

## Overview

This comprehensive cross-validation framework provides mathematical rigor and equivalence testing for HumpDay optimization algorithms. It ensures scientific correctness and consistency across implementations.

## Key Features

### 1. **Python vs 3rd Party Validation**
- Validates Python implementations against external reference packages (SciPy, PRIMA, etc.)
- Compares convergence behavior, not just final results
- Statistical analysis across multiple runs for robustness

### 2. **JavaScript vs Python Cross-Validation**
- Cross-language implementation consistency testing
- Same algorithm must produce similar results across languages
- Handles reasonable variation due to implementation differences

### 3. **Mathematical Correctness Verification**
- Ensures implementations match literature and theory
- Validates algorithm parameters and update equations
- Tests bounds handling and termination criteria
- Verifies mathematical properties (simplex validity, trust regions, etc.)

### 4. **Statistical Validation**
- Advanced statistical tests for algorithmic equivalence
- Convergence rate analysis and path correlation
- Distribution comparison using Kolmogorov-Smirnov and Mann-Whitney tests
- Non-parametric statistical validation methods

### 5. **Comprehensive Benchmark Suite**
- Standard optimization problems with known optima
- Multiple problem classes: smooth, multimodal, noisy, constrained
- Scalable benchmark problems (2D, 5D, 10D)
- Reference implementations from literature

## Quick Start

### Basic Demo
```bash
python demo_validation_framework.py
```

### Quick Validation
```bash
python run_comprehensive_validation.py --quick
```

### Full Validation Suite
```bash
python run_comprehensive_validation.py --trials 100 --runs 5
```

## Framework Components

### Core Modules

1. **`cross_validation_framework.py`** - Main validation framework
2. **`statistical_validation.py`** - Advanced statistical analysis
3. **`benchmark_suite.py`** - Standard benchmark problems
4. **`run_comprehensive_validation.py`** - Complete test runner

### Key Classes

- `CrossValidationFramework` - Main validation orchestration
- `StatisticalValidator` - Statistical equivalence testing
- `BenchmarkSuite` - Benchmark problem management
- `ValidationResult` - Result data structure

## Validation Categories

### 1. Python vs Reference Implementation
```python
# Example: Validate PRIMA UOBYQA against PDFO reference
framework = CrossValidationFramework()
results = framework.run_python_vs_reference_validation(n_trials=100, n_runs=5)
```

**Tests:**
- Mathematical equivalence (relative error < 10%)
- Convergence correlation (> 70%)
- Statistical distribution comparison

### 2. Cross-Language Consistency
```python
# Example: JavaScript vs Python implementation consistency
js_results = framework.run_cross_language_validation(n_trials=100, n_runs=5)
```

**Tests:**
- Mean performance difference (< 20% for cross-language)
- Standard deviation consistency
- Algorithmic behavior equivalence

### 3. Mathematical Correctness
```python
# Example: Verify algorithm mathematical properties
math_results = framework.run_mathematical_correctness_validation()
```

**Tests:**
- Algorithm-specific properties (simplex validity, trust regions)
- Convergence behavior on known problems
- Bounds handling compliance
- Parameter consistency with deterministic seeds

### 4. Benchmark Performance
```python
# Example: Standard benchmark evaluation
suite = BenchmarkSuite().create_standard_suite([2, 5, 10])
results = suite.evaluate_algorithm(algorithm_func, n_runs=5, n_trials=100)
```

**Problems:**
- Sphere (smooth, unimodal)
- Rosenbrock (smooth, ill-conditioned)
- Rastrigin (multimodal)
- Ackley (multimodal)
- Noisy variants

## Command Line Options

```bash
python run_comprehensive_validation.py [options]

Options:
  --trials N        Number of trials per run (default: 100)
  --runs N          Number of independent runs per test (default: 5)
  --output DIR      Output directory (default: validation_results)
  --skip-js         Skip JavaScript cross-validation
  --quick           Quick validation with reduced parameters
  --verbose         Enable verbose output
```

## Output Reports

### Comprehensive JSON Report
```json
{
  "timestamp": "2026-05-23 14:30:00",
  "validation_summary": {
    "total_tests_run": 45,
    "tests_passed": 42,
    "overall_pass_rate": 93.3
  },
  "algorithm_performance": {
    "NelderMead": 0.85,
    "PRIMA_UOBYQA": 0.92,
    "Powell": 0.78
  },
  "recommendations": [
    "EXCELLENT: All algorithms demonstrate strong mathematical consistency",
    "Best performing algorithms: PRIMA_UOBYQA, NelderMead, PRIMA_NEWUOA"
  ]
}
```

### Console Output
```
COMPREHENSIVE VALIDATION SUMMARY
==================================================
Total tests executed: 45
Tests passed: 42
Overall pass rate: 93.3%

VALIDATION CATEGORIES
  Framework Validation: 91.7%
  Benchmark Validation: 95.0%
  Cross Reference Validation: 92.3%

ALGORITHM PERFORMANCE
  PRIMA_UOBYQA: 92.0%
  NelderMead: 85.0%
  PRIMA_NEWUOA: 83.0%
  Powell: 78.0%
  ️ LBFGSB: 65.0%
```

## Mathematical Validation Approach

### Statistical Tests Used
1. **Kolmogorov-Smirnov Test** - Distribution similarity
2. **Mann-Whitney U Test** - Median differences
3. **Pearson Correlation** - Convergence path similarity
4. **Variance Ratio Test** - Spread consistency

### Equivalence Criteria
- **Relative Error**: < 10% for same-language, < 20% for cross-language
- **Convergence Correlation**: > 70% for equivalent algorithms
- **Distribution Similarity**: p-value > 0.05 in statistical tests
- **Bounds Compliance**: All solutions must respect [0,1]^n constraints

### Benchmark Problem Validation
```python
# All benchmark problems follow this pattern:
def problem_objective(x):
    x = np.asarray(x)
    # Transform [0,1]^n to problem domain
    x_scaled = transform_to_domain(x)
    # Return objective value
    return objective_function(x_scaled)
```

## Implementation Details

### Algorithm Wrapper Pattern
```python
class BaseOptimizer:
    def __init__(self, objective, n_trials, n_dim):
        self.objective = objective
        self.n_trials = n_trials
        self.n_dim = n_dim
        self.evaluations = 0
        self.best_value = float('inf')
        self.best_x = None

    def evaluate(self, x):
        self.evaluations += 1
        value = self.objective(np.clip(x, 0, 1))
        if value < self.best_value:
            self.best_value = value
            self.best_x = x.copy()
        return value

    def optimize(self):
        # Algorithm-specific implementation
        return self.best_value, self.best_x
```

### Reference Implementation Integration
```python
def scipy_nelder_mead(objective, n_trials, n_dim):
    from scipy.optimize import minimize

    result = minimize(
        objective,
        np.random.random(n_dim),
        method='Nelder-Mead',
        bounds=[(0, 1)] * n_dim,
        options={'maxfev': n_trials}
    )

    return result.fun, np.clip(result.x, 0, 1)
```

## Usage Examples

### Example 1: Basic Algorithm Validation
```python
from cross_validation_framework import CrossValidationFramework

# Initialize framework
framework = CrossValidationFramework("results")

# Run Python vs reference validation
results = framework.run_python_vs_reference_validation(
    n_trials=100,
    n_runs=5
)

# Generate report
report = framework.generate_report()
print(f"Pass rate: {report['summary']['pass_rate']:.1f}%")
```

### Example 2: Custom Benchmark Testing
```python
from benchmark_suite import BenchmarkSuite, SphereProblem

# Create custom benchmark
suite = BenchmarkSuite()
suite.add_problem(SphereProblem(dimension=3))

# Test your algorithm
def my_algorithm(objective, n_trials, n_dim):
    # Your implementation here
    return best_value, best_x

results = suite.evaluate_algorithm(my_algorithm, n_runs=10)
```

### Example 3: Statistical Analysis
```python
from statistical_validation import StatisticalValidator

validator = StatisticalValidator(significance_level=0.05)

# Compare two algorithms' results
comparison = validator.compare_performance_distributions(
    results_algorithm_a,
    results_algorithm_b,
    "Algorithm_A",
    "Algorithm_B"
)

print(f"Algorithms equivalent: {comparison.equivalent}")
```

## Interpreting Results

### Pass Rate Guidelines
- **≥ 90%**: Excellent mathematical consistency
- **75-89%**: Good validation with minor issues
- **50-74%**: Moderate consistency, review needed
- **< 50%**: Poor validation, significant issues

### Algorithm Performance Scores
- **> 0.8**: Excellent performance
- **0.6-0.8**: Good performance
- **0.4-0.6**: Moderate performance
- **< 0.4**: Poor performance, needs improvement

### Statistical Significance
- **p-value > 0.05**: No significant difference (good for equivalence)
- **p-value < 0.05**: Significant difference (potential issue)
- **Correlation > 0.7**: Strong similarity in behavior
- **Relative error < 0.1**: Close numerical agreement

## ️ Extending the Framework

### Adding New Algorithms
```python
from humpday.optimizers.base import BaseOptimizer

class MyNewOptimizer(BaseOptimizer):
    def optimize(self):
        # Your algorithm implementation
        # Must work on [0,1]^n domain
        # Must track self.best_value and self.best_x
        return self.best_value, self.best_x

# Add to test suite
test_algorithms = {
    'MyNewOptimizer': MyNewOptimizer,
    # ... other algorithms
}
```

### Adding New Benchmarks
```python
from benchmark_suite import BenchmarkProblem, BenchmarkMetadata

class MyBenchmark(BenchmarkProblem):
    def objective(self, x):
        # Transform [0,1]^n to appropriate domain
        # Return objective value
        pass

    def _get_metadata(self):
        return BenchmarkMetadata(
            name="MyBenchmark_2D",
            dimension=2,
            optimal_value=0.0,
            optimal_point=np.array([0.5, 0.5]),
            problem_class="smooth",
            difficulty="medium",
            literature_reference="Author (Year)"
        )
```

## Troubleshooting

### Common Issues

1. **JavaScript validation fails**
   - Ensure Node.js is installed and accessible
   - Check that `web/js/optimizers.js` exists
   - Use `--skip-js` flag to bypass

2. **SciPy reference tests fail**
   - Install SciPy: `pip install scipy`
   - Some tests will use internal fallbacks

3. **Memory issues with large problems**
   - Reduce `--runs` or `--trials` parameters
   - Use `--quick` mode for testing

4. **Import errors**
   - Ensure all modules are in the same directory
   - Check Python path configuration

### Performance Optimization

- Use `--quick` for development/testing
- Reduce problem dimensions for faster execution
- Skip JavaScript validation during development
- Use parallel execution for large validation suites

## Scientific Rigor

This framework emphasizes **mathematical correctness** over pure performance comparison:

1. **Algorithmic Equivalence** - Same algorithm should behave similarly regardless of implementation language
2. **Statistical Validation** - Multiple runs with proper statistical tests
3. **Convergence Analysis** - Focus on optimization behavior, not just final results
4. **Reference Standards** - Validation against established implementations
5. **Mathematical Properties** - Verify algorithm-specific theoretical properties

The goal is to ensure HumpDay algorithms are scientifically sound and mathematically equivalent to their reference implementations.

## References

- Powell, M.J.D. (2009). "The BOBYQA algorithm for bound constrained optimization without derivatives"
- Nelder, J.A. & Mead, R. (1965). "A simplex method for function minimization"
- Powell, M.J.D. (1964). "An efficient method for finding the minimum of a function of several variables"
- Rastrigin, L.A. (1974). "Systems of extremal control"
- Ackley, D.H. (1987). "A connectionist machine for genetic hillclimbing"