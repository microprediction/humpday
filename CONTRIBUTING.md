# Contributing to HumpDay

## CRITICAL: Library Intent - READ THIS FIRST

**HumpDay is a PURE PYTHON optimization library with NO external dependencies (except numpy).**

### What HumpDay IS:
- ✅ Pure Python implementations of established optimization algorithms
- ✅ Algorithms that work anywhere Python runs (no compilation needed)
- ✅ Small footprint, no dependencies beyond numpy
- ✅ Both Python and JavaScript implementations of the same algorithms

### What HumpDay is NOT:
- ❌ Wrappers around 3rd party optimization libraries (scipy, cvxpy, nlopt, etc.)
- ❌ Bindings to compiled optimization code
- ❌ Dependencies on external optimization packages

## Implementation Rules

### 1. Algorithm Implementation
- **Study reference implementations/papers** to understand the algorithm mathematically
- **Implement the algorithm correctly** in pure Python + numpy
- **Never import optimization libraries** in main implementation code
- **Match algorithmic behavior** of reference implementations, don't just make up your own version

### 2. Testing and Validation
- **Use 3rd party packages ONLY in testing** for validation/comparison
- **Compare against reference implementations** (e.g., SciPy, PDFO, etc.) in tests
- **Validate algorithmic correctness**, not just performance
- **Test that your pure Python implementation produces similar results** to established libraries

### 3. What This Means Practically

**WRONG:**
```python
# In main code - DON'T DO THIS
from scipy.optimize import minimize
def my_optimizer():
    return minimize(...)  # This is a wrapper, not implementation
```

**RIGHT:**
```python
# In main code - DO THIS
def my_optimizer():
    # Pure Python implementation of the algorithm
    for iteration in range(max_iter):
        # ... implement the actual algorithm steps
```

**Testing - THIS IS CORRECT:**
```python
# In test files - this is fine
def test_my_optimizer_vs_scipy():
    from scipy.optimize import minimize
    # Compare my implementation vs scipy to verify correctness
```

## Why This Matters

1. **Universal Compatibility**: Works anywhere Python runs
2. **No Dependencies**: Easy to install and deploy
3. **Educational Value**: Shows how algorithms actually work
4. **Reliability**: No external library version conflicts

## Common Mistakes to Avoid

1. **Testing algorithm X against algorithm Y**: Don't test PRIMA against Nelder-Mead - they're different algorithms!
2. **Using 3rd party in main code**: Keep imports out of implementation files
3. **Making up your own algorithm**: Implement established methods correctly
4. **Ignoring reference implementations**: Always validate against known-good implementations

## Adding New Algorithms

1. **Research**: Find paper/reference implementation
2. **Understand**: Study the mathematical algorithm
3. **Implement**: Write pure Python version
4. **Test**: Compare against reference implementation in tests
5. **Document**: Clear docstring explaining the method

Remember: The goal is to provide working optimization algorithms that don't require any external dependencies!