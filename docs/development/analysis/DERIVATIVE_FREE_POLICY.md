# HumpDay Derivative-Free Policy

## Core Constraint: NO DERIVATIVES

HumpDay is **exclusively focused on derivative-free optimization methods**. This is a fundamental design principle that must not be violated.

## Why Derivative-Free Only?

1. **Real-world applicability**: Most practical optimization problems involve:
   - Black-box functions where gradients are unavailable
   - Noisy or discontinuous objectives
   - Simulation-based optimization
   - Machine learning hyperparameter tuning

2. **Browser compatibility**: Derivative-free methods are generally:
   - Simpler to implement in pure Python/NumPy
   - More Pyodide-compatible
   - Easier for educational demonstrations

3. **Practical focus**: Addresses the common scenario where practitioners can evaluate a function but cannot compute derivatives

## Allowed Method Categories

✅ **ALLOWED - Derivative-Free Methods:**
- Nelder-Mead simplex
- Powell's direction set method  
- COBYLA (linear approximation)
- Differential Evolution
- Simulated Annealing variants
- Basin Hopping (even though it uses L-BFGS-B internally)
- Grid search / Brute force
- Random search
- Particle Swarm Optimization
- Genetic algorithms
- Pattern search methods
- Response surface methods

❌ **FORBIDDEN - Gradient-Based Methods:**
- BFGS, L-BFGS (these are already in via Powell/SLSQP wrappers only)
- Conjugate Gradient (CG)
- Newton methods
- Trust region methods requiring Jacobians
- Any method that computes or requires derivatives

## Implementation Guidelines

When adding new optimizers:

1. **Must be derivative-free**: Only uses function evaluations f(x)
2. **Must work on black-box functions**: No assumptions about smoothness, convexity, etc.
3. **Must handle bound constraints**: All methods work on [0,1]^n hypercube
4. **Must be Pyodide-compatible**: Pure Python or NumPy/SciPy only

## Exception Handling

If a method internally uses gradients (like Basin Hopping with L-BFGS-B), it's acceptable as long as:
- The **user interface** is derivative-free
- Gradients are approximated internally via finite differences
- The method works on arbitrary black-box functions

## Enforcement

This policy should be checked when:
- Adding new optimization methods
- Reviewing pull requests  
- Writing documentation
- Designing experiments

**Remember**: The goal is practical, accessible optimization for real-world black-box problems, not academic gradient-based algorithm development.