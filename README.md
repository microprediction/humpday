# Humpday: Pure Python or Javascript Derivative-Free Optimization

[![CI](https://github.com/microprediction/humpday/workflows/CI/badge.svg)](https://github.com/microprediction/humpday/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**[Documentation & Live Demos](https://humpday.microprediction.org)**

22 derivative-free optimization algorithms in pure Python. No compilation, no required dependencies.

## Install & Use

```bash
pip install humpday
```

Zero runtime dependencies. Every algorithm has a pure-Python implementation
that works wherever Python runs.

If you want the numpy-accelerated backend for higher dimensions:

```bash
pip install humpday[fast]
```

The same algorithm code runs either way; humpday transparently uses numpy
when it's available and falls back to pure Python when it isn't.

```python
from humpday import minimize

def objective(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

result = minimize(objective, bounds=[(-5, 5), (-5, 5)], method='DifferentialEvolution')
print(f"Solution: {result.x}")  # [2.0, 3.0]
```

## Algorithms

22 validated optimizers: **[See them in action](https://humpday.microprediction.org)** | **[Source code](humpday/optimizers/alloptimizers.py)**

Trust region methods, evolutionary algorithms, metaheuristics.

## Comparison

Marginal install footprint on top of a Python environment that already has numpy:

| Library | Adds on top of numpy | Global optimizers |
|---------|---------------------:|------------------:|
| **Humpday** | **~1 MB** (or zero without `[fast]`) | **22** |
| SciPy       | ~100 MB | 6 documented |
| Optuna      | ~30 MB  | 11 samplers |
| Nevergrad   | ~230 MB | 540+ registered (tuned variants of ~30 base methods) |

**Humpday's niche**: when you need optimization that works anywhere Python runs, without dependencies or compilation.

## License

MIT - Use freely in commercial and research projects.
