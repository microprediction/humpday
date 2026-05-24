# Humpday: Pure Python or Javascript Derivative-Free Optimization

[![Tests](https://github.com/microprediction/humpday/workflows/tests/badge.svg)](https://github.com/microprediction/humpday/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JavaScript](https://img.shields.io/badge/javascript-ES6+-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)

**[Documentation & Live Demos](https://humpday.microprediction.org)**

22 derivative-free optimization algorithms in pure Python. No compilation, just numpy.

## Install & Use

```bash
pip install humpday
```

```python
from humpday import minimize

def objective(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

result = minimize(objective, bounds=[(-5, 5), (-5, 5)], method='DifferentialEvolution')
print(f"Solution: {result.x}")  # [2.0, 3.0]
```

## Algorithms

22 validated optimizers: **[See them in action](https://humpday.microprediction.org)** | **[Source code](humpday/optimizers/optimizers.py)**

Trust region methods, evolutionary algorithms, metaheuristics. Each ranked by [Elo ratings](https://microprediction.github.io/optimizer-elo-ratings/html_leaderboards/overall.html) from head-to-head tournaments.

## Comparison

| Library | Dependencies | Install Size | Global Optimizers | Pure Python |
|---------|--------------|--------------|-------------------|-------------|
| **Humpday** | numpy only | ~1MB | 22 validated | ✅ |
| SciPy | C/Fortran libs | ~50MB | 3 | ❌ |
| Optuna | Many | ~100MB | 100+ | ❌ |
| Nevergrad | Many | ~200MB | 200+ | ❌ |

**Humpday's niche**: When you need optimization that works anywhere Python runs, without dependencies or compilation.

## License

MIT - Use freely in commercial and research projects.