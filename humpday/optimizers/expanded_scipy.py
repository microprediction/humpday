"""
Expanded SciPy optimizer collection for richer 3D Thurstone analysis.
All optimizers work directly on hypercube [0,1]^n (NO simplex transformation).
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing, basinhopping, brute

# DERIVATIVE-FREE optimizers working directly on hypercube [0,1]^n

def scipy_nelder_mead_cube(objective, n_trials=None, n_dim=None, with_count=False):
    """Nelder-Mead simplex method (derivative-free)"""
    if n_trials is None:
        n_trials = 100

    bounds = [(0, 1)] * n_dim
    x0 = np.random.uniform(0, 1, n_dim)

    options = {'maxfev': n_trials, 'disp': False}
    result = minimize(objective, x0, method='Nelder-Mead', options=options)

    if with_count:
        return result.fun, list(result.x), result.nfev
    else:
        return result.fun, list(result.x)


def scipy_cobyla_cube(objective, n_trials=None, n_dim=None, with_count=False):
    """COBYLA - derivative-free constrained optimizer"""
    if n_trials is None:
        n_trials = 100

    x0 = np.random.uniform(0, 1, n_dim)
    options = {'maxiter': n_trials, 'disp': False}
    result = minimize(objective, x0, method='COBYLA', options=options)

    if with_count:
        return result.fun, list(result.x), result.nfev
    else:
        return result.fun, list(result.x)


def scipy_differential_evolution_cube(objective, n_trials=None, n_dim=None, with_count=False):
    """Differential Evolution global optimizer"""
    if n_trials is None:
        n_trials = 100

    bounds = [(0, 1)] * n_dim

    # Adjust population size and generations based on budget
    popsize = max(15, min(int(n_trials / 20), 50))
    maxiter = max(1, n_trials // popsize)

    result = differential_evolution(objective, bounds, seed=42,
                                  maxiter=maxiter, popsize=popsize, disp=False)

    if with_count:
        return result.fun, list(result.x), result.nfev
    else:
        return result.fun, list(result.x)


def scipy_dual_annealing_cube(objective, n_trials=None, n_dim=None, with_count=False):
    """Dual Annealing global optimizer"""
    if n_trials is None:
        n_trials = 100

    bounds = [(0, 1)] * n_dim
    result = dual_annealing(objective, bounds, seed=42, maxfun=n_trials, no_local_search=False)

    if with_count:
        return result.fun, list(result.x), result.nfev
    else:
        return result.fun, list(result.x)


def scipy_basinhopping_cube(objective, n_trials=None, n_dim=None, with_count=False):
    """Basin Hopping global optimizer"""
    if n_trials is None:
        n_trials = 100

    x0 = np.random.uniform(0, 1, n_dim)
    niter = max(5, n_trials // 10)

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": [(0, 1)] * n_dim,
                       "options": {"maxfun": 10}}

    result = basinhopping(objective, x0, niter=niter,
                         minimizer_kwargs=minimizer_kwargs, seed=42)

    if with_count:
        return result.fun, list(result.x), result.nfev
    else:
        return result.fun, list(result.x)


def scipy_brute_cube(objective, n_trials=None, n_dim=None, with_count=False):
    """Brute force grid search"""
    if n_trials is None:
        n_trials = 100

    # Calculate grid points per dimension based on budget
    points_per_dim = max(3, int(n_trials ** (1.0 / n_dim)))
    grid_points = min(points_per_dim, 10)

    ranges = [(0, 1)] * n_dim
    result = brute(objective, ranges, Ns=grid_points, disp=False, finish=None)
    best_value = objective(result)

    if with_count:
        return best_value, list(result), grid_points ** n_dim
    else:
        return best_value, list(result)


def simple_random_search_cube(objective, n_trials=None, n_dim=None, with_count=False):
    """Pure random search on hypercube"""
    if n_trials is None:
        n_trials = 100

    best_value = float('inf')
    best_point = None

    for _ in range(n_trials):
        x = np.random.uniform(0, 1, n_dim)
        value = objective(x)

        if value < best_value:
            best_value = value
            best_point = x

    if with_count:
        return best_value, list(best_point), n_trials
    else:
        return best_value, list(best_point)


def simple_grid_search_cube(objective, n_trials=None, n_dim=None, with_count=False):
    """Simple grid search on hypercube"""
    if n_trials is None:
        n_trials = 100

    points_per_dim = max(2, int(n_trials ** (1.0 / n_dim)))
    ranges = [np.linspace(0, 1, points_per_dim) for _ in range(n_dim)]

    best_value = float('inf')
    best_point = None
    evaluations = 0

    import itertools
    for point in itertools.product(*ranges):
        if evaluations >= n_trials:
            break

        x = np.array(point)
        value = objective(x)
        evaluations += 1

        if value < best_value:
            best_value = value
            best_point = x

    if with_count:
        return best_value, list(best_point), evaluations
    else:
        return best_value, list(best_point)


# Collection of DERIVATIVE-FREE optimizers working on hypercube [0,1]^n
EXPANDED_SCIPY_OPTIMIZERS = [
    scipy_nelder_mead_cube,           # Simplex method
    scipy_cobyla_cube,                # Linear approximation
    scipy_differential_evolution_cube, # Evolutionary algorithm
    scipy_dual_annealing_cube,        # Simulated annealing
    scipy_basinhopping_cube,          # Basin hopping
    scipy_brute_cube,                 # Grid search
    simple_random_search_cube,        # Monte Carlo
    simple_grid_search_cube           # Systematic sampling
]


if __name__ == "__main__":
    # Test all methods on hypercube (no simplex transformation)
    from humpday.objectives.classic import rosenbrock_on_cube

    print("Testing expanded optimizers on hypercube [0,1]^n:")
    for opt in EXPANDED_SCIPY_OPTIMIZERS:
        try:
            result = opt(rosenbrock_on_cube, n_trials=10, n_dim=2, with_count=True)
            print(f"  ✓ {opt.__name__.replace('_cube', '')}: {result[0]:.4f}")
        except Exception as e:
            print(f"  ✗ {opt.__name__.replace('_cube', '')}: {e}")