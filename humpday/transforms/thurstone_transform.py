"""
Simplified cube-to-simplex transformations using Thurstone conventions.
Replaces the winning package dependency with thurstone-based implementations.
"""

import functools
import numpy as np
from typing import List
from scipy.stats import norm
from thurstone.conventions import STD_UNIT, STD_L

# Alternative implementation without winning package dependency
def cube_to_simplex_simple(u: List[float]) -> List[float]:
    """
    Convert point on hypercube to simplex using a simplified method.
    This replaces the winning package dependency with a direct implementation.

    :param u: point on the interior of the hyper-cube (0,1)^n
    :returns: a point p in (0,1)^{n+1} with sum(p)=1
    """
    # Convert to normal scores
    z_scores = [norm.ppf(max(1e-10, min(1-1e-10, ui))) for ui in u]
    z_scores = [0.0] + z_scores  # Add reference point

    # Convert to exponential weights (softmax-like transformation)
    exp_weights = [np.exp(z / STD_L) for z in z_scores]
    total_weight = sum(exp_weights)

    # Normalize to simplex
    probabilities = [w / total_weight for w in exp_weights]
    return probabilities


def simplex_to_cube_simple(p: List[float]):
    """
    Inverse transformation from simplex to cube.

    :param p: point in [0,1]^{n+1} with entries summing to unity
    :returns: point in (0,1)^n
    """
    # Take log ratios relative to first component
    p = np.array(p)
    p = np.maximum(p, 1e-10)  # Avoid log(0)

    # Compute log ratios
    log_ratios = [np.log(p[i] / p[0]) * STD_L for i in range(1, len(p))]

    # Convert back to uniform scores
    u = [norm.cdf(lr) for lr in log_ratios]
    return u


def lift_to_cube_simple(objective, fail_value=100000):
    """
    Modify a function's domain from the simplex to the cube.
    Uses the simplified transformation above.
    """
    @functools.wraps(objective)
    def wrapper(us):
        try:
            s = cube_to_simplex_simple(us)
        except:
            return fail_value

        try:
            return objective(s)
        except:
            warn_msg = f'WARNING: The func {objective.__name__} failed on the point {str(s)}'
            raise ValueError(warn_msg)

    return wrapper


def minimize_optimizer_on_simplex_simple(optimizer, objective, n_trials, n_dim,
                                       with_count=False, fail_value=100000,
                                       return_point_on_simplex=False, **kwargs):
    """
    Minimize objective on the n_dim-simplex using simplified transformations.

    :param optimizer: Optimizer function
    :param objective: Objective function expecting a k+1 vector
    :param n_trials: Number of trials
    :param n_dim: The manifold dimension of the simplex (1 less than actual dimension)
    :param with_count: Whether to return evaluation count
    :param fail_value: Value returned if mapping fails
    :param return_point_on_simplex: If True, return point on simplex
    :return: Same format as other optimizers
    """
    lifted_objective_on_cube = lift_to_cube_simple(objective=objective, fail_value=fail_value)
    f_best, x_best, feval_count = optimizer(lifted_objective_on_cube, n_trials=n_trials,
                                           n_dim=n_dim, with_count=True, **kwargs)

    if return_point_on_simplex:
        try:
            s_best = cube_to_simplex_simple(x_best)
        except:
            print(x_best)
            raise ValueError('Could not move optimal point back to simplex')
    else:
        s_best = np.copy(x_best)

    if with_count:
        return f_best, s_best, feval_count
    else:
        return f_best, s_best


# Simple horse racing objective that doesn't require winning package
def simple_ability_implied_dividends(ability: List[float]) -> List[float]:
    """
    Simplified version of ability to dividends conversion.
    This replaces the winning package function with a direct implementation.
    """
    ability = np.array(ability)

    # Convert abilities to probabilities using softmax
    exp_ability = np.exp(ability / STD_L)
    probabilities = exp_ability / np.sum(exp_ability)

    # Convert probabilities to dividends (inverse probabilities)
    dividends = 1.0 / np.maximum(probabilities, 1e-10)
    return dividends.tolist()


if __name__ == '__main__':
    # Test the transformations
    import random

    # Test cube to simplex and back
    for _ in range(5):
        u = [random.random() for _ in range(3)]
        s = cube_to_simplex_simple(u)
        u_back = simplex_to_cube_simple(s)

        print(f"Original u: {u}")
        print(f"Simplex s: {s} (sum={sum(s):.6f})")
        print(f"Recovered u: {u_back}")
        print(f"Error: {np.mean(np.abs(np.array(u) - np.array(u_back)))}")
        print()