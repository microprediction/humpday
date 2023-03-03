import functools
from winning.lattice_conventions import STD_UNIT, STD_L
from winning.std_calibration import std_ability_implied_state_prices, std_state_price_implied_ability
from scipy.stats import norm
from typing import List
import numpy as np

# Line by line explanation at https://medium.com/@mike.roweprediger/how-to-change-the-domain-of-a-function-from-a-simplex-to-a-cube-593161ab55aa

def cube_to_simplex(u: List[float]) -> List[float]:
    """
       :param  u is a point on the interior of the hyper-cube (0,1)^n
       :returns  a point p in (0,1)^{n+1} with sum(p)=1

    """
    a = [0] + [-norm.ppf(ui) for ui in u]
    p = std_ability_implied_state_prices(a, L=5 * STD_L, unit=STD_UNIT)
    return p


def simplex_to_cube(p: List[float]):
    """ The inverse map

         :param    p in [0,1]^{n+1}  with entries summing to unity
         :returns  (0,1)^n

    """
    x_mean_zero = std_state_price_implied_ability(p,L=5 * STD_L, unit=STD_UNIT)
    offset = x_mean_zero[0]
    a = [xi - offset for xi in x_mean_zero]
    return [norm.cdf(ai) for ai in a[1:]]


def lift_to_cube(objective, fail_value=100000):
    """ Modify a function's domain from the simplex to the cube

        Useful for minimizing on a simplex, even if the optimizer expects a rectangular domain or does not
        allow linear constraints.

        Remark: This will lower the dimension of the objective's input by 1
        Remark: Won't work for maximizing unless you set fail_vaue to negative

    See https://microprediction.medium.com/a-new-family-of-diffeomorphisms-from-the-simplex-to-the-cube-with-application-to-global-6d358714f429

    :param objective:
    :return:
    """

    # func defined on simplex
    @functools.wraps(objective)
    def wrapper(us):
        try:
            s = cube_to_simplex(us)
            # s is a point on the simplex in one higher dimension
        except:
            return fail_value

        try:
            return objective(s)
        except:
            warn_msg = 'WARNING: The func ' + objective.__name__ + ' failed on the point ' + str(s)
            raise ValueError(warn_msg)

    return wrapper


def minimize_optimizer_on_simplex(optimizer, objective, n_trials, n_dim,
                                  with_count=False, fail_value=100000,
                                  return_point_on_simplex=False, **kwargs):
    """
           Minimize objective on the (interior of the) n_dim-simplex in n_dim+1 dimensions by
           optimizing an objective function lifted to the n_dim-cube.

           See above. Not great for corners.

      :param optimizer:
      :param objective:   Expects a k+1 vector
      :param n_trials:
      :param n_dim:       The manifold dimension of the simplex (1 less)
      :param with_count:
      :param fail_value:  The value returned if the mapping fails
      :param return_point_on_simplex: If True, will return point on simplex. Otherwise the image on the cube. 
      :return: Same as any other optimizer
    """
    lifted_objective_on_cube = lift_to_cube(objective=objective, fail_value=fail_value)
    f_best, x_best, feval_count = optimizer(lifted_objective_on_cube, n_trials=n_trials, n_dim=n_dim, with_count=True, **kwargs)

    if return_point_on_simplex:
        try:
            s_best = cube_to_simplex(x_best)
        except:
            print(x_best)
            raise ValueError('Could not move optimal point back to simplex')
    else:
        s_best = np.copy(x_best)
    if with_count:
        return f_best, s_best, feval_count
    else:
        return f_best, s_best


if __name__=='__main__':
    from humpday.objectives.chatgptobjectives import CHATGPT_OBJECTIVES
    from humpday.optimizers.alloptimizers import OPTIMIZERS
    import random
    while True:
        opt = random.choice(OPTIMIZERS)
        objective = random.choice(CHATGPT_OBJECTIVES)
        n_dim = random.choice([3,5,8])
        f_best, x_best = minimize_optimizer_on_simplex(optimizer=opt, objective=objective, n_dim=n_dim, n_trials=100)
        print(f_best)
