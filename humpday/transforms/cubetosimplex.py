import functools
from winning.std_calibration import std_ability_implied_state_prices
from scipy.stats import norm
from typing import List
from winning.lattice_conventions import STD_UNIT, STD_L


def cube_to_simplex(u: List[float]) -> List[float]:
    """
       :param  u is a point on the interior of the hyper-cube (0,1)^n
       :returns  a point p in (0,1)^{n+1} with sum(p)=1

    """
    a = [0] + [-norm.ppf(ui) for ui in u]
    p = std_ability_implied_state_prices(a, L=5 * STD_L, unit=0.5 * STD_UNIT)
    return p


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


def minimize_optimizer_on_simplex(optimizer, objective, n_trials, n_dim, with_count=False, fail_value=100000, **kwargs):
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
      :return: Same as any other optimizer
    """
    lifted_objective_on_cube = lift_to_cube(objective=objective, fail_value=fail_value)
    return optimizer(lifted_objective_on_cube, n_trials=n_trials, n_dim=n_dim, with_count=with_count, **kwargs)


if __name__=='__main__':
    from humpday.objectives.chatgptobjectives import CHATGPT_OBJECTIVES
    from humpday.optimizers.freelunchcube import FREELUNCH_OPTIMIZERS
    import random
    while True:
        opt = random.choice(FREELUNCH_OPTIMIZERS)
        objective = random.choice(CHATGPT_OBJECTIVES)
        n_dim = random.choice([3,5,8])
        f_best, x_best = minimize_optimizer_on_simplex(optimizer=opt, objective=objective, n_dim=n_dim, n_trials=100)
        print(f_best)