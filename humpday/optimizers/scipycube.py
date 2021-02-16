from scipy.optimize import minimize
from humpday.objectives.classic import CLASSIC_OBJECTIVES
global feval_count

# TODO: standardize with SHGO and remove duplicate code
from copy import deepcopy
MINIMIZER_KWARGS = {'slsqp': {'method': 'SLSQP'},
                    'powell': {'method': 'Powell'},
                    'nelder': {'method': 'Nelder-Mead'},
                    'dogleg': {'method': 'dogleg'},
                    'lbfgsb': {'method': 'L-BFGS-B'},
}


def scipy_cube(objective, n_trials, n_dim, with_count=False, method=None):
    bounds = [(0,1) ]*n_dim

    options = deepcopy(MINIMIZER_KWARGS[method])
    options.update({'maxfev': n_trials,'maxiter':n_trials})

    global feval_count
    feval_count = 0

    def _objective(x):
        global feval_count
        feval_count +=1
        return objective(list(x))

    result = minimize(_objective, x0=[0]*n_dim, method=options['method'],bounds=bounds, options=options)
    best_x = result.x.tolist()
    best_val = _objective(result.x)
    return (best_val, best_x,  feval_count) if with_count else (best_val, best_x)


def scipy_slsqp_cube(objective, n_trials, n_dim, with_count=False ):
    return scipy_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='slsqp')


def scipy_powell_cube(objective, n_trials, n_dim, with_count=False):
    return scipy_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='powell')


def scipy_nelder_cube(objective, n_trials, n_dim, with_count=False):
    return scipy_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='nelder')


def scipy_lbfgsb_cube(objective, n_trials, n_dim, with_count=False):
    return scipy_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, method='lbfgsb')


SCIPY_OPTIMIZERS = [ scipy_slsqp_cube, scipy_powell_cube, scipy_nelder_cube, scipy_lbfgsb_cube ]


if __name__ == '__main__':
    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in SCIPY_OPTIMIZERS:
            print((optimizer.__name__,(optimizer(objective, n_trials=250, n_dim=6, with_count=True))))

